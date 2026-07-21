"""Close assignment likelihood precision on a frozen, unmodified Q1 candidate."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.autoregressive_assignment import GeometryAwareRemainingCountScorer
from scripts.train_h1a_assignment_iid import _bound_rows, _bound_summary, _load_examples


def _git_identity(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("assignment precision closure requires a clean committed tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _updated_checks(
    candidate_checks: dict[str, bool],
    summaries: dict[str, dict[str, float | int]],
    acceptance: dict[str, float],
) -> dict[str, bool]:
    checks = dict(candidate_checks)
    calibration = summaries["iid_calibration_supported"]
    test = summaries["iid_test_supported"]
    checks.update(
        {
            "iid_calibration_nll_reduction": calibration[
                "relative_nll_reduction_from_uniform"
            ]
            >= acceptance["iid_calibration_relative_nll_reduction_min"],
            "iid_test_nll_reduction": test["relative_nll_reduction_from_uniform"]
            >= acceptance["iid_test_relative_nll_reduction_min"],
            "iid_calibration_paired_bootstrap": calibration[
                "model_minus_uniform_nll_ucb95"
            ]
            <= acceptance["iid_calibration_model_minus_uniform_nll_ucb95_max"],
            "iid_test_paired_bootstrap": test["model_minus_uniform_nll_ucb95"]
            <= acceptance["iid_test_model_minus_uniform_nll_ucb95_max"],
            "iid_calibration_mc_precision": calibration[
                "maximum_order_elbo_mc_standard_error"
            ]
            <= acceptance["maximum_order_elbo_mc_standard_error"],
            "iid_test_mc_precision": test["maximum_order_elbo_mc_standard_error"]
            <= acceptance["maximum_order_elbo_mc_standard_error"],
        }
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--carrier-root", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_assignment_iid_precision_closure_v1" or protocol.get(
        "status_before_run"
    ) != "frozen_not_run":
        raise ValueError("unexpected or unfrozen assignment precision protocol")
    source = protocol["source"]
    if sha256_file(args.candidate_result) != source["candidate_result_sha256"]:
        raise ValueError("candidate assignment result identity changed")
    if sha256_file(args.candidate_checkpoint) != source["candidate_checkpoint_sha256"]:
        raise ValueError("candidate assignment checkpoint identity changed")
    candidate = load_json_object(args.candidate_result)
    failed = {name for name, passed in candidate["checks"].items() if not passed}
    if candidate.get("qualified") is not False or failed != {"iid_calibration_mc_precision"}:
        raise ValueError("candidate failure is not isolated to calibration MC precision")
    training_protocol = load_json_object(repository / source["training_protocol"])
    if canonical_json_hash(training_protocol) != source["training_protocol_sha256"]:
        raise ValueError("assignment training protocol identity changed")
    if candidate["protocol_sha256"] != source["training_protocol_sha256"]:
        raise ValueError("candidate result does not belong to the pinned training protocol")
    implementation_commit = _git_identity(repository)

    seed = int(protocol["evaluation"]["seed"])
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("assignment precision closure requires CUDA")
    device = torch.device("cuda", int(protocol["evaluation"]["cuda_device"]))
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    model_config = training_protocol["model"]
    role_path = repository / training_protocol["source"]["iid_role_result"]
    examples = _load_examples(
        args.carrier_root,
        role_path,
        maximum_sites=int(model_config["maximum_sites"]),
        radial_channels=int(model_config["radial_channels"]),
    )
    role_result = load_json_object(role_path)
    actions = {
        (str(row["material_id"]), str(row["embedding_key"])): str(row["action_signature"])
        for row in role_result["carrier_rows"]
    }
    fit_actions = {
        str(row["action_signature"])
        for row in role_result["carrier_rows"]
        if str(row["role"]) in {"iid_fit", "iid_fit_rare"}
    }
    by_role: dict[str, list[Any]] = defaultdict(list)
    for example in examples:
        key = (example.material_id_audit_only, example.embedding_key)
        if actions[key] in fit_actions and example.evidence_role_audit_only in {
            "iid_calibration",
            "iid_test",
        }:
            by_role[f"{example.evidence_role_audit_only}_supported"].append(example)
    if {key: len(value) for key, value in by_role.items()} != {
        "iid_calibration_supported": 35,
        "iid_test_supported": 35,
    }:
        raise ValueError("supported-IID precision panel changed")

    first = examples[0]
    model = GeometryAwareRemainingCountScorer(
        site_feature_dim=first.site_features.shape[1],
        graph_feature_dim=first.graph_features.shape[0],
        radial_channels=first.edge_rbf.shape[1],
        hidden_dim=int(model_config["hidden_dim"]),
        message_blocks=int(model_config["message_blocks"]),
        maximum_sites=int(model_config["maximum_sites"]),
        maximum_cell_index=int(model_config["maximum_cell_index"]),
    ).to(device)
    checkpoint = torch.load(args.candidate_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "parent_conditioned_assignment_iid_v3" or checkpoint.get(
        "protocol_sha256"
    ) != source["training_protocol_sha256"]:
        raise ValueError("candidate checkpoint provenance changed")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint

    evaluation = protocol["evaluation"]
    rows: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, float | int]] = {}
    for index, role in enumerate(("iid_calibration_supported", "iid_test_supported")):
        role_rows = _bound_rows(
            model,
            by_role[role],
            order_samples=int(evaluation["order_samples"]),
            batch_size=int(evaluation["carrier_batch_size"]),
            seed=seed + index,
            device=device,
        )
        rows[role] = role_rows
        summaries[role] = _bound_summary(
            role_rows,
            bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
            seed=seed + 100 + index,
        )
    acceptance = training_protocol["acceptance"]
    checks = _updated_checks(candidate["checks"], summaries, acceptance)
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "implementation_commit": implementation_commit,
        "qualified": qualified,
        "checks": checks,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
        "evaluation": {
            "seed": seed,
            "order_samples": int(evaluation["order_samples"]),
            "candidate_checkpoint_sha256": source["candidate_checkpoint_sha256"],
            "supported_likelihood": summaries,
            "supported_carrier_rows": rows,
        },
        "candidate_evidence_unchanged": {
            "sampling": candidate["sampling"],
            "exact_subset": candidate["exact_subset"],
            "relabel_logit_max_abs": candidate["relabel_logit_max_abs"],
            "training": candidate["training"],
            "stress_likelihood": {
                role: candidate["likelihood"][role]
                for role in (
                    "iid_calibration_unseen_action",
                    "iid_test_unseen_action",
                    "ood_validation",
                    "ood_test",
                )
            },
        },
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if qualified else 2)


if __name__ == "__main__":
    main()
