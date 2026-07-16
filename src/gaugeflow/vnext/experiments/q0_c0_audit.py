"""Run the checkpoint-preserving Q0 diagnosis of the frozen P5-C0 path."""

from __future__ import annotations

import argparse
import hashlib
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch, Data

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.geometry import GaussianRadialBasis, periodic_closest_image_edges
from gaugeflow.vnext.diagnostics import (
    analytic_endpoint_jacobians,
    variational_flow_jacobian,
)

ROOT = Path(__file__).resolve().parents[4]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _wsl_path(value: str) -> Path:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if match and platform.system() == "Linux":
        return Path("/mnt") / match.group(1).lower() / Path(match.group(2))
    return Path(value)


def _git_commit_from_metadata(root: Path) -> str:
    marker = root / ".git"
    gitdir = marker
    if marker.is_file():
        line = marker.read_text(encoding="utf-8").strip()
        if not line.startswith("gitdir: "):
            raise RuntimeError("worktree .git file has no gitdir entry")
        gitdir = _wsl_path(line.removeprefix("gitdir: "))
    head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    common = gitdir
    common_marker = gitdir / "commondir"
    if common_marker.is_file():
        common = (gitdir / common_marker.read_text(encoding="utf-8").strip()).resolve()
    for base in (gitdir, common):
        loose = base / reference
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip()
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", maxsplit=1)
                if name == reference:
                    return commit
    raise RuntimeError(f"cannot resolve git reference {reference}")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return _git_commit_from_metadata(ROOT)


def _endpoint_batch(count: int, device: torch.device) -> Batch:
    endpoint = Data(
        atom_types=torch.tensor((5, 7, 14, 32), dtype=torch.long, device=device),
        frac_coords=torch.tensor(
            ((0.06, 0.11, 0.19), (0.34, 0.22, 0.31), (0.72, 0.48, 0.41), (0.21, 0.79, 0.67)),
            dtype=torch.float32,
            device=device,
        ),
        lattice=torch.tensor(
            ((3.9, 0.2, 0.1), (0.3, 4.3, 0.4), (0.1, 0.4, 5.1)),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        num_nodes=4,
    )
    return Batch.from_data_list([endpoint.clone() for _ in range(count)]).to(device)


def _fixed_sources(batch: Batch, seed: int) -> torch.Tensor:
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    torch.manual_seed(seed)
    if batch.frac_coords.is_cuda:
        torch.cuda.manual_seed_all(seed)
    return matcher.random_state(batch).frac_coords


def _parse_int_vector(value: str, *, rows: int) -> torch.Tensor:
    numbers = [int(item) for item in value.split(",")]
    if len(numbers) != rows * 3:
        raise ValueError("frozen integer-lift row has the wrong length")
    return torch.tensor(numbers, dtype=torch.float64).reshape(rows, 3)


def _reduced(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1) @ basis


def _wrapped_representation(value: torch.Tensor) -> torch.Tensor:
    relative = value[:, 1:, :] - value[:, :1, :]
    angle = 2.0 * torch.pi * relative
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1).reshape(value.shape[0], -1)


def _input_representation(value: torch.Tensor, lattice: torch.Tensor, rbf_count: int, cutoff: float) -> torch.Tensor:
    rbf = GaussianRadialBasis(rbf_count, cutoff).to(dtype=torch.float64)
    rows = []
    for graph in range(value.shape[0]):
        graph_batch = torch.zeros(value.shape[1], dtype=torch.long)
        geometry = periodic_closest_image_edges(value[graph], lattice[graph : graph + 1], graph_batch)
        features = torch.cat((geometry.direction, rbf(geometry.distance)), dim=-1)
        rows.append(features.reshape(-1))
    return torch.stack(rows)


def _jacobian_rows(times: torch.Tensor, dimension: int, steps: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for time in times:
        result = analytic_endpoint_jacobians(dimension, time)
        terminal = float(time) == 1.0
        row: dict[str, Any] = {
            "time": float(time),
            "dimension": dimension,
            "vector_jacobian_spectral_norm": None
            if result.vector_jacobian is None
            else float(torch.linalg.matrix_norm(result.vector_jacobian, ord=2)),
            "flow_singular_value_min": float(result.singular_values.min()),
            "flow_singular_value_max": float(result.singular_values.max()),
            "flow_log_abs_det": float(result.log_abs_det),
            "endpoint_singular": terminal,
        }
        rows.append(row)
    end_time = 1.0 - 1.0e-6
    exact = analytic_endpoint_jacobians(dimension, torch.tensor(end_time, dtype=torch.float64)).flow_jacobian
    for count in steps:
        integrated = variational_flow_jacobian(
            lambda time: -torch.eye(dimension, dtype=torch.float64) / (1.0 - time),
            dimension=dimension,
            end_time=end_time,
            steps=int(count),
        )
        rows.append(
            {
                "time": end_time,
                "dimension": dimension,
                "variational_steps": int(count),
                "variational_flow_jacobian_max_error": float((integrated - exact).abs().max()),
            }
        )
    return rows


def _environment(device: torch.device) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if cuda else platform.processor(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_manifest(directory: Path) -> None:
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "manifest.sha256")
    lines = [f"{_sha256_file(path)}  {path.relative_to(directory).as_posix()}" for path in files]
    (directory / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path, *, device: torch.device) -> Path:
    raise RuntimeError("original Q0 is frozen at commit 42a34c5; use the versioned Q0.1 protocol")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gates/q0_c0_audit.yaml"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = run(config_path, device=torch.device(args.device))
    print(output)


if __name__ == "__main__":
    main()
