"""Verified runtime loading for the frozen stoichiometry-first composition law."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file

from .composition_state import StoichiometryFirstCompositionModel


def load_qualified_composition_model(
    checkpoint_path: Path,
    protocol_path: Path,
    *,
    device: torch.device | str,
    expected_checkpoint_sha256: str | None = None,
) -> StoichiometryFirstCompositionModel:
    """Load the frozen ``p(C|N)`` law with its exact support buffers.

    The composition checkpoint intentionally has its own small schema rather
    than pretending it is a joint-diffusion checkpoint.  Model dimensions,
    partition prior, and active vocabulary are recovered from the checkpoint
    itself, while the supplied protocol fixes the mathematical family.
    """

    protocol = load_json_object(protocol_path)
    if protocol.get("protocol") != "h1a_e1_absolute_likelihood_v1":
        raise ValueError("composition runtime requires the qualified absolute-likelihood E1 protocol")
    model_config = protocol.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("composition protocol does not contain a model contract")
    if expected_checkpoint_sha256 is not None and sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("composition checkpoint SHA-256 does not match the frozen runtime identity")
    payload: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise ValueError("unsupported composition checkpoint schema")
    if payload.get("protocol") != protocol["protocol"]:
        raise ValueError("composition checkpoint protocol does not match the requested runtime")
    if payload.get("protocol_sha256") != canonical_json_hash(protocol):
        raise ValueError("composition checkpoint was not trained against this exact frozen protocol")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("composition checkpoint does not contain a model state")
    partition_log_prior = state.get("partition_log_prior")
    active_vocabulary_mask = state.get("active_vocabulary_mask")
    if not isinstance(partition_log_prior, torch.Tensor) or not isinstance(active_vocabulary_mask, torch.Tensor):
        raise ValueError("composition checkpoint lacks its exact-support buffers")
    model = StoichiometryFirstCompositionModel(
        context_dim=int(model_config["context_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        partition_log_prior=partition_log_prior,
        maximum_atoms=int(model_config["maximum_atoms"]),
        maximum_species=int(model_config["maximum_species"]),
        vocabulary_size=int(model_config["vocabulary_size"]),
        active_vocabulary_mask=active_vocabulary_mask,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
