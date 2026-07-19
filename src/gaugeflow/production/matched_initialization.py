"""Matched shared initialization for causal angular-operator comparisons."""

from __future__ import annotations

from typing import Any

import torch

from .equivariant_denoiser import HybridCrystalDenoiser


def matched_angular_model(
    model_config: dict[str, Any], *, seed: int
) -> tuple[HybridCrystalDenoiser, int]:
    """Construct a variant with factorized-baseline values on shared tensors."""
    reference_config = dict(model_config)
    reference_config["angular_operator"] = "factorized"
    torch.manual_seed(seed)
    reference = HybridCrystalDenoiser(**reference_config)
    torch.manual_seed(seed)
    model = HybridCrystalDenoiser(**model_config)
    candidate_state = model.state_dict()
    shared = 0
    for name, value in reference.state_dict().items():
        if (
            ".angular_moments." not in name
            and name in candidate_state
            and candidate_state[name].shape == value.shape
        ):
            candidate_state[name].copy_(value)
            shared += value.numel()
    if shared < 4_000_000:
        raise ValueError("matched angular comparison copied too little shared state")
    model.load_state_dict(candidate_state, strict=True)
    return model, shared
