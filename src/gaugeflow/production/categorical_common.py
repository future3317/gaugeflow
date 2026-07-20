"""Shared validation for the fixed production element vocabulary."""

from __future__ import annotations

import torch

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def validate_element_tokens(tokens: torch.Tensor) -> None:
    if tokens.dtype != torch.long or tokens.ndim != 1:
        raise ValueError("clean element tokens must be a rank-one int64 tensor")
    if tokens.numel() and bool(((tokens < 0) | (tokens >= CHEMICAL_ELEMENT_COUNT)).any()):
        raise ValueError("clean element token lies outside 0..117")


def decode_element_tokens(tokens: torch.Tensor) -> torch.Tensor:
    validate_element_tokens(tokens)
    return tokens + 1


class FixedElementVocabulary:
    """Shared 118-element validation/decode interface for categorical paths."""

    element_count: int

    def validate_clean(self, tokens: torch.Tensor) -> None:
        validate_element_tokens(tokens)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return decode_element_tokens(tokens)
