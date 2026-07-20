"""Absorbing categorical diffusion for the 118-element production vocabulary."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT

from .categorical_common import FixedElementVocabulary
from .schedules import CosineNoiseSchedule


@dataclass(frozen=True)
class MaskedCategoricalState:
    tokens: torch.Tensor
    clean_mask: torch.Tensor


class AbsorbingMaskDiffusion(FixedElementVocabulary):
    """Continuous-time mask corruption and exact finite-step reverse kernel.

    Chemical states are dense element indices ``0..117``.  The absorbing mask
    is index ``118`` and is excluded from every clean posterior and decoded
    sample.
    """

    def __init__(
        self,
        schedule: CosineNoiseSchedule | None = None,
        *,
        element_count: int = CHEMICAL_ELEMENT_COUNT,
    ) -> None:
        if element_count != CHEMICAL_ELEMENT_COUNT:
            raise ValueError("production vocabulary is fixed to 118 chemical elements")
        self.element_count = element_count
        self.mask_index = element_count
        self.schedule = schedule or CosineNoiseSchedule()

    def corrupt(
        self,
        clean: torch.Tensor,
        time: torch.Tensor,
        batch: torch.Tensor,
        *,
        uniform: torch.Tensor | None = None,
    ) -> MaskedCategoricalState:
        self.validate_clean(clean)
        if batch.shape != clean.shape or batch.dtype != torch.long:
            raise ValueError("batch must provide one graph index per clean token")
        if batch.numel() and int(batch.max()) >= time.numel():
            raise ValueError("time does not cover every graph")
        keep_probability = self.schedule.alpha(time).square()[batch]
        if uniform is None:
            uniform = torch.rand(clean.shape, dtype=keep_probability.dtype, device=clean.device)
        if uniform.shape != clean.shape:
            raise ValueError("uniform corruption noise must match clean tokens")
        keep = uniform < keep_probability
        return MaskedCategoricalState(
            tokens=torch.where(keep, clean, torch.full_like(clean, self.mask_index)),
            clean_mask=keep,
        )

    def sample_prior(
        self,
        nodes: int,
        reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del generator
        return torch.full(
            (nodes,),
            self.mask_index,
            dtype=torch.long,
            device=reference.device,
        )

    def reverse_probabilities(
        self,
        current: torch.Tensor,
        clean_logits: torch.Tensor,
        time_from: torch.Tensor,
        time_to: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``p(x_time_to | x_time_from)`` for ``time_to < time_from``.

        Revealed chemical tokens are copied exactly.  A masked token unmasks
        according to the analytic absorbing bridge and the model's clean
        element posterior.  The result has 119 columns including MASK.
        """
        if current.ndim != 1 or current.dtype != torch.long:
            raise ValueError("current tokens must be a rank-one int64 tensor")
        if clean_logits.shape != (current.numel(), self.element_count):
            raise ValueError("clean logits must have shape [nodes, 118]")
        if batch.shape != current.shape:
            raise ValueError("batch must match current tokens")
        if time_from.shape != time_to.shape or time_from.ndim != 1:
            raise ValueError("time endpoints must be equal-shape graph vectors")
        if bool((time_to > time_from).any()):
            raise ValueError("reverse kernel requires time_to <= time_from")
        clean_survival_from = self.schedule.alpha(time_from).square()[batch]
        clean_survival_to = self.schedule.alpha(time_to).square()[batch]
        denominator = (1.0 - clean_survival_from).clamp_min(torch.finfo(clean_logits.dtype).eps)
        reveal = ((clean_survival_to - clean_survival_from) / denominator).clamp(0.0, 1.0)
        chemical = torch.softmax(clean_logits, dim=-1) * reveal.unsqueeze(-1)
        probabilities = clean_logits.new_zeros((current.numel(), self.element_count + 1))
        masked = current == self.mask_index
        if bool(((current < 0) | (current > self.mask_index)).any()):
            raise ValueError("current token lies outside the production vocabulary")
        probabilities[masked, : self.element_count] = chemical[masked]
        probabilities[masked, self.mask_index] = 1.0 - reveal[masked]
        revealed = ~masked
        if bool(revealed.any()):
            probabilities[revealed, current[revealed]] = 1.0
        return probabilities
