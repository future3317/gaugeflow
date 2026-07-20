"""Self-correcting uniform categorical diffusion for chemical elements."""

from __future__ import annotations

import torch

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT

from .categorical_common import FixedElementVocabulary
from .categorical_mask import MaskedCategoricalState
from .schedules import CosineNoiseSchedule


class UniformCategoricalDiffusion(FixedElementVocabulary):
    """D3PM-style uniform replacement with an O(nodes * elements) reverse.

    Unlike an absorbing mask path, every intermediate chemical token may be
    revised.  The symmetric transition matrix is represented by its scalar
    survival probability, so neither training nor sampling materializes a
    ``118 x 118`` matrix per node.
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
        self.schedule = schedule or CosineNoiseSchedule()

    @property
    def mask_index(self) -> int:
        """Reserved embedding index, never a state of the uniform path."""

        return self.element_count

    def sample_prior(
        self,
        nodes: int,
        reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randint(
            self.element_count,
            (nodes,),
            dtype=torch.long,
            device=reference.device,
            generator=generator,
        )

    def corrupt(
        self,
        clean: torch.Tensor,
        time: torch.Tensor,
        batch: torch.Tensor,
        *,
        uniform: torch.Tensor | None = None,
        token_uniform: torch.Tensor | None = None,
    ) -> MaskedCategoricalState:
        self.validate_clean(clean)
        if batch.shape != clean.shape or batch.dtype != torch.long:
            raise ValueError("batch must provide one graph index per clean token")
        if batch.numel() and int(batch.max()) >= time.numel():
            raise ValueError("time does not cover every graph")
        survival = self.schedule.alpha(time).square()[batch]
        if uniform is None:
            uniform = torch.rand(clean.shape, dtype=survival.dtype, device=clean.device)
        if token_uniform is None:
            token_uniform = torch.rand(clean.shape, dtype=survival.dtype, device=clean.device)
        if uniform.shape != clean.shape or token_uniform.shape != clean.shape:
            raise ValueError("uniform corruption noise must match clean tokens")
        keep = uniform < survival
        replacement = (token_uniform * self.element_count).floor().long().clamp_max(
            self.element_count - 1
        )
        return MaskedCategoricalState(
            tokens=torch.where(keep, clean, replacement),
            clean_mask=keep,
        )

    def reverse_probabilities(
        self,
        current: torch.Tensor,
        clean_logits: torch.Tensor,
        time_from: torch.Tensor,
        time_to: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Marginalize the exact symmetric posterior over predicted ``x_0``.

        For ``a_t = alpha(t)^2`` and ``b_t=(1-a_t)/K``, the posterior sum is
        evaluated using one rank-one term plus one diagonal term.  This is
        algebraically identical to a dense D3PM transition calculation but is
        linear rather than quadratic in the 118-element vocabulary.
        """

        if current.ndim != 1 or current.dtype != torch.long:
            raise ValueError("current tokens must be a rank-one int64 tensor")
        if clean_logits.shape != (current.numel(), self.element_count):
            raise ValueError("clean logits must have shape [nodes, 118]")
        if batch.shape != current.shape or batch.dtype != torch.long:
            raise ValueError("batch must match current tokens")
        if time_from.shape != time_to.shape or time_from.ndim != 1:
            raise ValueError("time endpoints must be equal-shape graph vectors")
        if bool((time_to > time_from).any()):
            raise ValueError("reverse kernel requires time_to <= time_from")
        self.validate_clean(current)

        survival_from = self.schedule.alpha(time_from).square()[batch]
        survival_to = self.schedule.alpha(time_to).square()[batch]
        transition_survival = survival_from / survival_to.clamp_min(1.0e-12)
        uniform_from = (1.0 - survival_from) / self.element_count
        uniform_to = (1.0 - survival_to) / self.element_count
        transition_uniform = (1.0 - transition_survival) / self.element_count

        clean_probability = torch.softmax(clean_logits, dim=-1)
        likelihood_from_clean = uniform_from.unsqueeze(-1).expand_as(clean_probability).clone()
        likelihood_from_clean.scatter_add_(
            1,
            current.unsqueeze(-1),
            survival_from.unsqueeze(-1),
        )
        weighted_clean = clean_probability / likelihood_from_clean.clamp_min(
            torch.finfo(clean_probability.dtype).tiny
        )
        posterior_to = survival_to.unsqueeze(-1) * weighted_clean
        posterior_to = posterior_to + uniform_to.unsqueeze(-1) * weighted_clean.sum(
            dim=-1,
            keepdim=True,
        )
        transition_likelihood = transition_uniform.unsqueeze(-1).expand_as(
            posterior_to
        ).clone()
        transition_likelihood.scatter_add_(
            1,
            current.unsqueeze(-1),
            transition_survival.unsqueeze(-1),
        )
        probability = transition_likelihood * posterior_to
        return probability / probability.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(probability.dtype).tiny
        )
