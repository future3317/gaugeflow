"""Task measures on the heterogeneous crystal modality-noise cube."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import torch


class ModalityRegime(IntEnum):
    """Low-cost cubature nodes used by the current five-regime task measure."""

    CLEAN_SIDE = 0
    NOISY_ELEMENT = 1
    NOISY_LATTICE = 2
    DIAGONAL = 3
    INTERIOR = 4


MODALITY_REGIME_NAMES = (
    "clean_clean",
    "noisy_element",
    "noisy_lattice",
    "diagonal",
    "interior",
)


@dataclass(frozen=True)
class ModalityNoiseTimes:
    """Graphwise coordinates in the ``(A,F,L)`` modality-noise cube.

    ``element``, ``coordinate`` and ``lattice`` are statistical noise levels
    for heterogeneous forward processes, not interchangeable wall-clock
    labels.  The optional regime index records which cubature component drew
    each graph; it is audit metadata and is never a denoiser input.
    """

    element: torch.Tensor
    coordinate: torch.Tensor
    lattice: torch.Tensor
    regime: torch.Tensor

    def validate(self, graph_count: int) -> None:
        """Validate static tensor contracts without synchronizing a GPU batch."""

        expected = (graph_count,)
        for name, value in (
            ("element", self.element),
            ("coordinate", self.coordinate),
            ("lattice", self.lattice),
        ):
            if value.shape != expected or not value.dtype.is_floating_point:
                raise ValueError(f"{name} time must be a floating graph vector")
        if self.regime.shape != expected or self.regime.dtype != torch.long:
            raise ValueError("regime must be an int64 graph vector")
        if not (
            self.element.device
            == self.coordinate.device
            == self.lattice.device
            == self.regime.device
        ):
            raise ValueError("all modality times must share one device")
        if not (self.element.dtype == self.coordinate.dtype == self.lattice.dtype):
            raise ValueError("all modality times must share one floating dtype")

    def as_afl(self) -> torch.Tensor:
        """Return the explicit ``[graphs,3]`` cube coordinate in A,F,L order."""

        return torch.stack((self.element, self.coordinate, self.lattice), dim=-1)


class FiveRegimeTaskMeasure:
    """Equal-mass cubature over one interior and four boundary task families.

    The 64-graph production batch has exact counts ``13/13/13/13/12``.  This
    is the current finite estimator of a task measure over modality-noise
    space; it is not a curriculum and the regime label is not model input.
    """

    components = len(MODALITY_REGIME_NAMES)

    def sample(
        self,
        graph_count: int,
        sample_time: Callable[[int], torch.Tensor],
        *,
        generator: torch.Generator | None = None,
    ) -> ModalityNoiseTimes:
        if graph_count < self.components:
            raise ValueError("five-regime task measure needs at least five graphs")

        # Preserve the preregistered RNG order: coordinate times, regime
        # assignment, then the two independent interior side times.
        coordinate = sample_time(graph_count)
        counts = torch.full(
            (self.components,),
            graph_count // self.components,
            dtype=torch.long,
            device=coordinate.device,
        )
        counts[: graph_count % self.components] += 1
        regime = torch.repeat_interleave(
            torch.arange(self.components, dtype=torch.long, device=coordinate.device),
            counts,
        )
        regime = regime[
            torch.randperm(graph_count, device=coordinate.device, generator=generator)
        ]

        element = torch.zeros_like(coordinate)
        lattice = torch.zeros_like(coordinate)
        element_noisy = (regime == int(ModalityRegime.NOISY_ELEMENT)) | (
            regime == int(ModalityRegime.DIAGONAL)
        )
        lattice_noisy = (regime == int(ModalityRegime.NOISY_LATTICE)) | (
            regime == int(ModalityRegime.DIAGONAL)
        )
        element[element_noisy] = coordinate[element_noisy]
        lattice[lattice_noisy] = coordinate[lattice_noisy]

        interior = regime == int(ModalityRegime.INTERIOR)
        # Remainder mass is assigned to the first components, so the final
        # interior component always has floor(G/5) graphs.  Derive the count
        # on the host instead of synchronizing ``interior.sum()`` from CUDA.
        interior_count = graph_count // self.components
        if interior_count:
            element[interior] = sample_time(interior_count)
            lattice[interior] = sample_time(interior_count)

        result = ModalityNoiseTimes(
            element=element,
            coordinate=coordinate,
            lattice=lattice,
            regime=regime,
        )
        result.validate(graph_count)
        return result
