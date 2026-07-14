"""Standalone Riemannian flow-matching objective and Euler sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .manifold import lattice_to_log_vector, torus_logmap, wrap01
from .manifold import log_vector_to_lattice
from .uncertainty import (
    SampleUncertainty,
    cartesian_isotropic_gaussian_nll,
    scalar_gaussian_nll,
)


@dataclass
class CrystalFlowState:
    type_state: torch.Tensor
    frac_coords: torch.Tensor
    lattice_log: torch.Tensor


class RiemannianCrystalFlowMatcher:
    def __init__(self, atom_types: int = 119, uncertainty_weight: float = 0.0):
        self.atom_types = atom_types
        if uncertainty_weight < 0:
            raise ValueError("uncertainty_weight must be non-negative")
        self.uncertainty_weight = uncertainty_weight

    def target_state(self, batch) -> CrystalFlowState:
        return CrystalFlowState(
            type_state=torch.nn.functional.one_hot(batch.atom_types, self.atom_types).float(),
            frac_coords=batch.frac_coords,
            lattice_log=lattice_to_log_vector(batch.lattice),
        )

    def random_state(self, batch) -> CrystalFlowState:
        device = batch.frac_coords.device
        return CrystalFlowState(
            type_state=torch.randn((batch.atom_types.numel(), self.atom_types), device=device),
            frac_coords=torch.rand_like(batch.frac_coords),
            lattice_log=torch.randn((batch.num_graphs, 6), device=device),
        )

    def loss(self, model, batch) -> dict[str, torch.Tensor]:
        target = self.target_state(batch)
        base = self.random_state(batch)
        time = torch.rand((batch.num_graphs,), device=batch.frac_coords.device)
        node_time = time[batch.batch].unsqueeze(-1)
        velocity_type = target.type_state - base.type_state
        velocity_coord = torus_logmap(base.frac_coords, target.frac_coords)
        velocity_lattice = target.lattice_log - base.lattice_log
        state = CrystalFlowState(
            type_state=base.type_state + node_time * velocity_type,
            frac_coords=wrap01(base.frac_coords + node_time * velocity_coord),
            lattice_log=base.lattice_log + time.unsqueeze(-1) * velocity_lattice,
        )
        outputs = model(
            state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
            batch.piezo_irreps, batch.condition_present,
            getattr(batch, "condition_orbit", None),
            return_uncertainty=self.uncertainty_weight > 0,
        )
        pred_type, pred_coord, pred_lattice, alignment = outputs[:4]
        terms = {
            "type": (pred_type - velocity_type).square().mean(),
            "coord": (pred_coord - velocity_coord).square().mean(),
            "lattice": (pred_lattice - velocity_lattice).square().mean(),
        }
        terms["loss"] = terms["type"] + terms["coord"] + terms["lattice"]
        if self.uncertainty_weight > 0:
            uncertainty = outputs[4]
            lattice_nodes = log_vector_to_lattice(state.lattice_log)[batch.batch]
            coordinate_residual_cartesian = torch.einsum(
                "ni,nij->nj", pred_coord - velocity_coord, lattice_nodes
            )
            terms["uncertainty_type"] = scalar_gaussian_nll(
                pred_type - velocity_type, uncertainty.type_log_std
            )
            terms["uncertainty_coord"] = cartesian_isotropic_gaussian_nll(
                coordinate_residual_cartesian, uncertainty.coord_log_std
            )
            terms["uncertainty_lattice"] = scalar_gaussian_nll(
                pred_lattice - velocity_lattice, uncertainty.lattice_log_std
            )
            terms["uncertainty"] = (
                terms["uncertainty_type"] + terms["uncertainty_coord"] + terms["uncertainty_lattice"]
            )
            terms["loss"] = terms["loss"] + self.uncertainty_weight * terms["uncertainty"]
        terms["alignment_entropy"] = -(alignment.clamp_min(1e-8) * alignment.clamp_min(1e-8).log()).sum(-1).mean()
        return terms

    @torch.no_grad()
    def sample(
        self, model, batch, *, steps: int = 100, guidance_scale: float = 0.0,
        return_uncertainty: bool = False,
    ) -> CrystalFlowState | tuple[CrystalFlowState, SampleUncertainty]:
        state = self.random_state(batch)
        dt = 1.0 / steps
        type_variance = state.type_state.new_zeros((state.type_state.shape[0], 1))
        coordinate_variance = state.frac_coords.new_zeros((state.frac_coords.shape[0], 1))
        lattice_variance = state.lattice_log.new_zeros((state.lattice_log.shape[0], 1))
        alignment_entropy = state.lattice_log.new_zeros((state.lattice_log.shape[0],))
        for step in range(steps):
            time = torch.full((batch.num_graphs,), step / steps, device=batch.frac_coords.device)
            conditional_outputs = model(
                state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                batch.piezo_irreps, batch.condition_present,
                getattr(batch, "condition_orbit", None),
                return_uncertainty=return_uncertainty,
            )
            conditional = conditional_outputs[:3]
            if guidance_scale:
                null = model(
                    state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                    batch.piezo_irreps, torch.zeros_like(batch.condition_present),
                    getattr(batch, "condition_orbit", None),
                )[:3]
                velocity = tuple((1 + guidance_scale) * c - guidance_scale * u for c, u in zip(conditional, null))
            else:
                velocity = conditional
            if return_uncertainty:
                uncertainty = conditional_outputs[4]
                type_variance += dt * dt * torch.exp(2.0 * uncertainty.type_log_std)
                coordinate_variance += dt * dt * torch.exp(2.0 * uncertainty.coord_log_std)
                lattice_variance += dt * dt * torch.exp(2.0 * uncertainty.lattice_log_std)
                alignment = conditional_outputs[3]
                alignment_entropy += dt * -(alignment.clamp_min(1e-8) * alignment.clamp_min(1e-8).log()).sum(-1)
            state = CrystalFlowState(
                type_state=state.type_state + dt * velocity[0],
                frac_coords=wrap01(state.frac_coords + dt * velocity[1]),
                lattice_log=state.lattice_log + dt * velocity[2],
            )
        if not return_uncertainty:
            return state
        return state, SampleUncertainty(
            type_variance=type_variance,
            coordinate_cartesian_variance=coordinate_variance,
            lattice_variance=lattice_variance,
            mean_alignment_entropy=alignment_entropy,
        )
