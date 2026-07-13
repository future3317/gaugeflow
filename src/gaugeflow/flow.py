"""Standalone Riemannian flow-matching objective and Euler sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .manifold import lattice_to_log_vector, torus_logmap, wrap01


@dataclass
class CrystalFlowState:
    type_state: torch.Tensor
    frac_coords: torch.Tensor
    lattice_log: torch.Tensor


class RiemannianCrystalFlowMatcher:
    def __init__(self, atom_types: int = 119):
        self.atom_types = atom_types

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
        pred_type, pred_coord, pred_lattice, alignment = model(
            state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
            batch.piezo_irreps, batch.condition_present,
            getattr(batch, "stabilizer_rotations", None),
            getattr(batch, "stabilizer_count", None),
        )
        terms = {
            "type": (pred_type - velocity_type).square().mean(),
            "coord": (pred_coord - velocity_coord).square().mean(),
            "lattice": (pred_lattice - velocity_lattice).square().mean(),
        }
        terms["loss"] = terms["type"] + terms["coord"] + terms["lattice"]
        terms["alignment_entropy"] = -(alignment.clamp_min(1e-8) * alignment.clamp_min(1e-8).log()).sum(-1).mean()
        return terms

    @torch.no_grad()
    def sample(
        self, model, batch, *, steps: int = 100, guidance_scale: float = 0.0
    ) -> CrystalFlowState:
        state = self.random_state(batch)
        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((batch.num_graphs,), step / steps, device=batch.frac_coords.device)
            conditional = model(
                state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                batch.piezo_irreps, batch.condition_present,
                getattr(batch, "stabilizer_rotations", None),
                getattr(batch, "stabilizer_count", None),
            )[:3]
            if guidance_scale:
                null = model(
                    state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                    batch.piezo_irreps, torch.zeros_like(batch.condition_present),
                    getattr(batch, "stabilizer_rotations", None),
                    getattr(batch, "stabilizer_count", None),
                )[:3]
                velocity = tuple((1 + guidance_scale) * c - guidance_scale * u for c, u in zip(conditional, null))
            else:
                velocity = conditional
            state = CrystalFlowState(
                type_state=state.type_state + dt * velocity[0],
                frac_coords=wrap01(state.frac_coords + dt * velocity[1]),
                lattice_log=state.lattice_log + dt * velocity[2],
            )
        return state
