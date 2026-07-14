"""Standalone Riemannian flow-matching objective and Euler sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.utils import scatter

from .coupling import periodic_assignment, remove_graphwise_translation
from .manifold import lattice_to_log_vector, project_simplex, simplex_tangent, torus_logmap, wrap01
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
    _HEADS = ("type", "coord", "lattice")

    def __init__(
        self,
        atom_types: int = 119,
        uncertainty_weight: float = 0.0,
        active_heads: tuple[str, ...] | list[str] = _HEADS,
        type_path: str = "euclidean_logits",
        target_coupling: str = "identity",
        coordinate_gauge: str = "absolute",
        loss_normalization: str = "none",
        endpoint_type_nll_weight: float = 0.0,
    ):
        self.atom_types = atom_types
        if uncertainty_weight < 0:
            raise ValueError("uncertainty_weight must be non-negative")
        active = tuple(active_heads)
        if not active or any(head not in self._HEADS for head in active):
            raise ValueError(f"active_heads must be a non-empty subset of {self._HEADS}")
        if type_path not in {"euclidean_logits", "simplex_probability", "riemannian_simplex"}:
            raise ValueError("unknown type_path")
        if target_coupling not in {"identity", "optimal_transport", "typewise_optimal_transport"}:
            raise ValueError("unknown target_coupling")
        if coordinate_gauge not in {"absolute", "no_drift"}:
            raise ValueError("coordinate_gauge must be 'absolute' or 'no_drift'")
        if loss_normalization not in {"none", "target_velocity_rms"}:
            raise ValueError("unknown loss_normalization")
        if endpoint_type_nll_weight < 0:
            raise ValueError("endpoint_type_nll_weight must be non-negative")
        if endpoint_type_nll_weight and type_path != "riemannian_simplex":
            raise ValueError("endpoint_type_nll_weight requires the riemannian_simplex type path")
        self.uncertainty_weight = uncertainty_weight
        self.active_heads = active
        self.type_path = type_path
        self.target_coupling = target_coupling
        self.coordinate_gauge = coordinate_gauge
        self.loss_normalization = loss_normalization
        self.endpoint_type_nll_weight = endpoint_type_nll_weight

    def target_state(self, batch) -> CrystalFlowState:
        return CrystalFlowState(
            type_state=torch.nn.functional.one_hot(batch.atom_types, self.atom_types).float(),
            frac_coords=batch.frac_coords,
            lattice_log=lattice_to_log_vector(batch.lattice),
        )

    def random_state(self, batch) -> CrystalFlowState:
        device = batch.frac_coords.device
        type_state = torch.randn((batch.atom_types.numel(), self.atom_types), device=device)
        if self.type_path == "simplex_probability":
            type_state = torch.softmax(type_state, dim=-1)
        elif self.type_path == "riemannian_simplex":
            type_state = torch.distributions.Dirichlet(
                torch.ones(self.atom_types, device=device)
            ).sample((batch.atom_types.numel(),))
        state = CrystalFlowState(
            type_state=type_state,
            frac_coords=torch.rand_like(batch.frac_coords),
            lattice_log=torch.randn((batch.num_graphs, 6), device=device),
        )
        if self.active_heads == self._HEADS:
            return state
        # A4 subspace qualification holds every inactive factor at its known
        # endpoint.  This makes type-only and geometry-only experiments test
        # the requested factor rather than an uncontrolled joint decoder.
        target = self.target_state(batch)
        return CrystalFlowState(
            type_state=state.type_state if "type" in self.active_heads else target.type_state,
            frac_coords=state.frac_coords if "coord" in self.active_heads else target.frac_coords,
            lattice_log=state.lattice_log if "lattice" in self.active_heads else target.lattice_log,
        )

    def _coupled_target(self, target: CrystalFlowState, base: CrystalFlowState, batch) -> CrystalFlowState:
        """Permute the paired endpoint only while constructing a training path."""
        if self.target_coupling == "identity":
            return target
        type_state, frac_coords = target.type_state.clone(), target.frac_coords.clone()
        for graph in range(batch.num_graphs):
            nodes = torch.nonzero(batch.batch == graph, as_tuple=False).flatten()
            source_types = target_types = None
            if self.target_coupling == "typewise_optimal_transport":
                source_types = batch.atom_types[nodes]
                target_types = batch.atom_types[nodes]
            assignment = periodic_assignment(
                base.frac_coords[nodes], target.frac_coords[nodes],
                source_types=source_types, target_types=target_types,
            )
            type_state[nodes] = target.type_state[nodes][assignment]
            frac_coords[nodes] = target.frac_coords[nodes][assignment]
        return CrystalFlowState(type_state, frac_coords, target.lattice_log)

    def _type_velocity(self, value: torch.Tensor) -> torch.Tensor:
        return simplex_tangent(value) if self.type_path == "riemannian_simplex" else value

    def _coordinate_velocity(self, value: torch.Tensor, batch) -> torch.Tensor:
        if self.coordinate_gauge == "no_drift":
            return remove_graphwise_translation(value, batch.batch, batch.num_graphs)
        return value

    def _head_loss(self, value: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loss = (value - target).square().mean()
        normalizer = target.square().mean().detach().clamp_min(1e-8)
        if self.loss_normalization == "target_velocity_rms":
            return loss, loss / normalizer
        return loss, loss

    def loss(
        self,
        model,
        batch,
        *,
        counterfactual_weight: float = 0.0,
        counterfactual_margin: float = 0.0,
        identification_weight: float = 0.0,
        identification_temperature: float = 1.0,
        identification_early_sigma: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Flow matching with an optional fixed-permutation tangent ranking term.

        The primary three-head flow loss is unchanged.  When enabled, the
        auxiliary term compares the current graph's own tensor condition to a
        cyclically shifted condition on exactly the same interpolant state.
        Null-conditioned examples are excluded from that comparison so a
        physical zero tensor (which is present) is never conflated with CFG's
        learned missing-condition token.
        """
        if counterfactual_weight < 0:
            raise ValueError("counterfactual_weight must be non-negative")
        if identification_weight < 0:
            raise ValueError("identification_weight must be non-negative")
        if identification_temperature <= 0:
            raise ValueError("identification_temperature must be positive")
        if identification_early_sigma is not None and identification_early_sigma <= 0:
            raise ValueError("identification_early_sigma must be positive when set")
        target = self.target_state(batch)
        base = self.random_state(batch)
        target = self._coupled_target(target, base, batch)
        time = torch.rand((batch.num_graphs,), device=batch.frac_coords.device)
        node_time = time[batch.batch].unsqueeze(-1)
        velocity_type = target.type_state - base.type_state
        velocity_coord = self._coordinate_velocity(
            torus_logmap(base.frac_coords, target.frac_coords), batch
        )
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
        pred_type = self._type_velocity(pred_type)
        pred_coord = self._coordinate_velocity(pred_coord, batch)
        type_loss, type_objective = self._head_loss(pred_type, velocity_type)
        coord_loss, coord_objective = self._head_loss(pred_coord, velocity_coord)
        lattice_loss, lattice_objective = self._head_loss(pred_lattice, velocity_lattice)
        terms = {"type": type_loss, "coord": coord_loss, "lattice": lattice_loss}
        terms["type_objective"] = type_objective
        terms["coord_objective"] = coord_objective
        terms["lattice_objective"] = lattice_objective
        terms["loss"] = sum(terms[f"{head}_objective"] for head in self.active_heads)
        terms["endpoint_type_nll"] = terms["loss"].new_zeros(())
        if self.endpoint_type_nll_weight:
            remaining = (1.0 - time[batch.batch]).unsqueeze(-1)
            endpoint_probability = project_simplex(state.type_state + remaining * pred_type)
            endpoint_index = target.type_state.argmax(dim=-1)
            terms["endpoint_type_nll"] = -endpoint_probability.clamp_min(1e-8).log().gather(
                -1, endpoint_index.unsqueeze(-1)
            ).mean()
            terms["loss"] = terms["loss"] + self.endpoint_type_nll_weight * terms["endpoint_type_nll"]
        terms["counterfactual"] = terms["loss"].new_zeros(())
        if counterfactual_weight > 0:
            permutation = torch.roll(
                torch.arange(batch.num_graphs, device=batch.batch.device), 1
            )
            wrong_outputs = model(
                state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                batch.piezo_irreps[permutation], batch.condition_present,
                getattr(batch, "condition_orbit", None),
                return_uncertainty=False,
            )
            wrong_type, wrong_coord, wrong_lattice = wrong_outputs[:3]
            wrong_type = self._type_velocity(wrong_type)
            wrong_coord = self._coordinate_velocity(wrong_coord, batch)
            own_type_error = (pred_type - velocity_type).square().mean(dim=-1)
            own_coord_error = (pred_coord - velocity_coord).square().mean(dim=-1)
            wrong_type_error = (wrong_type - velocity_type).square().mean(dim=-1)
            wrong_coord_error = (wrong_coord - velocity_coord).square().mean(dim=-1)
            own_graph_error = (
                scatter(
                    own_type_error + own_coord_error,
                    batch.batch,
                    dim=0,
                    dim_size=batch.num_graphs,
                    reduce="mean",
                )
                + (pred_lattice - velocity_lattice).square().mean(dim=-1)
            )
            wrong_graph_error = (
                scatter(
                    wrong_type_error + wrong_coord_error,
                    batch.batch,
                    dim=0,
                    dim_size=batch.num_graphs,
                    reduce="mean",
                )
                + (wrong_lattice - velocity_lattice).square().mean(dim=-1)
            )
            present = batch.condition_present.reshape(batch.num_graphs, -1).all(dim=-1)
            valid = present & present[permutation]
            if valid.any():
                terms["counterfactual"] = torch.nn.functional.softplus(
                    counterfactual_margin + own_graph_error[valid] - wrong_graph_error[valid]
                ).mean()
                terms["loss"] = terms["loss"] + counterfactual_weight * terms["counterfactual"]
        terms["identification"] = terms["loss"].new_zeros(())
        terms["identification_retrieval"] = terms["loss"].new_zeros(())
        if identification_weight > 0:
            # A3 uses every tensor in the batch as a candidate condition for
            # every own flow interpolant.  This is deliberately not a cyclic
            # negative: with two targets it is the exact all-negative softmax,
            # and it remains well-defined for a later, separately authorized
            # 4/8-target extension.
            present = batch.condition_present.reshape(batch.num_graphs, -1).all(dim=-1)
            if batch.num_graphs < 2:
                raise ValueError("All-negative identification requires at least two graphs")
            if not present.all():
                raise ValueError(
                    "All-negative identification requires present physical tensor conditions; "
                    "do not conflate it with the CFG null token"
                )
            candidate_errors = []
            for candidate in range(batch.num_graphs):
                candidate_condition = batch.piezo_irreps[candidate:candidate + 1].expand(
                    batch.num_graphs, -1
                )
                candidate_orbit = None
                condition_orbit = getattr(batch, "condition_orbit", None)
                if condition_orbit is not None:
                    candidate_orbit = condition_orbit[candidate:candidate + 1].expand(
                        batch.num_graphs, *condition_orbit.shape[1:]
                    )
                candidate_outputs = model(
                    state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                    candidate_condition, batch.condition_present, candidate_orbit,
                    return_uncertainty=False,
                )
                candidate_type, candidate_coord, candidate_lattice = candidate_outputs[:3]
                candidate_type = self._type_velocity(candidate_type)
                candidate_coord = self._coordinate_velocity(candidate_coord, batch)
                candidate_node_error = (
                    (candidate_type - velocity_type).square().mean(dim=-1)
                    + (candidate_coord - velocity_coord).square().mean(dim=-1)
                )
                candidate_errors.append(
                    scatter(
                        candidate_node_error,
                        batch.batch,
                        dim=0,
                        dim_size=batch.num_graphs,
                        reduce="mean",
                    )
                    + (candidate_lattice - velocity_lattice).square().mean(dim=-1)
                )
            # rows index the own interpolant x_t^i, columns index e_j.
            score = -torch.stack(candidate_errors, dim=-1)
            log_probability = torch.log_softmax(score / identification_temperature, dim=-1)
            own = torch.arange(batch.num_graphs, device=batch.batch.device)
            per_graph_identification = -log_probability[own, own]
            early_weight = (
                torch.exp(-time / identification_early_sigma)
                if identification_early_sigma is not None else torch.ones_like(time)
            )
            terms["identification"] = (early_weight * per_graph_identification).mean()
            terms["identification_retrieval"] = (score.argmax(dim=-1) == own).float().mean()
            terms["loss"] = terms["loss"] + identification_weight * terms["identification"]
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
        return_uncertainty: bool = False, initial_state: CrystalFlowState | None = None,
    ) -> CrystalFlowState | tuple[CrystalFlowState, SampleUncertainty]:
        if steps < 1:
            raise ValueError("steps must be positive")
        if initial_state is None:
            state = self.random_state(batch)
        else:
            expected = (
                (batch.atom_types.numel(), self.atom_types),
                tuple(batch.frac_coords.shape),
                (batch.num_graphs, 6),
            )
            actual = (
                tuple(initial_state.type_state.shape),
                tuple(initial_state.frac_coords.shape),
                tuple(initial_state.lattice_log.shape),
            )
            if actual != expected:
                raise ValueError(f"initial_state shapes {actual} do not match batch {expected}")
            state = CrystalFlowState(
                initial_state.type_state.clone(), initial_state.frac_coords.clone(), initial_state.lattice_log.clone()
            )
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
            conditional_raw = conditional_outputs[:3]
            conditional = (
                self._type_velocity(conditional_raw[0]),
                self._coordinate_velocity(conditional_raw[1], batch),
                conditional_raw[2],
            )
            if guidance_scale:
                null_raw = model(
                    state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                    batch.piezo_irreps, torch.zeros_like(batch.condition_present),
                    getattr(batch, "condition_orbit", None),
                )[:3]
                null = (
                    self._type_velocity(null_raw[0]),
                    self._coordinate_velocity(null_raw[1], batch),
                    null_raw[2],
                )
                velocity = tuple((1 + guidance_scale) * c - guidance_scale * u for c, u in zip(conditional, null))
            else:
                velocity = conditional
            velocity = (
                self._type_velocity(velocity[0]),
                self._coordinate_velocity(velocity[1], batch),
                velocity[2],
            )
            # Inactive A4 subspaces remain at the endpoint supplied by
            # random_state/initial_state.  Do not allow an untrained head to
            # contaminate the requested factor during a diagnostic sample.
            velocity = (
                velocity[0] if "type" in self.active_heads else torch.zeros_like(velocity[0]),
                velocity[1] if "coord" in self.active_heads else torch.zeros_like(velocity[1]),
                velocity[2] if "lattice" in self.active_heads else torch.zeros_like(velocity[2]),
            )
            if return_uncertainty:
                uncertainty = conditional_outputs[4]
                type_variance += dt * dt * torch.exp(2.0 * uncertainty.type_log_std)
                coordinate_variance += dt * dt * torch.exp(2.0 * uncertainty.coord_log_std)
                lattice_variance += dt * dt * torch.exp(2.0 * uncertainty.lattice_log_std)
                alignment = conditional_outputs[3]
                alignment_entropy += dt * -(alignment.clamp_min(1e-8) * alignment.clamp_min(1e-8).log()).sum(-1)
            type_state = state.type_state + dt * velocity[0]
            if self.type_path == "riemannian_simplex":
                type_state = project_simplex(type_state)
            elif self.type_path == "simplex_probability":
                # The exact linear interpolant remains in the simplex.  A
                # learned Euler field need not, so the diagnostic simplex path
                # projects only the probability factor back to its manifold.
                type_state = type_state.clamp_min(0.0)
                type_state = type_state / type_state.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            state = CrystalFlowState(
                type_state=type_state,
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
