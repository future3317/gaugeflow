"""Linear-complexity Cartesian angular moments for periodic edge states."""

from __future__ import annotations

import torch
from torch import nn

from .state_projection import sorted_segment_sum


class FactorizedCartesianAngularMoments(nn.Module):
    r"""Compress low-order triplet correlations without enumerating triplets.

    For scalar edge coefficients ``a`` and ``b`` on incoming edges ``k -> j``,
    the module forms

    ``m_j = sum_k a_k n_k / sqrt(d_j)`` and
    ``Q_j = sum_k b_k STF(n_k n_k^T) / sqrt(d_j)``.

    Each edge ``i -> j`` then receives the invariant contractions
    ``n_i . m_j`` and ``n_i^T Q_j n_i``. Expanding these contractions gives
    the first- and second-order angular kernels over all edge pairs at ``j``.
    The implementation stores only edge- and node-leading tensors, so its
    time and memory scale as ``O(E C)`` rather than ``O(sum_j d_j^2 C)``.
    """

    def __init__(self, edge_dim: int, channels: int) -> None:
        super().__init__()
        if edge_dim < 1 or channels < 1:
            raise ValueError("angular-moment dimensions must be positive")
        self.edge_dim = int(edge_dim)
        self.channels = int(channels)
        self.coefficient_projection = nn.Linear(edge_dim, 2 * channels, bias=False)

    @staticmethod
    def _quadratic_basis(direction: torch.Tensor) -> torch.Tensor:
        """Return six Cartesian components of ``STF(n n^T)``."""
        x, y, z = direction.unbind(dim=-1)
        one_third = direction.new_tensor(1.0 / 3.0)
        return torch.stack(
            (
                x.square() - one_third,
                y.square() - one_third,
                z.square() - one_third,
                x * y,
                x * z,
                y * z,
            ),
            dim=-1,
        )

    @staticmethod
    def _quadratic_contraction(
        moment: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        """Contract symmetric Cartesian components as ``n^T Q n``."""
        x, y, z = direction.unbind(dim=-1)
        return (
            moment[:, :, 0] * x[:, None].square()
            + moment[:, :, 1] * y[:, None].square()
            + moment[:, :, 2] * z[:, None].square()
            + 2.0 * moment[:, :, 3] * (x * y)[:, None]
            + 2.0 * moment[:, :, 4] * (x * z)[:, None]
            + 2.0 * moment[:, :, 5] * (y * z)[:, None]
        )

    @property
    def output_dim(self) -> int:
        return 2 * self.channels

    def forward(
        self,
        edge_state: torch.Tensor,
        edge_target: torch.Tensor,
        edge_direction: torch.Tensor,
        edge_envelope: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        if edge_state.ndim != 2 or edge_state.shape[1] != self.edge_dim:
            raise ValueError("angular moments received the wrong edge-state shape")
        if edge_target.shape != edge_state.shape[:1] or edge_target.dtype != torch.long:
            raise ValueError("angular moments require one int64 target per edge")
        if edge_direction.shape != (edge_state.shape[0], 3):
            raise ValueError("angular moments require one Cartesian direction per edge")
        if edge_envelope.shape != (edge_state.shape[0], 1):
            raise ValueError("angular moments require an [edges,1] envelope")
        if node_count < 0:
            raise ValueError("angular moments require a nonnegative node count")
        if edge_target.numel() and (
            int(edge_target.min()) < 0 or int(edge_target.max()) >= node_count
        ):
            raise ValueError("angular-moment target lies outside node support")
        if edge_state.shape[0] == 0:
            return edge_state.new_empty((0, self.output_dim))

        # The periodic graph is target-sorted, so both reductions are linear
        # contiguous segment sums.  No (i,j,k) index or E-by-degree expansion
        # is constructed anywhere in this path.
        coefficients = torch.tanh(self.coefficient_projection(edge_state))
        first_coefficients, second_coefficients = coefficients.split(
            self.channels, dim=-1
        )
        weighted_first = first_coefficients * edge_envelope
        weighted_second = second_coefficients * edge_envelope
        degree = torch.bincount(edge_target, minlength=node_count).to(edge_state)
        degree_scale = degree.clamp_min(1.0).rsqrt()

        quadratic_basis = self._quadratic_basis(edge_direction)
        # One wide contiguous reduction is faster on GPU than two small
        # reductions. The concatenated Cartesian components are algebraically
        # independent and split immediately after summation.
        all_moments = sorted_segment_sum(
            torch.cat(
                (
                    weighted_first[:, :, None] * edge_direction[:, None, :],
                    weighted_second[:, :, None] * quadratic_basis[:, None, :],
                ),
                dim=-1,
            ),
            edge_target,
            node_count,
        )
        all_moments = all_moments * degree_scale[:, None, None]
        first_moment, second_moment = all_moments.split((3, 6), dim=-1)

        linear = torch.einsum(
            "eci,ei->ec", first_moment[edge_target], edge_direction
        )
        quadratic = self._quadratic_contraction(
            second_moment[edge_target], edge_direction
        )

        return torch.cat((linear, quadratic), dim=-1)
