import torch

from scripts.audit_h1a_j0_side_information_sensitivity import (
    _cyclically_permute_node_tokens,
    _lattice_shape_donor_positions,
    _paired_relative_bootstrap,
)


def test_cyclic_token_intervention_is_graph_local_and_composition_preserving() -> None:
    tokens = torch.tensor([4, 7, 9, 2, 2], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    observed = _cyclically_permute_node_tokens(tokens, batch, 2)

    assert torch.equal(observed, torch.tensor([7, 9, 4, 2, 2]))
    assert torch.equal(torch.sort(observed[batch == 0]).values, torch.sort(tokens[batch == 0]).values)
    assert torch.equal(torch.sort(observed[batch == 1]).values, torch.sort(tokens[batch == 1]).values)


def test_lattice_donors_only_rotate_compatible_non_singleton_groups() -> None:
    counts = torch.tensor([4, 4, 4, 6, 6], dtype=torch.long)
    log_volumes = torch.tensor([2.01, 2.09, 2.51, 3.01, 3.51], dtype=torch.float64)

    donors, coverage = _lattice_shape_donor_positions(counts, log_volumes, bin_width=0.25)

    assert torch.equal(donors, torch.tensor([1, 0, 2, 3, 4]))
    assert coverage == 0.4
    assert torch.equal(counts[donors], counts)
    assert torch.equal(
        torch.floor(log_volumes[donors] / 0.25),
        torch.floor(log_volumes / 0.25),
    )


def test_paired_bootstrap_detects_uniform_relative_degradation() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0, 4.0])
    variant = 1.25 * reference

    result = _paired_relative_bootstrap(
        reference,
        variant,
        generator=torch.Generator().manual_seed(11),
        replicates=500,
    )

    assert abs(result["relative_mean_change"] - 0.25) < 1.0e-12
    assert abs(result["bootstrap_q025"] - 0.25) < 1.0e-12
    assert abs(result["bootstrap_q975"] - 0.25) < 1.0e-12
