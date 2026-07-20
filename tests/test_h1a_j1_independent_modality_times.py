import torch

from scripts.evaluate_h1a_j1_independent_modality_times import (
    CORNER_NAMES,
    _corner_side_times,
    _paired_bootstrap_mean_difference,
    _paired_bootstrap_ratio,
)


def test_j1_corner_time_contract() -> None:
    coordinate = torch.tensor([0.2, 0.7])
    interior_element = torch.tensor([0.3, 0.4])
    interior_lattice = torch.tensor([0.8, 0.1])
    expected = {
        "clean_clean": (torch.zeros(2), torch.zeros(2)),
        "noisy_element": (coordinate, torch.zeros(2)),
        "noisy_lattice": (torch.zeros(2), coordinate),
        "diagonal": (coordinate, coordinate),
        "interior": (interior_element, interior_lattice),
    }
    assert set(CORNER_NAMES) == set(expected)
    for name, values in expected.items():
        observed = _corner_side_times(
            name,
            coordinate,
            interior_element,
            interior_lattice,
        )
        torch.testing.assert_close(observed[0], values[0])
        torch.testing.assert_close(observed[1], values[1])


def test_j1_structure_bootstrap_is_paired_and_deterministic() -> None:
    initial = torch.tensor([1.0, 2.0, 4.0, 8.0])
    final = 0.5 * initial
    first = _paired_bootstrap_ratio(initial, final, seed=91, replicates=200)
    second = _paired_bootstrap_ratio(initial, final, seed=91, replicates=200)
    assert first == second
    assert first == {"q025": 0.5, "median": 0.5, "q975": 0.5}


def test_j1_structure_paired_difference_bootstrap_preserves_sign() -> None:
    left = torch.tensor([0.2, 0.4, 0.8, 1.6])
    right = left + 0.1
    result = _paired_bootstrap_mean_difference(left, right, seed=92, replicates=200)
    assert result["mean"] < 0.0
    assert result["q975"] < 0.0
