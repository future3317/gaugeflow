from scripts.audit_h1a_coordinate_memorization_scaling import _checks


def test_scaling_checks_apply_every_frozen_threshold():
    acceptance = {
        "coordinate_mse_max": 0.001,
        "explained_fraction_min": 0.995,
        "low_time_endpoint_rms_angstrom_max": 0.01,
        "tensor_candidates": 0,
    }
    metrics = {
        "coordinate_mse": 0.0009,
        "explained_fraction": 0.996,
        "low_time_endpoint_rms_angstrom": 0.009,
        "tensor_candidates": 0.0,
    }
    assert all(_checks(metrics, acceptance).values())
    metrics["explained_fraction"] = 0.9
    assert not _checks(metrics, acceptance)["explained_fraction"]
