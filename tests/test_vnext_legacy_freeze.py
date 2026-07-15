from pathlib import Path

from gaugeflow.vnext.legacy import (
    EXECUTION_CONTRACT_SHA256,
    LEGACY_SOURCE_COMMIT,
    load_manifest,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "vnext_legacy_frozen_v1" / "manifest.json"


def test_vnext_legacy_manifest_is_complete_and_unchanged():
    manifest = load_manifest(MANIFEST)
    assert manifest["legacy_source_commit"] == LEGACY_SOURCE_COMMIT
    assert manifest["execution_contract"]["sha256"] == EXECUTION_CONTRACT_SHA256
    assert manifest["file_count"] >= 250
    assert verify_manifest(ROOT, manifest) == []


def test_vnext_freeze_records_q0_checkpoint_blocker_and_prohibitions():
    manifest = load_manifest(MANIFEST)
    assert manifest["q0_input_availability"]["legacy_checkpoint"] is False
    status = manifest["scientific_status"]
    assert status["real_tensor_authorized"] is False
    assert status["oracle_authorized"] is False
    assert status["relaxation_dft_dfpt_authorized"] is False
