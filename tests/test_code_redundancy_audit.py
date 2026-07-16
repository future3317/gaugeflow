import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    path = ROOT / "scripts" / "audit_code_redundancy.py"
    spec = importlib.util.spec_from_file_location("audit_code_redundancy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_active_production_surface_has_no_static_redundancy_findings():
    result = _load_audit_module().audit()
    for finding in (
        "duplicate_normalized_bodies",
        "unreferenced_private_definitions",
        "stored_but_unread_self_attributes",
        "lexically_unreachable_statement_lines",
        "constant_boolean_branch_lines",
        "unused_cli_arguments",
    ):
        assert result[finding] in ([], {})
