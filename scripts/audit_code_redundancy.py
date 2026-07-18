"""Static redundancy audit for active GaugeFlow production code and entry points."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ENTRYPOINTS = (
    ROOT / "scripts" / "sample_production.py",
    ROOT / "scripts" / "train_production.py",
    ROOT / "scripts" / "build_tensororbit_v2_raw.py",
    ROOT / "scripts" / "audit_tensororbit_v2_build.py",
    ROOT / "scripts" / "prepare_v2_oracle_qualification.py",
    ROOT / "scripts" / "audit_alex_mp20_source.py",
    ROOT / "scripts" / "build_alex_h0_split.py",
    ROOT / "scripts" / "audit_alex_h0_split.py",
    ROOT / "scripts" / "build_phonondb_force_constants_v2.py",
    ROOT / "scripts" / "audit_phonondb_h0_b.py",
    ROOT / "scripts" / "audit_h0_activation.py",
    ROOT / "scripts" / "build_h0_d_opd_catalogue_v2.py",
    ROOT / "scripts" / "diagnose_h0_d_opd_catalogue_v2.py",
    ROOT / "scripts" / "audit_h0_d_opd_catalogue_v2.py",
    ROOT / "scripts" / "build_h0_e_maximal_embedding_catalogue_v2.py",
    ROOT / "scripts" / "audit_h0_e_maximal_embedding_catalogue_v2.py",
    ROOT / "scripts" / "build_h1a_p1_structure_cache.py",
    ROOT / "scripts" / "audit_h1a_p1_structure_cache.py",
    ROOT / "scripts" / "audit_h1a_generator_substrate.py",
    ROOT / "scripts" / "audit_h1a_joint_gradients.py",
    ROOT / "scripts" / "audit_h1a_coordinate_reverse_closure.py",
    ROOT / "scripts" / "evaluate_h1a_p1_protocol.py",
    ROOT / "scripts" / "evaluate_h1a_coordinate_pretraining.py",
    ROOT / "scripts" / "audit_h1a_coordinate_state_visibility.py",
    ROOT / "scripts" / "diagnose_h1a_coordinate_generator.py",
    ROOT / "scripts" / "benchmark_h1a_tensor_free.py",
)


@dataclass(frozen=True)
class Definition:
    path: str
    qualified_name: str
    kind: str
    line: int
    private: bool
    body_sha256: str


class ModuleAudit(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.scope: list[str] = []
        self.definitions: list[Definition] = []
        self.references: Counter[str] = Counter()
        self.self_stores: Counter[str] = Counter()
        self.self_loads: Counter[str] = Counter()
        self.unreachable_lines: list[int] = []
        self.constant_branch_lines: list[int] = []

    @staticmethod
    def _body_hash(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
        normalized = copy.deepcopy(node)
        normalized.name = "<definition>"
        normalized.decorator_list = []
        if normalized.body and isinstance(normalized.body[0], ast.Expr):
            value = normalized.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                normalized.body = normalized.body[1:]
        payload = ast.dump(normalized, annotate_fields=True, include_attributes=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _record_definition(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, kind: str
    ) -> None:
        qualified = ".".join((*self.scope, node.name))
        self.definitions.append(
            Definition(
                path=self.path.relative_to(ROOT).as_posix(),
                qualified_name=qualified,
                kind=kind,
                line=node.lineno,
                private=node.name.startswith("_") and not node.name.startswith("__"),
                body_sha256=self._body_hash(node),
            )
        )

    def _visit_scoped(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, kind: str
    ) -> None:
        self._record_definition(node, kind)
        self._scan_block(node.body)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node, "async_function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped(node, "class")

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.references[node.id] += 1
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            self.references[node.attr] += 1
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if isinstance(node.ctx, ast.Load):
                self.self_loads[node.attr] += 1
            elif isinstance(node.ctx, ast.Store):
                self.self_stores[node.attr] += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            self.constant_branch_lines.append(node.lineno)
        self._scan_block(node.body)
        self._scan_block(node.orelse)
        self.generic_visit(node)

    def _scan_block(self, statements: list[ast.stmt]) -> None:
        terminated = False
        for statement in statements:
            if terminated:
                self.unreachable_lines.append(statement.lineno)
            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                terminated = True
            for child_name in ("body", "orelse", "finalbody"):
                child = getattr(statement, child_name, None)
                if isinstance(child, list):
                    self._scan_block(child)


def _active_paths() -> list[Path]:
    production = sorted(
        path
        for path in (ROOT / "src" / "gaugeflow" / "production").glob("*.py")
        if path.name != "__init__.py"
    )
    catalogue = sorted(
        path
        for path in (ROOT / "src" / "gaugeflow" / "catalogue").glob("*.py")
        if path.name != "__init__.py"
    )
    return [*production, *catalogue, *ACTIVE_ENTRYPOINTS]


def _all_paths() -> list[Path]:
    return sorted(
        [*(ROOT / "src" / "gaugeflow").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]
    )


def _argparse_unused(path: Path, tree: ast.AST) -> list[str]:
    declared: set[str] = set()
    accessed: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            flags = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            explicit_dest = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant)
                ),
                None,
            )
            if explicit_dest:
                declared.add(str(explicit_dest))
            elif flags:
                option = next(
                    (value for value in flags if value.startswith("--")), flags[0]
                )
                declared.add(option.lstrip("-").replace("-", "_"))
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"args", "arguments"}
            and isinstance(node.ctx, ast.Load)
        ):
            accessed.add(node.attr)
    return sorted(declared - accessed)


def audit(paths: list[Path] | None = None) -> dict[str, object]:
    selected_paths = _active_paths() if paths is None else paths
    audits: list[ModuleAudit] = []
    cli_unused: dict[str, list[str]] = {}
    for path in selected_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = ModuleAudit(path)
        visitor._scan_block(tree.body)
        visitor.visit(tree)
        audits.append(visitor)
        if path.parent == ROOT / "scripts":
            unused = _argparse_unused(path, tree)
            if unused:
                cli_unused[path.relative_to(ROOT).as_posix()] = unused

    definitions = [
        definition for visitor in audits for definition in visitor.definitions
    ]
    references = sum((visitor.references for visitor in audits), Counter())
    private_unreferenced = [
        asdict(definition)
        for definition in definitions
        if definition.private
        and references[definition.qualified_name.rsplit(".", 1)[-1]] == 0
    ]
    body_groups: dict[str, list[Definition]] = defaultdict(list)
    for definition in definitions:
        short_name = definition.qualified_name.rsplit(".", 1)[-1]
        frozen_builder_provenance = (
            short_name == "sha256_file" and definition.path.startswith("scripts/build_")
        )
        if (
            short_name not in {"__init__", "forward", "main"}
            and not frozen_builder_provenance
        ):
            body_groups[definition.body_sha256].append(definition)
    duplicate_bodies = [
        [asdict(definition) for definition in group]
        for group in body_groups.values()
        if len(group) > 1
    ]
    unused_self_attributes = {
        visitor.path.relative_to(ROOT).as_posix(): sorted(
            set(visitor.self_stores) - set(visitor.self_loads)
        )
        for visitor in audits
        if set(visitor.self_stores) - set(visitor.self_loads)
    }
    unreachable = {
        visitor.path.relative_to(ROOT).as_posix(): sorted(
            set(visitor.unreachable_lines)
        )
        for visitor in audits
        if visitor.unreachable_lines
    }
    constant_branches = {
        visitor.path.relative_to(ROOT).as_posix(): sorted(
            set(visitor.constant_branch_lines)
        )
        for visitor in audits
        if visitor.constant_branch_lines
    }
    return {
        "scope": [path.relative_to(ROOT).as_posix() for path in selected_paths],
        "definition_count": len(definitions),
        "duplicate_normalized_bodies": duplicate_bodies,
        "unreferenced_private_definitions": private_unreferenced,
        "stored_but_unread_self_attributes": unused_self_attributes,
        "lexically_unreachable_statement_lines": unreachable,
        "constant_boolean_branch_lines": constant_branches,
        "unused_cli_arguments": cli_unused,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    parser.add_argument(
        "--scope",
        choices=("active", "all"),
        default="active",
        help="scan the active production surface or all source and scripts",
    )
    arguments = parser.parse_args()
    paths = _active_paths() if arguments.scope == "active" else _all_paths()
    print(
        json.dumps(
            audit(paths), indent=None if arguments.compact else 2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
