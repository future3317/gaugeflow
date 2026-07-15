"""Generate or verify the immutable vNext legacy-evidence inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gaugeflow.vnext.legacy import build_manifest, load_manifest, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts" / "vnext_legacy_frozen_v1" / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if args.verify:
        failures = verify_manifest(ROOT, load_manifest(manifest_path))
        if failures:
            raise SystemExit("\n".join(failures))
        print(json.dumps({"status": "verified", "manifest": str(manifest_path)}, indent=2))
        return
    if manifest_path.exists():
        raise SystemExit("refusing to overwrite an existing frozen manifest")
    manifest = build_manifest(ROOT)
    manifest_path.parent.mkdir(parents=True, exist_ok=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen", "file_count": manifest["file_count"]}, indent=2))


if __name__ == "__main__":
    main()
