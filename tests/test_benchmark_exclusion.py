from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.alex_p1_data import PACKED_ALEX_P1_PROTOCOL
from gaugeflow.production.benchmark_exclusion import build_alex_benchmark_exclusion


def test_alex_benchmark_exclusion_normalizes_and_deduplicates_ids(tmp_path: Path) -> None:
    cache = tmp_path / "alex"
    cache.mkdir()
    entries: dict[str, object] = {}
    for split, ids in (("val", ["alex<AGM1>", "mp-1"]), ("test", ["agm1", "MP-2"])):
        path = cache / f"{split}.parquet"
        pq.write_table(
            pa.table({"material_id": ids, "cache_row": list(range(len(ids)))}), path
        )
        entries[split] = {
            "index_file": path.name,
            "index_sha256": sha256_file(path),
            "rows": len(ids),
        }
    (cache / "manifest.json").write_text(
        json.dumps(
            {"protocol": PACKED_ALEX_P1_PROTOCOL, "qualified": True, "splits": entries}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "exclusion"
    result = build_alex_benchmark_exclusion(cache, output)
    assert result["qualified"] is True
    assert result["unique_normalized_ids"] == 3
    assert json.loads((output / "material_ids.json").read_text()) == [
        "agm1",
        "mp-1",
        "mp-2",
    ]
    assert sha256_file(output / "material_ids.json") == result["material_ids_sha256"]
