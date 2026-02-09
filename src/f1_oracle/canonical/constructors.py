"""Canonical constructors builder.

Reads raw Ergast constructors snapshots from disk and writes a canonical Parquet file
for a single season.

Output convention:
- Write a single Parquet file into a season-partition folder:
    data/canonical/constructors/season=YYYY/constructors.parquet

Schema note (important):
- We do NOT store a `season` column inside the Parquet file, because the season is
  already encoded in the partition folder name (`season=YYYY`). Keeping both can
  cause schema collisions when reading via pyarrow.dataset / pq.read_table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ConstructorRow:
    """One canonical constructor row.

    Note: `season` is intentionally NOT a column in the canonical file.
    The season is encoded in the output path partition (`season=YYYY`).
    """

    constructor_id: str
    constructor_name: str
    nationality: str | None
    url: str | None

    source: str
    source_path: str


def _read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def _get_nested(d: dict[str, Any], keys: list[str]) -> Any:
    """Safely walk a nested dict by keys; returns None if any key is missing."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _constructors_schema() -> pa.Schema:
    """Return the stable canonical schema for constructors."""
    return pa.schema(
        [
            pa.field("constructor_id", pa.string(), nullable=False),
            pa.field("constructor_name", pa.string(), nullable=False),
            pa.field("nationality", pa.string(), nullable=True),
            pa.field("url", pa.string(), nullable=True),
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
        ]
    )


def _row_from_ergast_constructor(constructor: dict[str, Any], source_path: Path) -> ConstructorRow:
    """Convert one Ergast 'Constructor' dict into a canonical ConstructorRow."""
    return ConstructorRow(
        constructor_id=str(constructor.get("constructorId", "")),
        constructor_name=str(constructor.get("name", "")),
        nationality=str(constructor.get("nationality")) if constructor.get("nationality") is not None else None,
        url=str(constructor.get("url")) if constructor.get("url") is not None else None,
        source="ergast",
        source_path=str(source_path),
    )


def build_constructors_for_season(season: int, raw_dir: str, canonical_dir: str, overwrite: bool = True) -> Path:
    """Build canonical constructors for one season from the raw Ergast constructors snapshot.

    Input:
      {raw_dir}/ergast/{season}/constructors.raw.json

    Output:
      {canonical_dir}/constructors/season={season}/constructors.parquet

    Overwrite behavior:
      - overwrite=True replaces the output file if it already exists (recommended default).
    """
    raw_path = Path(raw_dir) / "ergast" / str(season) / "constructors.raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw constructors file: {raw_path}")

    payload = json.loads(_read_text(raw_path))
    if not isinstance(payload, dict):
        raise ValueError("Expected raw JSON root to be an object")

    constructors = _get_nested(payload, ["MRData", "ConstructorTable", "Constructors"])
    if not isinstance(constructors, list):
        raise ValueError("Expected MRData.ConstructorTable.Constructors to be a list")

    rows: list[dict[str, Any]] = []
    for c in constructors:
        if isinstance(c, dict):
            rows.append(_row_from_ergast_constructor(c, raw_path).__dict__)

    out_dir = Path(canonical_dir) / "constructors" / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "constructors.parquet"

    if out_path.exists() and not overwrite:
        return out_path

    schema = _constructors_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path)

    return out_path