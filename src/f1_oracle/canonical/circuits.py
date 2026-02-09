"""Canonical circuits builder.

Reads raw Ergast circuits snapshots from disk and writes a canonical Parquet file
for a single season.

Output convention:
- Write a single Parquet file into a season-partition folder:
    data/canonical/circuits/season=YYYY/circuits.parquet

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
class CircuitRow:
    """One canonical circuit row.

    Note: `season` is intentionally NOT a column in the canonical file.
    The season is encoded in the output path partition (`season=YYYY`).
    """

    circuit_id: str
    circuit_name: str
    locality: str
    country: str
    lat: float | None
    long: float | None
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


def _to_float(value: Any) -> float | None:
    """Convert a value to float if possible; otherwise return None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _circuits_schema() -> pa.Schema:
    """Return the stable canonical schema for circuits.

    We define this explicitly so types are stable across seasons.
    """
    return pa.schema(
        [
            pa.field("circuit_id", pa.string(), nullable=False),
            pa.field("circuit_name", pa.string(), nullable=False),
            pa.field("locality", pa.string(), nullable=False),
            pa.field("country", pa.string(), nullable=False),
            pa.field("lat", pa.float64(), nullable=True),
            pa.field("long", pa.float64(), nullable=True),
            pa.field("url", pa.string(), nullable=True),
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
        ]
    )


def _row_from_ergast_circuit(circuit: dict[str, Any], source_path: Path) -> CircuitRow:
    """Convert one Ergast 'Circuit' dict into a canonical CircuitRow."""
    location = circuit.get("Location", {}) if isinstance(circuit.get("Location"), dict) else {}

    lat = _to_float(location.get("lat"))
    long = _to_float(location.get("long"))

    return CircuitRow(
        circuit_id=str(circuit.get("circuitId", "")),
        circuit_name=str(circuit.get("circuitName", "")),
        locality=str(location.get("locality", "")),
        country=str(location.get("country", "")),
        lat=lat,
        long=long,
        url=str(circuit.get("url")) if circuit.get("url") is not None else None,
        source="ergast",
        source_path=str(source_path),
    )


def build_circuits_for_season(season: int, raw_dir: str, canonical_dir: str, overwrite: bool = True) -> Path:
    """Build canonical circuits for one season from the raw Ergast circuits snapshot.

    Input:
      {raw_dir}/ergast/{season}/circuits.raw.json

    Output:
      {canonical_dir}/circuits/season={season}/circuits.parquet

    Overwrite behavior:
      - overwrite=True replaces the output file if it already exists (recommended default).
    """
    raw_path = Path(raw_dir) / "ergast" / str(season) / "circuits.raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw circuits file: {raw_path}")

    payload = json.loads(_read_text(raw_path))
    if not isinstance(payload, dict):
        raise ValueError("Expected raw JSON root to be an object")

    circuits = _get_nested(payload, ["MRData", "CircuitTable", "Circuits"])
    if not isinstance(circuits, list):
        raise ValueError("Expected MRData.CircuitTable.Circuits to be a list")

    rows: list[dict[str, Any]] = []
    for c in circuits:
        if isinstance(c, dict):
            rows.append(_row_from_ergast_circuit(c, raw_path).__dict__)

    out_dir = Path(canonical_dir) / "circuits" / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "circuits.parquet"

    if out_path.exists() and not overwrite:
        return out_path

    schema = _circuits_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path)

    return out_path