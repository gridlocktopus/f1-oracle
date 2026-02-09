"""Canonical drivers builder.

Reads raw Ergast drivers snapshots from disk and writes a canonical Parquet file
for a single season.

Output convention:
- Write a single Parquet file into a season-partition folder:
    data/canonical/drivers/season=YYYY/drivers.parquet

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
class DriverRow:
    """One canonical driver row.

    Note: `season` is intentionally NOT a column in the canonical file.
    The season is encoded in the output path partition (`season=YYYY`).
    """

    driver_id: str
    permanent_number: int | None
    code: str | None
    given_name: str
    family_name: str
    date_of_birth: str | None
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


def _to_int(value: Any) -> int | None:
    """Convert a value to int if possible; otherwise return None."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        try:
            return int(s)
        except ValueError:
            return None
    return None


def _drivers_schema() -> pa.Schema:
    """Return the stable canonical schema for drivers."""
    return pa.schema(
        [
            pa.field("driver_id", pa.string(), nullable=False),
            pa.field("permanent_number", pa.int64(), nullable=True),
            pa.field("code", pa.string(), nullable=True),
            pa.field("given_name", pa.string(), nullable=False),
            pa.field("family_name", pa.string(), nullable=False),
            pa.field("date_of_birth", pa.string(), nullable=True),
            pa.field("nationality", pa.string(), nullable=True),
            pa.field("url", pa.string(), nullable=True),
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
        ]
    )


def _row_from_ergast_driver(driver: dict[str, Any], source_path: Path) -> DriverRow:
    """Convert one Ergast 'Driver' dict into a canonical DriverRow."""
    return DriverRow(
        driver_id=str(driver.get("driverId", "")),
        permanent_number=_to_int(driver.get("permanentNumber")),
        code=str(driver.get("code")) if driver.get("code") is not None else None,
        given_name=str(driver.get("givenName", "")),
        family_name=str(driver.get("familyName", "")),
        date_of_birth=str(driver.get("dateOfBirth")) if driver.get("dateOfBirth") is not None else None,
        nationality=str(driver.get("nationality")) if driver.get("nationality") is not None else None,
        url=str(driver.get("url")) if driver.get("url") is not None else None,
        source="ergast",
        source_path=str(source_path),
    )


def build_drivers_for_season(season: int, raw_dir: str, canonical_dir: str, overwrite: bool = True) -> Path:
    """Build canonical drivers for one season from the raw Ergast drivers snapshot.

    Input:
      {raw_dir}/ergast/{season}/drivers.raw.json

    Output:
      {canonical_dir}/drivers/season={season}/drivers.parquet

    Overwrite behavior:
      - overwrite=True replaces the output file if it already exists (recommended default).
    """
    raw_path = Path(raw_dir) / "ergast" / str(season) / "drivers.raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw drivers file: {raw_path}")

    payload = json.loads(_read_text(raw_path))
    if not isinstance(payload, dict):
        raise ValueError("Expected raw JSON root to be an object")

    drivers = _get_nested(payload, ["MRData", "DriverTable", "Drivers"])
    if not isinstance(drivers, list):
        raise ValueError("Expected MRData.DriverTable.Drivers to be a list")

    rows: list[dict[str, Any]] = []
    for d in drivers:
        if isinstance(d, dict):
            rows.append(_row_from_ergast_driver(d, raw_path).__dict__)

    out_dir = Path(canonical_dir) / "drivers" / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "drivers.parquet"

    if out_path.exists() and not overwrite:
        return out_path

    schema = _drivers_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path)

    return out_path