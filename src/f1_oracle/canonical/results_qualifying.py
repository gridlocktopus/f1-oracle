"""Canonical qualifying results builder.

Reads raw Ergast qualifying results snapshots from disk and writes a canonical Parquet
file for a single season.

Output convention:
- Write a single Parquet file into a season-partition folder:
    data/canonical/results_qualifying/season=YYYY/results_qualifying.parquet

Schema note (important):
- We do NOT store a `season` column inside the Parquet file, because the season is
  already encoded in the partition folder name (`season=YYYY`).

Qualifying times:
- Ergast qualifying provides Q1/Q2/Q3 as strings like "1:23.456" (or may be missing).
- We store both:
  - q1_time / q2_time / q3_time (original strings)
  - q1_ms / q2_ms / q3_ms (best-effort parse to milliseconds)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class QualifyingResultRow:
    """One canonical qualifying result row (one driver in one qualifying session)."""

    # Race identity (join keys)
    round: int
    race_name: str
    circuit_id: str
    race_date: str  # Ergast race date (YYYY-MM-DD)

    # Entities
    driver_id: str
    constructor_id: str

    # Qualifying classification
    qualifying_position: int | None

    # Times (raw strings)
    q1_time: str | None
    q2_time: str | None
    q3_time: str | None

    # Times (best-effort parsed to milliseconds)
    q1_ms: int | None
    q2_ms: int | None
    q3_ms: int | None

    # Provenance
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


def _parse_laptime_ms(value: Any) -> int | None:
    """Parse a lap time string like '1:23.456' into milliseconds.

    Ergast uses mm:ss.mmm (minutes may be absent in some series, but F1 is usually present).
    Returns None if missing or unparseable.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None

    s = value.strip()
    if s == "":
        return None

    # Expected formats:
    #  - "1:23.456"
    #  - "59.123" (rare, but handle)
    #  - "0:59.123" (handle)
    try:
        if ":" in s:
            mins_str, rest = s.split(":", 1)
            mins = int(mins_str)
            secs_str, ms_str = rest.split(".", 1)
            secs = int(secs_str)
            ms = int(ms_str.ljust(3, "0")[:3])
            return (mins * 60 * 1000) + (secs * 1000) + ms
        else:
            secs_str, ms_str = s.split(".", 1)
            secs = int(secs_str)
            ms = int(ms_str.ljust(3, "0")[:3])
            return (secs * 1000) + ms
    except Exception:
        return None


def _results_qualifying_schema() -> pa.Schema:
    """Return the stable canonical schema for qualifying results."""
    return pa.schema(
        [
            # Race identity
            pa.field("round", pa.int64(), nullable=False),
            pa.field("race_name", pa.string(), nullable=False),
            pa.field("circuit_id", pa.string(), nullable=False),
            pa.field("race_date", pa.string(), nullable=False),
            # Entities
            pa.field("driver_id", pa.string(), nullable=False),
            pa.field("constructor_id", pa.string(), nullable=False),
            # Classification
            pa.field("qualifying_position", pa.int64(), nullable=True),
            # Times (raw strings)
            pa.field("q1_time", pa.string(), nullable=True),
            pa.field("q2_time", pa.string(), nullable=True),
            pa.field("q3_time", pa.string(), nullable=True),
            # Times (ms)
            pa.field("q1_ms", pa.int64(), nullable=True),
            pa.field("q2_ms", pa.int64(), nullable=True),
            pa.field("q3_ms", pa.int64(), nullable=True),
            # Provenance
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
        ]
    )


def _row_from_ergast_qualifying(
    *,
    race_round: int,
    race_name: str,
    circuit_id: str,
    race_date: str,
    entry: dict[str, Any],
    source_path: Path,
) -> QualifyingResultRow:
    """Convert one Ergast QualifyingResults entry into a canonical row."""
    driver = entry.get("Driver", {}) if isinstance(entry.get("Driver"), dict) else {}
    constructor = entry.get("Constructor", {}) if isinstance(entry.get("Constructor"), dict) else {}

    q1 = entry.get("Q1")
    q2 = entry.get("Q2")
    q3 = entry.get("Q3")

    q1_time = str(q1) if isinstance(q1, str) else None
    q2_time = str(q2) if isinstance(q2, str) else None
    q3_time = str(q3) if isinstance(q3, str) else None

    return QualifyingResultRow(
        round=race_round,
        race_name=race_name,
        circuit_id=circuit_id,
        race_date=race_date,
        driver_id=str(driver.get("driverId", "")),
        constructor_id=str(constructor.get("constructorId", "")),
        qualifying_position=_to_int(entry.get("position")),
        q1_time=q1_time,
        q2_time=q2_time,
        q3_time=q3_time,
        q1_ms=_parse_laptime_ms(q1_time),
        q2_ms=_parse_laptime_ms(q2_time),
        q3_ms=_parse_laptime_ms(q3_time),
        source="ergast",
        source_path=str(source_path),
    )


def build_results_qualifying_for_season(season: int, raw_dir: str, canonical_dir: str, overwrite: bool = True) -> Path:
    """Build canonical qualifying results for one season from the raw snapshot.

    Input:
      {raw_dir}/ergast/{season}/results_qualifying.raw.json

    Output:
      {canonical_dir}/results_qualifying/season={season}/results_qualifying.parquet
    """
    raw_path = Path(raw_dir) / "ergast" / str(season) / "results_qualifying.raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw qualifying results file: {raw_path}")

    payload = json.loads(_read_text(raw_path))
    if not isinstance(payload, dict):
        raise ValueError("Expected raw JSON root to be an object")

    races = _get_nested(payload, ["MRData", "RaceTable", "Races"])
    if not isinstance(races, list):
        raise ValueError("Expected MRData.RaceTable.Races to be a list")

    rows: list[dict[str, Any]] = []

    for race in races:
        if not isinstance(race, dict):
            continue

        race_round = _to_int(race.get("round"))
        race_name = str(race.get("raceName", "")).strip()
        race_date = str(race.get("date", "")).strip()

        circuit = race.get("Circuit", {}) if isinstance(race.get("Circuit"), dict) else {}
        circuit_id = str(circuit.get("circuitId", "")).strip()

        if race_round is None or race_name == "" or race_date == "" or circuit_id == "":
            continue

        quali = race.get("QualifyingResults")
        if not isinstance(quali, list):
            continue

        for entry in quali:
            if not isinstance(entry, dict):
                continue

            row = _row_from_ergast_qualifying(
                race_round=race_round,
                race_name=race_name,
                circuit_id=circuit_id,
                race_date=race_date,
                entry=entry,
                source_path=raw_path,
            )

            if row.driver_id == "" or row.constructor_id == "":
                continue

            rows.append(row.__dict__)

    out_dir = Path(canonical_dir) / "results_qualifying" / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_qualifying.parquet"

    if out_path.exists() and not overwrite:
        return out_path

    table = pa.Table.from_pylist(rows, schema=_results_qualifying_schema())
    pq.write_table(table, out_path)

    return out_path