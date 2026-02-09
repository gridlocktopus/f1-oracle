"""Canonical race results builder.

Reads raw Ergast race results snapshots from disk and writes a canonical Parquet file
for a single season.

Output convention:
- Write a single Parquet file into a season-partition folder:
    data/canonical/results_race/season=YYYY/results_race.parquet

Schema note (important):
- We do NOT store a `season` column inside the Parquet file, because the season is
  already encoded in the partition folder name (`season=YYYY`). Keeping both can
  cause schema collisions when reading via pyarrow.dataset / pq.read_table.

Status handling (important):
- Ergast provides a free-text-ish status string (e.g., "Finished", "+1 Lap", "Accident", "Engine", "Disqualified").
- We store:
  - status_raw: the original string from Ergast
  - status_category: a stable bucket we control (minimal taxonomy for now)
- We also store:
  - dnf: boolean derived from status_category != "CLASSIFIED"
  - finish_position: int position ONLY when status_category == "CLASSIFIED"
    (otherwise None; the model label layer will collapse non-classified to a single "DNF" class)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class RaceResultRow:
    """One canonical race result row (one driver in one race)."""

    # Race identity (join keys)
    round: int
    race_name: str
    circuit_id: str
    race_date: str  # ISO date string from Ergast (YYYY-MM-DD)

    # Entities
    driver_id: str
    constructor_id: str

    # Classification
    grid_position: int | None
    finish_position: int | None
    position_text: str | None

    # Race metrics
    points: float | None
    laps: int | None
    time_ms: int | None  # best-effort parse from Ergast "Time.millis" when present

    # Outcome
    status_raw: str
    status_category: str
    dnf: bool

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


def _normalize_status(status_raw: str) -> str:
    """Normalize status for robust matching."""
    return " ".join(status_raw.strip().lower().split())


def _status_category(status_raw: str) -> str:
    """Map Ergast status text into a stable category.

    Minimal taxonomy (expandable later without changing raw snapshots):
    - CLASSIFIED: finished / classified (+n laps, etc.)
    - MECHANICAL: engine/gearbox/hydraulics/etc.
    - INCIDENT: accident/collision/spun/crash/etc.
    - DNS: did not start / withdrew
    - DSQ: disqualified
    - OTHER: anything we don't recognize yet
    """
    s = _normalize_status(status_raw)

    # "Finished", "+1 Lap", and "Lapped" style statuses are considered classified.
    if s == "finished" or s.startswith("+") or s == "lapped":
        return "CLASSIFIED"

    # Disqualification / non-start markers.
    if "disqualified" in s:
        return "DSQ"
    if "did not start" in s or "withdrew" in s:
        return "DNS"

    # Incident-like (driver error / contact / crash).
    incident_keywords = (
        "accident",
        "collision",
        "crash",
        "spun",
        "spun off",
        "spin",
        "damage",
    )
    if any(k in s for k in incident_keywords):
        return "INCIDENT"

    # Mechanical-like failures.
    mechanical_keywords = (
        "engine",
        "gearbox",
        "hydraulics",
        "electrical",
        "electronics",
        "brakes",
        "brake",
        "suspension",
        "transmission",
        "clutch",
        "fuel",
        "oil",
        "water",
        "radiator",
        "pneumatics",
        "turbo",
        "power unit",
        "powerunit",
        "power loss",
        "puncture",
        "driveshaft",
        "steering",
        "battery",
        "mgu",
        "ers",
        "retired",
        "mechanical",
        "wheel",
        "tyre",
        "undertray",
        "rear wing",
        "front wing",
        "cooling system",
        "differential",
        "vibrations",
    )
    if any(k in s for k in mechanical_keywords):
        return "MECHANICAL"

    return "OTHER"


def _results_race_schema() -> pa.Schema:
    """Return the stable canonical schema for race results."""
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
            pa.field("grid_position", pa.int64(), nullable=True),
            pa.field("finish_position", pa.int64(), nullable=True),
            pa.field("position_text", pa.string(), nullable=True),
            # Metrics
            pa.field("points", pa.float64(), nullable=True),
            pa.field("laps", pa.int64(), nullable=True),
            pa.field("time_ms", pa.int64(), nullable=True),
            # Outcome
            pa.field("status_raw", pa.string(), nullable=False),
            pa.field("status_category", pa.string(), nullable=False),
            pa.field("dnf", pa.bool_(), nullable=False),
            # Provenance
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
        ]
    )


def _row_from_ergast_result(
    *,
    race_round: int,
    race_name: str,
    circuit_id: str,
    race_date: str,
    result: dict[str, Any],
    source_path: Path,
) -> RaceResultRow:
    """Convert one Ergast Result dict into a canonical RaceResultRow."""
    driver = result.get("Driver", {}) if isinstance(result.get("Driver"), dict) else {}
    constructors = result.get("Constructor", {}) if isinstance(result.get("Constructor"), dict) else {}

    status_raw = str(result.get("status", "")).strip()
    cat = _status_category(status_raw)
    dnf = cat != "CLASSIFIED"

    # Ergast "position" is usually numeric even for many non-finish statuses.
    # For our purposes, we only treat it as a real finish position if classified.
    finish_pos_raw = result.get("position")
    finish_pos = _to_int(finish_pos_raw) if (cat == "CLASSIFIED") else None

    # "positionText" can contain non-numeric markers; we preserve it.
    pos_text = result.get("positionText")
    position_text = str(pos_text) if pos_text is not None else None

    grid_position = _to_int(result.get("grid"))
    points = _to_float(result.get("points"))
    laps = _to_int(result.get("laps"))

    # Best-effort time parsing:
    # If "Time" exists with a "millis" field, capture it. Otherwise None.
    time_ms: int | None = None
    time_obj = result.get("Time")
    if isinstance(time_obj, dict):
        time_ms = _to_int(time_obj.get("millis"))

    return RaceResultRow(
        round=race_round,
        race_name=race_name,
        circuit_id=circuit_id,
        race_date=race_date,
        driver_id=str(driver.get("driverId", "")),
        constructor_id=str(constructors.get("constructorId", "")),
        grid_position=grid_position,
        finish_position=finish_pos,
        position_text=position_text,
        points=points,
        laps=laps,
        time_ms=time_ms,
        status_raw=status_raw if status_raw != "" else "UNKNOWN",
        status_category=cat,
        dnf=dnf,
        source="ergast",
        source_path=str(source_path),
    )


def build_results_race_for_season(season: int, raw_dir: str, canonical_dir: str, overwrite: bool = True) -> Path:
    """Build canonical race results for one season from the raw Ergast race results snapshot.

    Input:
      {raw_dir}/ergast/{season}/results_race.raw.json

    Output:
      {canonical_dir}/results_race/season={season}/results_race.parquet
    """
    raw_path = Path(raw_dir) / "ergast" / str(season) / "results_race.raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw race results file: {raw_path}")

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

        # Guardrails: these should exist for normal Ergast races.
        if race_round is None or race_name == "" or race_date == "" or circuit_id == "":
            # If the raw data is malformed, we skip this race deterministically.
            # (Better than emitting partial join keys into canonical.)
            continue

        results = race.get("Results")
        if not isinstance(results, list):
            # Some races might be missing results in odd historical seasons;
            # we treat it as "no rows" for that race.
            continue

        for result in results:
            if not isinstance(result, dict):
                continue

            row = _row_from_ergast_result(
                race_round=race_round,
                race_name=race_name,
                circuit_id=circuit_id,
                race_date=race_date,
                result=result,
                source_path=raw_path,
            )

            # Basic join-key integrity: driver_id and constructor_id should be present.
            # If either is missing, we skip deterministically.
            if row.driver_id == "" or row.constructor_id == "":
                continue

            rows.append(row.__dict__)

    out_dir = Path(canonical_dir) / "results_race" / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_race.parquet"

    if out_path.exists() and not overwrite:
        return out_path

    schema = _results_race_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path)

    return out_path
