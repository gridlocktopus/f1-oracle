"""Canonical sprint results builder.

Reads raw Ergast sprint results snapshots from disk and writes a canonical Parquet file
for a single season.

Output convention:
- data/canonical/results_sprint/season=YYYY/results_sprint.parquet

Not all seasons have sprints:
- In that case, the raw snapshot should exist but contain zero races / zero rows.
- Canonical build should still write a valid empty Parquet file with the correct schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class SprintResultRow:
    round: int
    race_name: str
    circuit_id: str
    race_date: str

    driver_id: str
    constructor_id: str

    grid_position: int | None
    finish_position: int | None
    position_text: str | None

    points: float | None
    laps: int | None
    time_ms: int | None

    status_raw: str
    status_category: str
    dnf: bool

    source: str
    source_path: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _get_nested(d: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _to_int(value: Any) -> int | None:
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
    return " ".join(status_raw.strip().lower().split())


def _status_category(status_raw: str) -> str:
    s = _normalize_status(status_raw)

    if s == "finished" or s.startswith("+"):
        return "CLASSIFIED"

    if "disqualified" in s:
        return "DSQ"
    if "did not start" in s or "withdrew" in s:
        return "DNS"

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

    mechanical_keywords = (
        "engine",
        "gearbox",
        "hydraulics",
        "electrical",
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
        "puncture",
        "driveshaft",
        "steering",
        "battery",
        "mgu",
        "ers",
    )
    if any(k in s for k in mechanical_keywords):
        return "MECHANICAL"

    return "OTHER"


def _results_sprint_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("round", pa.int64(), nullable=False),
            pa.field("race_name", pa.string(), nullable=False),
            pa.field("circuit_id", pa.string(), nullable=False),
            pa.field("race_date", pa.string(), nullable=False),
            pa.field("driver_id", pa.string(), nullable=False),
            pa.field("constructor_id", pa.string(), nullable=False),
            pa.field("grid_position", pa.int64(), nullable=True),
            pa.field("finish_position", pa.int64(), nullable=True),
            pa.field("position_text", pa.string(), nullable=True),
            pa.field("points", pa.float64(), nullable=True),
            pa.field("laps", pa.int64(), nullable=True),
            pa.field("time_ms", pa.int64(), nullable=True),
            pa.field("status_raw", pa.string(), nullable=False),
            pa.field("status_category", pa.string(), nullable=False),
            pa.field("dnf", pa.bool_(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
        ]
    )


def _row_from_ergast_sprint(
    *,
    race_round: int,
    race_name: str,
    circuit_id: str,
    race_date: str,
    entry: dict[str, Any],
    source_path: Path,
) -> SprintResultRow:
    driver = entry.get("Driver", {}) if isinstance(entry.get("Driver"), dict) else {}
    constructor = entry.get("Constructor", {}) if isinstance(entry.get("Constructor"), dict) else {}

    status_raw = str(entry.get("status", "")).strip()
    cat = _status_category(status_raw)
    dnf = cat != "CLASSIFIED"

    finish_pos = _to_int(entry.get("position")) if cat == "CLASSIFIED" else None
    pos_text_val = entry.get("positionText")
    position_text = str(pos_text_val) if pos_text_val is not None else None

    grid_position = _to_int(entry.get("grid"))
    points = _to_float(entry.get("points"))
    laps = _to_int(entry.get("laps"))

    time_ms: int | None = None
    time_obj = entry.get("Time")
    if isinstance(time_obj, dict):
        time_ms = _to_int(time_obj.get("millis"))

    return SprintResultRow(
        round=race_round,
        race_name=race_name,
        circuit_id=circuit_id,
        race_date=race_date,
        driver_id=str(driver.get("driverId", "")),
        constructor_id=str(constructor.get("constructorId", "")),
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


def build_results_sprint_for_season(season: int, raw_dir: str, canonical_dir: str, overwrite: bool = True) -> Path:
    """Build canonical sprint results for one season from the raw snapshot.

    Input:
      {raw_dir}/ergast/{season}/results_sprint.raw.json

    Output:
      {canonical_dir}/results_sprint/season={season}/results_sprint.parquet
    """
    raw_path = Path(raw_dir) / "ergast" / str(season) / "results_sprint.raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw sprint results file: {raw_path}")

    payload = json.loads(_read_text(raw_path))
    if not isinstance(payload, dict):
        raise ValueError("Expected raw JSON root to be an object")

    races = _get_nested(payload, ["MRData", "RaceTable", "Races"])
    if races is None:
        races = []
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

        sprint_results = race.get("SprintResults")
        if not isinstance(sprint_results, list):
            continue

        for entry in sprint_results:
            if not isinstance(entry, dict):
                continue

            row = _row_from_ergast_sprint(
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

    out_dir = Path(canonical_dir) / "results_sprint" / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_sprint.parquet"

    if out_path.exists() and not overwrite:
        return out_path

    table = pa.Table.from_pylist(rows, schema=_results_sprint_schema())
    pq.write_table(table, out_path)

    return out_path