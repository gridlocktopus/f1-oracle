"""Canonical weekends builder.

Reads raw Ergast calendar snapshots from disk and writes a canonical Parquet file
for a single season.

Important implementation detail:
- Avoid pandas' partitioned dataset writer (df.to_parquet(..., partition_cols=...)),
  which can leave background resources running and prevent the process from exiting
  cleanly on some setups.
- Instead, write a single Parquet file into a season-partition folder:
    data/canonical/weekends/season=YYYY/weekends.parquet

Schema note (important):
- We do NOT store a `season` column inside the Parquet file, because the season is
  already encoded in the partition folder name (`season=YYYY`). Keeping both causes
  schema collisions when reading via pyarrow.dataset / pq.read_table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class WeekendRow:
    """One canonical weekend row (one GP weekend).

    Note: `season` is intentionally NOT a column in the canonical file.
    The season is encoded in the output path partition (`season=YYYY`).
    """

    round: int
    weekend_id: str

    race_name: str
    circuit_id: str
    circuit_name: str
    locality: str
    country: str

    race_date: str | None
    fp1_date: str | None
    fp2_date: str | None
    fp3_date: str | None
    qualifying_date: str | None
    sprint_date: str | None

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


def _make_weekend_id(season: int, rnd: int) -> str:
    """Generate a stable weekend identifier.

    Even though `season` is not stored as a column, we bake it into weekend_id so
    the ID remains globally unique across seasons.
    """
    return f"{season}-{rnd:02d}"


def _row_from_ergast_race(season: int, race: dict[str, Any], source_path: Path) -> WeekendRow:
    """Convert one Ergast 'Race' dict into a canonical WeekendRow."""
    rnd = int(race["round"])
    weekend_id = _make_weekend_id(season, rnd)

    circuit = race.get("Circuit", {}) if isinstance(race.get("Circuit"), dict) else {}
    location = circuit.get("Location", {}) if isinstance(circuit.get("Location"), dict) else {}

    fp1_date = _get_nested(race, ["FirstPractice", "date"])
    fp2_date = _get_nested(race, ["SecondPractice", "date"])
    fp3_date = _get_nested(race, ["ThirdPractice", "date"])
    qualifying_date = _get_nested(race, ["Qualifying", "date"])

    # Ergast variants sometimes use "Sprint" or "SprintQualifying".
    sprint_date = _get_nested(race, ["Sprint", "date"]) or _get_nested(race, ["SprintQualifying", "date"])

    return WeekendRow(
        round=rnd,
        weekend_id=weekend_id,
        race_name=str(race.get("raceName", "")),
        circuit_id=str(circuit.get("circuitId", "")),
        circuit_name=str(circuit.get("circuitName", "")),
        locality=str(location.get("locality", "")),
        country=str(location.get("country", "")),
        race_date=str(race.get("date")) if race.get("date") is not None else None,
        fp1_date=str(fp1_date) if fp1_date is not None else None,
        fp2_date=str(fp2_date) if fp2_date is not None else None,
        fp3_date=str(fp3_date) if fp3_date is not None else None,
        qualifying_date=str(qualifying_date) if qualifying_date is not None else None,
        sprint_date=str(sprint_date) if sprint_date is not None else None,
        source="ergast",
        source_path=str(source_path),
    )


def _weekends_schema() -> pa.Schema:
    """Return the stable canonical schema for weekends.

    We define this explicitly so columns that are all-null for a season (e.g. sprint_date
    in 2018) still have the correct type (string), instead of being inferred as 'null'.
    """
    return pa.schema(
        [
            pa.field("round", pa.int64(), nullable=False),
            pa.field("weekend_id", pa.string(), nullable=False),
            pa.field("race_name", pa.string(), nullable=False),
            pa.field("circuit_id", pa.string(), nullable=False),
            pa.field("circuit_name", pa.string(), nullable=False),
            pa.field("locality", pa.string(), nullable=False),
            pa.field("country", pa.string(), nullable=False),
            pa.field("race_date", pa.string(), nullable=True),
            pa.field("fp1_date", pa.string(), nullable=True),
            pa.field("fp2_date", pa.string(), nullable=True),
            pa.field("fp3_date", pa.string(), nullable=True),
            pa.field("qualifying_date", pa.string(), nullable=True),
            pa.field("sprint_date", pa.string(), nullable=True),
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
        ]
    )


def build_weekends_for_season(season: int, raw_dir: str, canonical_dir: str, overwrite: bool = True) -> Path:
    """Build canonical weekends for one season from the raw Ergast calendar snapshot.

    Input:
      {raw_dir}/ergast/{season}/races.raw.json

    Output:
      {canonical_dir}/weekends/season={season}/weekends.parquet

    Overwrite behavior:
      - overwrite=True replaces the output file if it already exists (recommended default).
    """
    raw_path = Path(raw_dir) / "ergast" / str(season) / "races.raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw calendar file: {raw_path}")

    payload = json.loads(_read_text(raw_path))
    if not isinstance(payload, dict):
        raise ValueError("Expected raw JSON root to be an object")

    races = _get_nested(payload, ["MRData", "RaceTable", "Races"])
    if not isinstance(races, list):
        raise ValueError("Expected MRData.RaceTable.Races to be a list")

    rows: list[dict[str, Any]] = []
    for race in races:
        if isinstance(race, dict):
            rows.append(_row_from_ergast_race(season, race, raw_path).__dict__)

    out_dir = Path(canonical_dir) / "weekends" / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weekends.parquet"

    if out_path.exists() and not overwrite:
        return out_path

    # Build an Arrow table with an explicit schema (stable types across seasons),
    # then write a single Parquet file.
    schema = _weekends_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path)

    return out_path