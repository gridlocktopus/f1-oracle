"""
Ergast race results ingestion (raw layer).

Raw layer rules (per design.md):
- Store snapshots verbatim (no normalization, no prettifying).
- Treat raw snapshots as immutable inputs to canonical builders.

Naming (locked):
- CLI: f1-oracle ingest ergast results race --season YYYY
- Raw snapshot: data/raw/ergast/YYYY/results_race.raw.json

Endpoint (Ergast-style via Jolpica):
- {season}/results.json

Important implementation detail:
- The API enforces pagination (often caps limit at 100), even if we request larger limits.
- Therefore we must page through the dataset using (limit, offset) until we reach MRData.total.
- We write ONE season snapshot JSON file that contains the combined RaceTable.Races list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from f1_oracle.ingest.ergast_client import ErgastClient


def _ensure_dir(path: Path) -> None:
    """Create the directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _get_nested(d: dict[str, Any], keys: list[str]) -> Any:
    """Safely walk a nested dict by keys; returns None if any key is missing."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def ingest_results_race_for_season(
    season: int,
    raw_dir: str | Path = "data/raw",
    client: ErgastClient | None = None,
) -> Path:
    """
    Fetch and store the Ergast race results snapshot for a season.

    This endpoint is paginated. We fetch all pages and combine them into one
    season-level JSON snapshot.

    Args:
        season: F1 season year (e.g., 2018).
        raw_dir: Root raw data directory. Defaults to "data/raw".
        client: Optional ErgastClient instance (useful for testing).

    Returns:
        Path to the written raw snapshot JSON file.
    """
    raw_root = Path(raw_dir)
    out_dir = raw_root / "ergast" / str(season)
    _ensure_dir(out_dir)

    c = client or ErgastClient()

    # The API often caps limit at 100 regardless of what we request.
    # We choose a limit of 100 explicitly and page with offset.
    limit = 100
    offset = 0

    combined_payload: dict[str, Any] | None = None
    combined_races: list[dict[str, Any]] = []

    total_expected: int | None = None

    while True:
        raw_text = c.get_text(
            f"{season}/results.json",
            params={"limit": limit, "offset": offset},
        )

        page_payload = json.loads(raw_text)
        if not isinstance(page_payload, dict):
            raise ValueError("Expected raw JSON root to be an object")

        mrdata = page_payload.get("MRData", {})
        if not isinstance(mrdata, dict):
            raise ValueError("Expected MRData to be an object")

        # total is the total number of result rows (not races) for the season.
        # In 2018, for example, total=420 (21 races * 20 drivers).
        total_str = mrdata.get("total")
        if total_str is None:
            raise ValueError("Expected MRData.total to be present")
        try:
            total_expected = int(total_str)
        except (TypeError, ValueError):
            raise ValueError(f"Expected MRData.total to be an integer string, got: {total_str!r}")

        races = _get_nested(page_payload, ["MRData", "RaceTable", "Races"])
        if races is None:
            races = []
        if not isinstance(races, list):
            raise ValueError("Expected MRData.RaceTable.Races to be a list")

        # Initialize combined payload from the first page. We will replace its
        # RaceTable.Races with the combined list before writing to disk.
        if combined_payload is None:
            combined_payload = page_payload

        # Merge races from this page. Pagination for results yields disjoint
        # round ranges in practice, so we can safely extend.
        for r in races:
            if isinstance(r, dict):
                combined_races.append(r)

        offset += limit
        if offset >= total_expected:
            break

    if combined_payload is None:
        raise ValueError("No data returned from API")

    # Overwrite the races list with the combined full-season list.
    race_table = _get_nested(combined_payload, ["MRData", "RaceTable"])
    if not isinstance(race_table, dict):
        raise ValueError("Expected MRData.RaceTable to be an object")

    race_table["Races"] = combined_races

    # Update MRData metadata to reflect that the snapshot is complete.
    mrdata_final = combined_payload.get("MRData")
    if isinstance(mrdata_final, dict):
        mrdata_final["offset"] = "0"
        mrdata_final["limit"] = str(total_expected if total_expected is not None else len(combined_races))

    # Write as compact JSON (verbatim-ish). We are not "prettifying" or normalizing fields;
    # we are only combining pages into a single authoritative season snapshot.
    out_path = out_dir / "results_race.raw.json"
    out_path.write_text(json.dumps(combined_payload), encoding="utf-8")

    return out_path