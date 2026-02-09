"""
Ergast sprint results ingestion (raw layer).

Raw layer rules (per design.md):
- Store snapshots verbatim (no normalization, no prettifying).
- Treat raw snapshots as immutable inputs to canonical builders.

Naming (locked):
- CLI: f1-oracle ingest ergast results sprint --season YYYY
- Raw snapshot: data/raw/ergast/YYYY/results_sprint.raw.json

Endpoint (Ergast-style via Jolpica):
- {season}/sprint.json

Important implementation detail:
- The API enforces pagination (often caps limit at 100).
- Therefore we must page through using (limit, offset) until MRData.total.
- Not all seasons have sprint sessions. In that case:
  - The API usually returns MRData.total == "0" and an empty Races list.
  - We still write a raw snapshot file (empty, but valid), and we do NOT raise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from f1_oracle.ingest.ergast_client import ErgastClient


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_nested(d: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def ingest_results_sprint_for_season(
    season: int,
    raw_dir: str | Path = "data/raw",
    client: ErgastClient | None = None,
) -> Path:
    """Fetch and store the Ergast sprint results snapshot for a season (paginated)."""
    raw_root = Path(raw_dir)
    out_dir = raw_root / "ergast" / str(season)
    _ensure_dir(out_dir)

    c = client or ErgastClient()

    limit = 100
    offset = 0

    combined_payload: dict[str, Any] | None = None
    combined_races: list[dict[str, Any]] = []

    total_expected: int | None = None

    while True:
        raw_text = c.get_text(
            f"{season}/sprint.json",
            params={"limit": limit, "offset": offset},
        )

        page_payload = json.loads(raw_text)
        if not isinstance(page_payload, dict):
            raise ValueError("Expected raw JSON root to be an object")

        mrdata = page_payload.get("MRData", {})
        if not isinstance(mrdata, dict):
            raise ValueError("Expected MRData to be an object")

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

        if combined_payload is None:
            combined_payload = page_payload

        for r in races:
            if isinstance(r, dict):
                combined_races.append(r)

        # If the season has no sprints, total_expected will be 0.
        # In that case we should break immediately after the first fetch.
        if total_expected == 0:
            break

        offset += limit
        if offset >= total_expected:
            break

    if combined_payload is None:
        raise ValueError("No data returned from API")

    race_table = _get_nested(combined_payload, ["MRData", "RaceTable"])
    if not isinstance(race_table, dict):
        raise ValueError("Expected MRData.RaceTable to be an object")
    race_table["Races"] = combined_races

    mrdata_final = combined_payload.get("MRData")
    if isinstance(mrdata_final, dict):
        mrdata_final["offset"] = "0"
        mrdata_final["limit"] = str(total_expected if total_expected is not None else len(combined_races))

    out_path = out_dir / "results_sprint.raw.json"
    out_path.write_text(json.dumps(combined_payload), encoding="utf-8")

    return out_path