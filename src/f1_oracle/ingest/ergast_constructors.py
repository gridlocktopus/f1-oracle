"""
Ergast constructors ingestion (raw layer).

Raw layer rules (per design.md):
- Store snapshots verbatim (no normalization, no prettifying).
- Treat raw snapshots as immutable inputs to canonical builders.

This module mirrors the style of ergast_calendar.py / ergast_circuits.py / ergast_drivers.py,
but targets the constructors endpoint.
"""

from __future__ import annotations

from pathlib import Path

from f1_oracle.ingest.ergast_client import ErgastClient


def _ensure_dir(path: Path) -> None:
    """Create the directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def ingest_constructors_for_season(
    season: int,
    raw_dir: str | Path = "data/raw",
    client: ErgastClient | None = None,
) -> Path:
    """
    Fetch and store the Ergast constructors snapshot for a season.

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

    # IMPORTANT: Store response exactly as received (verbatim).
    c = client or ErgastClient()
    raw_text = c.get_text(
        f"{season}/constructors.json",
        params={"limit": 1000},
    )

    out_path = out_dir / "constructors.raw.json"
    out_path.write_text(raw_text, encoding="utf-8")

    return out_path