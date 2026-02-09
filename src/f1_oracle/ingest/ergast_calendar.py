"""Ergast calendar ingestion (raw snapshot).

Downloads a season calendar and stores the raw provider response on disk.
No prettifying, no schema changes, no parsing beyond writing bytes/text.

Downstream canonical builders should read from this raw snapshot.
"""

from __future__ import annotations

from pathlib import Path

from f1_oracle.ingest.ergast_client import ErgastClient


def _ensure_dir(dir_path: Path) -> None:
    """Create directory tree if needed."""
    dir_path.mkdir(parents=True, exist_ok=True)


def ingest_calendar_for_season(
    season: int,
    raw_dir: str,
    client: ErgastClient | None = None,
) -> Path:
    """Download and store the calendar for one season as a raw snapshot.

    Layout:
        {raw_dir}/ergast/{season}/races.raw.json

    Args:
        season: F1 season year (e.g. 2018).
        raw_dir: Root directory for raw data (typically from configs/paths.yaml).
        client: Optional injected ErgastClient (useful for tests).

    Returns:
        Path to the written raw JSON file.
    """
    c = client or ErgastClient()

    # Ergast responses are paginated; use a high limit so the full season comes back in one go.
    raw_text = c.get_text(f"{season}.json", params={"limit": 1000})

    out_dir = Path(raw_dir) / "ergast" / str(season)
    _ensure_dir(out_dir)

    out_path = out_dir / "races.raw.json"
    out_path.write_text(raw_text, encoding="utf-8")

    return out_path