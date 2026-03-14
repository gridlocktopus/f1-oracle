"""
Update + retrain orchestration.
"""

from __future__ import annotations

from pathlib import Path

from f1_oracle.common.io import load_yaml
from f1_oracle.ingest.ergast import (
    ingest_results_qualifying_for_season,
    ingest_results_race_for_season,
    ingest_results_sprint_for_season,
)
from f1_oracle.canonical.results_qualifying import build_results_qualifying_for_season
from f1_oracle.canonical.results_race import build_results_race_for_season
from f1_oracle.canonical.results_sprint import build_results_sprint_for_season
from f1_oracle.orchestrator.train_as_of import train_quali_as_of, train_race_as_of


def _load_paths() -> tuple[str, str]:
    paths = load_yaml(Path("configs") / "paths.yaml")
    raw_dir = paths.get("raw", {}).get("dir", "data/raw")
    canonical_dir = paths.get("canonical", {}).get("dir", "data/canonical")
    return raw_dir, canonical_dir


def update_quali_and_retrain(*, season: int, rnd: int) -> str:
    raw_dir, canonical_dir = _load_paths()

    ingest_results_qualifying_for_season(season=season, raw_dir=raw_dir)
    build_results_qualifying_for_season(season=season, raw_dir=raw_dir, canonical_dir=canonical_dir)
    ingest_results_sprint_for_season(season=season, raw_dir=raw_dir)
    build_results_sprint_for_season(season=season, raw_dir=raw_dir, canonical_dir=canonical_dir)

    # retrain for next round (include current round results)
    return train_quali_as_of(season=season, rnd=rnd + 1)


def update_race_and_retrain(*, season: int, rnd: int) -> tuple[str, str]:
    raw_dir, canonical_dir = _load_paths()

    ingest_results_race_for_season(season=season, raw_dir=raw_dir)
    build_results_race_for_season(season=season, raw_dir=raw_dir, canonical_dir=canonical_dir)
    ingest_results_sprint_for_season(season=season, raw_dir=raw_dir)
    build_results_sprint_for_season(season=season, raw_dir=raw_dir, canonical_dir=canonical_dir)

    # retrain for next round (include current round results)
    return train_race_as_of(season=season, rnd=rnd + 1)
