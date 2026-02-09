"""
Training orchestration (as-of round).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.dataset as ds

from f1_oracle.common.io import load_yaml
from f1_oracle.models.quali_model import train_quali_model
from f1_oracle.models.race_model import train_race_models
from f1_oracle.models.registry import save_model


def train_quali_as_of(*, season: int, rnd: int) -> str:
    model = train_quali_model(season=season, rnd=rnd)
    artifact = save_model("quali", model)
    return str(artifact.path)


def train_race_as_of(*, season: int, rnd: int) -> tuple[str, str]:
    models = train_race_models(season=season, rnd=rnd)
    finish_art = save_model("race_finish", models["finish"])
    dnf_art = save_model("race_dnf", models["dnf"])
    return str(finish_art.path), str(dnf_art.path)


def _canonical_dir() -> str:
    cfg = load_yaml(Path("configs") / "paths.yaml")
    return cfg.get("canonical", {}).get("dir", "data/canonical")


def _max_season_in_dataset(dataset: str) -> int | None:
    canonical_dir = _canonical_dir()
    root = Path(canonical_dir) / dataset
    if not root.exists():
        return None
    dsq = ds.dataset(str(root), format="parquet", partitioning="hive")
    seasons = dsq.to_table(columns=["season"]).to_pandas()["season"].dropna().unique().tolist()
    if not seasons:
        return None
    return int(max(seasons))


def train_all() -> tuple[str, str, str]:
    """
    Train qualifying and race models on all available canonical data
    up to the configured train_end_year.

    We train "as of" season = train_end_year + 1, round = 1, which includes all
    historical seasons and avoids any same-season leakage logic.
    """
    seasons_cfg = load_yaml(Path("configs") / "seasons.yaml")
    train_end_year = int(seasons_cfg["split"]["train_end_year"])

    max_q = _max_season_in_dataset("results_qualifying")
    max_r = _max_season_in_dataset("results_race")
    if max_q is None or max_r is None:
        raise ValueError("Missing canonical results data; cannot train.")

    max_season = min(max_q, max_r, train_end_year)
    target_season = max_season + 1

    quali_path = train_quali_as_of(season=target_season, rnd=1)
    race_finish, race_dnf = train_race_as_of(season=target_season, rnd=1)
    return quali_path, race_finish, race_dnf
