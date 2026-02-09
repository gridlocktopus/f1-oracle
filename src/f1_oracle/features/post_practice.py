"""
Post-practice feature builder for qualifying predictions.

Uses FastF1 practice results (FP1/FP2/FP3) when available and merges
with baseline qualifying features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from f1_oracle.common.io import load_yaml
from f1_oracle.features.baseline import build_features_qualifying_baseline_for_season


def _load_paths_cfg() -> dict[str, Any]:
    return load_yaml(Path("configs") / "paths.yaml")


def _load_practice_raw(season: int, rnd: int, raw_dir: str) -> pd.DataFrame:
    base = Path(raw_dir) / "fastf1" / f"season={season}" / f"round={rnd}"
    if not base.exists():
        return pd.DataFrame()

    frames = []
    for path in base.glob("practice_*.parquet"):
        df = pd.read_parquet(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _aggregate_practice(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["driver_id"])

    df = df.copy()
    df["best_lap_time_ms"] = pd.to_numeric(df["best_lap_time_ms"], errors="coerce")

    agg = (
        df.groupby("driver_id")
        .agg(
            practice_sessions_count=("session_type", "nunique"),
            practice_best_lap_ms=("best_lap_time_ms", "min"),
            practice_best_lap_ms_mean=("best_lap_time_ms", "mean"),
            practice_best_lap_ms_median=("best_lap_time_ms", "median"),
        )
        .reset_index()
    )

    # Rank within weekend (1 = fastest)
    agg["practice_best_lap_rank"] = agg["practice_best_lap_ms"].rank(method="min")
    best = agg["practice_best_lap_ms"].min()
    agg["practice_best_lap_gap_ms"] = agg["practice_best_lap_ms"] - best

    # Normalize best lap (z-score) within the round
    mean = agg["practice_best_lap_ms"].mean()
    std = agg["practice_best_lap_ms"].std()
    if pd.notna(std) and std > 0:
        agg["practice_best_lap_z"] = (agg["practice_best_lap_ms"] - mean) / std
    else:
        agg["practice_best_lap_z"] = 0.0

    return agg


def build_features_qualifying_post_practice_for_round(
    *,
    season: int,
    rnd: int,
    track_types_cfg: dict[str, Any] | None,
) -> pd.DataFrame:
    """
    Build post-practice qualifying features for a single season/round.

    Merges baseline qualifying features with practice-derived aggregates.
    """
    paths = _load_paths_cfg()
    canonical_dir = paths.get("canonical", {}).get("dir", "data/canonical")
    raw_dir = paths.get("raw", {}).get("dir", "data/raw")

    baseline = build_features_qualifying_baseline_for_season(
        season=season, canonical_dir=canonical_dir, track_types_cfg=track_types_cfg
    )
    if baseline.empty:
        return pd.DataFrame()

    base_round = baseline[baseline["round"] == rnd].copy()
    if base_round.empty:
        return pd.DataFrame()

    practice_raw = _load_practice_raw(season, rnd, raw_dir)
    practice = _aggregate_practice(practice_raw)

    out = base_round.merge(practice, on="driver_id", how="left")

    out["checkpoint"] = "post_practice"
    out["session_type"] = "Q"

    # Fill counts for missing practice data
    out["practice_sessions_count"] = (
        pd.to_numeric(out["practice_sessions_count"], errors="coerce").fillna(0).astype("Int64")
    )

    # If we have any practice data for the round, filter to drivers who participated
    # to avoid carrying over previous-season substitutes into a new season.
    if not practice.empty:
        out = out[out["practice_sessions_count"] > 0].copy()

    return out
