"""
Post-qualifying feature builder for race predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from f1_oracle.common.io import load_yaml
from f1_oracle.features.baseline import _load_track_types
from f1_oracle.features.post_practice import _load_practice_raw, _aggregate_practice


def _load_paths_cfg() -> dict[str, Any]:
    return load_yaml(Path("configs") / "paths.yaml")


def _read_canonical(canonical_dir: str, name: str) -> ds.Dataset:
    root = Path(canonical_dir) / name
    return ds.dataset(str(root), format="parquet", partitioning="hive")


def _load_laps_summary(season: int, rnd: int, raw_dir: str) -> pd.DataFrame:
    """
    Load long-run pace summaries from practice laps.
    Prefer FP2, then FP3, then FP1.
    """
    base = Path(raw_dir) / "fastf1" / f"season={season}" / f"round={rnd}"
    if not base.exists():
        return pd.DataFrame()

    for session_type in ("fp2", "fp3", "fp1"):
        path = base / f"practice_laps_{session_type}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "driver_id" in df.columns:
                return df
    return pd.DataFrame()


def build_features_race_post_quali_for_round(*, season: int, rnd: int) -> pd.DataFrame:
    """
    Build post-qualifying race features for a single season/round.

    Includes qualifying position for the round and historical race priors.
    """
    paths = _load_paths_cfg()
    canonical_dir = paths.get("canonical", {}).get("dir", "data/canonical")

    d_entries = _read_canonical(canonical_dir, "entries")
    d_weekends = _read_canonical(canonical_dir, "weekends")
    d_q = _read_canonical(canonical_dir, "results_qualifying")
    d_r = _read_canonical(canonical_dir, "results_race")

    entries = d_entries.to_table(
        filter=(ds.field("season") == season) & (ds.field("round") == rnd)
    ).to_pandas()
    if entries.empty:
        return pd.DataFrame()

    weekends = d_weekends.to_table(filter=ds.field("season") == season).to_pandas()

    q_round = d_q.to_table(
        filter=(ds.field("season") == season) & (ds.field("round") == rnd)
    ).to_pandas()

    r_all = d_r.to_table(filter=ds.field("season") <= season).to_pandas()

    for df in (entries, weekends, q_round, r_all):
        if "round" in df.columns:
            df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")

    base = entries.merge(
        weekends[["round", "weekend_id", "circuit_id", "race_date", "race_name"]],
        on=["round", "circuit_id", "race_date"],
        how="left",
        suffixes=("", "_wk"),
    )

    # Track type mapping (optional)
    base["track_type"] = pd.NA
    track_types_cfg = load_yaml(Path("configs") / "weekend_types.yaml")
    track_types_df = _load_track_types(track_types_cfg)
    if not track_types_df.empty:
        if "weekend_id" in track_types_df.columns:
            base = base.merge(track_types_df[["weekend_id", "track_type"]], on="weekend_id", how="left")
        elif {"season", "round", "track_type"}.issubset(track_types_df.columns):
            base["season"] = season
            base = base.merge(track_types_df[["season", "round", "track_type"]], on=["season", "round"], how="left")

    # Attach qualifying position for this round
    q_round["qualifying_position"] = pd.to_numeric(q_round["qualifying_position"], errors="coerce").astype("Int64")
    base = base.merge(
        q_round[["driver_id", "constructor_id", "qualifying_position"]],
        on=["driver_id", "constructor_id"],
        how="left",
    )
    # Only keep drivers with qualifying results for this round
    base = base[base["qualifying_position"].notna()].copy()

    # Teammate-relative qualifying delta for the round
    constructor_q_avg = base.groupby("constructor_id")["qualifying_position"].mean()
    base["constructor_q_avg_pos_round"] = base["constructor_id"].map(constructor_q_avg)
    base["driver_vs_teammate_q_delta_round"] = (
        base["qualifying_position"] - base["constructor_q_avg_pos_round"]
    )

    # Practice-derived pace (optional but useful for race)
    paths = _load_paths_cfg()
    raw_dir = paths.get("raw", {}).get("dir", "data/raw")
    practice_raw = _load_practice_raw(season, rnd, raw_dir)
    practice = _aggregate_practice(practice_raw)
    if not practice.empty:
        base = base.merge(practice, on="driver_id", how="left")
        base["practice_sessions_count"] = (
            pd.to_numeric(base["practice_sessions_count"], errors="coerce").fillna(0).astype("Int64")
        )

    # Long-run pace from FP2 (fallback to FP3/FP1 if needed)
    laps = _load_laps_summary(season, rnd, raw_dir)
    if not laps.empty:
        base = base.merge(laps, on="driver_id", how="left")

    # Prior race form: season-to-date only (rounds < r)
    r_prior = r_all[
        (r_all["season"] == season) & (r_all["round"] < rnd)
    ].copy()
    r_prior["finish_position"] = pd.to_numeric(r_prior["finish_position"], errors="coerce")
    r_prior["dnf"] = pd.to_numeric(r_prior["dnf"], errors="coerce").fillna(0)

    driver_finish_avg = r_prior.groupby("driver_id")["finish_position"].mean()
    driver_dnf_rate = r_prior.groupby("driver_id")["dnf"].mean()
    constructor_finish_avg = r_prior.groupby("constructor_id")["finish_position"].mean()

    base["driver_r_avg_pos_season_prior"] = base["driver_id"].map(driver_finish_avg)
    base["driver_r_dnf_rate_season_prior"] = base["driver_id"].map(driver_dnf_rate)
    base["constructor_r_avg_pos_season_prior"] = base["constructor_id"].map(constructor_finish_avg)

    base["season"] = season
    base["checkpoint"] = "post_quali"
    base["session_type"] = "RACE"

    return base
