"""
No-leakage tests for feature builders.

These tests are intentionally simple and enforce the most important invariant:

For season S and round r:
- any season-to-date feature must depend only on rounds < r (same season)
- career priors must depend only on seasons < S

We validate a couple of representative features against canonical data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd
import pyarrow.dataset as ds

from f1_oracle.common.io import load_yaml
from f1_oracle.features.baseline import build_features_qualifying_baseline_for_season


def _canonical_dir() -> str:
    paths = load_yaml(Path("configs") / "paths.yaml")
    return paths.get("canonical", {}).get("dir", "data/canonical")


def test_baseline_quali_no_leakage_driver_season_avg_matches_prior_rounds() -> None:
    """
    For a few rounds, verify driver_q_avg_pos_season_prior equals the mean qualifying_position
    from results_qualifying where round < r in the same season.
    """
    canonical_dir = _canonical_dir()
    season = 2024

    # Build features
    df = build_features_qualifying_baseline_for_season(
        season=season,
        canonical_dir=canonical_dir,
        track_types_cfg=load_yaml(Path("configs") / "weekend_types.yaml"),
    )
    assert not df.empty

    # Load canonical qualifying results for this season
    d_q = ds.dataset(f"{canonical_dir}/results_qualifying", format="parquet", partitioning="hive")
    q = d_q.to_table(filter=ds.field("season") == season).to_pandas()
    q["qualifying_position"] = pd.to_numeric(q["qualifying_position"], errors="coerce")
    q["round"] = pd.to_numeric(q["round"], errors="coerce")

    # Check a few rounds (only those that exist in the built feature frame)
    available_rounds = set(pd.to_numeric(df["round"], errors="coerce").dropna().astype(int).unique())
    rounds_to_check = [r for r in [1, 2, 5, 10] if r in available_rounds]
    assert rounds_to_check, "No expected rounds found in feature frame for no-leakage test"

    for r in rounds_to_check:
        df_r = df[df["round"] == r]
        assert not df_r.empty

        q_prior = q[q["round"] < r]
        prior_means = q_prior.groupby("driver_id")["qualifying_position"].mean()

        # For each driver row at round r:
        for _, row in df_r.iterrows():
            d = row["driver_id"]
            expected = prior_means.get(d, pd.NA)

            if r == 1 or pd.isna(expected):
                assert pd.isna(row["driver_q_avg_pos_season_prior"])
            else:
                # float32 comparisons: allow small tolerance
                assert float(row["driver_q_avg_pos_season_prior"]) == pytest.approx(float(expected), abs=1e-5)


def test_baseline_quali_career_priors_exclude_current_season() -> None:
    """
    For at least one driver, verify driver_q_avg_pos_career_prior excludes the current season.
    """
    canonical_dir = _canonical_dir()
    season = 2024

    df = build_features_qualifying_baseline_for_season(
        season=season,
        canonical_dir=canonical_dir,
        track_types_cfg=load_yaml(Path("configs") / "weekend_types.yaml"),
    )
    assert not df.empty

    # pick a driver from season 2024
    driver_id = df["driver_id"].iloc[0]

    d_q = ds.dataset(f"{canonical_dir}/results_qualifying", format="parquet", partitioning="hive")
    q = d_q.to_table(filter=ds.field("season") <= season).to_pandas()
    q["qualifying_position"] = pd.to_numeric(q["qualifying_position"], errors="coerce")
    q["season"] = pd.to_numeric(q["season"], errors="coerce")

    career_only = q[(q["season"] < season) & (q["driver_id"] == driver_id)]
    expected = career_only["qualifying_position"].mean()

    # Compare against round 1 row (baseline safest)
    row = df[(df["round"] == 1) & (df["driver_id"] == driver_id)].iloc[0]

    if pd.isna(expected):
        assert pd.isna(row["driver_q_avg_pos_career_prior"])
    else:
        assert float(row["driver_q_avg_pos_career_prior"]) == pytest.approx(float(expected), abs=1e-5)