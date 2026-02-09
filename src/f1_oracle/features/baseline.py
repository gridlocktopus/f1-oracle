"""
Baseline (pre-FP1) feature frame builders.

This module builds deterministic, checkpoint-specific feature rows. For v0.3 we
start with a baseline qualifying feature frame, using canonical-only data
(no FastF1, no weather) and strict no-leakage rules:

- For round r in season S:
  - "season-to-date" features use only (season == S and round < r)
  - "career" priors use only (season < S)
  - Track/circuit and track-type priors use (season < S) plus (season == S and round < r)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


@dataclass(frozen=True)
class BaselineQualiSchema:
    """
    Minimal schema contract for features_qualifying_baseline.

    NOTE: We enforce dtypes at the end of build to prevent inference drift.
    """

    # keys + context
    season: str = "int32"
    round: str = "int16"
    weekend_id: str = "string"
    race_name: str = "string"
    circuit_id: str = "string"
    race_date: str = "datetime64[ns]"  # canonical is a date-ish field; we keep it as datetime64 for safety
    driver_id: str = "string"
    constructor_id: str = "string"
    checkpoint: str = "string"
    session_type: str = "string"

    # counts (support null-safe stats)
    driver_season_quali_count_prior: str = "int16"
    constructor_season_quali_count_prior: str = "int16"
    driver_career_quali_count_prior: str = "int32"
    constructor_career_quali_count_prior: str = "int32"

    # season-to-date driver quali form
    driver_q_avg_pos_season_prior: str = "float32"
    driver_q_med_pos_season_prior: str = "float32"
    driver_q_best_pos_season_prior: str = "Int16"
    driver_q_worst_pos_season_prior: str = "Int16"

    # season-to-date constructor quali strength
    constructor_q_avg_pos_season_prior: str = "float32"
    constructor_q_best_pos_season_prior: str = "Int16"

    # teammate-relative season-to-date
    driver_vs_teammate_q_avg_delta_season_prior: str = "float32"

    # reliability (race-derived, season-to-date)
    driver_season_race_starts_prior: str = "int16"
    driver_season_dnfs_prior: str = "int16"
    driver_season_dnf_rate_prior: str = "float32"

    # career priors (seasons < S only)
    driver_q_avg_pos_career_prior: str = "float32"
    constructor_q_avg_pos_career_prior: str = "float32"

    # track (circuit) priors (seasons < S plus earlier rounds in S)
    driver_q_count_at_circuit_prior: str = "int16"
    driver_q_avg_pos_at_circuit_prior: str = "float32"
    constructor_q_count_at_circuit_prior: str = "int16"
    constructor_q_avg_pos_at_circuit_prior: str = "float32"

    # track type priors (seasons < S plus earlier rounds in S)
    track_type: str = "string"
    driver_q_count_at_track_type_prior: str = "int16"
    driver_q_avg_pos_at_track_type_prior: str = "float32"
    constructor_q_count_at_track_type_prior: str = "int16"
    constructor_q_avg_pos_at_track_type_prior: str = "float32"

    # driver-team tenure (all prior rounds in seasons <S plus earlier rounds in S)
    driver_constructor_tenure_rounds_prior: str = "int32"

    # racecraft proxy: finish minus grid (prior only)
    driver_finish_minus_grid_avg_season_prior: str = "float32"
    driver_finish_minus_grid_avg_career_prior: str = "float32"
    driver_finish_minus_grid_count_season_prior: str = "int16"
    driver_finish_minus_grid_count_career_prior: str = "int32"


def _read_canonical_dataset(canonical_dir: str, dataset_name: str) -> ds.Dataset:
    """
    Read a canonical dataset as a hive-partitioned pyarrow dataset.

    We rely on hive partitioning so 'season' can be inferred from season=YYYY folders.
    """
    root = Path(canonical_dir) / dataset_name
    return ds.dataset(str(root), format="parquet", partitioning="hive")


def _load_track_types(track_types_cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Load track types mapping from configs/weekend_types.yaml into a DataFrame.

    We support a few flexible shapes to avoid refactors:
      A) {"weekend_id": {"2024-01": "STREET", ...}}
      B) {"by_weekend_id": {...}}
      C) {"by_season_round": {"2024": {"1": "STREET", ...}, ...}}

    Output columns: weekend_id, track_type OR season, round, track_type
    """
    if "by_weekend_id" in track_types_cfg:
        mapping = track_types_cfg["by_weekend_id"]
        return pd.DataFrame(
            [{"weekend_id": k, "track_type": v} for k, v in mapping.items()],
            columns=["weekend_id", "track_type"],
        )

    if "weekend_id" in track_types_cfg and isinstance(track_types_cfg["weekend_id"], dict):
        mapping = track_types_cfg["weekend_id"]
        return pd.DataFrame(
            [{"weekend_id": k, "track_type": v} for k, v in mapping.items()],
            columns=["weekend_id", "track_type"],
        )

    if "by_season_round" in track_types_cfg:
        rows: list[dict[str, Any]] = []
        by_sr = track_types_cfg["by_season_round"]
        for season_str, by_round in by_sr.items():
            for round_str, ttype in by_round.items():
                rows.append({"season": int(season_str), "round": int(round_str), "track_type": ttype})
        return pd.DataFrame(rows, columns=["season", "round", "track_type"])

    # Unknown shape: return empty mapping; features will become null/0-count.
    return pd.DataFrame(columns=["weekend_id", "track_type"])


def _ensure_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce a column to numeric (nullable) safely."""
    return pd.to_numeric(df[col], errors="coerce")


def build_features_qualifying_baseline_for_season(
    *,
    season: int,
    canonical_dir: str,
    track_types_cfg: dict[str, Any] | None,
) -> pd.DataFrame:
    """
    Build the baseline qualifying features for a given season.

    Returns a DataFrame with one row per (season, round, driver_id), sorted
    deterministically and with explicit dtypes applied at the end.
    """
    # ---------
    # Load canonical datasets (as datasets) and then into pandas as needed.
    # ---------
    d_entries = _read_canonical_dataset(canonical_dir, "entries")
    d_weekends = _read_canonical_dataset(canonical_dir, "weekends")
    d_q = _read_canonical_dataset(canonical_dir, "results_qualifying")
    d_r = _read_canonical_dataset(canonical_dir, "results_race")

    # Base rows for this season: entries for season S
    t_entries_s = d_entries.to_table(filter=ds.field("season") == season)
    entries_s = t_entries_s.to_pandas()
    if entries_s.empty:
        # Deterministic empty output
        return pd.DataFrame()

    # Weekends for this season: used to attach weekend_id and (optionally) track_type
    weekends_s = d_weekends.to_table(filter=ds.field("season") == season).to_pandas()

    # Qualifying results all seasons up to S (we'll slice per round)
    q_all = d_q.to_table(filter=ds.field("season") <= season).to_pandas()
    # Race results all seasons up to S (for DNF and finish-minus-grid)
    r_all = d_r.to_table(filter=ds.field("season") <= season).to_pandas()

    # Ensure types for key columns we rely on
    for df in (entries_s, weekends_s, q_all, r_all):
        if "round" in df.columns:
            df["round"] = _ensure_numeric(df, "round").astype("Int64")
        if "race_date" in df.columns:
            df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    # Attach weekend_id to entries for season
    # Join keys: round + circuit_id + race_date (race_name can be noisy; we keep it as a debug column)
    base = (
        entries_s.merge(
            weekends_s[["round", "weekend_id", "circuit_id", "race_date", "race_name"]],
            on=["round", "circuit_id", "race_date"],
            how="left",
            suffixes=("", "_wk"),
        )
        .copy()
    )

    # Track type mapping
    base["track_type"] = pd.NA
    track_types_df = pd.DataFrame()
    if track_types_cfg is not None:
        track_types_df = _load_track_types(track_types_cfg)

    if not track_types_df.empty:
        if "weekend_id" in track_types_df.columns:
            base = base.merge(track_types_df[["weekend_id", "track_type"]], on="weekend_id", how="left")
        elif {"season", "round", "track_type"}.issubset(track_types_df.columns):
            base["season"] = season
            base = base.merge(track_types_df[["season", "round", "track_type"]], on=["season", "round"], how="left")

    # Add constants for checkpoint / target session type
    base["season"] = season
    base["checkpoint"] = "baseline"
    base["session_type"] = "Q"

    # Determine max round in this season from entries
    max_round = int(base["round"].max())

    # Precompute career-only qualifying priors (seasons < S)
    q_career = q_all[q_all["season"] < season].copy()
    if not q_career.empty:
        q_career["qualifying_position"] = _ensure_numeric(q_career, "qualifying_position").astype("Int64")
    driver_q_avg_career = q_career.groupby("driver_id")["qualifying_position"].mean()
    driver_q_count_career = q_career.groupby("driver_id")["qualifying_position"].count()

    constructor_q_avg_career = q_career.groupby("constructor_id")["qualifying_position"].mean()
    constructor_q_count_career = q_career.groupby("constructor_id")["qualifying_position"].count()

    # Precompute career-only finish-minus-grid (seasons < S)
    r_career = r_all[r_all["season"] < season].copy()
    if not r_career.empty:
        r_career["grid_position"] = _ensure_numeric(r_career, "grid_position").astype("Int64")
        r_career["finish_position"] = _ensure_numeric(r_career, "finish_position").astype("Int64")
        r_career["dnf"] = _ensure_numeric(r_career, "dnf").fillna(0).astype("Int64")
        r_career["finish_minus_grid"] = (r_career["finish_position"] - r_career["grid_position"]).where(
            (r_career["dnf"] == 0) & r_career["finish_position"].notna() & r_career["grid_position"].notna()
        )
    else:
        # Ensure the column exists so downstream groupby doesn't crash on empty slices.
        r_career["finish_minus_grid"] = pd.Series(dtype="float64")

    driver_fmg_avg_career = r_career.groupby("driver_id")["finish_minus_grid"].mean()
    driver_fmg_count_career = r_career.groupby("driver_id")["finish_minus_grid"].count()

    # We'll build round-by-round rows and concat.
    # Ensure each per-round chunk has the exact same columns in the same order to avoid
    # pandas concat dtype inference warnings (and future behavior changes).
    expected_cols = [f.name for f in fields(BaselineQualiSchema)]

    schema_dtype_map = BaselineQualiSchema().__dict__.copy()

    def _nullable_dtype(dtype_str: str) -> str:
        """Map schema dtype strings to pandas dtypes safe for per-round chunk construction."""
        if dtype_str == "int32":
            return "Int32"
        if dtype_str == "int16":
            return "Int16"
        # Keep declared nullable types as-is
        if dtype_str in {"Int16", "Int32", "Int64"}:
            return dtype_str
        # Strings and floats can be used directly
        if dtype_str in {"string", "float32"}:
            return dtype_str
        # Datetime
        if dtype_str.startswith("datetime64"):
            return "datetime64[ns]"
        # Fallback: let pandas handle
        return dtype_str

    def _schema_align_chunk(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure df has all schema columns with stable dtypes, then order columns."""
        n = len(df)
        for col, dtype_str in schema_dtype_map.items():
            if col not in df.columns:
                df[col] = pd.Series([pd.NA] * n, dtype=_nullable_dtype(dtype_str))
            else:
                # If present, coerce to a stable dtype when possible (safe for NA).
                tgt = _nullable_dtype(dtype_str)
                if tgt.startswith("datetime64"):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif tgt.startswith("Int"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(tgt)
                elif tgt == "float32":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
                elif tgt == "string":
                    df[col] = df[col].astype("string")
        return df.reindex(columns=expected_cols)

    out_rows: list[pd.DataFrame] = []

    for r in range(1, max_round + 1):
        # Rows for this round r
        base_r = base[base["round"] == r].copy()
        if base_r.empty:
            continue

        # Season-to-date slices: rounds < r
        q_season_prior = q_all[(q_all["season"] == season) & (q_all["round"] < r)].copy()
        r_season_prior = r_all[(r_all["season"] == season) & (r_all["round"] < r)].copy()
        entries_season_prior = entries_s[entries_s["round"] < r].copy()

        # Track/circuit and track_type priors: seasons < S plus earlier rounds in S
        q_track_prior = q_all[
            (q_all["circuit_id"].notna())
            & (
                (q_all["season"] < season)
                | ((q_all["season"] == season) & (q_all["round"] < r))
            )
        ].copy()

        # Attach track_type to qualifying results for track-type priors if we can
        q_track_prior["track_type"] = pd.NA
        if not track_types_df.empty:
            # join via weekends to get weekend_id first, then track_type
            w_prior = d_weekends.to_table(
                filter=(ds.field("season") < season) | ((ds.field("season") == season) & (ds.field("round") < r))
            ).to_pandas()
            w_prior["race_date"] = pd.to_datetime(w_prior["race_date"], errors="coerce")

            q_track_prior = q_track_prior.merge(
                w_prior[["season", "round", "circuit_id", "race_date", "weekend_id"]],
                on=["season", "round", "circuit_id", "race_date"],
                how="left",
            )

            if "weekend_id" in track_types_df.columns:
                q_track_prior = q_track_prior.merge(
                    track_types_df[["weekend_id", "track_type"]],
                    on="weekend_id",
                    how="left",
                )
            elif {"season", "round", "track_type"}.issubset(track_types_df.columns):
                q_track_prior = q_track_prior.merge(
                    track_types_df[["season", "round", "track_type"]],
                    on=["season", "round"],
                    how="left",
                )

        # Ensure numeric positions
        for df in (q_season_prior, q_track_prior):
            if not df.empty:
                df["qualifying_position"] = _ensure_numeric(df, "qualifying_position").astype("Int64")

        # -----------------------------
        # Season-to-date driver quali stats
        # -----------------------------
        driver_season_q = q_season_prior.groupby("driver_id")["qualifying_position"]
        driver_q_count_season = driver_season_q.count()
        driver_q_avg_season = driver_season_q.mean()
        driver_q_med_season = driver_season_q.median()
        driver_q_best_season = driver_season_q.min()
        driver_q_worst_season = driver_season_q.max()

        # Season-to-date constructor quali stats
        constructor_season_q = q_season_prior.groupby("constructor_id")["qualifying_position"]
        constructor_q_count_season = constructor_season_q.count()
        constructor_q_avg_season = constructor_season_q.mean()
        constructor_q_best_season = constructor_season_q.min()

        # -----------------------------
        # Teammate delta (season-to-date)
        # -----------------------------
        # Build a small frame with (driver_id, constructor_id, driver_avg_q_season)
        tmp_driver_avg = (
            q_season_prior.groupby(["constructor_id", "driver_id"])["qualifying_position"]
            .mean()
            .rename("driver_q_avg_pos_season_prior")
            .reset_index()
        )
        # For each constructor, compute teammate avg for each driver as mean of other drivers
        teammate_delta = {}
        if not tmp_driver_avg.empty:
            for ctor, g in tmp_driver_avg.groupby("constructor_id"):
                # Map driver -> avg
                m = dict(zip(g["driver_id"], g["driver_q_avg_pos_season_prior"]))
                drivers = list(m.keys())
                for d in drivers:
                    others = [m[o] for o in drivers if o != d]
                    if len(others) == 0:
                        teammate_delta[(ctor, d)] = pd.NA
                    else:
                        teammate_avg = float(sum(others) / len(others))
                        teammate_delta[(ctor, d)] = float(m[d] - teammate_avg)

        # -----------------------------
        # Reliability (season-to-date DNF)
        # -----------------------------
        if not r_season_prior.empty:
            r_season_prior["dnf"] = _ensure_numeric(r_season_prior, "dnf").fillna(0).astype("Int64")
        driver_race_starts_season = r_season_prior.groupby("driver_id")["dnf"].count()
        driver_dnfs_season = r_season_prior.groupby("driver_id")["dnf"].sum()

        # -----------------------------
        # Track/circuit priors (qualifying)
        # -----------------------------
        # For this round, circuit_id may differ per row if calendar is weird; compute per row later by join.
        # We'll compute groupby (driver_id, circuit_id) and (constructor_id, circuit_id)
        driver_circuit_q = q_track_prior.groupby(["driver_id", "circuit_id"])["qualifying_position"]
        driver_q_count_circuit = driver_circuit_q.count()
        driver_q_avg_circuit = driver_circuit_q.mean()

        ctor_circuit_q = q_track_prior.groupby(["constructor_id", "circuit_id"])["qualifying_position"]
        ctor_q_count_circuit = ctor_circuit_q.count()
        ctor_q_avg_circuit = ctor_circuit_q.mean()

        # -----------------------------
        # Track-type priors (qualifying)
        # -----------------------------
        driver_tt_q_count = pd.Series(dtype="int64")
        driver_tt_q_avg = pd.Series(dtype="float64")
        ctor_tt_q_count = pd.Series(dtype="int64")
        ctor_tt_q_avg = pd.Series(dtype="float64")

        if "track_type" in q_track_prior.columns and q_track_prior["track_type"].notna().any():
            driver_tt = q_track_prior.groupby(["driver_id", "track_type"])["qualifying_position"]
            driver_tt_q_count = driver_tt.count()
            driver_tt_q_avg = driver_tt.mean()

            ctor_tt = q_track_prior.groupby(["constructor_id", "track_type"])["qualifying_position"]
            ctor_tt_q_count = ctor_tt.count()
            ctor_tt_q_avg = ctor_tt.mean()

        # -----------------------------
        # Driver–constructor tenure (prior rounds)
        # -----------------------------
        # Count of all prior entries for this driver+constructor, across seasons <S plus earlier rounds in S.
        # We build it from entries dataset with hive-partitioned season.
        d_entries_all = d_entries.to_table(
            filter=(ds.field("season") < season) | ((ds.field("season") == season) & (ds.field("round") < r))
        ).to_pandas()
        if not d_entries_all.empty:
            d_entries_all["round"] = _ensure_numeric(d_entries_all, "round").astype("Int64")
        tenure = d_entries_all.groupby(["driver_id", "constructor_id"]).size()

        # -----------------------------
        # Finish-minus-grid (season-to-date)
        # -----------------------------
        if not r_season_prior.empty:
            r_season_prior["grid_position"] = _ensure_numeric(r_season_prior, "grid_position").astype("Int64")
            r_season_prior["finish_position"] = _ensure_numeric(r_season_prior, "finish_position").astype("Int64")
            r_season_prior["dnf"] = _ensure_numeric(r_season_prior, "dnf").fillna(0).astype("Int64")
            r_season_prior["finish_minus_grid"] = (r_season_prior["finish_position"] - r_season_prior["grid_position"]).where(
                (r_season_prior["dnf"] == 0) & r_season_prior["finish_position"].notna() & r_season_prior["grid_position"].notna()
            )
        else:
            # Ensure the column exists so downstream groupby doesn't crash on empty slices.
            r_season_prior["finish_minus_grid"] = pd.Series(dtype="float64")

        driver_fmg_avg_season = r_season_prior.groupby("driver_id")["finish_minus_grid"].mean()
        driver_fmg_count_season = r_season_prior.groupby("driver_id")["finish_minus_grid"].count()

        # -----------------------------
        # Populate features for rows in this round
        # -----------------------------
        def _series_lookup(series: pd.Series, key, default=pd.NA):
            try:
                val = series.loc[key]
                if pd.isna(val):
                    return default
                return val
            except KeyError:
                return default

        # counts
        base_r["driver_season_quali_count_prior"] = base_r["driver_id"].map(driver_q_count_season).fillna(0).astype("int64")
        base_r["constructor_season_quali_count_prior"] = base_r["constructor_id"].map(constructor_q_count_season).fillna(0).astype("int64")
        base_r["driver_career_quali_count_prior"] = base_r["driver_id"].map(driver_q_count_career).fillna(0).astype("int64")
        base_r["constructor_career_quali_count_prior"] = base_r["constructor_id"].map(constructor_q_count_career).fillna(0).astype("int64")

        # driver season stats
        base_r["driver_q_avg_pos_season_prior"] = base_r["driver_id"].map(driver_q_avg_season)
        base_r["driver_q_med_pos_season_prior"] = base_r["driver_id"].map(driver_q_med_season)
        base_r["driver_q_best_pos_season_prior"] = base_r["driver_id"].map(driver_q_best_season)
        base_r["driver_q_worst_pos_season_prior"] = base_r["driver_id"].map(driver_q_worst_season)

        # constructor season stats
        base_r["constructor_q_avg_pos_season_prior"] = base_r["constructor_id"].map(constructor_q_avg_season)
        base_r["constructor_q_best_pos_season_prior"] = base_r["constructor_id"].map(constructor_q_best_season)

        # teammate delta
        base_r["driver_vs_teammate_q_avg_delta_season_prior"] = base_r.apply(
            lambda row: teammate_delta.get((row["constructor_id"], row["driver_id"]), pd.NA),
            axis=1,
        )

        # reliability
        base_r["driver_season_race_starts_prior"] = base_r["driver_id"].map(driver_race_starts_season).fillna(0).astype("int64")
        base_r["driver_season_dnfs_prior"] = base_r["driver_id"].map(driver_dnfs_season).fillna(0).astype("int64")
        # rate: null when starts == 0
        base_r["driver_season_dnf_rate_prior"] = base_r.apply(
            lambda row: (row["driver_season_dnfs_prior"] / row["driver_season_race_starts_prior"])
            if row["driver_season_race_starts_prior"] > 0
            else pd.NA,
            axis=1,
        )

        # career priors
        base_r["driver_q_avg_pos_career_prior"] = base_r["driver_id"].map(driver_q_avg_career)
        base_r["constructor_q_avg_pos_career_prior"] = base_r["constructor_id"].map(constructor_q_avg_career)

        # track/circuit priors
        base_r["driver_q_count_at_circuit_prior"] = base_r.apply(
            lambda row: _series_lookup(driver_q_count_circuit, (row["driver_id"], row["circuit_id"]), default=0),
            axis=1,
        )
        base_r["driver_q_avg_pos_at_circuit_prior"] = base_r.apply(
            lambda row: _series_lookup(driver_q_avg_circuit, (row["driver_id"], row["circuit_id"]), default=pd.NA),
            axis=1,
        )
        base_r["constructor_q_count_at_circuit_prior"] = base_r.apply(
            lambda row: _series_lookup(ctor_q_count_circuit, (row["constructor_id"], row["circuit_id"]), default=0),
            axis=1,
        )
        base_r["constructor_q_avg_pos_at_circuit_prior"] = base_r.apply(
            lambda row: _series_lookup(ctor_q_avg_circuit, (row["constructor_id"], row["circuit_id"]), default=pd.NA),
            axis=1,
        )

        # track-type priors
        base_r["driver_q_count_at_track_type_prior"] = base_r.apply(
            lambda row: _series_lookup(driver_tt_q_count, (row["driver_id"], row.get("track_type")), default=0)
            if pd.notna(row.get("track_type")) else 0,
            axis=1,
        )
        base_r["driver_q_avg_pos_at_track_type_prior"] = base_r.apply(
            lambda row: _series_lookup(driver_tt_q_avg, (row["driver_id"], row.get("track_type")), default=pd.NA)
            if pd.notna(row.get("track_type")) else pd.NA,
            axis=1,
        )
        base_r["constructor_q_count_at_track_type_prior"] = base_r.apply(
            lambda row: _series_lookup(ctor_tt_q_count, (row["constructor_id"], row.get("track_type")), default=0)
            if pd.notna(row.get("track_type")) else 0,
            axis=1,
        )
        base_r["constructor_q_avg_pos_at_track_type_prior"] = base_r.apply(
            lambda row: _series_lookup(ctor_tt_q_avg, (row["constructor_id"], row.get("track_type")), default=pd.NA)
            if pd.notna(row.get("track_type")) else pd.NA,
            axis=1,
        )

        # driver-constructor tenure
        base_r["driver_constructor_tenure_rounds_prior"] = base_r.apply(
            lambda row: _series_lookup(tenure, (row["driver_id"], row["constructor_id"]), default=0),
            axis=1,
        )

        # finish-minus-grid (racecraft proxy)
        base_r["driver_finish_minus_grid_avg_season_prior"] = base_r["driver_id"].map(driver_fmg_avg_season)
        base_r["driver_finish_minus_grid_count_season_prior"] = base_r["driver_id"].map(driver_fmg_count_season).fillna(0).astype("int64")

        base_r["driver_finish_minus_grid_avg_career_prior"] = base_r["driver_id"].map(driver_fmg_avg_career)
        base_r["driver_finish_minus_grid_count_career_prior"] = base_r["driver_id"].map(driver_fmg_count_career).fillna(0).astype("int64")

        # Add to outputs (schema-aligned for deterministic concat)
        base_r = _schema_align_chunk(base_r)
        out_rows.append(base_r)

    # Pandas warns that concatenation behavior with empty or all-NA frames will change in the future.
    # Filter those out to keep schema/dtype inference stable across pandas versions.
    out_rows = [df for df in out_rows if df is not None and (not df.empty)]
    if not out_rows:
        # Deterministic empty output
        return pd.DataFrame()

    out = pd.concat(out_rows, ignore_index=True)

    # Deterministic ordering
    out = out.sort_values(["season", "round", "driver_id"], ascending=[True, True, True]).reset_index(drop=True)

    # Enforce schema dtypes (explicit and stable)
    schema = BaselineQualiSchema()
    dtype_map = schema.__dict__.copy()

    # Pandas uses "Int16" for nullable ints; we cast carefully.
    # First, standardize key columns.
    out["season"] = out["season"].astype(dtype_map["season"])
    out["round"] = _ensure_numeric(out, "round").astype("int64").astype(dtype_map["round"])
    out["driver_id"] = out["driver_id"].astype("string")
    out["constructor_id"] = out["constructor_id"].astype("string")
    out["circuit_id"] = out["circuit_id"].astype("string")
    out["race_name"] = out["race_name"].astype("string")
    out["weekend_id"] = out["weekend_id"].astype("string")
    out["checkpoint"] = out["checkpoint"].astype("string")
    out["session_type"] = out["session_type"].astype("string")
    out["track_type"] = out["track_type"].astype("string")

    # date
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")

    # counts (ints)
    out["driver_season_quali_count_prior"] = out["driver_season_quali_count_prior"].astype("int64").astype("int16")
    out["constructor_season_quali_count_prior"] = out["constructor_season_quali_count_prior"].astype("int64").astype("int16")
    out["driver_career_quali_count_prior"] = out["driver_career_quali_count_prior"].astype("int64").astype("int32")
    out["constructor_career_quali_count_prior"] = out["constructor_career_quali_count_prior"].astype("int64").astype("int32")

    out["driver_season_race_starts_prior"] = out["driver_season_race_starts_prior"].astype("int64").astype("int16")
    out["driver_season_dnfs_prior"] = out["driver_season_dnfs_prior"].astype("int64").astype("int16")

    out["driver_q_count_at_circuit_prior"] = out["driver_q_count_at_circuit_prior"].astype("int64").astype("int16")
    out["constructor_q_count_at_circuit_prior"] = out["constructor_q_count_at_circuit_prior"].astype("int64").astype("int16")

    out["driver_q_count_at_track_type_prior"] = out["driver_q_count_at_track_type_prior"].astype("int64").astype("int16")
    out["constructor_q_count_at_track_type_prior"] = out["constructor_q_count_at_track_type_prior"].astype("int64").astype("int16")

    out["driver_constructor_tenure_rounds_prior"] = out["driver_constructor_tenure_rounds_prior"].astype("int64").astype("int32")

    out["driver_finish_minus_grid_count_season_prior"] = out["driver_finish_minus_grid_count_season_prior"].astype("int64").astype("int16")
    out["driver_finish_minus_grid_count_career_prior"] = out["driver_finish_minus_grid_count_career_prior"].astype("int64").astype("int32")

    # floats (nullable)
    float_cols = [
        "driver_q_avg_pos_season_prior",
        "driver_q_med_pos_season_prior",
        "constructor_q_avg_pos_season_prior",
        "driver_vs_teammate_q_avg_delta_season_prior",
        "driver_season_dnf_rate_prior",
        "driver_q_avg_pos_career_prior",
        "constructor_q_avg_pos_career_prior",
        "driver_q_avg_pos_at_circuit_prior",
        "constructor_q_avg_pos_at_circuit_prior",
        "driver_q_avg_pos_at_track_type_prior",
        "constructor_q_avg_pos_at_track_type_prior",
        "driver_finish_minus_grid_avg_season_prior",
        "driver_finish_minus_grid_avg_career_prior",
    ]
    for c in float_cols:
        # `pd.NA` (nullable NA) cannot be cast directly to numpy float; coerce to NaN first.
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    # nullable small ints for best/worst positions
    for c in ["driver_q_best_pos_season_prior", "driver_q_worst_pos_season_prior", "constructor_q_best_pos_season_prior"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int16")

    return out