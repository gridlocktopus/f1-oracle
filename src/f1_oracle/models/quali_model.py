"""
Qualifying model (post-practice).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from f1_oracle.common.io import load_yaml
from f1_oracle.features.baseline import build_features_qualifying_baseline_for_season
from f1_oracle.features.post_practice import _load_practice_raw, _aggregate_practice


FEATURE_DROP = {
    "race_name",
    "race_date",
    "checkpoint",
    "session_type",
}


def _load_paths_cfg() -> dict[str, Any]:
    return load_yaml(Path("configs") / "paths.yaml")


def _read_canonical(canonical_dir: str, name: str) -> ds.Dataset:
    root = Path(canonical_dir) / name
    return ds.dataset(str(root), format="parquet", partitioning="hive")


def _load_practice_agg_for_season(season: int, raw_dir: str) -> pd.DataFrame:
    base = Path(raw_dir) / "fastf1" / f"season={season}"
    if not base.exists():
        return pd.DataFrame()

    frames = []
    for round_dir in base.glob("round=*"):
        try:
            rnd = int(round_dir.name.split("=")[-1])
        except ValueError:
            continue
        df = _load_practice_raw(season, rnd, raw_dir)
        if df.empty:
            continue
        agg = _aggregate_practice(df)
        agg["round"] = rnd
        agg["season"] = season
        frames.append(agg)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _merge_practice(features: pd.DataFrame, practice_agg: pd.DataFrame) -> pd.DataFrame:
    if practice_agg.empty:
        return features
    out = features.merge(practice_agg, on=["season", "round", "driver_id"], how="left")
    out["practice_sessions_count"] = (
        pd.to_numeric(out.get("practice_sessions_count"), errors="coerce").fillna(0).astype("Int64")
    )
    return out


def _build_features_for_season(season: int, canonical_dir: str, raw_dir: str) -> pd.DataFrame:
    track_types_cfg = load_yaml(Path("configs") / "weekend_types.yaml")
    baseline = build_features_qualifying_baseline_for_season(
        season=season, canonical_dir=canonical_dir, track_types_cfg=track_types_cfg
    )
    if baseline.empty:
        return baseline

    practice_agg = _load_practice_agg_for_season(season, raw_dir)
    return _merge_practice(baseline, practice_agg)


def _build_training_frame(*, season: int, rnd: int) -> pd.DataFrame:
    paths = _load_paths_cfg()
    canonical_dir = paths.get("canonical", {}).get("dir", "data/canonical")
    raw_dir = paths.get("raw", {}).get("dir", "data/raw")
    seasons_cfg = load_yaml(Path("configs") / "seasons.yaml")
    start_year = int(seasons_cfg["ingest"]["start_year"])
    train_end_year = int(seasons_cfg["split"]["train_end_year"])

    d_q = _read_canonical(canonical_dir, "results_qualifying")

    # Build features for all seasons <= season, then filter to allowed rounds.
    frames = []
    max_train_season = min(season, train_end_year)
    for s in range(start_year, max_train_season + 1):
        feats = _build_features_for_season(s, canonical_dir, raw_dir)
        if feats.empty:
            continue
        if s == season:
            feats = feats[feats["round"] < rnd]
        frames.append(feats)

    if not frames:
        return pd.DataFrame()

    features = pd.concat(frames, ignore_index=True)

    q_all = d_q.to_table(filter=ds.field("season") <= max_train_season).to_pandas()
    q_all["qualifying_position"] = pd.to_numeric(q_all["qualifying_position"], errors="coerce")
    q_all["round"] = pd.to_numeric(q_all["round"], errors="coerce")
    q_all["season"] = pd.to_numeric(q_all["season"], errors="coerce")

    # Safety: do not use seasons beyond configured training cutoff
    q_all = q_all[(q_all["season"] < season) | ((q_all["season"] == season) & (q_all["round"] < rnd))]
    q_all = q_all[q_all["season"] <= max_train_season]

    merged = features.merge(
        q_all[["season", "round", "driver_id", "constructor_id", "qualifying_position"]],
        on=["season", "round", "driver_id", "constructor_id"],
        how="inner",
    )

    merged = merged.dropna(subset=["qualifying_position"])
    return merged


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    y = df["qualifying_position"].to_numpy(dtype=float)
    x = df.drop(columns=["qualifying_position"], errors="ignore")
    x = x.drop(columns=[c for c in FEATURE_DROP if c in x.columns], errors="ignore")
    return x, y


def _build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    # Treat any non-numeric columns as categorical (e.g., weekend_id)
    non_numeric = set(x.select_dtypes(include=["object", "string"]).columns.tolist())
    categorical_cols = sorted(non_numeric.union({c for c in ["driver_id", "constructor_id", "circuit_id", "track_type"] if c in x.columns}))
    numeric_cols = [c for c in x.columns if c not in categorical_cols]
    # Drop numeric columns that are entirely missing
    numeric_cols = [c for c in numeric_cols if pd.to_numeric(x[c], errors="coerce").notna().any()]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def _sanitize_features(x: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric columns are numeric and categorical columns are strings,
    to avoid pandas NA object dtype issues inside sklearn.
    """
    non_numeric = set(x.select_dtypes(include=["object", "string"]).columns.tolist())
    categorical_cols = sorted(non_numeric.union({c for c in ["driver_id", "constructor_id", "circuit_id", "track_type"] if c in x.columns}))
    numeric_cols = [c for c in x.columns if c not in categorical_cols]

    out = x.copy()
    # Replace pandas NA with numpy nan globally
    out = out.replace({pd.NA: np.nan}).infer_objects(copy=False)
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in categorical_cols:
        out[col] = out[col].astype("string").fillna("__MISSING__")
    return out


def train_quali_model(*, season: int, rnd: int) -> Pipeline:
    df = _build_training_frame(season=season, rnd=rnd)
    if df.empty:
        raise ValueError("No training data available for qualifying model.")

    x, y = _split_features_target(df)
    x = _sanitize_features(x)
    pre = _build_preprocessor(x)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(x, y)
    return pipe


def predict_quali_scores(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    x = features.drop(columns=[c for c in FEATURE_DROP if c in features.columns], errors="ignore")
    x = _sanitize_features(x)
    preds = model.predict(x)
    return pd.Series(preds, index=features.index, name="score")
