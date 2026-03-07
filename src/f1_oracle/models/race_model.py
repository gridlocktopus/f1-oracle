"""
Race model (post-qualifying).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import warnings
import pandas as pd
import pyarrow.dataset as ds
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.isotonic import IsotonicRegression

from f1_oracle.common.io import load_yaml
from f1_oracle.features.post_quali import build_features_race_post_quali_for_round


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


def _build_training_frame(*, season: int, rnd: int) -> pd.DataFrame:
    paths = _load_paths_cfg()
    canonical_dir = paths.get("canonical", {}).get("dir", "data/canonical")
    seasons_cfg = load_yaml(Path("configs") / "seasons.yaml")
    start_year = int(seasons_cfg["ingest"]["start_year"])
    train_end_year = int(seasons_cfg["split"]["train_end_year"])

    d_r = _read_canonical(canonical_dir, "results_race")

    frames = []
    max_train_season = min(season, train_end_year)
    for s in range(start_year, max_train_season + 1):
        max_round = 999
        if s == season:
            max_round = rnd - 1
        if max_round < 1:
            continue
        for r in range(1, max_round + 1):
            feats = build_features_race_post_quali_for_round(season=s, rnd=r)
            if feats.empty:
                continue
            frames.append(feats)

    if not frames:
        return pd.DataFrame()

    features = pd.concat(frames, ignore_index=True)

    r_all = d_r.to_table(filter=ds.field("season") <= max_train_season).to_pandas()
    r_all["finish_position"] = pd.to_numeric(r_all["finish_position"], errors="coerce")
    r_all["dnf"] = pd.to_numeric(r_all["dnf"], errors="coerce").fillna(0)
    r_all["round"] = pd.to_numeric(r_all["round"], errors="coerce")
    r_all["season"] = pd.to_numeric(r_all["season"], errors="coerce")

    r_all = r_all[(r_all["season"] < season) | ((r_all["season"] == season) & (r_all["round"] < rnd))]
    r_all = r_all[r_all["season"] <= max_train_season]

    merged = features.merge(
        r_all[["season", "round", "driver_id", "constructor_id", "finish_position", "dnf"]],
        on=["season", "round", "driver_id", "constructor_id"],
        how="inner",
    )
    return merged


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    y_finish = df["finish_position"].to_numpy(dtype=float)
    y_dnf = df["dnf"].to_numpy(dtype=int)
    x = df.drop(columns=["finish_position", "dnf"], errors="ignore")
    x = x.drop(columns=[c for c in FEATURE_DROP if c in x.columns], errors="ignore")
    return x, y_finish, y_dnf


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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Downcasting behavior in `replace` is deprecated",
            category=FutureWarning,
        )
        out = out.replace({pd.NA: np.nan}).infer_objects(copy=False)
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in categorical_cols:
        out[col] = out[col].astype("string").fillna("__MISSING__")
    return out


def _align_features(x: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """
    Align prediction features to the columns seen during training.
    """
    expected = None
    if hasattr(model, "_input_columns"):
        expected = list(model._input_columns)  # type: ignore[attr-defined]
    else:
        pre = model.named_steps.get("pre")
        if pre is not None and hasattr(pre, "feature_names_in_"):
            expected = list(pre.feature_names_in_)
    if expected is None:
        return x
    out = x.copy()
    for col in expected:
        if col not in out.columns:
            out[col] = np.nan
    # Drop any extra columns
    out = out[expected]
    return out
def train_race_models(*, season: int, rnd: int) -> dict[str, Pipeline | dict]:
    df = _build_training_frame(season=season, rnd=rnd)
    if df.empty:
        raise ValueError("No training data available for race model.")

    # Split finish vs DNF training sets
    df_finish = df.dropna(subset=["finish_position"]).copy()
    if df_finish.empty:
        raise ValueError("No finish_position rows available for race model.")
    x_finish, y_finish, _ = _split_features_target(df_finish)
    x_finish = _sanitize_features(x_finish)

    df_dnf = df.copy()
    x_dnf, _, y_dnf = _split_features_target(df_dnf)
    x_dnf = _sanitize_features(x_dnf)

    pre_finish = _build_preprocessor(x_finish)
    pre_dnf = _build_preprocessor(x_dnf)

    finish_model = XGBRegressor(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.0,
        objective="reg:squarederror",
        random_state=42,
    )

    # If there is only one class in y_dnf, fall back to a constant predictor.
    # Base DNF rate: rolling window of the most recent 7 seasons in training data
    if "season" in df.columns:
        season_vals = pd.to_numeric(df["season"], errors="coerce")
        max_season = season_vals.max()
        min_season = max_season - 6 if pd.notna(max_season) else None
        if min_season is not None:
            recent_mask = season_vals >= min_season
            dnf_rate = float(pd.Series(y_dnf)[recent_mask].mean())
        else:
            dnf_rate = float(y_dnf.mean()) if len(y_dnf) else 0.0
    else:
        dnf_rate = float(y_dnf.mean()) if len(y_dnf) else 0.0
    if dnf_rate <= 0.0 or dnf_rate >= 1.0:
        dnf_model = DummyClassifier(strategy="most_frequent")
    else:
        scale_pos_weight = (1.0 - dnf_rate) / dnf_rate
        dnf_model = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            base_score=dnf_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )

    finish_pipe = Pipeline(steps=[("pre", pre_finish), ("model", finish_model)])
    dnf_pipe = Pipeline(steps=[("pre", pre_dnf), ("model", dnf_model)])

    # Weight recent seasons slightly higher
    if "season" in df_finish.columns:
        season_vals = pd.to_numeric(df_finish["season"], errors="coerce")
        min_season = season_vals.min()
        weights = 1.0 + 0.1 * (season_vals - min_season)
        weights = weights.fillna(1.0).to_numpy(dtype=float)
        finish_pipe.fit(x_finish, y_finish, model__sample_weight=weights)
        # DNF weights based on all rows (including DNFs)
        if "season" in df_dnf.columns:
            season_vals_dnf = pd.to_numeric(df_dnf["season"], errors="coerce")
            min_season_dnf = season_vals_dnf.min()
            weights_dnf = 1.0 + 0.1 * (season_vals_dnf - min_season_dnf)
            weights_dnf = weights_dnf.fillna(1.0).to_numpy(dtype=float)
        else:
            weights_dnf = None
        # Hold out most recent season for DNF calibration
        if "season" in df_dnf.columns:
            calib_season = pd.to_numeric(df_dnf["season"], errors="coerce").max()
            train_mask = pd.to_numeric(df_dnf["season"], errors="coerce") < calib_season
            calib_mask = pd.to_numeric(df_dnf["season"], errors="coerce") == calib_season
        else:
            train_mask = None
            calib_mask = None

        if train_mask is not None and calib_mask is not None and calib_mask.any():
            x_dnf_train = x_dnf[train_mask]
            y_dnf_train = y_dnf[train_mask]
            x_dnf_calib = x_dnf[calib_mask]
            y_dnf_calib = y_dnf[calib_mask]

            if weights_dnf is not None:
                dnf_pipe.fit(x_dnf_train, y_dnf_train, model__sample_weight=weights_dnf[train_mask])
            else:
                dnf_pipe.fit(x_dnf_train, y_dnf_train)

            probs_calib = dnf_pipe.predict_proba(x_dnf_calib)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(probs_calib, y_dnf_calib)
        else:
            if weights_dnf is not None:
                dnf_pipe.fit(x_dnf, y_dnf, model__sample_weight=weights_dnf)
            else:
                dnf_pipe.fit(x_dnf, y_dnf)
            calibrator = None
    else:
        finish_pipe.fit(x_finish, y_finish)
        dnf_pipe.fit(x_dnf, y_dnf)
        calibrator = None

    # Persist input column order for prediction alignment
    finish_pipe._input_columns = list(x_finish.columns)  # type: ignore[attr-defined]
    dnf_pipe._input_columns = list(x_dnf.columns)  # type: ignore[attr-defined]

    if calibrator is not None:
        return {
            "finish": finish_pipe,
            "dnf": {"model": dnf_pipe, "calibrator": calibrator, "base_rate": dnf_rate},
        }
    return {"finish": finish_pipe, "dnf": {"model": dnf_pipe, "base_rate": dnf_rate}}


def predict_race_scores(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    x = features.drop(columns=[c for c in FEATURE_DROP if c in features.columns], errors="ignore")
    x = _sanitize_features(x)
    x = _align_features(x, model)
    # Re-sanitize after alignment because missing columns are injected at this step.
    x = _sanitize_features(x)
    preds = model.predict(x)
    return pd.Series(preds, index=features.index, name="score")


def predict_dnf_probs(model: Pipeline | dict, features: pd.DataFrame) -> pd.Series:
    calibrator = None
    base_rate = None
    if isinstance(model, dict):
        calibrator = model.get("calibrator")
        base_rate = model.get("base_rate")
        model = model.get("model")
        if model is None:
            raise ValueError("DNF model is missing.")
    x = features.drop(columns=[c for c in FEATURE_DROP if c in features.columns], errors="ignore")
    x = _sanitize_features(x)
    x = _align_features(x, model)
    # Re-sanitize after alignment because missing columns are injected at this step.
    x = _sanitize_features(x)
    probs_all = model.predict_proba(x)
    if probs_all.shape[1] == 1:
        # Model was trained on a single class; return zeros
        probs = np.zeros(len(x), dtype=float)
    else:
        probs = probs_all[:, 1]
    if calibrator is not None:
        probs = calibrator.transform(probs)
    # Prior scaling to match historical base rate
    if base_rate is not None:
        mean_p = float(probs.mean()) if len(probs) else 0.0
        if mean_p > 0:
            scale = float(base_rate) / mean_p
            probs = (probs * scale).clip(0.0, 1.0)
    return pd.Series(probs, index=features.index, name="dnf_prob")
