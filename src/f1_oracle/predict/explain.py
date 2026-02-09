"""
Model explainability helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _get_feature_names(model: Pipeline) -> list[str]:
    pre = model.named_steps.get("pre")
    if pre is None:
        return []
    if hasattr(pre, "get_feature_names_out"):
        return list(pre.get_feature_names_out())
    return []


def global_feature_importance(model: Pipeline | dict, top_k: int = 20) -> pd.DataFrame:
    """
    Return top-k global feature importances as a DataFrame.
    """
    if isinstance(model, dict):
        model = model.get("model")
        if model is None:
            return pd.DataFrame(columns=["feature", "importance"])
    xgb = model.named_steps.get("model")
    if xgb is None or not hasattr(xgb, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importances = np.asarray(xgb.feature_importances_)
    feature_names = _get_feature_names(model)
    if len(feature_names) != len(importances):
        # Fallback to indexed names if we cannot align
        feature_names = [f"f{i}" for i in range(len(importances))]

    order = np.argsort(importances)[::-1][:top_k]
    rows = [{"feature": feature_names[i], "importance": float(importances[i])} for i in order]
    return pd.DataFrame(rows)


def top_feature_values(
    features: pd.DataFrame,
    model: Pipeline | dict,
    top_k: int = 10,
    id_col: str = "driver_id",
) -> pd.DataFrame:
    """
    For each row, return values of the top-k global features (if present).
    """
    if isinstance(model, dict):
        model = model.get("model")
        if model is None:
            return pd.DataFrame()
    imp = global_feature_importance(model, top_k=top_k)
    top_features = [f for f in imp["feature"].tolist() if f in features.columns]
    cols = [c for c in [id_col] if c in features.columns] + top_features
    if not cols:
        return pd.DataFrame()
    return features[cols].copy()
