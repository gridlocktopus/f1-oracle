"""
Qualifying prediction runner.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from f1_oracle.features.post_practice import build_features_qualifying_post_practice_for_round
from f1_oracle.models.quali_model import train_quali_model, predict_quali_scores
from f1_oracle.models.registry import load_model, save_model
from f1_oracle.predict.store import store_predictions
from f1_oracle.predict.utils import sample_plackett_luce, rankings_to_position_probs
from f1_oracle.predict.explain import global_feature_importance, top_feature_values
from f1_oracle.common.io import load_yaml
from pathlib import Path


def _load_track_types_cfg() -> dict:
    return load_yaml(Path("configs") / "weekend_types.yaml")


def _parse_tags(tags: str | None) -> set[str]:
    if tags is None or not tags.strip():
        return {"dist"}
    return {t.strip().lower() for t in tags.split(",") if t.strip()}


def run_quali_prediction(
    *,
    season: int,
    rnd: int,
    tags: str | None,
    samples: int,
    explain: bool,
    print_output: bool,
    print_limit: int,
) -> list[str]:
    track_types_cfg = _load_track_types_cfg()
    features = build_features_qualifying_post_practice_for_round(
        season=season, rnd=rnd, track_types_cfg=track_types_cfg
    )
    if features.empty:
        raise ValueError("No qualifying features available for this round.")

    model = load_model("quali")
    if model is None:
        model = train_quali_model(season=season, rnd=rnd)
        save_model("quali", model)

    scores = predict_quali_scores(model, features)
    features = features.copy()
    features["score"] = scores

    driver_ids = features["driver_id"].tolist()
    tags_set = _parse_tags(tags)

    written = []

    if "top" in tags_set:
        ordered = features.sort_values("score", ascending=True).reset_index(drop=True)
        ordered["predicted_position"] = np.arange(1, len(ordered) + 1)
        out = ordered[["driver_id", "predicted_position", "score"]]
        path = store_predictions(
            season=season,
            rnd=rnd,
            stage="post_practice",
            kind="quali_top",
            df=out,
            metadata={"tags": ",".join(sorted(tags_set))},
        )
        written.append(str(path))
        if print_output:
            print("\nQuali top prediction:")
            print(out.head(print_limit).to_string(index=False))

    if "dist" in tags_set:
        scores_arr = features["score"].to_numpy(dtype=float)
        rankings = sample_plackett_luce(scores_arr, n_samples=samples, temperature=1.0)
        probs = rankings_to_position_probs(rankings)

        rows = []
        for i, driver_id in enumerate(driver_ids):
            for pos in range(probs.shape[1]):
                rows.append(
                    {
                        "driver_id": driver_id,
                        "position": pos + 1,
                        "probability": probs[i, pos],
                    }
                )
        out = pd.DataFrame(rows)
        path = store_predictions(
            season=season,
            rnd=rnd,
            stage="post_practice",
            kind="quali_dist",
            df=out,
            metadata={"samples": samples, "tags": ",".join(sorted(tags_set))},
        )
        written.append(str(path))
        if print_output:
            print("\nQuali distribution (top probabilities):")
            view = out.sort_values("probability", ascending=False).head(print_limit)
            print(view.to_string(index=False))

    if explain:
        imp = global_feature_importance(model, top_k=25)
        if not imp.empty:
            path = store_predictions(
                season=season,
                rnd=rnd,
                stage="post_practice",
                kind="quali_explain_global",
                df=imp,
                metadata={"type": "global_importance"},
            )
            written.append(str(path))

        vals = top_feature_values(features, model, top_k=10, id_col="driver_id")
        if not vals.empty:
            path = store_predictions(
                season=season,
                rnd=rnd,
                stage="post_practice",
                kind="quali_explain_values",
                df=vals,
                metadata={"type": "top_feature_values"},
            )
            written.append(str(path))

    return written
