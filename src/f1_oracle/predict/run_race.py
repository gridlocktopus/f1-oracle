"""
Race prediction runner (post-qualifying).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_oracle.features.post_quali import build_features_race_post_quali_for_round
from f1_oracle.models.race_model import train_race_models, predict_race_scores, predict_dnf_probs
from f1_oracle.models.registry import load_model, save_model
from f1_oracle.predict.store import store_predictions
from f1_oracle.predict.utils import sample_plackett_luce, rankings_to_position_probs
from f1_oracle.predict.explain import global_feature_importance, top_feature_values


def _parse_tags(tags: str | None) -> set[str]:
    if tags is None or not tags.strip():
        return {"dist"}
    return {t.strip().lower() for t in tags.split(",") if t.strip()}


def run_race_prediction(
    *,
    season: int,
    rnd: int,
    tags: str | None,
    samples: int,
    explain: bool,
    print_output: bool,
    print_limit: int,
) -> list[str]:
    features = build_features_race_post_quali_for_round(season=season, rnd=rnd)
    if features.empty:
        raise ValueError("No race features available for this round.")
    if "qualifying_position" in features.columns and features["qualifying_position"].isna().any():
        raise ValueError("Missing qualifying results for this round; ingest/build qualifying results first.")

    finish_model = load_model("race_finish")
    dnf_model = load_model("race_dnf")
    if finish_model is None or dnf_model is None:
        models = train_race_models(season=season, rnd=rnd)
        finish_model = models["finish"]
        dnf_model = models["dnf"]
        save_model("race_finish", finish_model)
        save_model("race_dnf", dnf_model)

    try:
        scores = predict_race_scores(finish_model, features)
        dnf_probs = predict_dnf_probs(dnf_model, features)
    except ValueError as exc:
        if "Feature shape mismatch" in str(exc):
            # Retrain to align with current feature set
            models = train_race_models(season=season, rnd=rnd)
            finish_model = models["finish"]
            dnf_model = models["dnf"]
            save_model("race_finish", finish_model)
            save_model("race_dnf", dnf_model)
            scores = predict_race_scores(finish_model, features)
            dnf_probs = predict_dnf_probs(dnf_model, features)
        else:
            raise

    features = features.copy()
    features["score"] = scores
    features["dnf_prob"] = dnf_probs

    driver_ids = features["driver_id"].tolist()
    tags_set = _parse_tags(tags)
    written = []

    if "top" in tags_set:
        ordered = features.sort_values("score", ascending=True).reset_index(drop=True)
        ordered["predicted_position"] = np.arange(1, len(ordered) + 1)
        out = ordered[["driver_id", "predicted_position", "score", "dnf_prob"]]
        path = store_predictions(
            season=season,
            rnd=rnd,
            stage="post_quali",
            kind="race_top",
            df=out,
            metadata={"tags": ",".join(sorted(tags_set))},
        )
        written.append(str(path))
        if print_output:
            print("\nRace top prediction:")
            print(out.head(print_limit).to_string(index=False))

    if "dist" in tags_set:
        scores_arr = features["score"].to_numpy(dtype=float)
        dnf_arr = features["dnf_prob"].to_numpy(dtype=float)

        rng = np.random.default_rng(42)
        n_drivers = len(driver_ids)
        position_counts = np.zeros((n_drivers, n_drivers), dtype=float)
        dnf_counts = np.zeros(n_drivers, dtype=float)

        for _ in range(samples):
            dnf_mask = rng.random(n_drivers) < dnf_arr
            active_idx = np.where(~dnf_mask)[0]

            if active_idx.size > 0:
                active_scores = scores_arr[active_idx]
                rankings = sample_plackett_luce(active_scores, n_samples=1, temperature=1.0, rng=rng)
                order = rankings[0]
                for pos, idx_in_active in enumerate(order):
                    driver_idx = active_idx[idx_in_active]
                    position_counts[driver_idx, pos] += 1.0

            dnf_counts += dnf_mask.astype(float)

        position_probs = position_counts / float(samples)
        dnf_probs = dnf_counts / float(samples)

        rows = []
        for i, driver_id in enumerate(driver_ids):
            for pos in range(n_drivers):
                rows.append(
                    {
                        "driver_id": driver_id,
                        "position": str(pos + 1),
                        "probability": position_probs[i, pos],
                    }
                )
            rows.append(
                {
                    "driver_id": driver_id,
                    "position": "DNF",
                    "probability": dnf_probs[i],
                }
            )

        out = pd.DataFrame(rows)
        path = store_predictions(
            season=season,
            rnd=rnd,
            stage="post_quali",
            kind="race_dist",
            df=out,
            metadata={"samples": samples, "tags": ",".join(sorted(tags_set))},
        )
        written.append(str(path))
        if print_output:
            print("\nRace distribution (top probabilities):")
            view = out.sort_values("probability", ascending=False).head(print_limit)
            print(view.to_string(index=False))

    if explain:
        imp_finish = global_feature_importance(finish_model, top_k=25)
        if not imp_finish.empty:
            path = store_predictions(
                season=season,
                rnd=rnd,
                stage="post_quali",
                kind="race_explain_global_finish",
                df=imp_finish,
                metadata={"type": "global_importance", "model": "finish"},
            )
            written.append(str(path))

        imp_dnf = global_feature_importance(dnf_model, top_k=25)
        if not imp_dnf.empty:
            path = store_predictions(
                season=season,
                rnd=rnd,
                stage="post_quali",
                kind="race_explain_global_dnf",
                df=imp_dnf,
                metadata={"type": "global_importance", "model": "dnf"},
            )
            written.append(str(path))

        vals = top_feature_values(features, finish_model, top_k=10, id_col="driver_id")
        if not vals.empty:
            path = store_predictions(
                season=season,
                rnd=rnd,
                stage="post_quali",
                kind="race_explain_values",
                df=vals,
                metadata={"type": "top_feature_values"},
            )
            written.append(str(path))

    return written
