"""
Compare predictions to actual results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from f1_oracle.common.io import load_yaml


def _paths() -> tuple[Path, Path]:
    cfg = load_yaml(Path("configs") / "paths.yaml")
    pred_root = Path(cfg.get("predictions", {}).get("dir", "data/predictions"))
    canonical_root = Path(cfg.get("canonical", {}).get("dir", "data/canonical"))
    return pred_root, canonical_root


def _load_actual_quali(season: int, rnd: int) -> pd.DataFrame:
    _, canonical_root = _paths()
    d_q = ds.dataset(str(canonical_root / "results_qualifying"), format="parquet", partitioning="hive")
    q = d_q.to_table(
        filter=(ds.field("season") == season) & (ds.field("round") == rnd)
    ).to_pandas()
    q["qualifying_position"] = pd.to_numeric(q["qualifying_position"], errors="coerce")
    return q[["driver_id", "qualifying_position"]]


def _load_actual_race(season: int, rnd: int) -> pd.DataFrame:
    _, canonical_root = _paths()
    d_r = ds.dataset(str(canonical_root / "results_race"), format="parquet", partitioning="hive")
    r = d_r.to_table(
        filter=(ds.field("season") == season) & (ds.field("round") == rnd)
    ).to_pandas()
    r["finish_position"] = pd.to_numeric(r["finish_position"], errors="coerce")
    r["dnf"] = pd.to_numeric(r["dnf"], errors="coerce").fillna(0)
    return r[["driver_id", "finish_position", "dnf"]]


def compare_quali(season: int, rnd: int, kind: str) -> dict[str, Any]:
    pred_root, _ = _paths()
    if kind == "top":
        pred_path = pred_root / f"season={season}" / f"round={rnd}" / "post_practice" / "quali_top" / "predictions.parquet"
        pred = pd.read_parquet(pred_path)
        actual = _load_actual_quali(season, rnd)
        merged = pred.merge(actual, on="driver_id", how="left")
        merged["abs_error"] = (merged["predicted_position"] - merged["qualifying_position"]).abs()
        mae = merged["abs_error"].mean()
        exact = (merged["predicted_position"] == merged["qualifying_position"]).sum()
        summary = f"Quali top: MAE={mae:.3f}, exact={exact}/{len(merged)}"
        return {"summary": summary, "details": merged}

    pred_path = pred_root / f"season={season}" / f"round={rnd}" / "post_practice" / "quali_dist" / "predictions.parquet"
    pred = pd.read_parquet(pred_path)
    actual = _load_actual_quali(season, rnd)
    merged = pred.merge(actual, on="driver_id", how="left")
    # probability assigned to the actual finishing position
    merged["hit"] = merged["position"] == merged["qualifying_position"]
    p_hit = merged[merged["hit"]]["probability"]
    avg_p = p_hit.mean() if not p_hit.empty else float("nan")
    summary = f"Quali dist: avg P(actual)={avg_p:.4f}"
    # include only hits for readability
    details = merged[merged["hit"]][["driver_id", "position", "probability"]].sort_values("probability", ascending=False)
    return {"summary": summary, "details": details}


def compare_race(season: int, rnd: int, kind: str) -> dict[str, Any]:
    pred_root, _ = _paths()
    if kind == "top":
        pred_path = pred_root / f"season={season}" / f"round={rnd}" / "post_quali" / "race_top" / "predictions.parquet"
        pred = pd.read_parquet(pred_path)
        actual = _load_actual_race(season, rnd)
        merged = pred.merge(actual, on="driver_id", how="left")
        merged["abs_error"] = (merged["predicted_position"] - merged["finish_position"]).abs()
        mae = merged["abs_error"].mean()
        exact = (merged["predicted_position"] == merged["finish_position"]).sum()
        summary = f"Race top: MAE={mae:.3f}, exact={exact}/{len(merged)}"
        return {"summary": summary, "details": merged}

    pred_path = pred_root / f"season={season}" / f"round={rnd}" / "post_quali" / "race_dist" / "predictions.parquet"
    pred = pd.read_parquet(pred_path)
    actual = _load_actual_race(season, rnd)
    merged = pred.merge(actual, on="driver_id", how="left")

    def _actual_position(row: pd.Series) -> str:
        if row.get("dnf", 0) == 1:
            return "DNF"
        return str(int(row["finish_position"])) if pd.notna(row["finish_position"]) else "DNF"

    merged["actual_pos"] = merged.apply(_actual_position, axis=1)
    merged["hit"] = merged["position"] == merged["actual_pos"]
    p_hit = merged[merged["hit"]]["probability"]
    avg_p = p_hit.mean() if not p_hit.empty else float("nan")
    summary = f"Race dist: avg P(actual)={avg_p:.4f}"
    details = merged[merged["hit"]][["driver_id", "position", "probability"]].sort_values("probability", ascending=False)
    return {"summary": summary, "details": details}
