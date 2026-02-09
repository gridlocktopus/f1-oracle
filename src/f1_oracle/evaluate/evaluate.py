"""
Evaluation utilities for predictions vs actual results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime

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


def _pred_path(kind: str, mode: str, season: int, rnd: int) -> Path:
    pred_root, _ = _paths()
    if kind == "quali":
        stage = "post_practice"
        pred_kind = "quali_top" if mode == "top" else "quali_dist"
    else:
        stage = "post_quali"
        pred_kind = "race_top" if mode == "top" else "race_dist"
    return pred_root / f"season={season}" / f"round={rnd}" / stage / pred_kind / "predictions.parquet"


def evaluate_round(
    *,
    kind: str,
    mode: str,
    season: int,
    rnd: int,
) -> dict[str, Any] | None:
    pred_path = _pred_path(kind, mode, season, rnd)
    if not pred_path.exists():
        return None

    pred = pd.read_parquet(pred_path)

    if kind == "quali":
        actual = _load_actual_quali(season, rnd)
        if actual.empty:
            return None
        if mode == "top":
            merged = pred.merge(actual, on="driver_id", how="left")
            merged["abs_error"] = (merged["predicted_position"] - merged["qualifying_position"]).abs()
            mae = float(merged["abs_error"].mean())
            exact = int((merged["predicted_position"] == merged["qualifying_position"]).sum())
            top1 = int((merged.sort_values("predicted_position").head(1)["qualifying_position"] == 1).sum())
            top3 = int((merged.sort_values("predicted_position").head(3)["qualifying_position"] <= 3).sum())
            top5 = int((merged.sort_values("predicted_position").head(5)["qualifying_position"] <= 5).sum())
            return {
                "round": rnd,
                "mae": mae,
                "exact": exact,
                "top1": top1,
                "top3": top3,
                "top5": top5,
                "n": len(merged),
            }
        # dist
        merged = pred.merge(actual, on="driver_id", how="left")
        merged["hit"] = merged["position"] == merged["qualifying_position"]
        p_hit = merged[merged["hit"]]["probability"]
        avg_p = float(p_hit.mean()) if not p_hit.empty else float("nan")
        return {"round": rnd, "avg_p_actual": avg_p}

    # race
    actual = _load_actual_race(season, rnd)
    if actual.empty:
        return None
    if mode == "top":
        merged = pred.merge(actual, on="driver_id", how="left")
        merged["abs_error"] = (merged["predicted_position"] - merged["finish_position"]).abs()
        mae = float(merged["abs_error"].mean())
        exact = int((merged["predicted_position"] == merged["finish_position"]).sum())
        top1 = int((merged.sort_values("predicted_position").head(1)["finish_position"] == 1).sum())
        top3 = int((merged.sort_values("predicted_position").head(3)["finish_position"] <= 3).sum())
        top5 = int((merged.sort_values("predicted_position").head(5)["finish_position"] <= 5).sum())
        return {
            "round": rnd,
            "mae": mae,
            "exact": exact,
            "top1": top1,
            "top3": top3,
            "top5": top5,
            "n": len(merged),
        }
    # dist
    pred = pred.copy()
    pred["position"] = pred["position"].astype("string")

    def _actual_position(row: pd.Series) -> str:
        if row.get("dnf", 0) == 1:
            return "DNF"
        return str(int(row["finish_position"])) if pd.notna(row["finish_position"]) else "DNF"

    actual = actual.copy()
    actual["actual_pos"] = actual.apply(_actual_position, axis=1)
    merged = pred.merge(actual[["driver_id", "actual_pos"]], on="driver_id", how="left")
    merged["hit"] = merged["position"] == merged["actual_pos"]
    p_hit = merged[merged["hit"]]["probability"]
    avg_p = float(p_hit.mean()) if not p_hit.empty else float("nan")
    return {"round": rnd, "avg_p_actual": avg_p}


def evaluate_range(
    *,
    kind: str,
    mode: str,
    season: int,
    start_round: int,
    end_round: int,
) -> tuple[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for rnd in range(start_round, end_round + 1):
        res = evaluate_round(kind=kind, mode=mode, season=season, rnd=rnd)
        if res is not None:
            rows.append(res)

    if not rows:
        return "No rounds available for evaluation.", pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("round")
    if mode == "top":
        total_n = df["n"].sum()
        rounds = len(df)
        exact_rate = (df["exact"].sum() / total_n) if total_n else 0.0
        top1_rate = (df["top1"].sum() / (rounds * 1)) if rounds else 0.0
        top3_rate = (df["top3"].sum() / (rounds * 3)) if rounds else 0.0
        top5_rate = (df["top5"].sum() / (rounds * 5)) if rounds else 0.0
        summary = (
            f"{kind} top: rounds={rounds}, MAE={df['mae'].mean():.3f}, "
            f"exact={exact_rate:.1%} ({df['exact'].sum()}/{total_n}), "
            f"top1={top1_rate:.1%}, top3={top3_rate:.1%}, top5={top5_rate:.1%}"
        )
    else:
        summary = f"{kind} dist: rounds={len(df)}, avg P(actual)={df['avg_p_actual'].mean():.4f}"

    return summary, df


def save_evaluation(summary: str, df: pd.DataFrame, path: Path) -> tuple[Path, Path]:
    """
    Save evaluation details and summary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    summary_path = path.with_suffix(path.suffix + ".summary.txt")
    summary_path.write_text(summary + "\n")
    return path, summary_path


def snapshot_path(kind: str, mode: str, season: int, start_round: int, end_round: int, label: str | None) -> Path:
    base = Path("data") / "evaluation" / "baselines"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_part = f"_{label}" if label else ""
    name = f"{kind}_{mode}_season={season}_rounds={start_round}-{end_round}{label_part}_{ts}.csv"
    return base / name
