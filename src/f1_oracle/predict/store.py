"""
Prediction storage utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from f1_oracle.common.io import load_yaml


def _predictions_dir() -> Path:
    cfg = load_yaml(Path("configs") / "paths.yaml")
    root = cfg.get("predictions", {}).get("dir", "data/predictions")
    return Path(root)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def store_predictions(
    *,
    season: int,
    rnd: int,
    stage: str,
    kind: str,
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Store predictions as Parquet with optional metadata sidecar.
    """
    out_dir = _predictions_dir() / f"season={season}" / f"round={rnd}" / stage / kind
    _ensure_dir(out_dir)

    out_path = out_dir / "predictions.parquet"
    df.to_parquet(out_path, index=False)

    if metadata is not None:
        meta_path = out_dir / "metadata.json"
        pd.Series(metadata).to_json(meta_path)

    return out_path
