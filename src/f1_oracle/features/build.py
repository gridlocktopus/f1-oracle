"""
Feature dataset build runner.

This module:
- loads configs (paths + weekend_types)
- calls the appropriate feature builder
- writes deterministic Parquet output (one file per season partition folder)

We mirror canonical layout conventions:
  data/features/<dataset>/season=YYYY/<dataset>.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from f1_oracle.common.io import load_yaml
from f1_oracle.features.baseline import build_features_qualifying_baseline_for_season


def _load_paths_cfg() -> dict[str, Any]:
    return load_yaml(Path("configs") / "paths.yaml")


def _load_weekend_types_cfg() -> dict[str, Any]:
    return load_yaml(Path("configs") / "weekend_types.yaml")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_features_qualifying_baseline(*, season: int, canonical_dir: str, features_dir: str) -> Path:
    """
    Build and write features_qualifying_baseline for a single season.

    Returns the output parquet path.
    """
    weekend_types_cfg = _load_weekend_types_cfg()

    df = build_features_qualifying_baseline_for_season(
        season=season,
        canonical_dir=canonical_dir,
        track_types_cfg=weekend_types_cfg,
    )

    dataset_name = "features_qualifying_baseline"
    out_dir = Path(features_dir) / dataset_name / f"season={season}"
    _ensure_dir(out_dir)

    out_path = out_dir / f"{dataset_name}.parquet"

    # The output is written under a hive-style `season=YYYY/` folder, so `season` will be available
    # as a virtual partition column when reading via `pyarrow.dataset`. Storing `season` inside the
    # Parquet file can cause schema-merge conflicts (e.g., int32 vs dictionary<int32>) when readers
    # infer partitions. Drop it before writing.
    df_to_write = df.drop(columns=["season"]) if "season" in df.columns else df

    # Deterministic write: convert to arrow table and write.
    table = pa.Table.from_pandas(df_to_write, preserve_index=False)
    pq.write_table(table, out_path)

    return out_path


def build_features_dataset(*, dataset: str, season: int) -> Path:
    """
    Generic feature build entrypoint (dispatch by dataset name).
    """
    paths_cfg = _load_paths_cfg()
    canonical_dir = paths_cfg.get("canonical", {}).get("dir", "data/canonical")
    features_dir = paths_cfg.get("features", {}).get("dir", "data/features")

    if dataset == "qualifying-baseline":
        return build_features_qualifying_baseline(
            season=season, canonical_dir=canonical_dir, features_dir=features_dir
        )

    raise ValueError(f"Unknown feature dataset: {dataset}")