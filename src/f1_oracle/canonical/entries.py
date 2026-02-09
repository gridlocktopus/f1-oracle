"""
Canonical entries builder.

What is an "entry"?
- One row per (round, driver_id) indicating which constructor that driver represented
  in that weekend.

Why entries exist:
- This is the backbone join table for the entire project.
- Results (race/quali/sprint), features, predictions, and evaluations should all
  join through entries rather than trying to infer participation ad-hoc.

Design decisions (important):
- We INCLUDE DNS/DSQ/etc. as long as the driver appears in canonical race results.
  This aligns with "entered for the weekend" rather than "took the start".
- We do NOT store a `season` column inside the Parquet file.
  Season is encoded in the output partition folder name: season=YYYY

Input:
- Canonical race results:
  {canonical_dir}/results_race/season=YYYY/results_race.parquet

Output:
- Canonical entries:
  {canonical_dir}/entries/season=YYYY/entries.parquet

Implementation detail:
- We prefer pyarrow-native operations.
- If a pyarrow feature isn't available in the installed version (e.g., drop_duplicates),
  we fall back to a pandas approach ONLY for the final dedupe step.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd

from f1_oracle.common.io import load_yaml


def _entries_schema() -> pa.Schema:
    """Stable schema for canonical entries."""
    return pa.schema(
        [
            pa.field("round", pa.int64(), nullable=False),
            pa.field("race_name", pa.string(), nullable=False),
            pa.field("circuit_id", pa.string(), nullable=False),
            pa.field("race_date", pa.string(), nullable=False),
            pa.field("driver_id", pa.string(), nullable=False),
            pa.field("constructor_id", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
        ]
    )


def build_entries_for_season(season: int, raw_dir: str, canonical_dir: str, overwrite: bool = True) -> Path:
    """
    Build canonical entries for one season.

    Args:
        season: Season year (e.g., 2018).
        raw_dir: Unused here (kept for consistent builder signature).
        canonical_dir: Canonical root directory.
        overwrite: If False and output exists, return existing output path.

    Returns:
        Path to the written entries Parquet file.
    """
    canonical_root = Path(canonical_dir)

    in_path = canonical_root / "results_race" / f"season={season}" / "results_race.parquet"

    out_dir = canonical_root / "entries" / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "entries.parquet"

    if out_path.exists() and not overwrite:
        return out_path

    if in_path.exists():
        # Read canonical race results (single file per season).
        t = pq.read_table(in_path)
        source = "results_race"
        source_path = str(in_path)
    else:
        # Fallbacks for "pre-season / pre-race" entry construction:
        # 1) configs/entries_override.yaml (explicit driver->constructor pairs)
        # 2) previous season results_race (best-effort)
        override_path = Path("configs") / "entries_override.yaml"
        if override_path.exists():
            cfg = load_yaml(override_path)
            season_cfg = cfg.get(str(season)) or cfg.get(season)
            if season_cfg and "drivers" in season_cfg:
                rows = season_cfg["drivers"]
                if rows:
                    weekends = _load_weekends_for_season(canonical_root, season)
                    t = _expand_entries_from_mapping(weekends, rows)
                    source = "manual_override"
                    source_path = str(override_path)
                else:
                    raise ValueError(f"entries_override.yaml has empty drivers list for season {season}")
            else:
                t = _fallback_entries_from_previous_season(canonical_root, season)
                source = "prev_season_results"
                source_path = str(canonical_root / "results_race" / f"season={season-1}" / "results_race.parquet")
        else:
            t = _fallback_entries_from_previous_season(canonical_root, season)
            source = "prev_season_results"
            source_path = str(canonical_root / "results_race" / f"season={season-1}" / "results_race.parquet")

    # Select only the participation/join columns we need.
    # NOTE: We do NOT use finish_position, status, etc. in entries.
    needed_cols = [
        "round",
        "race_name",
        "circuit_id",
        "race_date",
        "driver_id",
        "constructor_id",
        "source",
        "source_path",
    ]

    missing = [c for c in needed_cols if c not in t.column_names]
    # Allow source/source_path to be missing; we'll inject them.
    missing_required = [c for c in missing if c not in {"source", "source_path"}]
    if missing_required:
        raise ValueError(f"results_race is missing required columns for entries: {missing_required}")

    # Select only existing columns first; we'll add source/source_path if missing.
    existing_cols = [c for c in needed_cols if c in t.column_names]
    t = t.select(existing_cols)

    # Deduplicate to one row per (round, driver_id).
    # There should already be exactly one row per driver per round in results_race,
    # but we enforce this for safety and future-proofing.
    try:
        # Newer pyarrow versions support this.
        t = t.drop_duplicates(subset=["round", "driver_id"])
    except AttributeError:
        # Fallback: pandas-based dedupe (still deterministic, but not ideal for huge tables).
        df = t.to_pandas()
        df = df.drop_duplicates(subset=["round", "driver_id"], keep="first")
        # Ensure source metadata exists before casting
        if "source" not in df.columns:
            df["source"] = source
        if "source_path" not in df.columns:
            df["source_path"] = source_path
        t = pa.Table.from_pandas(df, schema=_entries_schema(), preserve_index=False)

    # Sort deterministically (stable output file).
    # This makes diffs/test comparisons far easier.
    try:
        sort_idx = pc.sort_indices(t, sort_keys=[("round", "ascending"), ("driver_id", "ascending")])
        t = t.take(sort_idx)
    except Exception:
        # If sort_indices behaves unexpectedly on this pyarrow version, we still write
        # a valid table; it will just be unsorted.
        pass

    # Enforce schema and write.
    # Ensure source metadata exists (for fallback paths).
    if "source" not in t.column_names:
        t = t.append_column("source", pa.array([source] * t.num_rows, type=pa.string()))
    if "source_path" not in t.column_names:
        t = t.append_column("source_path", pa.array([source_path] * t.num_rows, type=pa.string()))

    t = t.cast(_entries_schema())
    pq.write_table(t, out_path)

    return out_path


def _load_weekends_for_season(canonical_root: Path, season: int) -> pa.Table:
    d_weekends = ds.dataset(str(canonical_root / "weekends"), format="parquet", partitioning="hive")
    t = d_weekends.to_table(filter=ds.field("season") == season)
    if t.num_rows == 0:
        raise FileNotFoundError(f"Missing canonical weekends for season {season}. Build weekends first.")
    return t


def _expand_entries_from_mapping(weekends: pa.Table, rows: list[dict]) -> pa.Table:
    w = weekends.to_pandas()
    mapping = pd.DataFrame(rows)
    if not {"driver_id", "constructor_id"}.issubset(mapping.columns):
        raise ValueError("entries_override.yaml must include driver_id and constructor_id for each row")

    w = w[["round", "race_name", "circuit_id", "race_date"]]
    w["key"] = 1
    mapping["key"] = 1
    df = w.merge(mapping, on="key", how="outer").drop(columns=["key"])

    t = pa.Table.from_pandas(df, preserve_index=False)
    return t


def _fallback_entries_from_previous_season(canonical_root: Path, season: int) -> pa.Table:
    prev_path = canonical_root / "results_race" / f"season={season-1}" / "results_race.parquet"
    if not prev_path.exists():
        raise FileNotFoundError(
            f"Missing canonical race results for season {season} and previous season {season-1}. "
            "Provide configs/entries_override.yaml or build previous season results_race."
        )
    prev = pq.read_table(prev_path)
    prev_df = prev.to_pandas()[["driver_id", "constructor_id"]].drop_duplicates()

    weekends = _load_weekends_for_season(canonical_root, season).to_pandas()
    weekends = weekends[["round", "race_name", "circuit_id", "race_date"]]
    weekends["key"] = 1
    prev_df["key"] = 1
    df = weekends.merge(prev_df, on="key", how="outer").drop(columns=["key"])

    t = pa.Table.from_pandas(df, preserve_index=False)
    return t
