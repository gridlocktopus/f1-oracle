"""
FastF1 ingestion utilities.

We ingest practice session results into a lightweight raw layer to support
post-practice qualifying predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import time

import pandas as pd
from f1_oracle.common.io import load_yaml


@dataclass(frozen=True)
class PracticeIngestConfig:
    cache_dir: Path
    raw_dir: Path
    sessions: tuple[str, ...] = ("FP1", "FP2", "FP3")
    session_delay_seconds: float = 2.0
    round_delay_seconds: float = 1.0


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _resolve_practice_out_path(raw_dir: Path, season: int, rnd: int, session_type: str) -> Path:
    out_dir = raw_dir / "fastf1" / f"season={season}" / f"round={rnd}"
    _ensure_dir(out_dir)
    return out_dir / f"practice_{session_type.lower()}.parquet"

def _resolve_laps_out_path(raw_dir: Path, season: int, rnd: int, session_type: str) -> Path:
    out_dir = raw_dir / "fastf1" / f"season={season}" / f"round={rnd}"
    _ensure_dir(out_dir)
    return out_dir / f"practice_laps_{session_type.lower()}.parquet"


def _canonical_drivers_path() -> Path:
    cfg = load_yaml(Path("configs") / "paths.yaml")
    canonical_root = Path(cfg.get("canonical", {}).get("dir", "data/canonical"))
    return canonical_root / "drivers"


def _load_driver_lookup_for_season(season: int) -> tuple[dict[int, str], dict[str, str]]:
    """
    Return lookup maps:
      - permanent_number (int) -> canonical driver_id
      - code (upper str) -> canonical driver_id
    """
    path = _canonical_drivers_path() / f"season={season}" / "drivers.parquet"
    if not path.exists():
        return {}, {}

    df = pd.read_parquet(path)
    by_number: dict[int, str] = {}
    by_code: dict[str, str] = {}

    if "permanent_number" in df.columns and "driver_id" in df.columns:
        nums = pd.to_numeric(df["permanent_number"], errors="coerce")
        for n, d in zip(nums, df["driver_id"]):
            if pd.notna(n) and pd.notna(d):
                by_number[int(n)] = str(d)

    if "code" in df.columns and "driver_id" in df.columns:
        for c, d in zip(df["code"], df["driver_id"]):
            if pd.notna(c) and pd.notna(d):
                by_code[str(c).strip().upper()] = str(d)

    return by_number, by_code


def _resolve_driver_ids(df: pd.DataFrame, season: int) -> pd.Series:
    """
    Resolve canonical driver_id from FastF1 frames using multiple fallbacks.
    """
    out = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    cols = set(df.columns)
    by_number, by_code = _load_driver_lookup_for_season(season)

    # 1) Native driver id from FastF1 when present and not empty.
    for key in ("DriverId", "driverId"):
        if key in cols:
            s = df[key].astype("string").str.strip().str.lower()
            s = s.where(s.notna() & (s != ""))
            out = out.where(out.notna(), s)
            break

    # 2) Fallback from three-letter code fields.
    for key in ("Abbreviation", "Driver"):
        if key in cols:
            code = df[key].astype("string").str.strip().str.upper()
            mapped = code.map(by_code)
            out = out.where(out.notna(), mapped)

    # 3) Fallback from driver number.
    if "DriverNumber" in cols:
        num = pd.to_numeric(df["DriverNumber"], errors="coerce")
        mapped = num.map(lambda x: by_number.get(int(x)) if pd.notna(x) else pd.NA)
        out = out.where(out.notna(), mapped)

    return out.astype("string")


def _normalize_results(
    df: pd.DataFrame,
    season: int,
    rnd: int,
    session_type: str,
    laps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Normalize FastF1 session results to a compact schema.
    Expected columns in df: DriverId, Time, Position (fallbacks supported).
    """
    if df.empty:
        return pd.DataFrame()

    cols = set(df.columns)
    driver_id = _resolve_driver_ids(df, season=season)

    if "Time" in cols:
        best_time = df["Time"]
    elif "BestTime" in cols:
        best_time = df["BestTime"]
    else:
        best_time = pd.Series([pd.NaT] * len(df))

    if "Position" in cols:
        position = df["Position"]
    elif "position" in cols:
        position = df["position"]
    else:
        position = pd.Series([pd.NA] * len(df))

    out = pd.DataFrame(
        {
            "season": season,
            "round": rnd,
            "session_type": session_type,
            "driver_id": driver_id,
            "best_lap_time": best_time,
            "position": position,
        }
    )
    out["best_lap_time"] = pd.to_timedelta(out["best_lap_time"], errors="coerce")
    out["best_lap_time_ms"] = out["best_lap_time"].dt.total_seconds() * 1000.0
    out["position"] = pd.to_numeric(out["position"], errors="coerce").astype("Int64")

    # Fallback: if best lap values are missing in results, derive from laps.
    if laps is not None and not laps.empty:
        l = laps.copy()
        l["driver_id"] = _resolve_driver_ids(l, season=season)
        if "LapTime" in l.columns:
            l["lap_time_ms"] = pd.to_timedelta(l["LapTime"], errors="coerce").dt.total_seconds() * 1000.0
            l_best = (
                l.dropna(subset=["driver_id", "lap_time_ms"])
                .groupby("driver_id", as_index=False)["lap_time_ms"]
                .min()
                .rename(columns={"lap_time_ms": "lap_best_ms"})
            )
            if not l_best.empty:
                out = out.merge(l_best, on="driver_id", how="left")
                out["best_lap_time_ms"] = out["best_lap_time_ms"].where(out["best_lap_time_ms"].notna(), out["lap_best_ms"])
                out = out.drop(columns=["lap_best_ms"])

    # If no explicit position, rank by best lap time.
    if out["position"].isna().all() and out["best_lap_time_ms"].notna().any():
        out["position"] = out["best_lap_time_ms"].rank(method="min").astype("Int64")

    # Keep only rows with resolved driver id.
    out["driver_id"] = out["driver_id"].astype("string").str.strip().str.lower()
    out = out[out["driver_id"].notna() & (out["driver_id"] != "")]

    # Deduplicate driver rows (best lap / best position)
    if not out.empty:
        out = out.sort_values(["driver_id", "best_lap_time_ms"], na_position="last")
        out = out.drop_duplicates(subset=["driver_id"], keep="first").reset_index(drop=True)

    return out


def _summarize_laps(laps: pd.DataFrame, season: int, rnd: int, session_type: str) -> pd.DataFrame:
    """
    Summarize lap-level data into per-driver long-run pace metrics.
    """
    if laps is None or laps.empty:
        return pd.DataFrame()

    df = laps.copy()
    df["driver_id"] = _resolve_driver_ids(df, season=season)

    if "LapTime" in df.columns:
        df["lap_time_ms"] = pd.to_timedelta(df["LapTime"], errors="coerce").dt.total_seconds() * 1000.0
    else:
        return pd.DataFrame()

    # Exclude in/out laps where possible
    for col in ("PitInTime", "PitOutTime"):
        if col in df.columns:
            df = df[df[col].isna()]

    df = df[df["driver_id"].notna() & (df["driver_id"].astype("string").str.strip() != "")]
    df = df[df["lap_time_ms"].notna()]
    if df.empty:
        return pd.DataFrame()

    def _trimmed_mean(x: pd.Series) -> float:
        x = x.sort_values()
        n = len(x)
        if n < 5:
            return float(x.mean())
        k = int(n * 0.2)
        if k * 2 >= n:
            return float(x.mean())
        return float(x.iloc[k : n - k].mean())

    agg = (
        df.groupby("driver_id")
        .agg(
            practice_laps_count=("lap_time_ms", "count"),
            practice_longrun_med_ms=("lap_time_ms", "median"),
            practice_longrun_mean_ms=("lap_time_ms", "mean"),
            practice_longrun_trimmed_ms=("lap_time_ms", _trimmed_mean),
        )
        .reset_index()
    )
    agg["season"] = season
    agg["round"] = rnd
    agg["session_type"] = session_type
    return agg


def ingest_practice_for_weekend(
    *,
    season: int,
    rnd: int,
    raw_dir: str,
    cache_dir: str,
    sessions: Iterable[str] | None = None,
    session_delay_seconds: float = 2.0,
    only_missing: bool = False,
) -> list[Path]:
    """
    Ingest practice sessions (FP1/FP2/FP3 by default) for a season round.

    Returns list of written parquet paths.
    """
    # Import fastf1 lazily to keep import-time fast when not used.
    import fastf1  # type: ignore

    cfg = PracticeIngestConfig(
        cache_dir=Path(cache_dir),
        raw_dir=Path(raw_dir),
        sessions=tuple(sessions) if sessions is not None else ("FP1", "FP2", "FP3"),
        session_delay_seconds=session_delay_seconds,
    )

    _ensure_dir(cfg.cache_dir)
    fastf1.Cache.enable_cache(str(cfg.cache_dir))

    written: list[Path] = []

    for session_type in cfg.sessions:
        try:
            session = fastf1.get_session(season, rnd, session_type)
        except Exception as exc:  # FastF1 raises ValueError when schedule fetch fails
            print(f"FastF1: failed to resolve session for season {season} round {rnd} {session_type}: {exc}")
            continue

        try:
            # Load minimal data needed for session.results + laps (avoid heavy telemetry/weather/messages)
            session.load(telemetry=False, weather=False, messages=False)
        except Exception as exc:
            print(f"FastF1: failed to load session data for {season} round {rnd} {session_type}: {exc}")
            continue

        if session.results is None or session.results.empty:
            continue

        out_path = _resolve_practice_out_path(cfg.raw_dir, season, rnd, session_type)
        laps_path = _resolve_laps_out_path(cfg.raw_dir, season, rnd, session_type)

        # Skip work if requested and outputs already exist
        if only_missing and out_path.exists() and laps_path.exists():
            time.sleep(cfg.session_delay_seconds)
            continue

        norm = _normalize_results(session.results, season, rnd, session_type, laps=session.laps)
        if not norm.empty:
            if not (only_missing and out_path.exists()):
                norm.to_parquet(out_path, index=False)
                written.append(out_path)

        # Laps summary (long-run pace)
        try:
            laps = session.laps
            laps_summary = _summarize_laps(laps, season, rnd, session_type)
            if not laps_summary.empty:
                if not (only_missing and laps_path.exists()):
                    laps_summary.to_parquet(laps_path, index=False)
                    written.append(laps_path)
        except Exception as exc:
            print(f"FastF1: failed to summarize laps for {season} round {rnd} {session_type}: {exc}")

        time.sleep(cfg.session_delay_seconds)

    return written


def ingest_practice_for_rounds(
    *,
    season: int,
    rounds: Iterable[int],
    raw_dir: str,
    cache_dir: str,
    sessions: Iterable[str] | None = None,
    session_delay_seconds: float = 2.0,
    round_delay_seconds: float = 1.0,
    only_missing: bool = False,
) -> list[Path]:
    """
    Ingest practice sessions for multiple rounds with throttling between rounds.
    """
    written: list[Path] = []
    for rnd in rounds:
        written.extend(
            ingest_practice_for_weekend(
                season=season,
                rnd=int(rnd),
                raw_dir=raw_dir,
                cache_dir=cache_dir,
                sessions=sessions,
                session_delay_seconds=session_delay_seconds,
                only_missing=only_missing,
            )
        )
        time.sleep(round_delay_seconds)
    return written
