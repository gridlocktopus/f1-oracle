"""Command-line interface for the f1-oracle project.

This module defines the console script entrypoint (`f1-oracle`) and a small
subcommand tree.

Current capabilities:
- Print an ingestion plan from config
- Ingest raw Ergast calendar snapshots
- Ingest raw Ergast circuits snapshots
- Ingest raw Ergast drivers snapshots
- Ingest raw Ergast constructors snapshots
- Ingest raw Ergast race results snapshots
- Ingest raw Ergast qualifying results snapshots
- Ingest raw Ergast sprint results snapshots
- Build canonical weekends (Parquet) from raw snapshots
- Build canonical circuits (Parquet) from raw snapshots
- Build canonical drivers (Parquet) from raw snapshots
- Build canonical constructors (Parquet) from raw snapshots
- Build canonical race results (Parquet) from raw snapshots
- Build canonical qualifying results (Parquet) from raw snapshots
- Build canonical sprint results (Parquet) from raw snapshots
- Build canonical entries (Parquet) from raw snapshots
- Build feature frames (Parquet) for modeling (v0.3+)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from f1_oracle.common.io import load_yaml

from f1_oracle.canonical.circuits import build_circuits_for_season
from f1_oracle.canonical.constructors import build_constructors_for_season
from f1_oracle.canonical.drivers import build_drivers_for_season
from f1_oracle.canonical.entries import build_entries_for_season
from f1_oracle.canonical.results_qualifying import build_results_qualifying_for_season
from f1_oracle.canonical.results_race import build_results_race_for_season
from f1_oracle.canonical.results_sprint import build_results_sprint_for_season
from f1_oracle.canonical.weekends import build_weekends_for_season

from f1_oracle.features.build import build_features_dataset

from f1_oracle.ingest.ergast import (
    ingest_calendar_for_season,
    ingest_circuits_for_season,
    ingest_constructors_for_season,
    ingest_drivers_for_season,
    ingest_results_qualifying_for_season,
    ingest_results_race_for_season,
    ingest_results_sprint_for_season,
)
from f1_oracle.ingest.fastf1 import ingest_practice_for_weekend, ingest_practice_for_rounds

from f1_oracle.predict.run_quali import run_quali_prediction
from f1_oracle.predict.run_race import run_race_prediction
from f1_oracle.orchestrator.update_and_retrain import update_quali_and_retrain, update_race_and_retrain
from f1_oracle.orchestrator.train_as_of import train_all

import pyarrow.dataset as ds
import pandas as pd

from f1_oracle.predict.compare import compare_quali, compare_race
from f1_oracle.evaluate.evaluate import evaluate_range, save_evaluation, snapshot_path

# Canonical session-type labels used throughout the project.
#
# IMPORTANT: We keep these as ALL-CAPS enums to avoid mixed-case values
# (e.g., "Sprint" vs "SPRINT") drifting into joins, features, and models.
SESSION_TYPES = ("FP1", "FP2", "FP3", "SQ", "SPRINT", "Q", "RACE")


def _load_configs() -> tuple[dict, dict]:
    """Load and return (seasons_cfg, paths_cfg) from YAML config files."""
    seasons_cfg = load_yaml(Path("configs") / "seasons.yaml")
    paths_cfg = load_yaml(Path("configs") / "paths.yaml")
    return seasons_cfg, paths_cfg


def _resolve_season_and_raw_dir(args: argparse.Namespace) -> tuple[int, str]:
    """Resolve (season, raw_dir) for ingest commands."""
    seasons_cfg, paths_cfg = _load_configs()
    start_year = int(seasons_cfg["ingest"]["start_year"])

    season = int(args.season) if args.season is not None else start_year
    raw_dir = paths_cfg.get("raw", {}).get("dir", "data/raw")

    return season, raw_dir


def _cmd_ingest_ergast(args: argparse.Namespace) -> int:
    """Generic Ergast ingest runner; calls the ingest function attached to the parser."""
    season, raw_dir = _resolve_season_and_raw_dir(args)

    # `ingest_fn` is attached via argparse's set_defaults(...) for each subcommand.
    ingest_fn = args.ingest_fn
    out_path = ingest_fn(season=season, raw_dir=raw_dir)

    print(f"Wrote: {out_path}")
    return 0


def _cmd_ingest_plan(_: argparse.Namespace) -> int:
    """Print the ingestion plan derived from configuration."""
    seasons_cfg, paths_cfg = _load_configs()

    ingest = seasons_cfg.get("ingest", {})
    split = seasons_cfg.get("split", {})

    start_year = int(ingest["start_year"])
    end_year = int(ingest["end_year"])
    train_end_year = int(split["train_end_year"])
    backtest_year = int(split["backtest_year"])

    raw_dir = paths_cfg.get("raw", {}).get("dir", "data/raw")

    print("Ingestion plan")
    print("-------------")
    print(f"Ingest years: {start_year}–{end_year}")
    print(f"Train cutoff: ≤{train_end_year}")
    print(f"Backtest year: {backtest_year}")
    print(f"Raw output dir: {raw_dir}")
    print(f"Session types (canonical, ALL-CAPS): {', '.join(SESSION_TYPES)}")

    return 0


def _resolve_raw_and_canonical_dirs() -> tuple[str, str]:
    """Resolve (raw_dir, canonical_dir) for canonical build commands."""
    _, paths_cfg = _load_configs()

    raw_dir = paths_cfg.get("raw", {}).get("dir", "data/raw")
    canonical_dir = paths_cfg.get("canonical", {}).get("dir", "data/canonical")

    return raw_dir, canonical_dir


def _resolve_seasons_for_build(args: argparse.Namespace) -> list[int]:
    """Resolve season list for canonical build commands."""
    seasons_cfg, _ = _load_configs()

    ingest_cfg = seasons_cfg.get("ingest", {})
    start_year = int(ingest_cfg["start_year"])
    end_year = int(ingest_cfg["end_year"])

    # If --season is provided, build one season. Otherwise build the configured range.
    if args.season is not None:
        return [int(args.season)]
    return list(range(start_year, end_year + 1))


def _cmd_build_canonical(args: argparse.Namespace) -> int:
    """Canonical build runner; calls the build function attached to the parser."""
    seasons = _resolve_seasons_for_build(args)
    raw_dir, canonical_dir = _resolve_raw_and_canonical_dirs()

    # `build_fn` is attached via argparse's set_defaults(...) for each dataset.
    build_fn = args.build_fn

    for s in seasons:
        out_path = build_fn(season=s, raw_dir=raw_dir, canonical_dir=canonical_dir)
        print(f"Wrote: {out_path}")

    return 0


def _cmd_build_features(args: argparse.Namespace) -> int:
    """Feature build runner; dispatches by dataset name."""
    season = int(args.season)
    dataset = str(args.dataset)

    out_path = build_features_dataset(dataset=dataset, season=season)
    print(f"Wrote: {out_path}")
    return 0


def _cmd_ingest_fastf1_practice(args: argparse.Namespace) -> int:
    _, paths_cfg = _load_configs()
    raw_dir = paths_cfg.get("raw", {}).get("dir", "data/raw")
    cache_dir = paths_cfg.get("artifacts", {}).get("dir", "data/artifacts")
    cache_dir = str(Path(cache_dir) / "fastf1_cache")

    written = ingest_practice_for_weekend(
        season=int(args.season),
        rnd=int(args.round),
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        sessions=tuple(args.sessions.split(",")),
        session_delay_seconds=float(args.session_delay),
        only_missing=bool(args.only_missing),
    )

    for p in written:
        print(f"Wrote: {p}")

    return 0


def _cmd_ingest_fastf1_practice_range(args: argparse.Namespace) -> int:
    _, paths_cfg = _load_configs()
    raw_dir = paths_cfg.get("raw", {}).get("dir", "data/raw")
    cache_dir = paths_cfg.get("artifacts", {}).get("dir", "data/artifacts")
    cache_dir = str(Path(cache_dir) / "fastf1_cache")

    canonical_dir = paths_cfg.get("canonical", {}).get("dir", "data/canonical")
    d_weekends = ds.dataset(f"{canonical_dir}/weekends", format="parquet", partitioning="hive")

    written = []
    for season in range(int(args.start_season), int(args.end_season) + 1):
        w = d_weekends.to_table(filter=ds.field("season") == season).to_pandas()
        rounds = sorted(pd.to_numeric(w["round"], errors="coerce").dropna().astype(int).unique().tolist())
        if not rounds:
            continue
        written.extend(
            ingest_practice_for_rounds(
                season=season,
                rounds=rounds,
                raw_dir=raw_dir,
                cache_dir=cache_dir,
                sessions=tuple(args.sessions.split(",")),
                session_delay_seconds=float(args.session_delay),
                round_delay_seconds=float(args.round_delay),
                only_missing=bool(args.only_missing),
            )
        )

    for p in written:
        print(f"Wrote: {p}")
    return 0


def _cmd_predict_quali(args: argparse.Namespace) -> int:
    paths = run_quali_prediction(
        season=int(args.season),
        rnd=int(args.round),
        tags=args.tags,
        samples=int(args.samples),
        explain=bool(args.explain),
        print_output=bool(args.print_output),
        print_limit=int(args.print_limit),
    )
    for p in paths:
        print(f"Wrote: {p}")
    return 0


def _cmd_predict_race(args: argparse.Namespace) -> int:
    paths = run_race_prediction(
        season=int(args.season),
        rnd=int(args.round),
        tags=args.tags,
        samples=int(args.samples),
        explain=bool(args.explain),
        print_output=bool(args.print_output),
        print_limit=int(args.print_limit),
    )
    for p in paths:
        print(f"Wrote: {p}")
    return 0


def _cmd_update_quali(args: argparse.Namespace) -> int:
    path = update_quali_and_retrain(season=int(args.season), rnd=int(args.round))
    print(f"Updated + retrained quali model: {path}")
    return 0


def _cmd_update_race(args: argparse.Namespace) -> int:
    path_finish, path_dnf = update_race_and_retrain(season=int(args.season), rnd=int(args.round))
    print(f"Updated + retrained race models: {path_finish}, {path_dnf}")
    return 0


def _cmd_compare_quali(args: argparse.Namespace) -> int:
    report = compare_quali(
        season=int(args.season),
        rnd=int(args.round),
        kind=str(args.kind),
    )
    print(report["summary"])
    df = report["details"]
    if df is not None:
        limit = int(args.print_limit)
        if limit > 0:
            print(df.head(limit).to_string(index=False))
        else:
            print(df.to_string(index=False))
    return 0


def _cmd_compare_race(args: argparse.Namespace) -> int:
    report = compare_race(
        season=int(args.season),
        rnd=int(args.round),
        kind=str(args.kind),
    )
    print(report["summary"])
    df = report["details"]
    if df is not None:
        limit = int(args.print_limit)
        if limit > 0:
            print(df.head(limit).to_string(index=False))
        else:
            print(df.to_string(index=False))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    summary, df = evaluate_range(
        kind=str(args.kind),
        mode=str(args.mode),
        season=int(args.season),
        start_round=int(args.start_round),
        end_round=int(args.end_round),
    )
    print(summary)
    if not df.empty:
        print(df.to_string(index=False))
    if args.save:
        path = Path(str(args.save))
        save_evaluation(summary, df, path)
        print(f"Saved evaluation to: {path}")
    if args.snapshot:
        snap_path = snapshot_path(
            kind=str(args.kind),
            mode=str(args.mode),
            season=int(args.season),
            start_round=int(args.start_round),
            end_round=int(args.end_round),
            label=str(args.label) if args.label else None,
        )
        save_evaluation(summary, df, snap_path)
        print(f"Saved snapshot to: {snap_path}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    if args.fastf1:
        # ingest practice data for range before training
        start_season = int(args.start_season)
        end_season = int(args.end_season)
        _cmd_ingest_fastf1_practice_range(
            argparse.Namespace(
                start_season=start_season,
                end_season=end_season,
                sessions=args.sessions,
                session_delay=args.session_delay,
                round_delay=args.round_delay,
            )
        )

    quali_path, race_finish, race_dnf = train_all()
    print(f"Trained models: {quali_path}, {race_finish}, {race_dnf}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    _, paths_cfg = _load_configs()
    canonical_dir = paths_cfg.get("canonical", {}).get("dir", "data/canonical")
    raw_dir = paths_cfg.get("raw", {}).get("dir", "data/raw")

    season = int(args.season)
    rnd = int(args.round) if args.round is not None else None

    def _has_canonical(ds_name: str) -> bool:
        root = Path(canonical_dir) / ds_name / f"season={season}"
        return root.exists()

    print(f"Season {season} status")
    print(f"canonical/weekends: {_has_canonical('weekends')}")
    print(f"canonical/entries: {_has_canonical('entries')}")
    print(f"canonical/results_qualifying: {_has_canonical('results_qualifying')}")
    print(f"canonical/results_sprint: {_has_canonical('results_sprint')}")
    print(f"canonical/results_race: {_has_canonical('results_race')}")

    if rnd is not None:
        practice_dir = Path(raw_dir) / "fastf1" / f"season={season}" / f"round={rnd}"
        print(f"fastf1 practice for round {rnd}: {practice_dir.exists()}")
        weekends_path = Path(canonical_dir) / "weekends" / f"season={season}" / "weekends.parquet"
        if weekends_path.exists():
            weekends = pd.read_parquet(weekends_path)
            row = weekends[pd.to_numeric(weekends["round"], errors="coerce") == rnd]
            if not row.empty and "sprint_date" in row.columns:
                sprint_date = row.iloc[0]["sprint_date"]
                print(f"round {rnd} sprint weekend: {pd.notna(sprint_date)}")

    return 0


def _cmd_run_round(args: argparse.Namespace) -> int:
    season = int(args.season)
    rnd = int(args.round)
    tags = args.tags
    samples = int(args.samples)
    explain = bool(args.explain)
    print_output = bool(args.print_output)
    print_limit = int(args.print_limit)

    # 1) Practice ingest
    _cmd_ingest_fastf1_practice(
        argparse.Namespace(
            season=season,
            round=rnd,
            sessions=args.sessions,
            session_delay=args.session_delay,
            only_missing=args.only_missing,
        )
    )

    # 2) Predict quali
    run_quali_prediction(
        season=season,
        rnd=rnd,
        tags=tags,
        samples=samples,
        explain=explain,
        print_output=print_output,
        print_limit=print_limit,
    )

    # 3) Update quali + compare
    update_quali_and_retrain(season=season, rnd=rnd)
    for kind in ("top", "dist"):
        report = compare_quali(season=season, rnd=rnd, kind=kind)
        print(report["summary"])
        if report["details"] is not None:
            print(report["details"].head(print_limit).to_string(index=False))

    # 4) Predict race
    run_race_prediction(
        season=season,
        rnd=rnd,
        tags=tags,
        samples=samples,
        explain=explain,
        print_output=print_output,
        print_limit=print_limit,
    )

    # 5) Update race + compare
    update_race_and_retrain(season=season, rnd=rnd)
    for kind in ("top", "dist"):
        report = compare_race(season=season, rnd=rnd, kind=kind)
        print(report["summary"])
        if report["details"] is not None:
            print(report["details"].head(print_limit).to_string(index=False))

    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "f1_oracle.dashboard.app:app",
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(prog="f1-oracle")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------------
    # Ingestion command group
    # -----------------------
    ingest = subparsers.add_parser("ingest", help="Ingestion commands")
    ingest_sub = ingest.add_subparsers(dest="ingest_cmd", required=True)

    plan = ingest_sub.add_parser("plan", help="Print ingestion plan from config")
    plan.set_defaults(func=_cmd_ingest_plan)

    ergast = ingest_sub.add_parser("ergast", help="Ergast/Jolpica ingestion")
    ergast_sub = ergast.add_subparsers(dest="ergast_cmd", required=True)

    calendar = ergast_sub.add_parser("calendar", help="Ingest season race calendar (raw snapshot)")
    calendar.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: ingest.start_year from configs/seasons.yaml)",
    )
    calendar.set_defaults(func=_cmd_ingest_ergast, ingest_fn=ingest_calendar_for_season)

    circuits = ergast_sub.add_parser("circuits", help="Ingest season circuits list (raw snapshot)")
    circuits.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: ingest.start_year from configs/seasons.yaml)",
    )
    circuits.set_defaults(func=_cmd_ingest_ergast, ingest_fn=ingest_circuits_for_season)

    drivers = ergast_sub.add_parser("drivers", help="Ingest season drivers list (raw snapshot)")
    drivers.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: ingest.start_year from configs/seasons.yaml)",
    )
    drivers.set_defaults(func=_cmd_ingest_ergast, ingest_fn=ingest_drivers_for_season)

    constructors = ergast_sub.add_parser("constructors", help="Ingest season constructors list (raw snapshot)")
    constructors.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: ingest.start_year from configs/seasons.yaml)",
    )
    constructors.set_defaults(func=_cmd_ingest_ergast, ingest_fn=ingest_constructors_for_season)

    results = ergast_sub.add_parser("results", help="Ingest Ergast classification/results datasets (raw snapshots)")
    results_sub = results.add_subparsers(dest="results_cmd", required=True)

    race = results_sub.add_parser("race", help="Ingest season race results (raw snapshot)")
    race.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: ingest.start_year from configs/seasons.yaml)",
    )
    race.set_defaults(func=_cmd_ingest_ergast, ingest_fn=ingest_results_race_for_season)

    qualifying = results_sub.add_parser("qualifying", help="Ingest season qualifying results (raw snapshot)")
    qualifying.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: ingest.start_year from configs/seasons.yaml)",
    )
    qualifying.set_defaults(func=_cmd_ingest_ergast, ingest_fn=ingest_results_qualifying_for_season)

    sprint = results_sub.add_parser("sprint", help="Ingest season sprint results (raw snapshot, if present)")
    sprint.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: ingest.start_year from configs/seasons.yaml)",
    )
    sprint.set_defaults(func=_cmd_ingest_ergast, ingest_fn=ingest_results_sprint_for_season)

    fastf1 = ingest_sub.add_parser("fastf1", help="FastF1 ingestion")
    fastf1_sub = fastf1.add_subparsers(dest="fastf1_cmd", required=True)

    practice = fastf1_sub.add_parser("practice", help="Ingest practice session results (FP1/FP2/FP3)")
    practice.add_argument("--season", type=int, required=True, help="Season year")
    practice.add_argument("--round", type=int, required=True, help="Round number")
    practice.add_argument(
        "--sessions",
        type=str,
        default="FP1,FP2,FP3",
        help="Comma-separated session types (default: FP1,FP2,FP3)",
    )
    practice.add_argument(
        "--session-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between FastF1 session downloads (default: 2.0)",
    )
    practice.add_argument(
        "--only-missing",
        action="store_true",
        help="Only ingest missing practice and lap summaries",
    )
    practice.set_defaults(func=_cmd_ingest_fastf1_practice)

    practice_range = fastf1_sub.add_parser("practice-range", help="Ingest practice sessions for a season range")
    practice_range.add_argument("--start-season", type=int, required=True, help="Start season (inclusive)")
    practice_range.add_argument("--end-season", type=int, required=True, help="End season (inclusive)")
    practice_range.add_argument(
        "--sessions",
        type=str,
        default="FP1,FP2,FP3",
        help="Comma-separated session types (default: FP1,FP2,FP3)",
    )
    practice_range.add_argument(
        "--session-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between FastF1 session downloads (default: 2.0)",
    )
    practice_range.add_argument(
        "--round-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between rounds (default: 1.0)",
    )
    practice_range.add_argument(
        "--only-missing",
        action="store_true",
        help="Only ingest missing practice and lap summaries",
    )
    practice_range.set_defaults(func=_cmd_ingest_fastf1_practice_range)

    # -----------------------
    # Build command group
    # -----------------------
    build = subparsers.add_parser("build", help="Build derived datasets")
    build_sub = build.add_subparsers(dest="build_cmd", required=True)

    canonical = build_sub.add_parser("canonical", help="Build canonical datasets")
    canonical_sub = canonical.add_subparsers(dest="canonical_ds", required=True)

    weekends = canonical_sub.add_parser("weekends", help="Build canonical weekends dataset (Parquet)")
    weekends.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: configured ingest start/end range)",
    )
    weekends.set_defaults(func=_cmd_build_canonical, build_fn=build_weekends_for_season)

    circuits = canonical_sub.add_parser("circuits", help="Build canonical circuits dataset (Parquet)")
    circuits.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: configured ingest start/end range)",
    )
    circuits.set_defaults(func=_cmd_build_canonical, build_fn=build_circuits_for_season)

    drivers = canonical_sub.add_parser("drivers", help="Build canonical drivers dataset (Parquet)")
    drivers.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: configured ingest start/end range)",
    )
    drivers.set_defaults(func=_cmd_build_canonical, build_fn=build_drivers_for_season)

    constructors = canonical_sub.add_parser("constructors", help="Build canonical constructors dataset (Parquet)")
    constructors.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: configured ingest start/end range)",
    )
    constructors.set_defaults(func=_cmd_build_canonical, build_fn=build_constructors_for_season)

    results = canonical_sub.add_parser("results", help="Build canonical results datasets (Parquet)")
    results_sub = results.add_subparsers(dest="results_ds", required=True)

    race = results_sub.add_parser("race", help="Build canonical race results dataset (Parquet)")
    race.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: configured ingest start/end range)",
    )
    race.set_defaults(func=_cmd_build_canonical, build_fn=build_results_race_for_season)

    qualifying = results_sub.add_parser("qualifying", help="Build canonical qualifying results dataset (Parquet)")
    qualifying.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: configured ingest start/end range)",
    )
    qualifying.set_defaults(func=_cmd_build_canonical, build_fn=build_results_qualifying_for_season)

    sprint = results_sub.add_parser("sprint", help="Build canonical sprint results dataset (Parquet)")
    sprint.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: configured ingest start/end range)",
    )
    sprint.set_defaults(func=_cmd_build_canonical, build_fn=build_results_sprint_for_season)

    entries = canonical_sub.add_parser("entries", help="Build canonical entries table (Parquet)")
    entries.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: configured ingest start/end range)",
    )
    entries.set_defaults(func=_cmd_build_canonical, build_fn=build_entries_for_season)

    # -----------------------
    # Features build group (v0.3+)
    # -----------------------
    features = build_sub.add_parser("features", help="Build feature frames for modeling")
    features.add_argument(
        "dataset",
        type=str,
        choices=["qualifying-baseline"],
        help="Feature dataset identifier",
    )
    features.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year to build features for",
    )
    features.set_defaults(func=_cmd_build_features)

    # -----------------------
    # Prediction commands
    # -----------------------
    predict = subparsers.add_parser("predict", help="Prediction commands")
    predict_sub = predict.add_subparsers(dest="predict_cmd", required=True)

    pred_quali = predict_sub.add_parser("quali", help="Predict qualifying (post-practice)")
    pred_quali.add_argument("--season", type=int, required=True, help="Season year")
    pred_quali.add_argument("--round", type=int, required=True, help="Round number")
    pred_quali.add_argument("--tags", type=str, default="dist", help="Comma tags: dist,top (default: dist)")
    pred_quali.add_argument("--samples", type=int, default=5000, help="Monte Carlo samples for dist output")
    pred_quali.add_argument("--explain", action="store_true", help="Write explainability outputs")
    pred_quali.add_argument("--print", dest="print_output", action="store_true", help="Print predictions to stdout")
    pred_quali.add_argument("--print-limit", type=int, default=20, help="Max rows to print per output")
    pred_quali.set_defaults(func=_cmd_predict_quali)

    pred_race = predict_sub.add_parser("race", help="Predict race (post-qualifying)")
    pred_race.add_argument("--season", type=int, required=True, help="Season year")
    pred_race.add_argument("--round", type=int, required=True, help="Round number")
    pred_race.add_argument("--tags", type=str, default="dist", help="Comma tags: dist,top (default: dist)")
    pred_race.add_argument("--samples", type=int, default=5000, help="Monte Carlo samples for dist output")
    pred_race.add_argument("--explain", action="store_true", help="Write explainability outputs")
    pred_race.add_argument("--print", dest="print_output", action="store_true", help="Print predictions to stdout")
    pred_race.add_argument("--print-limit", type=int, default=20, help="Max rows to print per output")
    pred_race.set_defaults(func=_cmd_predict_race)

    # -----------------------
    # Compare commands
    # -----------------------
    compare = subparsers.add_parser("compare", help="Compare predictions to actual results")
    compare_sub = compare.add_subparsers(dest="compare_cmd", required=True)

    cmp_quali = compare_sub.add_parser("quali", help="Compare qualifying predictions")
    cmp_quali.add_argument("--season", type=int, required=True, help="Season year")
    cmp_quali.add_argument("--round", type=int, required=True, help="Round number")
    cmp_quali.add_argument("--kind", type=str, choices=["top", "dist"], default="top", help="Prediction kind")
    cmp_quali.add_argument("--print-limit", type=int, default=0, help="Max rows to print (0 = all)")
    cmp_quali.set_defaults(func=_cmd_compare_quali)

    cmp_race = compare_sub.add_parser("race", help="Compare race predictions")
    cmp_race.add_argument("--season", type=int, required=True, help="Season year")
    cmp_race.add_argument("--round", type=int, required=True, help="Round number")
    cmp_race.add_argument("--kind", type=str, choices=["top", "dist"], default="top", help="Prediction kind")
    cmp_race.add_argument("--print-limit", type=int, default=0, help="Max rows to print (0 = all)")
    cmp_race.set_defaults(func=_cmd_compare_race)

    # -----------------------
    # Update + retrain commands
    # -----------------------
    update = subparsers.add_parser("update", help="Update with actuals and retrain")
    update_sub = update.add_subparsers(dest="update_cmd", required=True)

    upd_quali = update_sub.add_parser("quali", help="Update qualifying results and retrain")
    upd_quali.add_argument("--season", type=int, required=True, help="Season year")
    upd_quali.add_argument("--round", type=int, required=True, help="Round number (just completed)")
    upd_quali.set_defaults(func=_cmd_update_quali)

    upd_race = update_sub.add_parser("race", help="Update race results and retrain")
    upd_race.add_argument("--season", type=int, required=True, help="Season year")
    upd_race.add_argument("--round", type=int, required=True, help="Round number (just completed)")
    upd_race.set_defaults(func=_cmd_update_race)

    # -----------------------
    # Training + status commands
    # -----------------------
    train = subparsers.add_parser("train", help="Train models on all available data")
    train.add_argument("--fastf1", action="store_true", help="Ingest FastF1 practice data before training")
    train.add_argument(
        "--start-season",
        type=int,
        default=int(_load_configs()[0]["ingest"]["start_year"]),
        help="Start season for FastF1 ingest (default: configs ingest.start_year)",
    )
    train.add_argument(
        "--end-season",
        type=int,
        default=int(_load_configs()[0]["ingest"]["end_year"]),
        help="End season for FastF1 ingest (default: configs ingest.end_year)",
    )
    train.add_argument(
        "--sessions",
        type=str,
        default="FP1,FP2,FP3",
        help="Comma-separated session types (default: FP1,FP2,FP3)",
    )
    train.add_argument(
        "--session-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between FastF1 session downloads (default: 2.0)",
    )
    train.add_argument(
        "--round-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between rounds (default: 1.0)",
    )
    train.set_defaults(func=_cmd_train)

    status = subparsers.add_parser("status", help="Show data availability for a season/round")
    status.add_argument("--season", type=int, required=True, help="Season year")
    status.add_argument("--round", type=int, default=None, help="Round number (optional)")
    status.set_defaults(func=_cmd_status)

    # -----------------------
    # Evaluation commands
    # -----------------------
    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions vs actuals over a round range")
    evaluate.add_argument("--season", type=int, required=True, help="Season year")
    evaluate.add_argument("--start-round", type=int, required=True, help="Start round (inclusive)")
    evaluate.add_argument("--end-round", type=int, required=True, help="End round (inclusive)")
    evaluate.add_argument("--kind", type=str, choices=["quali", "race"], required=True, help="Prediction kind")
    evaluate.add_argument("--mode", type=str, choices=["top", "dist"], required=True, help="Prediction mode")
    evaluate.add_argument("--save", type=str, default=None, help="Save evaluation details to a CSV/Parquet file")
    evaluate.add_argument("--snapshot", action="store_true", help="Save a baseline snapshot under data/evaluation")
    evaluate.add_argument("--label", type=str, default=None, help="Optional label for snapshot filename")
    evaluate.set_defaults(func=_cmd_evaluate)

    # -----------------------
    # Run-round command
    # -----------------------
    run_round = subparsers.add_parser("run-round", help="Run a full round: ingest, predict, update, compare")
    run_round.add_argument("--season", type=int, required=True, help="Season year")
    run_round.add_argument("--round", type=int, required=True, help="Round number")
    run_round.add_argument("--tags", type=str, default="dist,top", help="Comma tags: dist,top (default: dist,top)")
    run_round.add_argument("--samples", type=int, default=5000, help="Monte Carlo samples for dist output")
    run_round.add_argument("--explain", action="store_true", help="Write explainability outputs")
    run_round.add_argument("--print", dest="print_output", action="store_true", help="Print predictions to stdout")
    run_round.add_argument("--print-limit", type=int, default=20, help="Max rows to print per output")
    run_round.add_argument(
        "--sessions",
        type=str,
        default="FP1,FP2,FP3",
        help="Comma-separated session types (default: FP1,FP2,FP3)",
    )
    run_round.add_argument(
        "--session-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between FastF1 session downloads (default: 2.0)",
    )
    run_round.add_argument(
        "--only-missing",
        action="store_true",
        help="Only ingest missing practice and lap summaries",
    )
    run_round.set_defaults(func=_cmd_run_round)

    # -----------------------
    # Dashboard command
    # -----------------------
    dashboard = subparsers.add_parser("dashboard", help="Run local web dashboard")
    dashboard.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    dashboard.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    dashboard.add_argument("--reload", action="store_true", help="Enable autoreload for development")
    dashboard.set_defaults(func=_cmd_dashboard)

    return parser


def main() -> None:
    """Console script entrypoint for the `f1-oracle` command."""
    parser = _build_parser()
    args = parser.parse_args()

    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
