"""Local dashboard server for f1-oracle."""

from __future__ import annotations

import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from f1_oracle.common.io import load_yaml
from f1_oracle.evaluate.evaluate import evaluate_range
from f1_oracle.predict.compare import compare_quali, compare_race


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[2]
TEMPLATES_DIR = APP_ROOT / "templates"
STATIC_DIR = APP_ROOT / "static"

app = FastAPI(title="f1-oracle dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@dataclass
class Job:
    id: str
    commands: list[list[str]]
    status: str = "queued"
    output: str = ""
    returncode: int | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, text: str) -> None:
        with self._lock:
            self.output += text


JOBS: dict[str, Job] = {}
ALLOWED_TOP_LEVEL = {
    "train",
    "run-round",
    "predict",
    "update",
    "ingest",
    "evaluate",
    "compare",
    "status",
    "build",
}


class CommandRequest(BaseModel):
    argv: list[str] = Field(default_factory=list, description="Arguments after `f1-oracle`")
    argv_batch: list[list[str]] = Field(default_factory=list, description="Batch commands after `f1-oracle`")


def _paths() -> tuple[Path, Path, Path]:
    cfg = load_yaml(REPO_ROOT / "configs" / "paths.yaml")
    pred = REPO_ROOT / cfg.get("predictions", {}).get("dir", "data/predictions")
    canonical = REPO_ROOT / cfg.get("canonical", {}).get("dir", "data/canonical")
    evaluation = REPO_ROOT / cfg.get("evaluation", {}).get("dir", "data/evaluation")
    return pred, canonical, evaluation


def _pred_path(season: int, rnd: int, kind: str, mode: str) -> Path:
    pred_root, _, _ = _paths()
    if kind == "quali":
        stage = "post_practice"
        pred_kind = "quali_top" if mode == "top" else "quali_dist"
    else:
        stage = "post_quali"
        pred_kind = "race_top" if mode == "top" else "race_dist"
    return pred_root / f"season={season}" / f"round={rnd}" / stage / pred_kind / "predictions.parquet"


def _run_job(job: Job) -> None:
    try:
        job.status = "running"
        last_rc = 0
        for idx, cmd in enumerate(job.commands, start=1):
            job.append(f"\n$ f1-oracle {' '.join(cmd)}\n")
            proc = subprocess.Popen(
                ["f1-oracle", *cmd],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                job.append(line)
            proc.wait()
            last_rc = int(proc.returncode or 0)
            if last_rc != 0:
                job.returncode = last_rc
                job.status = "failed"
                job.append(f"\nCommand {idx}/{len(job.commands)} failed with rc={last_rc}\n")
                return
        job.returncode = last_rc
        job.status = "completed"
    except Exception as exc:  # pragma: no cover
        job.error = str(exc)
        job.status = "failed"


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/predictions")
def api_predictions(season: int, rnd: int, kind: str, mode: str, limit: int = 100) -> dict[str, Any]:
    path = _pred_path(season, rnd, kind, mode)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Prediction file not found: {path}")
    df = pd.read_parquet(path).head(limit)
    return {"path": str(path), "rows": df.to_dict(orient="records")}


@app.get("/api/compare")
def api_compare(season: int, rnd: int, kind: str, mode: str, limit: int = 100) -> dict[str, Any]:
    if kind not in {"quali", "race"}:
        raise HTTPException(status_code=400, detail="kind must be 'quali' or 'race'")
    if mode not in {"top", "dist"}:
        raise HTTPException(status_code=400, detail="mode must be 'top' or 'dist'")

    if kind == "quali":
        out = compare_quali(season=season, rnd=rnd, kind=mode)
    else:
        out = compare_race(season=season, rnd=rnd, kind=mode)
    details = out["details"]
    rows = [] if details is None else details.head(limit).to_dict(orient="records")
    return {"summary": out["summary"], "rows": rows}


@app.get("/api/evaluate")
def api_evaluate(
    season: int,
    start_round: int,
    end_round: int,
    kind: str,
    mode: str,
) -> dict[str, Any]:
    summary, df = evaluate_range(
        kind=kind,
        mode=mode,
        season=season,
        start_round=start_round,
        end_round=end_round,
    )
    rows = [] if df.empty else df.to_dict(orient="records")
    return {"summary": summary, "rows": rows}


def _read_season_rounds(base: Path) -> pd.DataFrame:
    if not base.exists():
        return pd.DataFrame(columns=["season", "round"])
    t = ds.dataset(str(base), format="parquet", partitioning="hive").to_table(columns=["season", "round"])
    df = t.to_pandas()
    if df.empty:
        return pd.DataFrame(columns=["season", "round"])
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    return df.dropna(subset=["season", "round"])[["season", "round"]].astype({"season": int, "round": int})


@app.get("/api/training-coverage")
def api_training_coverage() -> dict[str, Any]:
    _, canonical_root, _ = _paths()
    weekends = _read_season_rounds(canonical_root / "weekends")
    race_results = _read_season_rounds(canonical_root / "results_race")

    if weekends.empty:
        return {
            "trained_range": "none",
            "completed_seasons": [],
            "current_season": None,
            "current_progress": "no canonical weekends data",
            "rows": [],
        }

    total_rounds = weekends.groupby("season")["round"].nunique().rename("weekends_rounds")
    race_rounds = (
        race_results.groupby("season")["round"].nunique().rename("race_results_rounds")
        if not race_results.empty
        else pd.Series(dtype=int, name="race_results_rounds")
    )
    frame = pd.concat([total_rounds, race_rounds], axis=1).fillna(0).reset_index()
    frame["race_results_rounds"] = frame["race_results_rounds"].astype(int)
    frame["complete"] = frame["race_results_rounds"] >= frame["weekends_rounds"]

    completed = sorted(frame.loc[frame["complete"], "season"].tolist())
    if completed:
        trained_range = f"{completed[0]} -> {completed[-1]}"
    else:
        trained_range = "none"

    current_season = int(frame["season"].max())
    current_row = frame.loc[frame["season"] == current_season].iloc[0]
    current_progress = f"{int(current_row['race_results_rounds'])}/{int(current_row['weekends_rounds'])} rounds with race results"

    return {
        "trained_range": trained_range,
        "completed_seasons": completed,
        "current_season": current_season,
        "current_progress": current_progress,
        "rows": frame.sort_values("season", ascending=False).to_dict(orient="records"),
    }


@app.post("/api/jobs")
def api_create_job(payload: CommandRequest) -> dict[str, Any]:
    commands: list[list[str]] = []
    if payload.argv_batch:
        commands = payload.argv_batch
    elif payload.argv:
        commands = [payload.argv]
    else:
        raise HTTPException(status_code=400, detail="argv/argv_batch must not be empty")

    for cmd in commands:
        if not cmd:
            raise HTTPException(status_code=400, detail="empty command in batch")
        if cmd[0] not in ALLOWED_TOP_LEVEL:
            raise HTTPException(status_code=400, detail=f"unsupported command: {cmd[0]}")

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, commands=commands)
    JOBS[job_id] = job
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job.id,
        "commands": job.commands,
        "status": job.status,
        "returncode": job.returncode,
        "error": job.error,
        "output": job.output,
    }
