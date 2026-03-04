"""Local dashboard server for f1-oracle."""

from __future__ import annotations

import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
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
    command: list[str]
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
        proc = subprocess.Popen(
            ["f1-oracle", *job.command],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            job.append(line)
        proc.wait()
        job.returncode = proc.returncode
        job.status = "completed" if proc.returncode == 0 else "failed"
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


@app.post("/api/jobs")
def api_create_job(payload: CommandRequest) -> dict[str, Any]:
    if not payload.argv:
        raise HTTPException(status_code=400, detail="argv must not be empty")
    if payload.argv[0] not in ALLOWED_TOP_LEVEL:
        raise HTTPException(status_code=400, detail="unsupported command")

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, command=payload.argv)
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
        "command": job.command,
        "status": job.status,
        "returncode": job.returncode,
        "error": job.error,
        "output": job.output,
    }

