"""
Model registry: save/load model artifacts by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from f1_oracle.common.io import load_yaml


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    path: Path


def _artifacts_dir() -> Path:
    cfg = load_yaml(Path("configs") / "paths.yaml")
    root = cfg.get("artifacts", {}).get("dir", "data/artifacts")
    return Path(root) / "models"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_model(name: str, model: Any) -> ModelArtifact:
    root = _artifacts_dir() / name
    _ensure_dir(root)
    path = root / "model.joblib"
    joblib.dump(model, path)
    return ModelArtifact(name=name, path=path)


def load_model(name: str) -> Any | None:
    root = _artifacts_dir() / name
    path = root / "model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)
