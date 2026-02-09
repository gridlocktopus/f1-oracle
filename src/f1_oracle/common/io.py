"""I/O utilities.

This module is intentionally small. It centralizes config loading so that:
- YAML parsing behavior is consistent across the codebase
- missing/invalid config errors are raised early and clearly
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    The configuration files in this project are expected to be YAML mappings.
    Empty files are treated as empty mappings.

    Args:
        path: File path to read (string or Path).

    Returns:
        A dictionary representing the YAML mapping.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML content is not a mapping.
    """

    # Normalize input to a Path for reliable filesystem operations.
    p = Path(path)

    # Fail fast if the config file is missing.
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    # Use safe_load to avoid constructing arbitrary Python objects from YAML.
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Enforce the expected shape: configs are YAML mappings (objects).
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping (YAML object) in {p}, got {type(data).__name__}"
        )

    return data