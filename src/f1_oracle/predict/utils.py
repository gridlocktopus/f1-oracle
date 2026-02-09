"""
Shared prediction utilities.
"""

from __future__ import annotations

import numpy as np


def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = scores / temperature
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / exp.sum()


def sample_plackett_luce(
    scores: np.ndarray,
    n_samples: int,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample rankings using a Plackett-Luce-style sequential draw.

    Lower scores are better; we invert to utilities internally.
    Returns array of shape (n_samples, n_items) with ranks (0-based positions).
    """
    n_items = len(scores)
    if n_items == 0:
        return np.empty((0, 0), dtype=int)

    utilities = -scores.astype(float)

    if rng is None:
        rng = np.random.default_rng(42)
    rankings = np.zeros((n_samples, n_items), dtype=int)

    for i in range(n_samples):
        remaining = list(range(n_items))
        pos = 0
        while remaining:
            rem_utils = utilities[remaining]
            probs = _softmax(rem_utils, temperature=temperature)
            pick = rng.choice(len(remaining), p=probs)
            rankings[i, pos] = remaining[pick]
            remaining.pop(pick)
            pos += 1

    return rankings


def rankings_to_position_probs(rankings: np.ndarray) -> np.ndarray:
    """
    Convert rankings (n_samples x n_items) to position probabilities (n_items x n_items).
    """
    if rankings.size == 0:
        return np.empty((0, 0))
    n_samples, n_items = rankings.shape
    probs = np.zeros((n_items, n_items), dtype=float)
    for s in range(n_samples):
        for pos in range(n_items):
            driver_idx = rankings[s, pos]
            probs[driver_idx, pos] += 1.0
    probs /= float(n_samples)
    return probs
