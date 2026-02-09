"""
Ergast-compatible HTTP client (Jolpica / Ergast mirror).

Why this exists:
- Centralize HTTP behavior (base URL, timeouts, retries, backoff).
- Bulk ingestion will hit API rate limits (HTTP 429). We handle that here so
  callers don't need to implement retry logic.

Design constraints (per project rules):
- Raw ingestion files remain verbatim snapshots of API responses.
- Retrying must NOT change semantics; it only improves robustness.

Key behavior:
- Retries on transient errors (429, 5xx, some connection issues)
- Respects Retry-After header if present (common for 429)
- Exponential backoff with small jitter
- Conservative defaults so we don't hammer the API
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ErgastClientConfig:
    """
    Tunables for HTTP behavior.

    These defaults are intentionally conservative to keep the API happy during bulk runs.
    """

    base_url: str = "https://api.jolpi.ca/ergast/f1"
    timeout_seconds: float = 30.0

    # Retry behavior
    max_retries: int = 8  # total attempts = 1 initial + up to max_retries retries
    backoff_initial_seconds: float = 1.0
    backoff_max_seconds: float = 60.0

    # Status codes to retry
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)

    # If the server does not provide Retry-After for 429, we use backoff + jitter.
    jitter_fraction: float = 0.2  # +/- 20% jitter


class ErgastClient:
    """
    Simple HTTP client for Ergast-compatible endpoints.

    Public API:
    - get_text(path, params=None) -> str

    Example:
    - get_text("2018/results.json", params={"limit": 100, "offset": 0})
    """

    def __init__(self, config: ErgastClientConfig | None = None) -> None:
        self.config = config or ErgastClientConfig()

        # Session improves performance and keeps things consistent across calls.
        self._session = requests.Session()

        # A polite User-Agent helps some providers; also useful for debugging.
        self._session.headers.update(
            {
                "User-Agent": "f1-oracle/0.x (ErgastClient; https://github.com/)",
                "Accept": "application/json",
            }
        )

    def _full_url(self, path: str) -> str:
        """Join base_url and a relative endpoint path."""
        base = self.config.base_url.rstrip("/")
        rel = path.lstrip("/")
        return f"{base}/{rel}"

    def _parse_retry_after_seconds(self, response: requests.Response) -> float | None:
        """
        Parse Retry-After header if present.

        Retry-After can be:
        - integer seconds
        - HTTP-date (rare; we don't implement date parsing here)

        We only support integer seconds because that's what Jolpica typically returns (if any).
        """
        ra = response.headers.get("Retry-After")
        if not ra:
            return None

        ra = ra.strip()
        try:
            seconds = float(ra)
            if seconds < 0:
                return None
            return seconds
        except ValueError:
            # If it's an HTTP-date, we ignore and fall back to backoff.
            return None

    def _compute_backoff(self, attempt_index: int) -> float:
        """
        Compute exponential backoff for a given retry attempt index (0-based for first retry).
        """
        # Exponential: initial * 2^attempt, capped.
        base = self.config.backoff_initial_seconds * (2 ** attempt_index)
        base = min(base, self.config.backoff_max_seconds)

        # Jitter to avoid thundering herd if multiple retries happen.
        jitter = base * self.config.jitter_fraction
        return max(0.0, base + random.uniform(-jitter, +jitter))

    def get_text(self, path: str, params: dict[str, Any] | None = None) -> str:
        """
        GET an endpoint and return response text.

        Retries:
        - 429: respects Retry-After if available, otherwise exponential backoff.
        - 5xx: exponential backoff.
        - Connection/timeouts: exponential backoff.

        Raises:
        - requests.exceptions.HTTPError if the response is non-success and retries exhausted.
        - requests.exceptions.RequestException for non-HTTP failures if retries exhausted.
        """
        url = self._full_url(path)

        # Total attempts = 1 + max_retries
        for attempt in range(self.config.max_retries + 1):
            try:
                r = self._session.get(url, params=params, timeout=self.config.timeout_seconds)

                # Success path
                if 200 <= r.status_code < 300:
                    return r.text

                # Retryable status codes
                if r.status_code in self.config.retry_statuses and attempt < self.config.max_retries:
                    # Prefer Retry-After for 429 when provided.
                    if r.status_code == 429:
                        wait = self._parse_retry_after_seconds(r)
                        if wait is None:
                            wait = self._compute_backoff(attempt_index=attempt)
                    else:
                        wait = self._compute_backoff(attempt_index=attempt)

                    # Sleep before retrying.
                    time.sleep(wait)
                    continue

                # Non-retryable, or retries exhausted: raise
                r.raise_for_status()
                return r.text  # unreachable, but keeps type checkers happy

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                # Retry transient network errors
                if attempt < self.config.max_retries:
                    wait = self._compute_backoff(attempt_index=attempt)
                    time.sleep(wait)
                    continue
                raise e

        # Should never reach here, but raise a clear error if we do.
        raise RuntimeError("Unexpected retry loop exit in ErgastClient.get_text()")