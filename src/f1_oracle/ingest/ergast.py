"""Ergast ingestion façade.

This module re-exports the primary Ergast ingestion functions so other parts of
the codebase can import from a single place.
"""

from __future__ import annotations

from f1_oracle.ingest.ergast_calendar import ingest_calendar_for_season
from f1_oracle.ingest.ergast_circuits import ingest_circuits_for_season
from f1_oracle.ingest.ergast_drivers import ingest_drivers_for_season
from f1_oracle.ingest.ergast_constructors import ingest_constructors_for_season
from f1_oracle.ingest.ergast_results_race import ingest_results_race_for_season
from f1_oracle.ingest.ergast_results_qualifying import ingest_results_qualifying_for_season
from f1_oracle.ingest.ergast_results_sprint import ingest_results_sprint_for_season
from f1_oracle.ingest.ergast_client import ErgastClient

__all__ = [
    "ErgastClient",
    "ingest_calendar_for_season",
    "ingest_circuits_for_season",
    "ingest_drivers_for_season",
    "ingest_constructors_for_season",
    "ingest_results_race_for_season",
    "ingest_results_qualifying_for_season",
    "ingest_results_sprint_for_season",
]