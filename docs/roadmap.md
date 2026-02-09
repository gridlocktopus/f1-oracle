# F1 Oracle — Roadmap
Version: v1.2  
Last updated: 2025-12-28

---

## v0 — Project foundation ✅ COMPLETE

- Repo scaffold
- CLI (`f1-oracle`) working
- Config-driven paths
- Python environment stabilized
  - Pinned to Python **3.12** for pandas/pyarrow stability
- Docs: design + roadmap + README

**Commits:**
- chore: scaffold project structure
- feat(cli): add f1-oracle command
- docs: add initial project docs
- chore(py): pin project to Python 3.12

---

## v0.1 — Historical ingestion (raw layer) ✅ COMPLETE

### Implemented
- Ergast-compatible API ingestion (Jolpica)
- Raw snapshots stored verbatim (no parsing or normalization)
- One immutable snapshot per dataset per season:
  - `data/raw/ergast/<season>/calendar.raw.json`
  - `data/raw/ergast/<season>/circuits.raw.json`
  - `data/raw/ergast/<season>/drivers.raw.json`
  - `data/raw/ergast/<season>/constructors.raw.json`
  - `data/raw/ergast/<season>/results_race.raw.json`
  - `data/raw/ergast/<season>/results_qualifying.raw.json`
  - `data/raw/ergast/<season>/results_sprint.raw.json`
- Pagination handled explicitly for large datasets
- Sprint ingestion handles seasons with **no sprint events** gracefully

**Design guarantees:**
- Raw layer is immutable
- Re-ingest is deterministic
- No data files tracked in git

---

## v0.2 — Canonical datasets (core tables) ⏳ IN PROGRESS

### Completed
- Canonical **weekends**
- Canonical **circuits**
- Canonical **drivers**
- Canonical **constructors**
- Canonical **results**
  - race
  - qualifying
  - sprint
- Deterministic raw → canonical conversion
- Parquet output:
  - One file per season
  - Explicit season partition folders (`season=YYYY`)
- Explicit avoidance of pandas partition writers

**Canonical output examples:**
- `data/canonical/weekends/season=YYYY/weekends.parquet`
- `data/canonical/results_race/season=YYYY/results_race.parquet`
- `data/canonical/results_qualifying/season=YYYY/results_qualifying.parquet`
- `data/canonical/results_sprint/season=YYYY/results_sprint.parquet`

### Next (current focus)
- Canonical **entries** table  
  *(driver ↔ constructor ↔ weekend participation)*

This table becomes the backbone join for:
- results
- features
- season-aware modeling

---

## v0.3 — Feature assembly & joins

- Join-safe canonical tables
- Explicit season scoping
- Weekend-level feature windows
- Baseline feature set (no FastF1 yet)
- Deterministic feature schema (stable column names & meanings)

---

## v0.4 — FastF1 integration

- Session-level ingestion
- Lap times, stints, compounds
- Practice-derived signals
- Sprint-aware feature gating

---

## v0.5 — Modeling v1

- Qualifying model
- Race model
- Probabilistic outputs per driver
- Frozen prediction storage
- DNF treated as a first-class outcome

---

## v0.6 — Replay & evaluation

- Chronological replay of 2025
- Train ≤2024, replay 2025
- Evaluation metrics
- Calibration diagnostics
- Training cutoff enforcement

---

## v1.0 — Live-ready architecture

- Update mode for 2026+
- Retraining cadence
- Versioned predictions
- Long-term schema stability