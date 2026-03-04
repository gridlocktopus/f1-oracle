# f1-oracle

A probabilistic Formula 1 weekend forecasting system. It produces driver‑level
ranked predictions (P1…Pn) and full position distributions for qualifying and
race sessions, with an update loop that mirrors a live season.

## What it does
- Predicts qualifying and race outcomes for a given round
- Outputs per‑driver probability distributions (P1–Pn, plus DNF for race)
- Updates models as real results arrive (qualifying → race → next round)
- Supports live usage or historical replay/backtesting

## Data sources
- **Ergast/Jolpica** (historical results, calendar, drivers, constructors)
- **FastF1** (practice sessions + lap‑level pace features)

## Setup
```
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Quick start (live season workflow)

### 1) Pre‑season setup
```
f1-oracle ingest ergast calendar --season 2026
f1-oracle ingest ergast circuits --season 2026
f1-oracle ingest ergast drivers --season 2026
f1-oracle ingest ergast constructors --season 2026

# Ingest historical results for training (recommended range)
for y in $(seq 2010 2025); do
  f1-oracle ingest ergast results race --season $y
  f1-oracle ingest ergast results qualifying --season $y
done

f1-oracle build canonical weekends --season 2026
f1-oracle build canonical circuits --season 2026
f1-oracle build canonical drivers --season 2026
f1-oracle build canonical constructors --season 2026
f1-oracle build canonical entries --season 2026

# Optional: ingest practice data for training seasons (recommended 2018+)
f1-oracle ingest fastf1 practice-range --start-season 2018 --end-season 2025 --only-missing

f1-oracle train
```

### 2) Each race weekend (round r)
```
# Practice ingest
f1-oracle ingest fastf1 practice --season 2026 --round r

# Predict qualifying (post‑practice)
f1-oracle predict quali --season 2026 --round r --tags dist,top --explain --print

# Update with actual qualifying results
f1-oracle update quali --season 2026 --round r

# Compare qualifying predictions vs actuals
f1-oracle compare quali --season 2026 --round r --kind top
f1-oracle compare quali --season 2026 --round r --kind dist

# Predict race (post‑qualifying)
f1-oracle predict race --season 2026 --round r --tags dist,top --explain --print

# Update with actual race results
f1-oracle update race --season 2026 --round r

# Compare race predictions vs actuals
f1-oracle compare race --season 2026 --round r --kind top
f1-oracle compare race --season 2026 --round r --kind dist
```

### One‑command round runner
```
f1-oracle run-round --season 2026 --round r --print --explain
```

## Evaluate performance over a range
```
f1-oracle evaluate --season 2026 --start-round 1 --end-round 10 --kind quali --mode top
f1-oracle evaluate --season 2026 --start-round 1 --end-round 10 --kind race --mode dist
```

## Web dashboard
The project includes a local dashboard to run commands and inspect predictions/evaluation.

### Start dashboard
```
source .venv/bin/activate
python -m pip install -e .
f1-oracle dashboard --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`.

### What you can do in the dashboard
- Load prediction outputs (`quali`/`race`, `top`/`dist`)
- Compare predictions vs actuals by round
- Evaluate a round range (`top`/`dist`)
- Run operational commands (`train`, `run-round`, `predict`, `update`)

### Quick test checklist
1. Open the dashboard and load a known prediction file from a completed round.
2. Run `compare` for the same round and verify a summary + table appear.
3. Run `evaluate` for a short range (for example rounds 1-3) and verify metrics render.
4. Launch a `predict race` job from Operations and confirm logs stream to completion.

## Outputs
- Predictions are written to `data/predictions/season=YYYY/round=R/...`
- Evaluation snapshots are written to `data/evaluation/baselines/`

## How the model learns
The system is designed for **live** usage:
- **Initial training** on historical seasons up to `train_end_year`
- **Round‑by‑round updates** as real qualifying and race results arrive

This means the model adapts to in‑season form shifts (upgrades, driver form, team strength)
while still leveraging the full historical baseline.

## Configuration
- `configs/seasons.yaml` — ingestion range and train/backtest split
- `configs/paths.yaml` — data and artifact directory layout
- `configs/apis.yaml` — API base URLs (Ergast/Jolpica)

## Docs
- `docs/tutorial.md` — step‑by‑step walkthrough
- `docs/design.md` — design notes
- `docs/roadmap.md` — implementation plan
