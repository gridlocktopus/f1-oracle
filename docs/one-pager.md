# F1 Weekend Forecasting System — One-Pager (v1.1)

## What it does
Forecasts Formula 1 qualifying and race outcomes using historical data, practice sessions, and qualifying results.

Outputs:
- Per-driver probability distributions
- A derived best-guess finishing order

DNF is modeled as a single outcome class.

---

## Prediction checkpoints

1) **Baseline (pre-FP1)**  
   Historical context → early race prior

2) **Post-practice**  
   Practice data → qualifying prediction

3) **Post-qualifying**  
   Practice + qualifying → race prediction

Predictions are frozen and stored immutably.

---

## Data sources
- Official results & structure: Ergast-compatible API (stored as raw snapshots)
- Session data: FastF1
- Weather: forecast only (as-of checkpoint)

---

## Train / backtest split (initial build)
- **Train:** all seasons through end of **2024**
- **Backtest:** **2025**, replayed chronologically with strict “as-of” rules

2025 data is ingested but excluded from training until explicitly enabled.

---

## Learning loop
After each completed session or weekend:
1) Freeze and store predictions  
2) Ingest official results  
3) Evaluate predictions and store labeled outcomes  
4) Retrain on schedule and export the “current best” model  

Designed to operate indefinitely (2026+).

---

## Modeling
- Separate models for qualifying and race
- Probabilistic outputs, not deterministic rankings
- Rankings derived from probability distributions

---

## Design principles
- No data leakage
- Immutable predictions
- Chronologically correct replay
- Reproducible (models tied to git + data cutoffs)

Status: **Implementation in progress (foundation complete)**