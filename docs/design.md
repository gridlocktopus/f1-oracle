# F1 Weekend Forecasting System
Design Document (Canonical)

Owner: Stefan  
Status: v1.1  
Last updated: 2025-12-27

---

## 1. Purpose

Build a long-lived forecasting system for Formula 1 that:

- Predicts **qualifying and race outcomes**
- Outputs **per-driver probability distributions**
- Updates predictions throughout a race weekend
- Learns continuously as new results arrive
- Remains valid across future seasons (2026+)

The system is designed to be:
- probabilistic (not just rankings),
- chronologically correct (no data leakage),
- auditable and reproducible.

---

## 2. Outputs

For each driver, at each prediction checkpoint:

- Probability distribution over finishing positions  
  - Qualifying: P1–P20  
  - Race: P1–P20 + DNF
- A derived “best-guess” finishing position/order

### DNF handling
DNF is treated as a **single outcome class**. Ordering among DNFs is not modeled in v1.

---

## 3. Prediction checkpoints

Predictions are generated at three explicit moments:

1) **Baseline (pre-FP1)**
   - Historical + season context only
   - Produces an early race prior

2) **Post-practice**
   - Uses practice session data
   - Produces **qualifying predictions**

3) **Post-qualifying**
   - Uses practice + official qualifying results
   - Produces **race predictions**

All predictions are **frozen and stored immutably** once generated.

---

## 4. Learning & lifecycle model

### Continuous learning philosophy

The system is designed to run indefinitely:

- After each completed session or weekend:
  - ingest official results,
  - evaluate prior predictions,
  - append labeled data,
  - optionally retrain on a defined cadence.

Once data is ingested, it becomes part of the permanent historical record.

There is **no season-specific logic** baked into training.

---

### Train / backtest split (initial build)

For the initial implementation and evaluation workflow:

- **Training set:** all seasons through end of **2024 (≤2024)**
- **Backtest set:** the **2025** season, replayed chronologically

Raw data for 2025 is ingested and stored, but it is **excluded from training** until an explicit “train-as-of end of 2025” snapshot is created (e.g. to seed the 2026 season).

---

## 5. Data sources

### Official structure & results (authoritative)
- Ergast-compatible API (Ergast / Jolpica)
  - Calendar, circuits
  - Drivers & constructors
  - Official qualifying, race, sprint results
  - Status codes (DNF, DSQ, etc.)

### Session performance
- FastF1
  - Practice, qualifying, sprint, race sessions
  - Lap times, compounds, stints, conditions

### Weather (v1)
- Forecast data only
- Stored with “as-of” timestamp
- Minimal fields (rain probability, temperatures)

Raw data is stored **verbatim** before any transformation.

---

## 6. System architecture

### Data layers (final decision)

1) **Raw**
   - Immutable snapshots fetched from APIs
   - Stored exactly as received (no parsing or prettification)
   - Used as the single source of truth

2) **Canonical**
   - Deterministic, normalized tables derived from raw
   - Join-safe IDs and schemas
   - Stored as Parquet

There is **no persistent “clean” layer**. Debug transforms may exist ad-hoc but are not part of the pipeline.

---

### Core components

1) **Ingestion**
   - Fetch raw data
   - Store immutable snapshots

2) **Canonicalization**
   - Normalize IDs, names, session labels
   - Produce join-safe canonical tables

3) **Feature builder**
   - Builds checkpoint-specific feature rows
   - Enforces strict “as-of” constraints

4) **Models**
   - Qualifying model
   - Race model
   - Optional baseline race prior

5) **Prediction store**
   - Immutable storage of predictions
   - Includes model version, git hash, timestamp

6) **Evaluator**
   - Scores probability distributions and rankings
   - Appends labeled rows to training data

7) **Orchestrator**
   - Runs replay, training, and update workflows

---

## 7. Canonical storage layout (implemented)

Canonical datasets are stored as: 
`data/canonical/<dataset>/season=YYYY/<dataset>.parquet`

Examples:
- `data/canonical/weekends/season=2018/weekends.parquet`

**Design decision:**  
Avoid Parquet dataset writers with automatic partitioning.  
Write **one Parquet file per season partition** for reliability and explicit control.

---

## 8. Canonical identifiers

- driver_id
- constructor_id
- circuit_id
- season
- round
- session_type ∈ {FP1, FP2, FP3, SQ, Sprint, Q, Race}
- checkpoint ∈ {baseline, post_practice, post_quali}
- as_of_timestamp

---

## 9. Modeling strategy

- Separate models for:
  - **Qualifying** (post-practice → quali distribution)
  - **Race** (post-qualifying → race distribution)
- Models output **full probability distributions**
- Rankings are derived from probabilities

---

## 10. Evaluation & calibration

### Probability quality
- Log loss / NLL (primary)
- Brier scores (top-3, top-10, DNF)

### Ranking quality
- Top-k inclusion
- Rank correlation
- Mean absolute error

---

## 11. Weekend execution flows

### Normal weekend
1) Baseline build → optional baseline race prediction  
2) Practice ingested → qualifying prediction (frozen)  
3) Qualifying ingested → race prediction (frozen)  
4) Race ingested → evaluation → optional retraining  

### Sprint weekend (v1)
- Sprint sessions ingested once completed
- Sprint signals become allowed known inputs
- Sprint/SQ predictions are optional (v1.5+)

---

## 12. Non-goals (v1)

- In-race live prediction
- Strategy simulation
- Ordering among DNFs
- Betting / odds integration

---

## 13. Open questions (explicitly deferred)

(unchanged; still valid — intentionally kept open)
- Weather provider choice
- Sprint signal handling depth
- Tyre legality modeling
- Reliability decomposition
- Retraining cadence defaults