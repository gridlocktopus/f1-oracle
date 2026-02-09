 # F1 Oracle — Live Season Tutorial

This is a step‑by‑step checklist to run the project as if it were a live season.

## 0) One‑time setup
```
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 1) Beginning of a new season (pre‑season)

### 1.1 Ingest season metadata
```
f1-oracle ingest ergast calendar --season 2025
f1-oracle ingest ergast circuits --season 2025
f1-oracle ingest ergast drivers --season 2025
f1-oracle ingest ergast constructors --season 2025
```

### 1.2 Build canonical metadata tables
```
f1-oracle build canonical weekends --season 2025
f1-oracle build canonical circuits --season 2025
f1-oracle build canonical drivers --season 2025
f1-oracle build canonical constructors --season 2025
```

### 1.3 Build entries for the season
If you already have some 2025 race results, entries can be built directly:
```
f1-oracle build canonical entries --season 2025
```

If you do NOT have any 2025 race results yet, provide a lineup override:
```
# edit configs/entries_override.yaml and add 2025 driver_id -> constructor_id pairs
f1-oracle build canonical entries --season 2025
```

### 1.4 (Optional) Ingest practice data for training seasons
FastF1 has better coverage from 2018+.
```
f1-oracle ingest fastf1 practice-range --start-season 2018 --end-season 2024
```

### 1.5 Train models (uses train_end_year from configs/seasons.yaml)
```
f1-oracle train
```

## 2) Each race weekend (round r)

### 2.1 Ingest practice sessions for the round
```
f1-oracle ingest fastf1 practice --season 2025 --round r
```

### 2.2 Predict qualifying after practice
```
f1-oracle predict quali --season 2025 --round r --tags dist,top --explain --print
```

### 2.3 After qualifying: update + retrain
```
f1-oracle update quali --season 2025 --round r
```

### 2.4 Compare qualifying predictions vs actuals
```
f1-oracle compare quali --season 2025 --round r --kind top
f1-oracle compare quali --season 2025 --round r --kind dist
```

### 2.5 Predict race after qualifying
```
f1-oracle predict race --season 2025 --round r --tags dist,top --explain --print
```

### 2.6 After race: update + retrain
```
f1-oracle update race --season 2025 --round r
```

### 2.7 Compare race predictions vs actuals
```
f1-oracle compare race --season 2025 --round r --kind top
f1-oracle compare race --season 2025 --round r --kind dist
```

## 3) Evaluate accuracy across multiple rounds
```
f1-oracle evaluate --season 2025 --start-round 1 --end-round 5 --kind quali --mode top
f1-oracle evaluate --season 2025 --start-round 1 --end-round 5 --kind quali --mode dist
f1-oracle evaluate --season 2025 --start-round 1 --end-round 5 --kind race --mode top
f1-oracle evaluate --season 2025 --start-round 1 --end-round 5 --kind race --mode dist
```

## 4) Helpful checks
```
f1-oracle status --season 2025 --round r
```

## Notes
- `update` pulls results from Ergast/Jolpica and retrains for the next round.
- `train` always respects `configs/seasons.yaml -> split.train_end_year`.
- If entries look wrong (extra drivers), update `configs/entries_override.yaml` and rebuild entries.
