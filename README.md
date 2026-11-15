# ons-forecast
 
Forecasting daily peak electricity demand for Brazil's National Interconnected System (SIN), using open data from ONS.
 
> **Status:** work in progress. Results section is a placeholder until the first full run.
 
---
 
## The question
 
Brazil's grid operator (ONS) publishes daily maximum demand per subsystem. The practical question this repo answers:
 
**How far ahead can daily peak demand be predicted, and does a machine learning model actually beat a well-chosen naive baseline?**
 
The second half of that question matters more than the first. Most forecasting portfolios skip the baseline and report an impressive-looking error in isolation. Here the baseline is the benchmark every model has to clear.
 
---
 
## Why this dataset
 
Daily peak demand drives real decisions in the Brazilian power sector: generation dispatch, reserve margin, transmission constraints, and contract exposure for free-market energy traders. An error of a few percent on the peak is not an abstract metric — it is the difference between covering load and buying energy on the short-term market.
 
---
 
## Data
 
**Source:** [ONS Dados Abertos — Demanda Máxima Diária por Subsistema](https://dados.ons.org.br/dataset/demanda_maxima_di)
 
| Property | Value |
|---|---|
| Granularity | Daily, by subsystem |
| Subsystems | Southeast/Midwest, South, Northeast, North |
| Period used | 2024 – 2026 |
| Formats published | CSV, Parquet, XLSX |
| Encoding | UTF-8, delimited |
| License | Open data (ONS) |
 
The dataset reports both instantaneous and integrated demand per subsystem, and includes an estimated contribution from distributed micro and mini generation (MMGD), which ONS derives from forecast weather data.
 
> **TODO:** confirm exact column names and types against `DicionarioDados_DemandaMaximaDiaria.json` after the first download, then update the table below.
 
| Column | Type | Description |
|---|---|---|
| `din_instante` | date | Reference day |
| `id_subsistema` | text | Subsystem code |
| `nom_subsistema` | text | Subsystem name |
| `val_*` | numeric | Demand values (instantaneous / integrated / MMGD) |
 
### Data volume — read this before judging the models
 
Three years of daily data gives roughly **1,100 observations per subsystem**. That is enough for seasonal structure (weekly and annual) but modest for anything data-hungry. This constraint shapes every modeling choice here, and the Limitations section says what it rules out.
 
---
 
## Branch strategy
 
The same pipeline is implemented twice, against the two formats ONS publishes. This is deliberate: it isolates the format decision and makes the trade-off measurable instead of asserted.
 
| Branch | Source format | Purpose |
|---|---|---|
| `main` | Parquet | Reference implementation |
| `csv-pipeline` | CSV | Same logic, raw text ingestion |
 
**What differs between branches:** only the ingestion layer. The dbt models, the feature engineering, the validation strategy and the models themselves are identical. Everything downstream of ingestion is shared.
 
**What the comparison measures:**
 
| Dimension | Reported in |
|---|---|
| File size on disk | `docs/format-comparison.md` |
| Ingestion wall time | same |
| Type fidelity (dates and numerics arriving correctly typed) | same |
| Lines of parsing code required | same |
 
The expected conclusion is that Parquet wins on all four. Publishing both anyway is the point: a stated preference is weaker evidence than a measured one.
 
---
 
## Repository structure
 
```
ons-forecast/
├── data/
│   ├── raw/                      # as downloaded from ONS (gitignored)
│   └── ons.duckdb                # local warehouse (gitignored)
├── dbt/
│   ├── models/
│   │   ├── staging/
│   │   │   └── stg_demanda_maxima.sql
│   │   ├── marts/
│   │   │   ├── fct_demanda_diaria.sql
│   │   │   └── fct_demanda_features.sql
│   │   └── schema.yml            # data quality tests
│   ├── dbt_project.yml
│   └── profiles.yml
├── src/
│   ├── ingest.py                 # the only file that differs across branches
│   ├── baselines.py
│   ├── models.py
│   ├── validate.py               # walk-forward cross-validation
│   └── evaluate.py
├── notebooks/
│   └── 01_exploration.ipynb
├── docs/
│   ├── format-comparison.md
│   └── figures/
├── requirements.txt
└── README.md
```
 
---
 
## Stack
 
| Layer | Tool | Why |
|---|---|---|
| Warehouse | DuckDB | Runs locally, no server, no cost; anyone can clone and run |
| Transformation | dbt | SQL models, dependency graph, built-in data quality tests |
| Modeling | Python — pandas, statsmodels, scikit-learn, XGBoost | |
| Visualization | matplotlib | |
 
The transformation layer is SQL rather than pandas on purpose. Lags, rolling means and calendar features are expressed as window functions, which is how this work is actually done in a warehouse.
 
---
 
## Data layer (dbt)
 
### `stg_demanda_maxima`
 
Typing, column renaming, deduplication, and a filter to the analysis period.
 
### `fct_demanda_diaria`
 
One row per subsystem per day. Gaps made explicit rather than silently dropped.
 
### `fct_demanda_features`
 
Temporal features built with window functions:
 
```sql
SELECT
    din_instante,
    id_subsistema,
    demanda_maxima_mw,
 
    LAG(demanda_maxima_mw, 1) OVER w   AS lag_1d,
    LAG(demanda_maxima_mw, 7) OVER w   AS lag_7d,
    LAG(demanda_maxima_mw, 364) OVER w AS lag_52w,
 
    AVG(demanda_maxima_mw) OVER (
        PARTITION BY id_subsistema
        ORDER BY din_instante
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS media_movel_7d,
 
    EXTRACT(DOW   FROM din_instante) AS dia_semana,
    EXTRACT(MONTH FROM din_instante) AS mes
 
FROM {{ ref('fct_demanda_diaria') }}
WINDOW w AS (PARTITION BY id_subsistema ORDER BY din_instante)
```
 
**Every window is strictly backward-looking.** Note the `ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING` — excluding the current row is what keeps the target out of its own feature. This is the single most common source of leakage in time series work, and it is enforced here in SQL rather than trusted to a later step.
 
### Data quality tests (`schema.yml`)
 
| Test | Column |
|---|---|
| `not_null` | `din_instante`, `id_subsistema`, `demanda_maxima_mw` |
| `unique` (combination) | `din_instante` + `id_subsistema` |
| `accepted_values` | `id_subsistema` |
| custom range check | `demanda_maxima_mw` within plausible bounds |
 
---
 
## Method
 
Models are introduced in order of increasing complexity. Each has to beat the one before it to justify its existence.
 
| Tier | Model | Rationale |
|---|---|---|
| 0 | Seasonal naive — same weekday, previous week | The honest floor. Demand is strongly weekly |
| 0 | Previous-day persistence | Secondary reference |
| 1 | SARIMA | Classical statistical benchmark; interpretable, works with limited history |
| 2 | XGBoost on the feature table | Handles calendar effects and multiple inputs jointly |
 
**Forecast horizon:** 7 days ahead.
 
### Validation
 
**Walk-forward (expanding window) cross-validation.** The training set grows forward in time; the test window always sits strictly after it. No shuffling, no random splits.
 
```
Fold 1: train [2024-01 → 2025-06]  test [2025-07 → 2025-09]
Fold 2: train [2024-01 → 2025-09]  test [2025-10 → 2025-12]
Fold 3: train [2024-01 → 2025-12]  test [2026-01 → 2026-03]
...
```
 
The final period is held out entirely and touched once, at the end, for the reported result.
 
### Metrics
 
| Metric | Why |
|---|---|
| **WAPE** | Primary. Robust where MAPE is distorted by scale differences between subsystems |
| MAPE | Secondary, for comparability with published literature |
| **Skill score vs. seasonal naive** | The number that actually matters: relative improvement over the baseline |
 
Absolute error is reported, but the headline is always relative to Tier 0.
 
---
 
## Results
 
> Placeholder. Filled after the first complete run.
 
### Southeast / Midwest subsystem, 7-day horizon
 
| Model | WAPE | MAPE | vs. seasonal naive |
|---|---|---|---|
| Seasonal naive | — | — | baseline |
| Previous-day persistence | — | — | — |
| SARIMA | — | — | — |
| XGBoost | — | — | — |
 
### Format comparison
 
| | CSV | Parquet |
|---|---|---|
| Size on disk | — | — |
| Ingestion time | — | — |
| Typed on read | no | yes |
 
---
 
## Limitations
 
Stated up front, not buried.
 
- **No weather data.** Temperature is the dominant driver of peak demand, and it is absent here. This caps achievable accuracy, and any model in this repo would improve materially with a temperature series. Adding one is the highest-value next step.
- **Three years of daily data** (~1,100 points per subsystem) is thin. Annual seasonality is observed roughly three times, which is too few to estimate it confidently. This is why no deep learning model appears here — the data does not support one.
- **Point forecasts only.** The models return a single number, not a distribution. For dispatch or trading decisions, uncertainty bounds (P10/P50/P90) matter more than the central estimate. See Roadmap.
- **Daily peak, not the load curve.** Predicting the daily maximum is a different, easier problem than predicting the full hourly shape.
- **Revisions.** ONS notes that published values may be updated after release. This analysis uses data as retrieved on `YYYY-MM-DD` and does not model revision behaviour.
- **MMGD is itself estimated.** The distributed generation component is modelled by ONS from forecast weather, so part of the target carries upstream model error.
---
 
## Roadmap
 
- [ ] Add temperature data from INMET as an exogenous variable
- [ ] Probabilistic forecasting: quantile regression for P10 / P50 / P90
- [ ] Extend from daily peak to the full hourly load curve
- [ ] Per-subsystem model comparison (the South behaves differently from the Northeast)
- [ ] Holiday calendar as an explicit feature
---
 
## Running it
 
```bash
git clone https://github.com/dogeraldi05/ons-forecast.git
cd ons-forecast
 
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
 
# download raw files from ONS
python src/ingest.py --years 2024 2025 2026
 
# build the warehouse
cd dbt && dbt build && cd ..
 
# baselines, models, evaluation
python src/validate.py
python src/evaluate.py
```
 
To run the CSV variant:
 
```bash
git checkout csv-pipeline
python src/ingest.py --years 2024 2025 2026
```
 
---
 
## Author
 
**Douglas Geraldi** — Electrical Engineer (UNICAMP, B.Sc. and M.Sc.), 15 years in Brazil's power sector.
 
The domain framing in this repo — why daily peak matters, what drives it, which errors are expensive — comes from that background rather than from the dataset documentation.
 
[LinkedIn](https://linkedin.com/in/douglas-geraldi) · [GitHub](https://github.com/dogeraldi05)
 
---
 
## License
 
Code: MIT. Data: open data published by ONS, subject to its own terms.
 
