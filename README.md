# ST-DSS — AI-Driven Smart Traffic Decision Support System (RIAPR)

M.E. Big Data Analytics — Project Phase-I implementation
Proposed mechanism: **Recovery-Impact-Aware Prediction Refreshing (RIAPR)**

```
METR-LA ─► Spark/PySpark preprocessing ─► XGBoost / LSTM forecasting (MAE, RMSE)
                                                     │
SUMO network + incidents ─► edge observations ─► RIAPR controller ──(refresh?)──► edge forecaster
                                   ▲                                                   │
                                   └──── recovery-oriented routing ◄───── forecast ────┘
                                                     │
                                  metrics + Streamlit dashboard
```

## 1. Setup

```bash
pip install -r requirements.txt        # needs Python 3.10+ and Java 17+ (for Spark)
```
SUMO comes from the `eclipse-sumo` pip package. If you installed SUMO yourself, set `SUMO_HOME`.

**Dataset:** download `metr-la.h5` (DCRNN release, Li et al. 2018) and put it at `data/metr-la.h5`.
Without it the code uses a *synthetic* METR-LA-shaped dataset so you can test the pipeline —
the dashboard shows a warning. **Never report synthetic numbers as results.**

## 2. Run

```bash
python run_pipeline.py --quick --stage all     # ~10 min smoke test (small settings)

python run_pipeline.py --stage data            # Objective 01 — Spark preprocessing
python run_pipeline.py --stage forecast        # Objective 02 — HA / LastValue / XGBoost / LSTM
python run_pipeline.py --stage sim             # Objectives 03–05 — RIAPR vs 3 baselines in SUMO
python run_pipeline.py --stage sim --sweep 0.2 0.3 0.4 0.5 0.6   # tune theta on validation seeds

streamlit run dashboard/app.py                 # prototype dashboard
```
All settings (thresholds, seeds, scenarios, horizons) are in `config.yaml`.

## 2b. Real cities — Western Tamil Nadu (OpenStreetMap)

```bash
python run_pipeline.py --stage sim --city erode        # also: coimbatore, tiruppur, salem, namakkal
python run_pipeline.py --stage sim --city erode --quick
```
* Downloads the drivable roads of a 2.6 km × 2.6 km box around the city centre (needs internet;
  cached in `data/osm/<city>.osm`). If the download fails, the error message prints the exact
  box to export manually from openstreetmap.org/export.
* `netconvert` with **left-hand traffic**, signal guessing, joined junctions, street names.
* Indian vehicle mix (`city_sim.vehicle_mix: india`): 55 % two-wheelers, 28 % cars, 12 % autos, 5 % buses.
* Results go to `outputs_<city>/`; choose the network in the dashboard sidebar.
* If a city gridlocks (many teleports), raise `city_sim.demand_period` (e.g. 1.0–1.5).

## 3. Code map

| Objective (review slide 9) | File |
|---|---|
| 01 Spark preprocessing (cleaning, alignment, windowing, chronological split) | `st_dss/data/spark_preprocess.py`, `st_dss/data/metr_la.py` |
| 02 XGBoost vs LSTM on MAE / RMSE (+ HA, LastValue references) | `st_dss/models/forecasting.py` |
| 03 RIAPR mechanism + baseline refresh policies | `st_dss/riapr/controller.py` |
| 04 Recovery-oriented routing + incidents in SUMO | `st_dss/sim/runner.py`, `st_dss/sim/network.py`, `st_dss/sim/edge_forecaster.py` |
| 05 Comparison with three baselines | `st_dss/sim/experiment.py` |
| Real city networks (OSM) | `st_dss/sim/osm_city.py` |
| Route advisor (best route on current forecast) | `st_dss/route_advisor.py` |
| Streamlit prototype | `dashboard/app.py` |

## 4. RIAPR in one paragraph

Every decision step (90 s = one signal cycle) the controller computes
* **A** forecast age / horizon,
* **D** flow-weighted deviation between observed and forecast edge speed,
* **I** expected routing impact = relative change of the load-weighted route cost if routing used
  the latest observation instead of the forecast (congested edges weighted ×(1+κ)).

`score = w_age·A + w_dev·D + w_impact·I` → refresh if `score ≥ θ` (bounded by `min_age`, `max_age`).
`I` re-uses edge costs that routing already needs, so deciding is much cheaper than running the
forecaster. Baselines: **B1** fixed-interval + travel-time routing, **B2** fixed-interval +
recovery-oriented routing, **B3** age-threshold (= horizon) + recovery-oriented routing.
`NoGuidance` is included only as a reference.

Recovery-oriented edge cost = `tt_fc · (1 + α(1 − v_fc/v_free)) · (1 + γ(load/capacity)²)` — the
load term discourages sending every rerouted vehicle to the same alternative road.

## 5. Metrics (PICO "P")

| Metric | Definition in code |
|---|---|
| Network recovery time (min) | onset → 3-step-smoothed mean network speed ≥ `recovery_fraction` × pre-incident (10 min) level, sustained `recovery_sustain_steps`; censored at `sim_end` (`recovered=0`) |
| Total vehicle delay (veh-min) | Σ SUMO `timeLoss` (travel time − free-flow time), unfinished vehicles included |
| Refresh count | forecaster runs per simulation |
| Computational latency (ms) | mean wall-clock of one decision cycle incl. any refresh (same machine for all) |
| MAE / RMSE | mph, masked on missing ground truth |

## 6. Outputs (`outputs/`)

* `models/forecast_metrics.csv` — MAE/RMSE/MAPE per model × horizon
* `sim/results_runs.csv` — every SUMO run (no filtering)
* `sim/results_summary.csv`, `sim/riapr_vs_baselines.csv` — means and % differences (paired by seed)
* `sim/timeseries.csv` — per-step speed, refresh events, RIAPR components
* `figures/sim_comparison.png`, `figures/sim_speed_timeseries.png`
* `pipeline.log`

## 7. Honest limitations (say these in the review)

* City networks cover a 2.6 km centre box, not the whole city; demand is synthetic (randomTrips), not counted traffic.
* Sub-lane behaviour of two-wheelers (lane sharing) is not modelled.
* METR-LA sensors are not mapped to SUMO edges; the in-simulation forecaster (XGBoost) is trained on
  calibration simulations with **different seeds** from the test runs.
* θ and weights must be tuned on validation seeds only (`--sweep`), then fixed for test seeds.
* Results may or may not favour RIAPR — report all runs, including the ones where it loses.
