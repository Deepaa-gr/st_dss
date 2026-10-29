#!/usr/bin/env python3
"""ST-DSS end-to-end pipeline.

  python run_pipeline.py --stage data        # Spark preprocessing (Objective 01)
  python run_pipeline.py --stage forecast    # XGBoost vs LSTM      (Objective 02)
  python run_pipeline.py --stage sim         # RIAPR + SUMO + baselines (Objectives 03-05)
  python run_pipeline.py --stage report      # rebuild tables + graphs from saved sim results
  python run_pipeline.py --stage all
  python run_pipeline.py --stage sim --sweep 0.2 0.3 0.4 0.5   # tune theta on validation seeds
  python run_pipeline.py --stage sim --city erode   # real OSM roads (coimbatore/erode/tiruppur/salem/namakkal)
  python run_pipeline.py --stage forecast --city erode   # XGBoost vs LSTM on the city's roads only
  python run_pipeline.py --quick ...         # small settings for a fast smoke test
"""
import argparse
import json
import os
import sys
import time

from st_dss.config import load_config, out_path


def log_to(path):
    f = open(path, "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def quick(cfg):
    cfg["data"].update(max_sensors=30, max_days=21)
    cfg["models"]["xgboost"].update(n_estimators=120, max_train_rows=200000)
    cfg["models"]["lstm"].update(epochs=3, max_train_rows=100000)
    cfg["sim"].update(calibration_seeds=[101, 102], validation_seeds=[201], test_seeds=[1, 2])
    return cfg


def city_setup(cfg, city):
    """Real-city run: own output folder (outputs_<city>/), Indian vehicle mix,
    demand scaled for a ~2.6 km city centre. Forecasting results stay in outputs/."""
    from st_dss.config import ROOT
    cfg["base_output_dir"] = cfg["output_dir"]
    cfg["sim"].update(city=city, **cfg.get("city_sim", {}))
    cfg["output_dir"] = os.path.join(ROOT, f"outputs_{city}")
    for sub in ["sim", "figures"]:
        os.makedirs(os.path.join(cfg["output_dir"], sub), exist_ok=True)
    return cfg


def stage_data(cfg, log):
    from st_dss.data.metr_la import load_metr_la, to_long_parquet
    from st_dss.data.spark_preprocess import preprocess
    df, source = load_metr_la(cfg, log)
    long_path = out_path(cfg, "data", "metr_la_long.parquet")
    to_long_parquet(df, long_path)
    d = cfg["data"]
    rep = preprocess(long_path, out_path(cfg, "data", "processed"), d["n_lags"], d["horizons"],
                     d["split"], log)
    rep["data_source"] = source
    with open(out_path(cfg, "data", "preprocess_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)


def stage_forecast(cfg, log):
    from st_dss.models.forecasting import run_forecasting
    with open(out_path(cfg, "data", "preprocess_report.json")) as f:
        source = json.load(f)["data_source"]
    run_forecasting(cfg, out_path(cfg, "data", "processed"), cfg["output_dir"], source, log)


def stage_sim(cfg, log, workers, sweep):
    from st_dss.sim.experiment import run_experiment
    run_experiment(cfg, workers=workers, sweep=sweep, log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "forecast", "sim", "report", "all"], default="all")
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--sweep", type=float, nargs="*", default=None, help="theta values to sweep")
    ap.add_argument("--city", default=None,
                    help="real OSM road network: coimbatore | erode | tiruppur | salem | namakkal")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.quick:
        cfg = quick(cfg)
    if a.city:
        cfg = city_setup(cfg, a.city.lower())
    log = log_to(out_path(cfg, "pipeline.log"))
    log(f"=== ST-DSS stage={a.stage} quick={a.quick} ===")
    # data + forecasting always use the main outputs/ folder (they do not depend on the city)
    base = dict(cfg, output_dir=cfg.get("base_output_dir", cfg["output_dir"]))
    if a.stage in ("data", "all"):
        stage_data(base, log)
    if a.stage == "forecast" and a.city:
        # forecasting on the city's own road network (Objective 02 for that city)
        from st_dss.sim.experiment import city_forecast_stage
        city_forecast_stage(cfg, a.workers, log)
    elif a.stage in ("forecast", "all"):
        stage_forecast(base, log)
    if a.stage in ("sim", "all"):
        stage_sim(cfg, log, a.workers, a.sweep)
    if a.stage == "report":
        from st_dss.sim.experiment import make_report
        make_report(cfg, log)
    log("done.")


if __name__ == "__main__":
    sys.exit(main())
