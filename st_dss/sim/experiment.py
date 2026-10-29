"""Objective 05 — compare RIAPR with the three baselines in SUMO.

Strategies (C in PICO):
  NoGuidance          reference only (no forecast, no rerouting)
  B1 Fixed+TT         fixed-interval refresh + travel-time routing
  B2 Fixed+Recovery   fixed-interval refresh + recovery-oriented routing
  B3 Age+Recovery     age-threshold refresh + recovery-oriented routing
  RIAPR+Recovery      proposed (I in PICO)

Protocol
  1. calibration seeds  -> choose incident edges (auto) + train the edge forecaster
  2. validation seeds   -> optional theta sweep for RIAPR (--sweep)
  3. test seeds         -> final comparison; every run is reported (no filtering)
"""
import json
import os
import tempfile
import pickle
import time
import multiprocessing as mp


def Pool(n):
    # 'spawn' avoids OpenMP (XGBoost) deadlocks in forked workers
    return mp.get_context("spawn").Pool(n)

import numpy as np
import pandas as pd

from ..config import out_path
from ..riapr.controller import make_policy
from .edge_forecaster import PersistenceForecaster, XGBEdgeForecaster, build_training_set
import sumolib

from .network import build_network, build_trips, interior_edges, street_name
from .runner import SimRun

STRATEGIES = {
    "NoGuidance": ("none", None),
    "B1 Fixed+TT": ("travel_time", "fixed"),
    "B2 Fixed+Recovery": ("recovery", "fixed"),
    "B3 Age+Recovery": ("recovery", "age"),
    "RIAPR+Recovery": ("recovery", "riapr"),
}


def _job(args):
    cfg, net, trips, scenario, inc, seed, strategy, fc_path, theta = args
    routing, pkind = STRATEGIES[strategy]
    if theta is not None:
        cfg = json.loads(json.dumps(cfg))
        cfg["refresh"]["riapr"]["theta"] = theta
    fc = XGBEdgeForecaster(fc_path) if fc_path else PersistenceForecaster()
    pol = make_policy(pkind, cfg) if pkind else None
    t0 = time.time()
    r = SimRun(cfg, net, trips, scenario, inc, seed, routing=routing, policy=pol,
               forecaster=fc, out_prefix=os.path.join(tempfile.gettempdir(), f"stdss_{scenario}_{seed}_{os.getpid()}")).run()
    r.update(scenario=scenario, seed=seed, strategy=strategy, theta=theta, wall_s=time.time() - t0)
    return r


def _pool_map(jobs, workers, log):
    out = []
    with Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(_job, jobs), 1):
            log(f"  [{i}/{len(jobs)}] {r['scenario']:<12} seed={r['seed']} {r['strategy']:<18}"
                f" delay={r['total_delay_veh_min']:.0f} rec={r['recovery_time_min']:.1f}min"
                f" refresh={r['refresh_count']} ({r['wall_s']:.0f}s)")
            out.append(r)
    return out


def prepare(cfg, log=print):
    sim_dir = out_path(cfg, "sim")
    s = cfg["sim"]
    net = build_network(cfg, sim_dir, log)
    seeds = s["calibration_seeds"] + s["validation_seeds"] + s["test_seeds"]
    trips = {sd: build_trips(cfg, net, sim_dir, sd) for sd in seeds}

    # incident edges: most-used interior edges in a no-incident calibration run
    inc = s["incident_edges"]
    if inc == "auto":
        sd = s["calibration_seeds"][0]
        r = SimRun(cfg, net, trips[sd], "none", [], sd, routing="none", record=True).run()
        use = r["_record"]["d"].sum(0)
        edges = r["_record"]["edges"]
        inner = set(interior_edges(net))
        snet = sumolib.net.readNet(net)
        ranked = [edges[i] for i in np.argsort(-use) if edges[i] in inner]
        multi = [e for e in ranked if snet.getEdge(e).getLaneNumber() >= 2]
        inc = (multi + [e for e in ranked if e not in multi])[: s["n_incident_edges"]]
    snet = sumolib.net.readNet(net)
    log(f"[sim] network={os.path.basename(net)} edges={len(snet.getEdges())} "
        f"incident edges={[street_name(snet, e) for e in inc]}")
    return net, trips, inc


def train_edge_forecaster(cfg, net, trips, inc, workers, log=print):
    """Calibration runs (persistence forecaster + fixed refresh + TT routing, all
    scenarios) -> XGBoost edge forecaster. Uses calibration seeds only."""
    s = cfg["sim"]
    path = out_path(cfg, "sim", "edge_forecaster_xgb.json")
    jobs = []
    for sd in s["calibration_seeds"]:
        for sc in s["scenarios"] + ["none"]:
            jobs.append((sc, sd))
    recs = []
    with Pool(workers) as p:
        recs = p.map(_calib_job, [(cfg, net, trips[sd], sc, inc, sd) for sc, sd in jobs])
    K, H = s["history_steps"], s["horizon_steps"]
    try:   # Objective 02 on this road network: Persistence / Moving avg / XGBoost / LSTM
        from .city_forecast import compare_models
        compare_models(cfg, recs, jobs, inc, sumolib.net.readNet(net), out_path(cfg, "sim"), log)
    except Exception as e:
        log(f"[forecast] model comparison skipped ({type(e).__name__}: {e})")
    X, y = build_training_set(recs, K, H)
    n_val = len(X) // 5
    fc = XGBEdgeForecaster().fit(X[:-n_val], y[:-n_val])
    p = fc.model.predict(X[-n_val:])
    mae = float(np.mean(np.abs(p - y[-n_val:])))
    mae_persist = float(np.mean(np.abs(X[-n_val:, 0] - y[-n_val:])))
    fc = XGBEdgeForecaster().fit(X, y)
    fc.save(path)
    info = dict(samples=int(len(X)), holdout_MAE_ratio=mae, persistence_MAE_ratio=mae_persist)
    log(f"[sim] edge forecaster trained: {info}")
    with open(out_path(cfg, "sim", "edge_forecaster_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return path


def city_forecast_stage(cfg, workers=None, log=print):
    """Only the forecasting part for a road network: calibration runs + model comparison."""
    workers = workers or max(1, os.cpu_count())
    net, trips, inc = prepare(cfg, log)
    return train_edge_forecaster(cfg, net, trips, inc, workers, log)


def _calib_job(a):
    cfg, net, trips, sc, inc, sd = a
    routing = "travel_time"
    return SimRun(cfg, net, trips, sc if sc != "none" else "blockage", inc if sc != "none" else [],
                  sd, routing=routing, policy=make_policy("fixed", cfg),
                  forecaster=PersistenceForecaster(), record=True).run()["_record"]


def sweep_theta(cfg, net, trips, inc, fc_path, thetas, workers, log=print):
    s = cfg["sim"]
    jobs = [(cfg, net, trips[sd], sc, inc, sd, "RIAPR+Recovery", fc_path, th)
            for th in thetas for sd in s["validation_seeds"] for sc in s["scenarios"]]
    jobs += [(cfg, net, trips[sd], sc, inc, sd, "B2 Fixed+Recovery", fc_path, None)
             for sd in s["validation_seeds"] for sc in s["scenarios"]]
    res = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in _pool_map(jobs, workers, log)])
    res.to_csv(out_path(cfg, "sim", "theta_sweep.csv"), index=False)
    summ = res.groupby(["strategy", "theta"], dropna=False)[
        ["total_delay_veh_min", "recovery_time_min", "refresh_count"]].mean()
    log("\n[sweep] validation seeds (mean over scenarios and seeds)\n" + summ.round(2).to_string())
    return summ


def run_experiment(cfg, workers=None, sweep=None, log=print):
    workers = workers or max(1, os.cpu_count())
    net, trips, inc = prepare(cfg, log)
    fc_path = train_edge_forecaster(cfg, net, trips, inc, workers, log)
    if sweep:
        sweep_theta(cfg, net, trips, inc, fc_path, sweep, workers, log)

    s = cfg["sim"]
    jobs = [(cfg, net, trips[sd], sc, inc, sd, st, fc_path, None)
            for sc in s["scenarios"] for sd in s["test_seeds"] for st in STRATEGIES]
    log(f"[sim] {len(jobs)} test runs on {workers} worker(s)")
    results = _pool_map(jobs, workers, log)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    df.to_csv(out_path(cfg, "sim", "results_runs.csv"), index=False)
    ts = pd.concat([pd.DataFrame(r["_timeseries"]).assign(scenario=r["scenario"], seed=r["seed"],
                                                          strategy=r["strategy"]) for r in results])
    ts.to_csv(out_path(cfg, "sim", "timeseries.csv"), index=False)
    snap = {f"{r['scenario']}|{r['seed']}|{r['strategy']}": r["_final_fc"] for r in results}
    peak = {f"{r['scenario']}|{r['seed']}|{r['strategy']}": r.get("_peak_obs", {}) for r in results}
    with open(out_path(cfg, "sim", "final_forecasts.pkl"), "wb") as f:
        pickle.dump(dict(net=net, incident_edges=inc, snapshots=snap, peak_snapshots=peak), f)

    summary, cmp = summarize(df)
    summary.to_csv(out_path(cfg, "sim", "results_summary.csv"))
    cmp.to_csv(out_path(cfg, "sim", "riapr_vs_baselines.csv"), index=False)
    log("\n" + summary.round(2).to_string())
    log("\n" + cmp.round(1).to_string())
    plot_results(cfg, df, ts)
    return summary


def make_report(cfg, log=print):
    """Rebuild summary tables and figures from saved run files (no simulation)."""
    df = pd.read_csv(out_path(cfg, "sim", "results_runs.csv"))
    ts = pd.read_csv(out_path(cfg, "sim", "timeseries.csv"))
    summary, cmp = summarize(df)
    summary.to_csv(out_path(cfg, "sim", "results_summary.csv"))
    cmp.to_csv(out_path(cfg, "sim", "riapr_vs_baselines.csv"), index=False)
    log("\n" + summary[["recovery_time_min", "total_delay_veh_min", "refresh_count",
                         "decision_latency_ms"]].round(2).to_string())
    log("\n" + cmp.round(1).to_string())
    plot_results(cfg, df, ts)
    log("[report] tables -> outputs/sim, figures -> outputs/figures")


def summarize(df):
    """Mean/std per scenario x strategy, plus RIAPR-vs-baseline % differences."""
    m = ["recovery_time_min", "total_delay_veh_min", "refresh_count", "decision_latency_ms",
         "forecast_time_s", "reroutes", "mean_travel_time_s", "recovered", "teleports"]
    g = df.groupby(["scenario", "strategy"])[m]
    out = g.mean().join(g.std()[["recovery_time_min", "total_delay_veh_min"]], rsuffix="_std")
    # % difference vs each baseline (positive = RIAPR lower)
    rows = []
    for sc, d in df.groupby("scenario"):
        ri = d[d.strategy == "RIAPR+Recovery"].set_index("seed").sort_index()
        for b in ["B1 Fixed+TT", "B2 Fixed+Recovery", "B3 Age+Recovery"]:
            bd = d[d.strategy == b].set_index("seed").reindex(ri.index)
            for col in ["recovery_time_min", "total_delay_veh_min", "refresh_count"]:
                base = bd[col].mean()
                rows.append(dict(scenario=sc, baseline=b, metric=col,
                                 riapr_vs_baseline_pct=100 * (base - ri[col].mean()) / base if base else np.nan,
                                 paired_seeds_riapr_better=int((ri[col].to_numpy() < bd[col].to_numpy()).sum()),
                                 n_seeds=len(ri)))
    return out, pd.DataFrame(rows)


def plot_results(cfg, df, ts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_path(cfg, "figures")
    order = list(STRATEGIES)
    colors = ["#9aa0a6", "#f29900", "#1a73e8", "#9334e6", "#188038"]
    metrics = [("recovery_time_min", "Recovery time (min)"),
               ("total_delay_veh_min", "Total vehicle delay (veh-min)"),
               ("refresh_count", "Prediction refresh count"),
               ("decision_latency_ms", "Decision latency (ms / cycle)")]
    scen = cfg["sim"]["scenarios"]
    fig, axes = plt.subplots(len(metrics), len(scen), figsize=(4.2 * len(scen), 3.2 * len(metrics)))
    axes = np.atleast_2d(axes)
    for j, sc in enumerate(scen):
        d = df[df.scenario == sc]
        for i, (m, lab) in enumerate(metrics):
            ax = axes[i, j]
            g = d.groupby("strategy")[m]
            mu = g.mean().reindex(order)
            sd = g.std().reindex(order).fillna(0)
            ax.bar(range(len(order)), mu.values, yerr=sd.values, color=colors, capsize=3)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([o.split()[0] for o in order], fontsize=8)
            ax.set_title(f"{sc}: {lab}" if i == 0 else lab, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "sim_comparison.png"), dpi=150)
    plt.close(fig)

    on = cfg["sim"]["incident_start"]
    off = on + cfg["sim"]["incident_duration"]
    fig, axes = plt.subplots(1, len(scen), figsize=(5 * len(scen), 3.4), sharey=True)
    for ax, sc in zip(np.atleast_1d(axes), scen):
        d = ts[ts.scenario == sc]
        for st, c in zip(order, colors):
            g = d[d.strategy == st].groupby("t")["net_speed"].mean()
            ax.plot(g.index / 60, g.values, label=st, color=c, lw=1.6)
        ax.axvspan(on / 60, off / 60, color="red", alpha=0.08, label="incident")
        ax.set_title(sc)
        ax.set_xlabel("time (min)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("mean network speed (m/s)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "sim_speed_timeseries.png"), dpi=150)
    plt.close(fig)
