"""Objective 02 on the city road network: compare forecasting models for road speed.

Data: per-road speed observations from the calibration simulations of the city
(incident + normal runs). The last calibration seed is held out completely as the
test set (an independent simulation the models never saw).

Models (all predict each road's speed `horizon_steps` decision steps ahead):
  * Persistence      — "no change": future speed = current speed
  * Moving average   — mean of the last `history_steps` observations
  * XGBoost          — gradient-boosted trees on recent history + road features
  * LSTM             — recurrent network on the recent speed sequence + road features
Errors are reported in km/h and as % of each road's free-flow speed.
"""
import os

import numpy as np
import pandas as pd

from .edge_forecaster import XGBEdgeForecaster, build_training_set, edge_features


def _kmh(ratio, vfree_ms):
    return ratio * vfree_ms * 3.6


def _lstm_xy(X, K):
    seq = X[:, :K][:, ::-1]                              # oldest -> newest
    extra = X[:, K:]                                     # density, trend, time, road features
    Xs = np.concatenate([seq[..., None], np.repeat(extra[:, None, :], K, axis=1)], axis=2)
    return Xs.astype(np.float32)


def _train_lstm(Xtr, ytr, K, log):
    import tensorflow as tf
    tf.random.set_seed(0)
    Xs = _lstm_xy(Xtr, K)
    inp = tf.keras.Input(shape=Xs.shape[1:])
    h = tf.keras.layers.LSTM(48)(inp)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    out = tf.keras.layers.Dense(1)(h)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss="mae")
    m.fit(Xs, ytr, epochs=8, batch_size=512, validation_split=0.1, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
    return m


def compare_models(cfg, recs, labels, inc, net, out_dir, log=print):
    s = cfg["sim"]
    K, H = s["history_steps"], s["horizon_steps"]
    seeds = [lab[1] for lab in labels]
    test_seed = seeds[-1]
    tr = [r for r, lab in zip(recs, labels) if lab[1] != test_seed]
    te = [r for r, lab in zip(recs, labels) if lab[1] == test_seed]
    if not tr:                       # only one calibration seed: fall back to last record as test
        tr, te = recs[:-1], recs[-1:]
    Xtr, ytr = build_training_set(tr, K, H)
    Xte, yte = build_training_set(te, K, H)
    vfree = Xte[:, K + 4] * 13.89                       # static[0] = v_free / 13.89

    preds = {"Persistence (no change)": Xte[:, 0],
             f"Moving average (last {K})": Xte[:, :K].mean(axis=1)}
    xgb = XGBEdgeForecaster().fit(Xtr, ytr)
    preds["XGBoost"] = xgb.predict_matrix(Xte)
    lstm = None
    try:
        lstm = _train_lstm(Xtr, ytr, K, log)
        preds["LSTM"] = np.clip(lstm.predict(_lstm_xy(Xte, K), batch_size=4096, verbose=0)[:, 0], 0.02, 1.05)
    except Exception as e:  # TensorFlow missing or failed: keep the other models
        log(f"[forecast] LSTM skipped ({type(e).__name__}: {e})")

    rows = []
    for name, p in preds.items():
        err = _kmh(p, vfree) - _kmh(yte, vfree)
        rows.append(dict(model=name, MAE_kmh=float(np.mean(np.abs(err))),
                         RMSE_kmh=float(np.sqrt(np.mean(err ** 2))),
                         MAE_pct_of_free_speed=float(100 * np.mean(np.abs(p - yte))),
                         test_samples=int(len(yte))))
    met = pd.DataFrame(rows).sort_values("MAE_kmh")
    met["horizon_min"] = H * s["step_s"] / 60
    met.to_csv(os.path.join(out_dir, "city_forecast_metrics.csv"), index=False)
    log("[forecast] road-speed forecasting on held-out simulation:\n" + met.round(2).to_string(index=False))

    # actual vs predicted on an incident road in the held-out blockage run
    try:
        idx = [lab[0] for lab in labels].index("blockage")
        test_idx = [i for i, lab in enumerate(labels) if lab[1] == test_seed and lab[0] == "blockage"]
        rec = recs[test_idx[0] if test_idx else idx]
        e = inc[0]
        j = rec["edges"].index(e)
        r, d, st = rec["r"][:, [j]], rec["d"][:, [j]], rec["static"][[j]]
        v = st[0, 0] * 13.89
        out = []
        for k in range(K - 1, r.shape[0] - H):
            t_frac = (k + 1) * rec["step_s"] / rec["sim_end"]
            Xk = edge_features(r[k - K + 1:k + 1], d[k - K + 1:k + 1], t_frac, st)
            row = dict(t_min=(k + 1 + H) * rec["step_s"] / 60, actual=_kmh(r[k + H, 0], v),
                       XGBoost=_kmh(float(xgb.predict_matrix(Xk)[0]), v),
                       **{"Persistence (no change)": _kmh(r[k, 0], v)})
            if lstm is not None:
                row["LSTM"] = _kmh(float(lstm.predict(_lstm_xy(Xk, K), verbose=0)[0, 0]), v)
            out.append(row)
        smp = pd.DataFrame(out)
        smp["road"] = net.getEdge(e).getName() or e
        smp.to_csv(os.path.join(out_dir, "city_forecast_sample.csv"), index=False)
    except Exception as ex:
        log(f"[forecast] sample series skipped ({ex})")
    return met
