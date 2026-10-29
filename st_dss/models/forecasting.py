"""Objective 02 — XGBoost vs LSTM traffic-speed forecasting (MAE / RMSE / MAPE).

Reference baselines: Historical Average (sensor x weekday/weekend x 5-min slot)
and Last Value (persistence). One XGBoost model per horizon; one multi-output
LSTM for all horizons. Evaluation is masked: steps whose true value is missing
in the raw data are excluded.
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd

SCALE = 70.0


def load_split(processed_dir, split):
    files = glob.glob(os.path.join(processed_dir, f"split={split}", "*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values(["sensor", "ts"])


def feature_cols(n_lags):
    return [f"lag_{k}" for k in range(n_lags)] + ["tod", "dow", "weekend"]


def metrics(y, p):
    m = ~np.isnan(y)
    y, p = y[m], p[m]
    if len(y) == 0:
        return dict(MAE=np.nan, RMSE=np.nan, MAPE=np.nan, n=0)
    err = p - y
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=float(np.sqrt(np.mean(err ** 2))),
                MAPE=float(100 * np.mean(np.abs(err) / np.maximum(y, 1.0))), n=int(len(y)))


class HistoricalAverage:
    name = "HistoricalAverage"

    def fit(self, train):
        tr = train.copy()
        tr["slot"] = (tr["tod"] * 288).round().astype(int)
        self.table = tr.groupby(["sensor", "weekend", "slot"])["lag_0"].mean()
        self.fallback = tr.groupby("sensor")["lag_0"].mean()
        return self

    def predict(self, df, h):
        slot = ((df["tod"] * 288).round().astype(int) + h) % 288
        wk = df["weekend"]  # (ignores midnight weekend change — negligible)
        key = pd.MultiIndex.from_arrays([df["sensor"], wk, slot])
        p = self.table.reindex(key).to_numpy()
        fb = self.fallback.reindex(df["sensor"]).to_numpy()
        return np.where(np.isnan(p), fb, p)


class LastValue:
    name = "LastValue"

    def fit(self, train):
        return self

    def predict(self, df, h):
        return df["lag_0"].to_numpy()


class XGBForecaster:
    name = "XGBoost"

    def __init__(self, params, n_lags, horizons):
        self.params, self.n_lags, self.horizons = dict(params), n_lags, horizons
        self.max_rows = self.params.pop("max_train_rows", None)
        self.models = {}

    def fit(self, train, val, log=print):
        import xgboost as xgb
        X_cols = feature_cols(self.n_lags)
        for h in self.horizons:
            tr = train.dropna(subset=[f"y_{h}"])
            if self.max_rows and len(tr) > self.max_rows:
                tr = tr.sample(self.max_rows, random_state=0)
            va = val.dropna(subset=[f"y_{h}"])
            m = xgb.XGBRegressor(tree_method="hist", early_stopping_rounds=20,
                                 n_jobs=os.cpu_count(), **self.params)
            t0 = time.time()
            m.fit(tr[X_cols], tr[f"y_{h}"], eval_set=[(va[X_cols], va[f"y_{h}"])], verbose=False)
            log(f"[xgb] h={h} trained on {len(tr):,} rows in {time.time() - t0:.1f}s "
                f"(best iteration {m.best_iteration})")
            self.models[h] = m
        return self

    def predict(self, df, h):
        return self.models[h].predict(df[feature_cols(self.n_lags)])

    def save(self, d):
        for h, m in self.models.items():
            m.save_model(os.path.join(d, f"xgb_h{h}.json"))


class LSTMForecaster:
    name = "LSTM"

    def __init__(self, params, n_lags, horizons):
        self.p, self.n_lags, self.horizons = params, n_lags, horizons

    def _xy(self, df):
        lags = df[[f"lag_{k}" for k in reversed(range(self.n_lags))]].to_numpy(np.float32) / SCALE
        cal = np.stack([np.sin(2 * np.pi * df["tod"]), np.cos(2 * np.pi * df["tod"]),
                        df["weekend"]], axis=1).astype(np.float32)
        X = np.concatenate([lags[..., None], np.repeat(cal[:, None, :], self.n_lags, 1)], axis=2)
        Y = df[[f"y_{h}" for h in self.horizons]].to_numpy(np.float32) / SCALE
        return X, Y

    def fit(self, train, val, log=print):
        import tensorflow as tf
        tf.random.set_seed(0)
        tr = train.dropna(subset=[f"y_{h}" for h in self.horizons])
        if len(tr) > self.p["max_train_rows"]:
            tr = tr.sample(self.p["max_train_rows"], random_state=0)
        va = val.dropna(subset=[f"y_{h}" for h in self.horizons])
        va = va.sample(min(len(va), 200000), random_state=0)
        Xtr, Ytr = self._xy(tr)
        Xva, Yva = self._xy(va)
        inp = tf.keras.Input(shape=Xtr.shape[1:])
        x = tf.keras.layers.LSTM(self.p["units"], return_sequences=True)(inp)
        x = tf.keras.layers.LSTM(self.p["units"])(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        out = tf.keras.layers.Dense(len(self.horizons))(x)
        self.model = tf.keras.Model(inp, out)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
        t0 = time.time()
        self.model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=self.p["epochs"],
                       batch_size=self.p["batch_size"], verbose=0,
                       callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
        log(f"[lstm] trained on {len(tr):,} rows in {time.time() - t0:.1f}s")
        return self

    def predict_all(self, df):
        X, _ = self._xy(df)
        return self.model.predict(X, batch_size=4096, verbose=0) * SCALE

    def predict(self, df, h):
        return self.predict_all(df)[:, self.horizons.index(h)]

    def save(self, d):
        self.model.save(os.path.join(d, "lstm.keras"))


def run_forecasting(cfg, processed_dir, out_dir, source, log=print):
    dcfg, mcfg = cfg["data"], cfg["models"]
    L, H = dcfg["n_lags"], dcfg["horizons"]
    train, val, test = (load_split(processed_dir, s) for s in ["train", "val", "test"])
    log(f"[models] train={len(train):,} val={len(val):,} test={len(test):,}")

    models = [HistoricalAverage().fit(train), LastValue().fit(train),
              XGBForecaster(mcfg["xgboost"], L, H).fit(train, val, log),
              LSTMForecaster(mcfg["lstm"], L, H).fit(train, val, log)]

    rows, preds = [], {}
    for m in models:
        if isinstance(m, LSTMForecaster):
            allp = m.predict_all(test)
        for i, h in enumerate(H):
            p = allp[:, i] if isinstance(m, LSTMForecaster) else m.predict(test, h)
            preds[(m.name, h)] = p
            r = metrics(test[f"y_{h}"].to_numpy(), np.asarray(p, dtype=float))
            rows.append(dict(model=m.name, horizon_min=5 * h, **r))
        if hasattr(m, "save"):
            m.save(os.path.join(out_dir, "models"))

    res = pd.DataFrame(rows)
    res["data_source"] = source
    res.to_csv(os.path.join(out_dir, "models", "forecast_metrics.csv"), index=False)
    log("\n" + res.pivot(index="model", columns="horizon_min", values="MAE").round(3).to_string())

    # sample predictions for the dashboard (one sensor, first 3 test days)
    s0 = test["sensor"].iloc[0]
    msk = (test["sensor"] == s0).to_numpy()
    sample = test.loc[msk, ["ts", "sensor"] + [f"y_{h}" for h in H]].copy()
    for (name, h), p in preds.items():
        sample[f"{name}_h{h}"] = np.asarray(p)[msk]
    sample.head(864).to_csv(os.path.join(out_dir, "models", "forecast_sample.csv"), index=False)
    with open(os.path.join(out_dir, "models", "forecast_meta.json"), "w") as f:
        json.dump(dict(data_source=source, n_lags=L, horizons=H), f)
    return res
