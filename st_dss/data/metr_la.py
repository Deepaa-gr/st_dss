"""Load METR-LA (207 sensors x 34,272 five-minute steps, speed in mph, 0 = missing).

The official file is `metr-la.h5` from the DCRNN release (Li et al., ICLR 2018):
a pandas HDF5 store whose single DataFrame has a DatetimeIndex and one column
per sensor. If it is not present we generate a synthetic dataset with the same
shape and basic properties (daily/weekly cycles, random incidents, ~8 % zeros)
so the code can be exercised. Synthetic numbers must never be reported as results.
"""
import os
import numpy as np
import pandas as pd

from ..config import root_path


def load_metr_la(cfg, log=print):
    dcfg = cfg["data"]
    path = root_path(dcfg["metr_la_path"])
    if os.path.exists(path):
        df = pd.read_hdf(path)
        source = "METR-LA"
        log(f"[data] loaded {path}: {df.shape}")
    elif dcfg.get("allow_synthetic", True):
        df = synthetic_metr_la(n_sensors=dcfg["max_sensors"], n_days=dcfg["max_days"] or 119,
                               seed=cfg["seed"])
        source = "SYNTHETIC"
        log(f"[data] {path} not found -> using SYNTHETIC METR-LA-shaped data {df.shape}")
    else:
        raise FileNotFoundError(path)

    df = df.iloc[:, : dcfg["max_sensors"]]
    if dcfg.get("max_days"):
        df = df.iloc[: int(dcfg["max_days"]) * 288]
    df.columns = [str(c) for c in df.columns]
    return df, source


def synthetic_metr_la(n_sensors=207, n_days=119, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-03-01", periods=n_days * 288, freq="5min")
    t = np.arange(len(idx))
    tod = (t % 288) / 288.0
    dow = idx.dayofweek.values
    weekday = (dow < 5).astype(float)

    free = rng.uniform(55, 70, n_sensors)
    am = rng.uniform(0.15, 0.55, n_sensors)
    pm = rng.uniform(0.15, 0.55, n_sensors)
    am_c = rng.normal(8 / 24, 0.02, n_sensors)
    pm_c = rng.normal(17.5 / 24, 0.02, n_sensors)

    dip = (am[None] * np.exp(-((tod[:, None] - am_c[None]) ** 2) / (2 * 0.04 ** 2))
           + pm[None] * np.exp(-((tod[:, None] - pm_c[None]) ** 2) / (2 * 0.05 ** 2)))
    dip *= (0.35 + 0.65 * weekday)[:, None]
    speed = free[None] * (1 - dip)

    # AR(1) noise
    noise = np.zeros_like(speed)
    e = rng.normal(0, 2.0, speed.shape)
    for i in range(1, len(t)):
        noise[i] = 0.85 * noise[i - 1] + e[i]
    speed += noise

    # incidents: sudden drops that propagate to neighbouring sensors and decay
    n_inc = int(n_days * n_sensors * 0.02)
    for _ in range(n_inc):
        s = rng.integers(n_sensors)
        st = rng.integers(len(t) - 30)
        dur = rng.integers(4, 24)
        sev = rng.uniform(0.3, 0.7)
        prof = np.concatenate([np.full(dur, sev), sev * np.exp(-np.arange(12) / 4)])
        for nb, w in [(s, 1.0), ((s + 1) % n_sensors, 0.5), ((s - 1) % n_sensors, 0.5)]:
            seg = slice(st, st + len(prof))
            speed[seg, nb] *= 1 - w * prof[: len(speed[seg, nb])]

    speed = np.clip(speed, 3, 75)
    # missing values (0) as blocks and isolated points, as in METR-LA
    mask = rng.random(speed.shape) < 0.03
    for _ in range(int(n_sensors * n_days * 0.05)):
        s = rng.integers(n_sensors)
        st = rng.integers(len(t) - 50)
        mask[st: st + rng.integers(5, 50), s] = True
    speed[mask] = 0.0
    cols = [str(773869 + 13 * i) for i in range(n_sensors)]
    return pd.DataFrame(speed.astype(np.float32), index=idx, columns=cols)


def to_long_parquet(df, path):
    long = df.stack().rename("speed").reset_index()
    long.columns = ["ts", "sensor", "speed"]
    long["speed"] = long["speed"].astype("float32")
    long.to_parquet(path, index=False)
    return path
