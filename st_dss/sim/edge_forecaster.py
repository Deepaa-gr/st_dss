"""Online edge-speed forecaster used inside SUMO.

METR-LA sensors do not map onto SUMO edges, so (as stated in the review) the
integrated routing experiments use SUMO-generated observations: an XGBoost
model is trained on calibration simulations (different seeds from the test
runs) to predict each edge's speed ratio H steps ahead from its recent history.
A "refresh" = one run of this model over all edges.
"""
import os

import numpy as np


def edge_features(hist_r, hist_d, t_frac, static):
    """hist_r, hist_d: (K, E) arrays of the last K speed ratios / densities."""
    K = hist_r.shape[0]
    cols = [hist_r[K - 1 - j] for j in range(K)]
    cols += [hist_d[-1], hist_d[-2] if K > 1 else hist_d[-1],
             hist_r[-1] - hist_r[0], np.full(hist_r.shape[1], t_frac)]
    return np.column_stack(cols + [static])


class PersistenceForecaster:
    name = "persistence"

    def predict(self, hist_r, hist_d, t_frac, static):
        return hist_r[-1].copy()


class XGBEdgeForecaster:
    name = "xgboost"

    def __init__(self, model_path=None):
        import xgboost as xgb
        self.model = xgb.XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.08,
                                      subsample=0.8, tree_method="hist", n_jobs=1)
        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, hist_r, hist_d, t_frac, static):
        p = self.model.predict(edge_features(hist_r, hist_d, t_frac, static))
        return np.clip(p, 0.02, 1.05)

    def predict_matrix(self, X):
        return np.clip(self.model.predict(X), 0.02, 1.05)

    def save(self, path):
        self.model.save_model(path)


def build_training_set(records, K, H):
    """records: list of dicts with 'r' (T,E), 'd' (T,E), 'static' (E,s), 'T_end'."""
    Xs, ys = [], []
    for rec in records:
        r, d, st = rec["r"], rec["d"], rec["static"]
        T = r.shape[0]
        for k in range(K - 1, T - H):
            t_frac = (k + 1) * rec["step_s"] / rec["sim_end"]
            Xs.append(edge_features(r[k - K + 1:k + 1], d[k - K + 1:k + 1], t_frac, st))
            ys.append(r[k + H])
    return np.vstack(Xs), np.concatenate(ys)
