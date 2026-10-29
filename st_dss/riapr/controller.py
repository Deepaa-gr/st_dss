"""Objective 03 — Recovery-Impact-Aware Prediction Refreshing (RIAPR).

At every decision step the controller decides: REFRESH the forecast, or KEEP
the current (ageing) one. Three inputs, as defined in the review:

  A  forecast age            a / H                         (H = forecast horizon)
  D  forecast-observation    flow-weighted mean |v_obs - v_fc| / v_free
     deviation
  I  expected routing        relative change in the route-weighted edge cost if
     impact                  routing used the latest observation instead of the
                             forecast:  sum_e w_e |c_e(obs) - c_e(fc)| / sum_e w_e c_e(fc)

  w_e = assigned load on edge e  x  (1 + kappa * congested_e)   (recovery relevance:
        edges already congested — i.e. the ones that matter for recovery — count more)

  score = w_age*A + w_dev*D + w_impact*I ;  refresh if score >= theta
  (always refresh if age >= max_age, never if age < min_age)

I is cheap: it re-uses the edge costs routing already needs and does NOT run
the forecasting model — the point is to skip model runs whose result would not
change routing much.

The baselines use the same interface:
  FixedInterval  — refresh every T seconds
  AgeThreshold   — refresh when the forecast age reaches the forecast horizon
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Decision:
    refresh: bool
    score: float = np.nan
    age: float = 0.0
    dev: float = np.nan
    impact: float = np.nan
    reason: str = ""


class RefreshPolicy:
    name = "base"

    def decide(self, t, last_refresh_t, ctx) -> Decision:
        raise NotImplementedError


class FixedIntervalPolicy(RefreshPolicy):
    def __init__(self, interval_s):
        self.T = interval_s
        self.name = f"fixed_{interval_s}s"

    def decide(self, t, last_refresh_t, ctx):
        age = t - last_refresh_t
        return Decision(age >= self.T, age=age, reason="interval")


class AgeThresholdPolicy(RefreshPolicy):
    def __init__(self, age_s):
        self.A = age_s
        self.name = f"age_{age_s}s"

    def decide(self, t, last_refresh_t, ctx):
        age = t - last_refresh_t
        return Decision(age >= self.A, age=age, reason="age")


@dataclass
class RIAPRPolicy(RefreshPolicy):
    horizon_s: float
    w_age: float = 0.3
    w_dev: float = 1.0
    w_impact: float = 1.0
    congestion_boost: float = 2.0
    theta: float = 0.35
    min_age_s: float = 60
    max_age_s: float = 600
    name: str = field(default="riapr")

    def components(self, ctx):
        """ctx: dict of numpy arrays over edges:
        v_obs, v_fc, v_free (m/s), load (vehicles assigned), cost_fc, cost_obs."""
        v_obs, v_fc, v_free = ctx["v_obs"], ctx["v_fc"], ctx["v_free"]
        congested = (v_obs / v_free) < 0.5
        w = (ctx["load"] + 1e-3) * (1.0 + self.congestion_boost * congested)
        dev = float(np.sum(w * np.abs(v_obs - v_fc) / v_free) / np.sum(w))
        c_fc, c_obs = ctx["cost_fc"], ctx["cost_obs"]
        impact = float(np.sum(w * np.abs(c_obs - c_fc)) / max(np.sum(w * c_fc), 1e-9))
        return dev, min(impact, 2.0)

    def decide(self, t, last_refresh_t, ctx):
        age = t - last_refresh_t
        if age < self.min_age_s:
            return Decision(False, age=age, reason="min_age")
        dev, imp = self.components(ctx)
        score = self.w_age * age / self.horizon_s + self.w_dev * dev + self.w_impact * imp
        if age >= self.max_age_s:
            return Decision(True, score, age, dev, imp, "max_age")
        return Decision(score >= self.theta, score, age, dev, imp, "score")


def make_policy(kind, cfg):
    r = cfg["refresh"]
    H = cfg["sim"]["horizon_steps"] * cfg["sim"]["step_s"]
    if kind == "fixed":
        return FixedIntervalPolicy(r["fixed_interval_s"])
    if kind == "age":
        return AgeThresholdPolicy(r["age_threshold_s"])
    if kind == "riapr":
        return RIAPRPolicy(horizon_s=H, **r["riapr"])
    raise ValueError(kind)
