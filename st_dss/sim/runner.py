"""Objective 04 — one SUMO run with incidents, forecasting, refresh policy and routing.

Routing modes
  none        : no guidance (reference only; vehicles keep their departure route)
  travel_time : edge cost = length / forecast speed
  recovery    : recovery-oriented cost = travel time x (1 + alpha (1 - r_fc))
                x (1 + gamma (load / capacity)^2). The load term counts vehicles
                already assigned to the edge, so rerouting does not simply move
                the queue to the next-best road (congestion redistribution).

Metrics per run
  recovery_time_min  : onset -> mean network speed back to >= frac x pre-incident
                       level (sustained); censored at sim_end if never reached
  total_delay_veh_min: sum of SUMO timeLoss (travel time - free-flow time), incl.
                       vehicles unfinished at the end
  refresh_count, decision_latency_ms (mean wall-clock per decision cycle incl.
  any refresh), forecast_time_s (total model time), reroutes, mean_travel_time_s
"""
import os
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import sumolib
import traci
import traci.constants as tc

from .network import binary, boundary_edges
from ..riapr.controller import make_policy

INCIDENT_SPEED = 0.3
ROAD_CLASSES = ["passenger", "motorcycle", "taxi", "bus", "truck", "delivery", "bicycle"]   # m/s on a blocked edge (near-closure, vehicles can still clear)


class SimRun:
    def __init__(self, cfg, net_path, trips_path, scenario, incident_edges, seed,
                 routing="recovery", policy=None, forecaster=None, out_prefix=None,
                 record=False):
        self.cfg, self.s, self.rc = cfg, cfg["sim"], cfg["routing"]
        self.net_path, self.trips_path = net_path, trips_path
        self.scenario, self.inc_edges, self.seed = scenario, list(incident_edges), seed
        self.routing, self.policy, self.forecaster = routing, policy, forecaster
        self.out_prefix, self.record = out_prefix, record

    # ------------------------------------------------------------------ costs
    def cost(self, v, load):
        tt = self.length / np.maximum(v, 0.5)
        if self.routing == "travel_time":
            return tt
        r = np.clip(v / self.v_free, 0, 1)
        return (tt * (1 + self.rc["recovery_alpha"] * (1 - r))
                * (1 + self.rc["recovery_gamma"] * (load / self.cap) ** 2))

    # ------------------------------------------------------------- incidents
    def _incident_on(self):
        for e in self.inc_edges:
            for i in range(self.n_lanes[e]):
                lane = f"{e}_{i}"
                if self.scenario == "blockage":
                    traci.lane.setMaxSpeed(lane, INCIDENT_SPEED)
                elif self.scenario == "lane_closure":
                    if i == 0 and self.n_lanes[e] > 1:       # close one lane (keep >= 1 open)
                        self._orig_disallowed[lane] = traci.lane.getDisallowed(lane)
                        traci.lane.setDisallowed(lane, ROAD_CLASSES)
                    else:
                        traci.lane.setMaxSpeed(lane, 0.5 * self.lane_speed[lane])

    def _incident_off(self):
        for e in self.inc_edges:
            for i in range(self.n_lanes[e]):
                lane = f"{e}_{i}"
                traci.lane.setMaxSpeed(lane, self.lane_speed[lane])
                if lane in self._orig_disallowed:
                    traci.lane.setDisallowed(lane, list(self._orig_disallowed.pop(lane)))

    def _surge_routes(self, rng):
        src, snk = boundary_edges(self.net)
        routes = []
        for inc in self.inc_edges:
            for _ in range(60):
                a, b = rng.choice(src), rng.choice(snk)
                r1 = traci.simulation.findRoute(a, inc).edges
                r2 = traci.simulation.findRoute(inc, b).edges
                if r1 and r2 and len(r2) > 1:
                    routes.append(list(r1) + list(r2[1:]))
                if len(routes) >= 20 * len(self.inc_edges):
                    break
        for i, r in enumerate(routes):
            traci.route.add(f"surge_r{i}", r)
        return len(routes)

    # ------------------------------------------------------------------- run
    def run(self):
        s = self.s
        rng = np.random.default_rng(self.seed)
        self.net = sumolib.net.readNet(self.net_path)
        tag = self.out_prefix or os.path.join(tempfile.gettempdir(), f"stdss_{os.getpid()}")
        tripinfo = f"{tag}_tripinfo.xml"
        cmd = [binary("sumo"), "-n", self.net_path, "-r", self.trips_path,
               "--tripinfo-output", tripinfo, "--tripinfo-output.write-unfinished",
               "--no-step-log", "--no-warnings", "--seed", str(self.seed),
               "--time-to-teleport", str(s["time_to_teleport"]),
               "--device.rerouting.probability", "0", "--end", str(s["sim_end"] + 1800),
               "--ignore-route-errors", "true"]   # skip a vehicle with no valid route (OSM access rules)
        vt = os.path.join(os.path.dirname(self.trips_path), "vtypes.add.xml")
        self.vtype = None
        if os.path.exists(vt) and self.cfg["sim"].get("vehicle_mix") == "india":
            cmd += ["-a", vt]
            self.vtype = "india_mix"
        self._orig_disallowed = {}
        traci.start(cmd)
        try:
            raw = self._loop(rng)
        finally:
            traci.close()          # tripinfo is flushed on close
        return self._metrics(tripinfo, raw)

    def _loop(self, rng):
        s, step = self.s, self.s["step_s"]
        edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
        E = len(edges)
        eidx = {e: i for i, e in enumerate(edges)}
        net_e = [self.net.getEdge(e) for e in edges]
        self.length = np.array([e.getLength() for e in net_e])
        self.v_free = np.array([e.getSpeed() for e in net_e])
        lanes = np.array([e.getLaneNumber() for e in net_e])
        self.cap = np.maximum(lanes * self.length / 7.5, 1.0)
        self.n_lanes = {e.getID(): e.getLaneNumber() for e in net_e}
        self.lane_speed = {l.getID(): l.getSpeed() for e in net_e for l in e.getLanes()}
        static = np.column_stack([self.v_free / 13.89, self.length / 200.0, lanes])
        for e in edges:
            traci.edge.subscribe(e, [tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_VEHICLE_NUMBER])

        guided = self.routing != "none"
        self.probe = make_policy("riapr", self.cfg)
        K, H = s["history_steps"], s["horizon_steps"]
        v_fc = self.v_free.copy()
        last_refresh, refreshes, refreshes_obs = None, 0, 0
        acc_sn, acc_n = np.zeros(E), np.zeros(E)
        hist_r, hist_d, ts_rows, decisions = [], [], [], []
        last_rr, n_reroutes, teleports = {}, 0, 0
        lat, fc_time = [], 0.0
        n_surge_routes = self._surge_routes(rng) if self.scenario == "demand_surge" else 0
        surge_id, t = 0, 0
        on, off = s["incident_start"], s["incident_start"] + s["incident_duration"]
        peak_obs = None
        surge_rate = s.get("surge_rate_per_min", 30) / 60.0

        while True:
            traci.simulationStep()
            t += 1
            if t == on and self.scenario != "demand_surge":
                self._incident_on()
            if t == off and self.scenario != "demand_surge":
                self._incident_off()
            if self.scenario == "demand_surge" and on <= t < off and n_surge_routes:
                for _ in range(rng.poisson(surge_rate)):
                    traci.vehicle.add(f"surge{surge_id}", f"surge_r{rng.integers(n_surge_routes)}",
                                      typeID=self.vtype or "DEFAULT_VEHTYPE",
                                      departLane="best", departSpeed="max")
                    surge_id += 1
            teleports += traci.simulation.getStartingTeleportNumber()
            res = traci.edge.getAllSubscriptionResults()
            for e, d in res.items():
                i = eidx.get(e)
                if i is not None:
                    n = d[tc.LAST_STEP_VEHICLE_NUMBER]
                    acc_sn[i] += d[tc.LAST_STEP_MEAN_SPEED] * n
                    acc_n[i] += n

            if t % step == 0:
                v_obs = np.where(acc_n > 0, acc_sn / np.maximum(acc_n, 1e-9), self.v_free)
                dens = (acc_n / step) / self.cap
                net_speed = acc_sn.sum() / acc_n.sum() if acc_n.sum() > 0 else np.nan
                hist_r.append(v_obs / self.v_free)
                hist_d.append(dens)
                row = dict(t=t, net_speed=net_speed, vehicles=acc_n.sum() / step,
                           refreshed=0, score=np.nan, dev=np.nan, impact=np.nan, age=np.nan)

                if guided and len(hist_r) >= K:
                    load, remaining = self._load(edges, eidx)
                    t0 = time.perf_counter()
                    if last_refresh is None:
                        dec_refresh, dec = True, None
                    else:
                        cost_fc = self.cost(v_fc, load)
                        cost_obs = self.cost(v_obs, load)
                        ctx = dict(v_obs=v_obs, v_fc=v_fc, v_free=self.v_free, load=load,
                                   cost_fc=cost_fc, cost_obs=cost_obs)
                        dec = self.policy.decide(t, last_refresh, ctx)
                        dec_refresh = dec.refresh
                        dev, imp = self.probe.components(ctx)       # logged for every policy
                        row.update(dev=dev, impact=imp, age=t - last_refresh,
                                   score=self.probe.w_age * (t - last_refresh) / self.probe.horizon_s
                                   + self.probe.w_dev * dev + self.probe.w_impact * imp)
                    if dec_refresh:
                        tf0 = time.perf_counter()
                        r_fc = self.forecaster.predict(np.array(hist_r[-K:]), np.array(hist_d[-K:]),
                                                       t / s["sim_end"], static)
                        fc_time += time.perf_counter() - tf0
                        v_fc = r_fc * self.v_free
                        last_refresh, refreshes = t, refreshes + 1
                        refreshes_obs += t <= s["sim_end"]
                        row["refreshed"] = 1
                    lat.append(time.perf_counter() - t0)
                    cost = self.cost(v_fc, load)
                    for e, c in zip(edges, cost):
                        traci.edge.adaptTraveltime(e, float(c))
                    n_reroutes += self._reroute(v_fc / self.v_free, remaining, t, last_rr)

                ts_rows.append(row)
                # snapshot of the actual congestion ~60 % into the incident (for the map view)
                if peak_obs is None and t >= on + 0.6 * s["incident_duration"]:
                    peak_obs = dict(zip(edges, (v_obs / self.v_free).round(3).tolist()))
                    peak_obs["_t"] = t
                acc_sn[:], acc_n[:] = 0, 0
                if t >= s["sim_end"] and traci.simulation.getMinExpectedNumber() == 0:
                    break
            if t >= s["sim_end"] + 1800:
                break

        out = dict(ts_rows=ts_rows, refreshes=refreshes, refreshes_obs=refreshes_obs, lat=lat, fc_time=fc_time,
                   n_reroutes=n_reroutes, teleports=teleports)
        if self.record:
            out["_record"] = dict(r=np.array(hist_r), d=np.array(hist_d), static=static,
                                  step_s=step, sim_end=s["sim_end"], edges=edges)
        out["_timeseries"] = ts_rows
        out["_final_fc"] = dict(zip(edges, (v_fc / self.v_free).round(3).tolist()))
        out["_peak_obs"] = peak_obs or {}
        return out

    def _load(self, edges, eidx):
        load = np.zeros(len(edges))
        remaining = {}
        for v in traci.vehicle.getIDList():
            route = traci.vehicle.getRoute(v)
            idx = max(traci.vehicle.getRouteIndex(v), 0)
            rem = [eidx[e] for e in route[idx:] if e in eidx]
            remaining[v] = rem
            load[rem] += 1
        return load, remaining

    def _reroute(self, r_fc, remaining, t, last_rr):
        """Vehicle selection: only vehicles whose remaining route contains an edge
        forecast to be congested (speed ratio < trigger), most affected first."""
        cand = []
        for v, rem in remaining.items():
            if len(rem) < 2 or t - last_rr.get(v, -1e9) < self.rc["min_reroute_gap_s"]:
                continue
            m = r_fc[rem[1:]].min()
            if m < self.rc["reroute_trigger"]:
                cand.append((m, v))
        cand.sort()
        n = 0
        for _, v in cand[: self.rc["max_reroutes_per_step"]]:
            try:
                traci.vehicle.rerouteTraveltime(v, False)
                last_rr[v] = t
                n += 1
            except traci.TraCIException:
                pass
        return n

    def _metrics(self, tripinfo, raw):
        s = self.s
        ts_rows, lat = raw["ts_rows"], raw["lat"]
        delays, durs, unfinished = [], [], 0
        for _, el in ET.iterparse(tripinfo):
            if el.tag == "tripinfo":
                delays.append(float(el.get("timeLoss")))
                durs.append(float(el.get("duration")))
                unfinished += el.get("vaporized", "") != ""
                el.clear()
        rec = recovery_time(ts_rows, s["incident_start"], s["sim_end"], s["step_s"],
                            s["recovery_fraction"], s["recovery_sustain_steps"])
        out = dict(total_delay_veh_min=sum(delays) / 60.0, mean_travel_time_s=float(np.mean(durs)),
                   vehicles=len(delays), unfinished=unfinished, teleports=raw["teleports"],
                   refresh_count=raw["refreshes_obs"],          # within the 0..sim_end window
                   refresh_count_total=raw["refreshes"],        # incl. drain phase
                   decision_latency_ms=1000 * float(np.mean(lat)) if lat else 0.0,
                   forecast_time_s=raw["fc_time"], reroutes=raw["n_reroutes"], **rec)
        for k in ["_record", "_timeseries", "_final_fc", "_peak_obs"]:
            if k in raw:
                out[k] = raw[k]
        try:
            os.remove(tripinfo)
        except OSError:
            pass
        return out


def recovery_time(ts_rows, onset, sim_end, step, frac, sustain):
    t = np.array([r["t"] for r in ts_rows])
    v = np.array([r["net_speed"] for r in ts_rows], dtype=float)
    v = np.convolve(np.nan_to_num(v, nan=np.nanmean(v)), np.ones(3) / 3, mode="same")
    pre_mask = (t > onset - 600) & (t <= onset)
    pre = float(np.mean(v[pre_mask]))
    thr = frac * pre
    idx = np.where((t > onset) & (t <= sim_end))[0]
    dropped, min_speed = False, float(v[idx].min()) if len(idx) else np.nan
    for j, i in enumerate(idx):
        if v[i] < thr:
            dropped = True
            continue
        if dropped and all(v[k] >= thr for k in idx[j: j + sustain]) and len(idx[j: j + sustain]) == sustain:
            return dict(recovery_time_min=(t[i] - onset) / 60.0, recovered=1,
                        pre_incident_speed=pre, min_speed=min_speed)
    if not dropped:
        return dict(recovery_time_min=0.0, recovered=1, pre_incident_speed=pre, min_speed=min_speed)
    return dict(recovery_time_min=(sim_end - onset) / 60.0, recovered=0,
                pre_incident_speed=pre, min_speed=min_speed)
