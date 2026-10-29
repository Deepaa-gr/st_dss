"""ST-DSS Streamlit prototype dashboard.   Run:  streamlit run dashboard/app.py"""
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from st_dss.config import load_config, out_path  # noqa: E402

st.set_page_config(page_title="ST-DSS Dashboard", layout="wide")
cfg = load_config()
OUT = cfg["output_dir"]


CITY_LABELS = {"coimbatore": "Coimbatore", "erode": "Erode", "tiruppur": "Tiruppur",
               "salem": "Salem", "namakkal": "Namakkal"}
networks = {"Test grid (6x6)": OUT}
for c, lab in CITY_LABELS.items():
    d = os.path.join(ROOT, f"outputs_{c}")
    if os.path.exists(os.path.join(d, "sim", "results_runs.csv")):
        networks[f"{lab} (OpenStreetMap)"] = d
with st.sidebar:
    st.header("Road network")
    net_choice = st.radio("Show simulation results for", list(networks),
                          index=len(networks) - 1)
    st.caption("Run a city with:\n`python run_pipeline.py --stage sim --city erode`")
SIM_OUT = networks[net_choice]


def read_csv(*p):
    # forecasting results live in outputs/; fall back to the selected city folder
    for base in (OUT, SIM_OUT):
        f = os.path.join(base, *p)
        if os.path.exists(f):
            return pd.read_csv(f)
    return None


def resolve_net(p):
    """Network path saved by the simulation may be from another computer
    (e.g. C:\\Users\\...). Fall back to the copy next to the results."""
    if os.path.exists(p):
        return p
    import ntpath
    local = os.path.join(SIM_OUT, "sim", ntpath.basename(p))
    return local if os.path.exists(local) else p


def read_sim(*p):
    f = os.path.join(SIM_OUT, "sim", *p)
    return pd.read_csv(f) if os.path.exists(f) else None


st.title("ST-DSS — Smart Traffic Decision Support System")
st.caption(f"Recovery-Impact-Aware Prediction Refreshing (RIAPR) · research prototype · SUMO simulation · network: **{net_choice}**")

meta_f = os.path.join(OUT, "data", "preprocess_report.json")
if os.path.exists(meta_f):
    meta = json.load(open(meta_f))
    if meta.get("data_source") == "SYNTHETIC" and SIM_OUT == OUT:
        st.warning("Forecasting results below use SYNTHETIC data (metr-la.h5 not found). "
                   "They only show that the pipeline works — do not report them.")

tab5, tab1, tab2, tab3, tab4 = st.tabs(["Route advisor", "Forecasting", "Simulation results",
                                        "RIAPR decisions", "Route map"])

# ------------------------------------------------------------------ forecasting
with tab1:
    IS_CITY = SIM_OUT != OUT
    if IS_CITY:
        st.subheader(f"Road-speed forecasting — {net_choice}")
        cm = read_sim("city_forecast_metrics.csv")
        if cm is None:
            st.info("Forecasting results for this city not found yet. In the project folder run:\n\n"
                    f"`python run_pipeline.py --stage forecast --city {net_choice.split()[0].lower()} --quick`\n\n"
                    "then refresh this page.")
        else:
            hz = cm["horizon_min"].iloc[0]
            st.caption(f"Each model predicts every road's speed **{hz:.0f} minutes ahead**. Trained on simulated "
                       "observations of this road network; tested on a separate simulation run the models never saw. "
                       "Lower error is better.")
            best = cm.iloc[0]
            base_row = cm[cm.model.str.startswith("Persistence")].iloc[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("Best model", best.model)
            k2.metric("Its error (MAE)", f"{best.MAE_kmh:.1f} km/h")
            k3.metric("vs 'no change' guess", f"{100 * (1 - best.MAE_kmh / base_row.MAE_kmh):.0f}% lower error")
            c1, c2 = st.columns([1.1, 1])
            c1.dataframe(cm[["model", "MAE_kmh", "RMSE_kmh", "MAE_pct_of_free_speed"]].rename(columns={
                "model": "Model", "MAE_kmh": "MAE (km/h)", "RMSE_kmh": "RMSE (km/h)",
                "MAE_pct_of_free_speed": "MAE (% of free speed)"}).round(2), hide_index=True, width="stretch")
            fig, ax = plt.subplots(figsize=(5, 2.8))
            colors = {"XGBoost": "#188038", "LSTM": "#1a73e8"}
            ax.barh(cm.model, cm.MAE_kmh, color=[colors.get(m_, "#9aa0a6") for m_ in cm.model])
            ax.invert_yaxis()
            ax.set_xlabel("MAE (km/h) — lower is better")
            ax.grid(axis="x", alpha=0.3)
            c2.pyplot(fig)
            smp = read_sim("city_forecast_sample.csv")
            if smp is not None:
                st.markdown(f"**Actual vs predicted speed on an incident road: {smp.road.iloc[0]}** "
                            "(held-out blockage run; pink = incident period)")
                fig, ax = plt.subplots(figsize=(10, 3.2))
                ax.plot(smp.t_min, smp.actual, color="black", lw=2.2, label="actual")
                for col, c in [("XGBoost", "#188038"), ("LSTM", "#1a73e8"), ("Persistence (no change)", "#9aa0a6")]:
                    if col in smp:
                        ax.plot(smp.t_min, smp[col], color=c, lw=1.5, ls="--" if col.startswith("Pers") else "-",
                                label=col)
                on = cfg["sim"]["incident_start"] / 60
                ax.axvspan(on, on + cfg["sim"]["incident_duration"] / 60, color="red", alpha=0.08)
                ax.set_xlabel("time (min)")
                ax.set_ylabel("speed (km/h)")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8, ncol=4)
                st.pyplot(fig)
                st.caption("Prediction is made "
                           f"{hz:.0f} min earlier. Note how forecasts lag when the incident starts: "
                           "this is exactly when RIAPR's deviation (D) rises and triggers a refresh.")
    else:
        st.subheader("Model comparison on METR-LA (public traffic dataset)")
        st.caption("Choose a city in the sidebar to see forecasting on its own road network.")
        m = read_csv("models", "forecast_metrics.csv")
        if m is None:
            st.info("Forecasting results not found yet. In the project folder run:\n\n"
                    "`python run_pipeline.py --quick --stage data`\n\n"
                    "`python run_pipeline.py --quick --stage forecast`\n\n"
                    "(without --city), then refresh this page.")
        else:
            c1, c2 = st.columns([1, 1])
            metric = c1.selectbox("Metric", ["MAE", "RMSE", "MAPE"])
            c1.dataframe(m.pivot(index="model", columns="horizon_min", values=metric).round(3))
            fig, ax = plt.subplots(figsize=(5, 3))
            for name, g in m.groupby("model"):
                ax.plot(g.horizon_min, g[metric], marker="o", label=name)
            ax.set_xlabel("horizon (min)")
            ax.set_ylabel(metric + " (mph)" if metric != "MAPE" else "MAPE (%)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            c2.pyplot(fig)
            s = read_csv("models", "forecast_sample.csv")
            if s is not None:
                h = st.select_slider("Horizon (steps of 5 min)", options=cfg["data"]["horizons"])
                s["ts"] = pd.to_datetime(s["ts"])
                cols = [c for c in s.columns if c.endswith(f"_h{h}") and "LastValue" not in c]
                st.line_chart(s.set_index("ts")[[f"y_{h}"] + cols].rename(columns={f"y_{h}": "actual"}))


# ------------------------------------------------------------------ simulation
runs = read_sim("results_runs.csv")
ts = read_sim("timeseries.csv")
with tab2:
    if runs is None:
        st.info("Run: python run_pipeline.py --stage sim")
    else:
        sc = st.selectbox("Incident scenario", sorted(runs.scenario.unique()))
        d = runs[runs.scenario == sc]
        cols = ["recovery_time_min", "total_delay_veh_min", "refresh_count", "decision_latency_ms",
                "reroutes", "mean_travel_time_s", "recovered"]
        st.dataframe(d.groupby("strategy")[cols].mean().round(2), width="stretch")
        cmp = read_sim("riapr_vs_baselines.csv")
        if cmp is not None:
            st.markdown("**RIAPR vs baselines** (positive % = RIAPR lower; counted over paired seeds)")
            st.dataframe(cmp[cmp.scenario == sc].round(1), width="stretch")
        fig_f = os.path.join(SIM_OUT, "figures", "sim_comparison.png")
        if os.path.exists(fig_f):
            st.image(fig_f)
        if ts is not None:
            dd = ts[ts.scenario == sc].groupby(["t", "strategy"])["net_speed"].mean().unstack()
            dd.index = dd.index / 60
            st.markdown("Mean network speed (m/s) vs time (min)")
            st.line_chart(dd)

# ------------------------------------------------------------------ RIAPR decisions
with tab3:
    if ts is None:
        st.info("Run the simulation stage first.")
    else:
        c1, c2 = st.columns(2)
        sc = c1.selectbox("Scenario ", sorted(ts.scenario.unique()))
        seed = c2.selectbox("Seed", sorted(ts.seed.unique()))
        d = ts[(ts.scenario == sc) & (ts.seed == seed)]
        theta = cfg["refresh"]["riapr"]["theta"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        r = d[d.strategy == "RIAPR+Recovery"]
        axes[0].plot(r.t / 60, r.score, label="RIAPR score")
        axes[0].plot(r.t / 60, r.dev, label="deviation D", alpha=0.6)
        axes[0].plot(r.t / 60, r.impact, label="routing impact I", alpha=0.6)
        axes[0].axhline(theta, color="k", ls="--", lw=1, label=f"theta={theta}")
        axes[0].legend(fontsize=7, ncol=4)
        for i, stg in enumerate(["B2 Fixed+Recovery", "B3 Age+Recovery", "RIAPR+Recovery"]):
            e = d[(d.strategy == stg) & (d.refreshed == 1)]
            axes[1].scatter(e.t / 60, np.full(len(e), i), marker="|", s=200)
        axes[1].set_yticks([0, 1, 2])
        axes[1].set_yticklabels(["Fixed", "Age", "RIAPR"])
        axes[1].set_xlabel("time (min)")
        axes[1].set_title("forecast refresh events", fontsize=9)
        on = cfg["sim"]["incident_start"] / 60
        for ax in axes:
            ax.axvspan(on, on + cfg["sim"]["incident_duration"] / 60, color="red", alpha=0.08)
            ax.grid(alpha=0.3)
        st.pyplot(fig)

# ------------------------------------------------------------------ route map
with tab4:
    pk = os.path.join(SIM_OUT, "sim", "final_forecasts.pkl")
    if not os.path.exists(pk):
        st.info("Run the simulation stage first.")
    else:
        import sumolib
        from st_dss.route_advisor import lonlat_path
        snap = pickle.load(open(pk, "rb"))
        net = sumolib.net.readNet(resolve_net(snap["net"]))
        keys = sorted(snap["snapshots"])
        scen = sorted({k.split("|")[0] for k in keys})
        strat = sorted({k.split("|")[2] for k in keys})
        seeds = sorted({k.split("|")[1] for k in keys})
        c1, c2, c3, c4 = st.columns(4)
        sc = c1.selectbox("Incident", scen, key="rm_sc")
        sg = c2.selectbox("Strategy", strat, index=strat.index("RIAPR+Recovery") if "RIAPR+Recovery" in strat else 0,
                          key="rm_st")
        sd = c3.selectbox("Seed", seeds, key="rm_sd")
        peak_all = snap.get("peak_snapshots", {})
        k = f"{sc}|{sd}|{sg}"
        has_peak = bool(peak_all.get(k))
        when = c4.radio("Time", ["During incident", "End of run"] if has_peak else ["End of run"], key="rm_when")
        if when == "During incident":
            ratios = {e: v for e, v in peak_all[k].items() if not e.startswith("_")}
            t_min = peak_all[k].get("_t", 0) / 60
            what = f"observed speed at minute {t_min:.0f} (incident active)"
        else:
            ratios = snap["snapshots"][k]
            what = "forecast speed at the end of the run (incident cleared)"
        if not has_peak:
            st.caption("Tip: re-run the simulation stage to also store the congestion *during* the incident.")
        inc = set(snap["incident_edges"]) if sc != "demand_surge" else set(snap["incident_edges"])

        # ---- summary numbers
        rows = []
        for e in net.getEdges():
            r = min(float(ratios.get(e.getID(), 1.0)), 1.0)
            rows.append(dict(id=e.getID(), road=e.getName() or "unnamed street", ratio=r,
                             kmh=r * e.getSpeed() * 3.6, free=e.getSpeed() * 3.6, length=e.getLength()))
        df_e = pd.DataFrame(rows)
        jam = df_e[df_e.ratio < 0.4]
        k1, k2, k3 = st.columns(3)
        k1.metric("Roads shown", f"{len(df_e):,}")
        k2.metric("Heavily congested (< 40 % of normal speed)", f"{len(jam):,}",
                  f"{100 * jam.length.sum() / df_e.length.sum():.1f} % of road length", delta_color="off")
        k3.metric("Average speed (length-weighted)",
                  f"{np.average(df_e.kmh, weights=df_e.length):.0f} km/h")

        cmap = plt.get_cmap("RdYlGn")

        def col(r):
            return [int(255 * v) for v in cmap(min(r, 1.0))[:3]] + [230]

        geo = bool(net.hasGeoProj()) and lonlat_path(net, df_e.id.iloc[0]) is not None
        if geo:
            import pydeck as pdk
            paths = []
            for row in df_e.itertuples():
                p = lonlat_path(net, row.id)
                if not p:
                    continue
                is_inc = row.id in inc
                paths.append(dict(path=p, color=[200, 0, 0, 255] if is_inc else col(row.ratio),
                                  width=26 if is_inc else (9 if row.ratio < 0.4 else 5),
                                  name=row.road, what=("INCIDENT · " if is_inc else "")
                                  + f"{row.kmh:.0f} km/h (normal {row.free:.0f} km/h)"))
            allp = [pt for r_ in paths for pt in r_["path"]]
            view = pdk.ViewState(longitude=float(np.mean([p_[0] for p_ in allp])),
                                 latitude=float(np.mean([p_[1] for p_ in allp])), zoom=14.2)
            deck = pdk.Deck(
                layers=[pdk.Layer("PathLayer", paths, get_path="path", get_color="color", get_width="width",
                                  width_scale=1, width_min_pixels=2, pickable=True, cap_rounded=True,
                                  joint_rounded=True)],
                initial_view_state=view, map_provider="carto",
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                tooltip={"text": "{name}\n{what}"})
            st.pydeck_chart(deck, height=600)
            st.markdown("🟩 free flow &nbsp; 🟨 slow &nbsp; 🟥 congested &nbsp; **thick red = incident road** "
                        f"&nbsp;·&nbsp; showing {what}. Hover a road for its name and speed.")
        else:
            st.caption("Real-map background needs the `pyproj` package and internet; showing the offline map.")
        with st.expander("Offline map (no internet needed)", expanded=not geo):
            fig, ax = plt.subplots(figsize=(8, 8))
            for row in df_e.itertuples():
                xs, ys = zip(*net.getEdge(row.id).getShape())
                is_inc = row.id in inc
                ax.plot(xs, ys, color="red" if is_inc else cmap(row.ratio),
                        lw=6 if is_inc else (2.5 if row.ratio < 0.4 else 1.2))
            named = df_e[df_e.road != "unnamed street"].groupby("road").length.sum().sort_values(ascending=False)
            for road in named.index[:12]:
                eid = df_e[df_e.road == road].sort_values("length").id.iloc[-1]
                shp = net.getEdge(eid).getShape()
                x, y = shp[len(shp) // 2]
                ax.annotate(road, (x, y), fontsize=7, ha="center",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, lw=0))
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{sc} · {sg} · seed {sd} — {what}", fontsize=9)
            st.pyplot(fig)

        st.markdown("**Most congested named roads**")
        worst = (df_e[df_e.road != "unnamed street"].groupby("road")
                 .apply(lambda g: pd.Series({"Speed now (km/h)": np.average(g.kmh, weights=g.length),
                                             "Normal speed (km/h)": np.average(g.free, weights=g.length),
                                             "Length (m)": g.length.sum()}), include_groups=False)
                 .assign(**{"% of normal": lambda d: 100 * d["Speed now (km/h)"] / d["Normal speed (km/h)"]})
                 .sort_values("% of normal").head(8).round(0))
        st.dataframe(worst, width="stretch")

# ------------------------------------------------------------------ route advisor
with tab5:
    pk = os.path.join(SIM_OUT, "sim", "final_forecasts.pkl")
    ts_all = read_sim("timeseries.csv")
    if not os.path.exists(pk):
        st.info("Run the simulation stage first: python run_pipeline.py --stage sim")
    else:
        import sumolib
        from st_dss.route_advisor import best_route
        from st_dss.sim.network import boundary_edges, street_name

        snap = pickle.load(open(pk, "rb"))
        net = sumolib.net.readNet(resolve_net(snap["net"]))
        src_list, dst_list = boundary_edges(net)
        st.markdown("Pick where you start and where you want to go. The advisor compares the "
                    "**normal shortest route** (free-flow) with the **recommended route** "
                    "computed on the current traffic forecast.")
        c1, c2, c3 = st.columns(3)
        fmt = lambda e: street_name(net, e)  # noqa: E731
        src = c1.selectbox("From (road)", src_list, format_func=fmt)
        dst = c2.selectbox("To (road)", dst_list, index=len(dst_list) // 2, format_func=fmt)
        cond = c3.selectbox("Traffic condition", ["Normal (no incident)"] + sorted(snap["snapshots"]))
        mode = st.radio("Routing", ["recovery", "travel_time"], horizontal=True,
                        format_func=lambda m: "Recovery-oriented" if m == "recovery" else "Fastest (travel time)")

        ratios = {} if cond.startswith("Normal") else snap["snapshots"][cond]
        inc = set() if cond.startswith("Normal") else set(snap["incident_edges"])
        if not cond.startswith("Normal") and cond.split("|")[0] != "demand_surge":
            # make the advisor aware of the incident itself
            ratios = {**ratios, **{e: min(ratios.get(e, 1.0), 0.05) for e in inc}}
        base, base_eta_free = best_route(net, {}, src, dst, "travel_time")
        rec, rec_eta = best_route(net, ratios, src, dst, mode, cfg["routing"]["recovery_alpha"])

        if rec is None:
            st.error("No route between these segments.")
        else:
            base_eta_now = sum(net.getEdge(e).getLength() / max(min(ratios.get(e, 1.0), 1.0)
                                                                * net.getEdge(e).getSpeed(), 0.5)
                               for e in base)
            from st_dss.route_advisor import directions, roads_of, lonlat_path
            d_base = directions(net, base, ratios)
            d_rec = directions(net, rec, ratios)
            saved = base_eta_now - rec_eta
            blocked = [net.getEdge(e).getName() or e for e in inc if e in base]

            # ---- headline for the user / control-room operator
            if blocked:
                st.error(f"🚧 Incident on your normal route: **{', '.join(sorted(set(blocked)))}**")
            if rec == base:
                st.success("✅ Your normal route is still the best route right now. "
                           f"ETA **{rec_eta / 60:.1f} min**.")
            else:
                new_roads = [r for r in roads_of(d_rec) if r not in roads_of(d_base)]
                avoid = [r for r in roads_of(d_base) if r not in roads_of(d_rec)]
                msg = "✅ **Take the alternate route:** " + " → ".join(dict.fromkeys(roads_of(d_rec)) or ["(local streets)"])
                if blocked:
                    msg += f"  \n⛔ **Avoid the blocked section of:** {', '.join(sorted(set(blocked)))}"
                elif avoid:
                    msg += f"  \n⛔ **Avoid:** {', '.join(dict.fromkeys(avoid))}"
                if new_roads:
                    msg += f"  \n🆕 **Via:** {', '.join(dict.fromkeys(new_roads))}"
                msg += f"  \n⏱️ Saves about **{saved / 60:.1f} min** ({base_eta_now / 60:.1f} → {rec_eta / 60:.1f} min)"
                st.success(msg)

            m1, m2, m3 = st.columns(3)
            m1.metric("Normal route ETA (current traffic)", f"{base_eta_now / 60:.1f} min")
            m2.metric("Recommended route ETA", f"{rec_eta / 60:.1f} min",
                      delta=f"{(rec_eta - base_eta_now) / 60:+.1f} min", delta_color="inverse")
            m3.metric("Recommended route length", f"{sum(net.getEdge(e).getLength() for e in rec) / 1000:.2f} km")

            # ---- real map (OpenStreetMap basemap) when the network is geo-referenced
            geo = bool(net.hasGeoProj()) and lonlat_path(net, rec[0]) is not None
            if geo:
                import pydeck as pdk

                def layer_rows(edges, color, width, label):
                    rows = []
                    for e in edges:
                        p = lonlat_path(net, e)
                        if p:
                            rows.append(dict(path=p, color=color, width=width,
                                             name=(net.getEdge(e).getName() or "unnamed street"),
                                             what=label))
                    return rows

                cmap = plt.get_cmap("RdYlGn")
                bg = []
                for e in net.getEdges():
                    r = min(ratios.get(e.getID(), 1.0), 1.0)
                    c = [int(255 * v) for v in cmap(r)[:3]] + [110]
                    bg += layer_rows([e.getID()], c, 4, f"forecast speed {r * 100:.0f}% of normal")
                rows = (bg + layer_rows(base, [95, 99, 104, 220], 22, "normal route")
                        + layer_rows(rec, [24, 128, 56, 240], 13, "recommended route")
                        + layer_rows(list(inc), [220, 0, 0, 230], 28, "INCIDENT"))
                ends = []
                for e, lab, col in [(src, "START", [0, 0, 0]), (dst, "DESTINATION", [26, 115, 232])]:
                    p = lonlat_path(net, e)
                    ends.append(dict(pos=p[len(p) // 2], label=lab, color=col,
                                     name=net.getEdge(e).getName() or "unnamed street", what=lab))
                allp = [pt for r in rows for pt in r["path"]]
                view = pdk.ViewState(longitude=float(np.mean([p[0] for p in allp])),
                                     latitude=float(np.mean([p[1] for p in allp])), zoom=14.3)
                deck = pdk.Deck(
                    layers=[pdk.Layer("PathLayer", rows, get_path="path", get_color="color",
                                      get_width="width", width_scale=1, width_min_pixels=2,
                                      pickable=True, cap_rounded=True, joint_rounded=True),
                            pdk.Layer("ScatterplotLayer", ends, get_position="pos", get_fill_color="color",
                                      get_radius=40, pickable=True),
                            pdk.Layer("TextLayer", ends, get_position="pos", get_text="label",
                                      get_size=15, get_color=[0, 0, 0], get_pixel_offset=[0, -22])],
                    initial_view_state=view, map_provider="carto",
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    tooltip={"text": "{name}\n{what}"})
                st.pydeck_chart(deck, height=560)
                st.caption("Grey = normal route · Green = recommended · Red = incident · thin lines coloured by "
                           "forecast speed (red = slow). Hover a road to see its name. Basemap © OpenStreetMap / CARTO "
                           "(needs internet; the offline map is below).")

            # ---- turn-by-turn directions
            c1, c2 = st.columns(2)
            for col, title, steps in [(c1, "Normal route", d_base), (c2, "Recommended route", d_rec)]:
                col.markdown(f"**{title}**")
                col.dataframe(pd.DataFrame([dict(Road=s["road"], **{"Distance (m)": round(s["distance_m"]),
                                                                    "Time now (min)": round(s["time_min"], 1)})
                                            for s in steps]), hide_index=True, width="stretch")

            with st.expander("Offline map (no internet needed)", expanded=not geo):
                fig, ax = plt.subplots(figsize=(7, 7))
                cmap = plt.get_cmap("RdYlGn")
                for e in net.getEdges():
                    xs, ys = zip(*e.getShape())
                    ax.plot(xs, ys, color=cmap(min(ratios.get(e.getID(), 1.0), 1.0)), lw=1.2, alpha=0.5)
                for path, colr, lw, lab in [(base, "#5f6368", 7, "normal route"), (rec, "#188038", 4, "recommended")]:
                    for i, eid in enumerate(path):
                        xs, ys = zip(*net.getEdge(eid).getShape())
                        ax.plot(xs, ys, color=colr, lw=lw, solid_capstyle="round", label=lab if i == 0 else None)
                for i, eid in enumerate(sorted(inc)):
                    xs, ys = zip(*net.getEdge(eid).getShape())
                    ax.plot(xs, ys, color="red", lw=8, alpha=0.6, label="incident" if i == 0 else None)
                for eid, mk, lab in [(src, "o", "start"), (dst, "*", "destination")]:
                    shp = net.getEdge(eid).getShape()
                    x, y = shp[len(shp) // 2]
                    ax.scatter([x], [y], s=180, marker=mk, color="black", zorder=5, label=lab)
                # label main roads of both routes
                for steps in (d_base, d_rec):
                    for s_ in steps:
                        if s_["road"] != "unnamed street" and s_["distance_m"] > 150:
                            shp = net.getEdge(s_["edges"][len(s_["edges"]) // 2]).getShape()
                            x, y = shp[len(shp) // 2]
                            ax.annotate(s_["road"], (x, y), fontsize=7, ha="center",
                                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, lw=0))
                ax.set_aspect("equal")
                ax.axis("off")
                ax.legend(loc="upper left", fontsize=8)
                st.pyplot(fig)
