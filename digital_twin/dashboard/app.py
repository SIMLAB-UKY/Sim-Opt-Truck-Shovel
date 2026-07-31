"""Streamlit dashboard for the Adaptive Truck-Shovel Digital Twin.

Pages:
    1. Run a Scenario   — configure and run a single simulation
    2. Compare Policies — multi-replication policy comparison
    3. Adaptive Inspector — EWMA estimates and decision scores
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truck_shovel_dt.config import load_scenario
from truck_shovel_dt.metrics import KPICalculator
from truck_shovel_dt.simulation import TruckShovelSimulation, Sampler
from truck_shovel_dt.estimators import EstimatorRegistry

SCENARIOS_DIR = ROOT / "data" / "scenarios"
ROUTES_PATH = SCENARIOS_DIR / "routes.csv"

st.set_page_config(
    page_title="Truck-Shovel Digital Twin",
    page_icon="🚛",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_scenario_files() -> list[str]:
    return sorted([f.name for f in SCENARIOS_DIR.glob("*.json")])


def run_simulation(
    scenario_name: str,
    policy: str,
    n_trucks: int | None,
    duration: float | None,
    seed: int | None,
    enable_disruptions: bool,
    ewma_alpha: float | None = None,
) -> tuple:
    scenario_path = SCENARIOS_DIR / scenario_name
    config = load_scenario(str(scenario_path), str(ROUTES_PATH))

    from dataclasses import replace

    if duration is not None:
        sim = replace(config.simulation, duration_minutes=float(duration))
        config = replace(config, simulation=sim)
    if ewma_alpha is not None:
        learning = replace(config.learning, ewma_alpha=ewma_alpha)
        config = replace(config, learning=learning)

    _seed = seed if seed is not None else config.simulation.seed
    rng = np.random.default_rng(_seed)
    sampler = Sampler(config=config, rng=rng)

    model = TruckShovelSimulation(
        config=config,
        sampler=sampler,
        policy=policy,
        number_of_trucks=n_trucks,
        enable_disruptions=enable_disruptions,
    )
    result = model.run()

    df = result.event_log.to_dataframe()
    calc = KPICalculator(
        event_log=df,
        warmup_minutes=config.simulation.warmup_minutes,
        simulation_duration_minutes=config.simulation.duration_minutes,
    )
    kpis = calc.calculate()

    return result, df, kpis, calc, config


# ---------------------------------------------------------------------------
# Page 1: Run a Scenario
# ---------------------------------------------------------------------------


def page_run_scenario():
    st.title("🚛 Run a Scenario")

    with st.sidebar:
        st.header("Simulation Settings")
        scenario = st.selectbox("Scenario", get_scenario_files())
        policy = st.selectbox(
            "Dispatch Policy",
            ["fixed", "shortest_queue", "adaptive_ect"],
        )
        n_trucks = st.slider("Number of Trucks", 1, 12, 6)
        duration = st.slider("Duration (minutes)", 60, 960, 480, step=60)
        seed = st.number_input("Random Seed", value=20260715, step=1)
        ewma_alpha = st.slider("EWMA Alpha (α)", 0.05, 1.0, 0.20, step=0.05)
        enable_disruptions = st.checkbox("Enable Shovel Failures", value=False)
        run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Configure settings in the sidebar and click **Run Simulation**.")
        return

    with st.spinner("Running simulation..."):
        try:
            result, df, kpis, calc, config = run_simulation(
                scenario_name=scenario,
                policy=policy,
                n_trucks=n_trucks,
                duration=duration,
                seed=int(seed),
                enable_disruptions=enable_disruptions,
                ewma_alpha=ewma_alpha,
            )
        except Exception as e:
            st.error(f"Simulation error: {e}")
            return

    # ── KPI cards ────────────────────────────────────────────────────────
    st.subheader("Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Production", f"{kpis.production.total_production_tonnes:,.0f} t")
    c2.metric("Tonnes per Hour", f"{kpis.production.tonnes_per_operating_hour:.1f} t/h")
    c3.metric("Completed Trips", f"{kpis.production.completed_trips}")
    c4.metric("Mean Cycle Time", f"{kpis.cycle.mean_cycle_time_min:.1f} min")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Shovel Queue Wait", f"{kpis.queue.mean_shovel_queue_wait_min:.2f} min")
    c6.metric("Dump Queue Wait", f"{kpis.queue.mean_dump_queue_wait_min:.2f} min")
    c7.metric("Mean Truck Util.", f"{kpis.utilization.mean_truck_utilization:.1%}")
    c8.metric("EWMA Alpha", f"{ewma_alpha:.2f}")

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # ── Row 1: Production + Queue length over time ───────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cumulative Production")
            dump_events = df[df["event_type"] == "DUMPING_END"].copy()
            if not dump_events.empty and "payload_tonnes" in dump_events.columns:
                dump_events = dump_events.sort_values("sim_time_min")
                dump_events["cumulative_tonnes"] = dump_events["payload_tonnes"].cumsum()
                fig = px.line(
                    dump_events,
                    x="sim_time_min",
                    y="cumulative_tonnes",
                    labels={"sim_time_min": "Time (min)", "cumulative_tonnes": "Tonnes"},
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Queue Length Over Time")
            queue_events = df[
                df["event_type"].isin(["QUEUE_FOR_SHOVEL"])
                & df["queue_length"].notna()
                & df["shovel_id"].notna()
            ].copy()
            if not queue_events.empty:
                fig = px.scatter(
                    queue_events,
                    x="sim_time_min",
                    y="queue_length",
                    color="shovel_id",
                    labels={
                        "sim_time_min": "Time (min)",
                        "queue_length": "Queue Length",
                        "shovel_id": "Shovel",
                    },
                    opacity=0.6,
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Row 2: Truck + Shovel utilization ────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Truck Utilization")
            truck_table = calc.truck_kpi_table()
            fig = px.bar(
                truck_table,
                x="truck_id",
                y="utilization",
                labels={"truck_id": "Truck", "utilization": "Utilization"},
                color="utilization",
                color_continuous_scale="Blues",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("Shovel Utilization")
            resource_table = calc.resource_kpi_table()
            fig = px.bar(
                resource_table,
                x="resource_id",
                y="utilization",
                labels={"resource_id": "Shovel", "utilization": "Utilization"},
                color="utilization",
                color_continuous_scale="Greens",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 3: Assignment counts ──────────────────────────────────────
        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Assignment Counts by Shovel")
            if "shovel_id" in df.columns:
                assignment_counts = (
                    df[df["event_type"] == "LOADING_START"]["shovel_id"]
                    .value_counts()
                    .reset_index()
                )
                assignment_counts.columns = ["shovel_id", "count"]
                fig = px.bar(
                    assignment_counts,
                    x="shovel_id",
                    y="count",
                    labels={"shovel_id": "Shovel", "count": "Loads"},
                    color="count",
                    color_continuous_scale="Oranges",
                )
                fig.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        with col6:
            st.subheader("Shovel KPI Table")
            st.dataframe(calc.resource_kpi_table(), use_container_width=True)

    except ImportError:
        st.warning("Plotly not available — charts skipped.")

    # ── Tables ───────────────────────────────────────────────────────────
    st.subheader("Truck KPI Table")
    st.dataframe(calc.truck_kpi_table(), use_container_width=True)

    # ── Event log preview ────────────────────────────────────────────────
    st.subheader("Event Log Preview (last 50 events)")
    st.dataframe(df.tail(50), use_container_width=True)

    # ── Disruption events ────────────────────────────────────────────────
    if enable_disruptions:
        disruption_events = df[
            df["event_type"].isin(["SHOVEL_FAILED", "SHOVEL_REPAIRED", "SHOVEL_REPAIR_START"])
        ]
        if not disruption_events.empty:
            st.subheader("Disruption Events")
            st.dataframe(
                disruption_events[["sim_time_min", "event_type", "shovel_id", "notes"]],
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Page 2: Compare Policies
# ---------------------------------------------------------------------------


def page_compare_policies():
    st.title("📊 Compare Policies")

    with st.sidebar:
        st.header("Experiment Settings")
        scenario = st.selectbox("Scenario", get_scenario_files())
        policies = st.multiselect(
            "Policies to Compare",
            ["fixed", "shortest_queue", "adaptive_ect"],
            default=["fixed", "shortest_queue"],
        )
        n_reps = st.slider("Replications", 5, 30, 20)
        ewma_alpha = st.slider("EWMA Alpha (α)", 0.05, 1.0, 0.20, step=0.05)
        enable_disruptions = st.checkbox("Enable Shovel Failures", value=False)
        run_btn = st.button("▶ Run Experiment", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Select policies and click **Run Experiment**.")
        return

    if not policies:
        st.warning("Select at least one policy.")
        return

    scenario_path = SCENARIOS_DIR / scenario
    config = load_scenario(str(scenario_path), str(ROUTES_PATH))
    base_seed = config.simulation.seed

    all_rows = []
    progress = st.progress(0)
    total = len(policies) * n_reps

    for pi, policy in enumerate(policies):
        for rep in range(n_reps):
            seed = base_seed + hash(f"{policy}_{rep}") % (2**31)
            try:
                _, _, kpis, _, _ = run_simulation(
                    scenario_name=scenario,
                    policy=policy,
                    n_trucks=None,
                    duration=None,
                    seed=seed,
                    enable_disruptions=enable_disruptions,
                    ewma_alpha=ewma_alpha,
                )
                all_rows.append(
                    {
                        "policy": policy,
                        "replication": rep + 1,
                        "total_production_tonnes": kpis.production.total_production_tonnes,
                        "tonnes_per_operating_hour": kpis.production.tonnes_per_operating_hour,
                        "completed_trips": kpis.production.completed_trips,
                        "mean_cycle_time_min": kpis.cycle.mean_cycle_time_min,
                        "mean_shovel_queue_wait_min": kpis.queue.mean_shovel_queue_wait_min,
                        "mean_truck_utilization": kpis.utilization.mean_truck_utilization,
                    }
                )
            except Exception as e:
                st.warning(f"Rep {rep+1} failed for {policy}: {e}")
            progress.progress((pi * n_reps + rep + 1) / total)

    df = pd.DataFrame(all_rows)

    st.subheader("Results Summary")
    agg = (
        df.groupby("policy")
        .agg(
            production_mean=("total_production_tonnes", "mean"),
            production_std=("total_production_tonnes", "std"),
            tph_mean=("tonnes_per_operating_hour", "mean"),
            cycle_mean=("mean_cycle_time_min", "mean"),
            queue_mean=("mean_shovel_queue_wait_min", "mean"),
            util_mean=("mean_truck_utilization", "mean"),
        )
        .reset_index()
    )
    st.dataframe(agg.round(2), use_container_width=True)

    try:
        import plotly.express as px

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Production Distribution")
            fig = px.box(
                df,
                x="policy",
                y="total_production_tonnes",
                labels={"policy": "Policy", "total_production_tonnes": "Production (t)"},
                color="policy",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Cycle Time Distribution")
            fig = px.box(
                df,
                x="policy",
                y="mean_cycle_time_min",
                labels={"policy": "Policy", "mean_cycle_time_min": "Cycle Time (min)"},
                color="policy",
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Shovel Queue Wait")
            fig = px.box(
                df,
                x="policy",
                y="mean_shovel_queue_wait_min",
                labels={"policy": "Policy", "mean_shovel_queue_wait_min": "Queue Wait (min)"},
                color="policy",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("Truck Utilization")
            fig = px.box(
                df,
                x="policy",
                y="mean_truck_utilization",
                labels={"policy": "Policy", "mean_truck_utilization": "Utilization"},
                color="policy",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("All Replications")
        st.dataframe(df, use_container_width=True)

    except ImportError:
        st.warning("Plotly not available — charts skipped.")


# ---------------------------------------------------------------------------
# Page 3: Adaptive Inspector
# ---------------------------------------------------------------------------


def page_adaptive_inspector():
    st.title("🔍 Adaptive Controller Inspector")

    with st.sidebar:
        st.header("Settings")
        scenario = st.selectbox("Scenario", get_scenario_files())
        seed = st.number_input("Random Seed", value=20260715, step=1)
        n_trucks = st.slider("Number of Trucks", 1, 12, 6)
        duration = st.slider("Duration (minutes)", 60, 480, 480, step=60)
        ewma_alpha = st.slider("EWMA Alpha (α)", 0.05, 1.0, 0.20, step=0.05)
        enable_disruptions = st.checkbox("Enable Shovel Failures", value=False)
        run_btn = st.button("▶ Run Adaptive ECT", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Click **Run Adaptive ECT** to inspect the controller.")
        return

    with st.spinner("Running adaptive simulation..."):
        try:
            result, df, kpis, calc, config = run_simulation(
                scenario_name=scenario,
                policy="adaptive_ect",
                n_trucks=n_trucks,
                duration=duration,
                seed=int(seed),
                enable_disruptions=enable_disruptions,
                ewma_alpha=ewma_alpha,
            )
        except Exception as e:
            st.error(f"Simulation error: {e}")
            return

    # ── KPI summary ──────────────────────────────────────────────────────
    st.subheader("Simulation KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Production", f"{kpis.production.total_production_tonnes:,.0f} t")
    c2.metric("Trips", f"{kpis.production.completed_trips}")
    c3.metric("Cycle Time", f"{kpis.cycle.mean_cycle_time_min:.1f} min")
    c4.metric("EWMA Alpha", f"{ewma_alpha:.2f}")

    # ── EWMA estimate table ──────────────────────────────────────────────
    st.subheader("EWMA Estimates (initialized from scenario)")
    reg = EstimatorRegistry(
        alpha=ewma_alpha,
        minimum_observations=config.learning.minimum_observations,
    )
    for shovel in config.shovels:
        reg.init_loading(shovel.id, shovel.loading.mode)
    for dump in config.dumps:
        reg.init_dumping(dump.id, dump.dump.mode)

    summary = reg.summary()
    loading_rows = [
        {"resource": k, "type": "loading", "initial_estimate_min": round(v["estimate"], 3)}
        for k, v in summary["loading"].items()
    ]
    dumping_rows = [
        {"resource": k, "type": "dumping", "initial_estimate_min": round(v["estimate"], 3)}
        for k, v in summary["dumping"].items()
    ]
    st.dataframe(pd.DataFrame(loading_rows + dumping_rows), use_container_width=True)

    # ── Queue length over time ───────────────────────────────────────────
    try:
        import plotly.express as px

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Queue Length Over Time")
            queue_events = df[
                df["event_type"].isin(["QUEUE_FOR_SHOVEL"])
                & df["queue_length"].notna()
                & df["shovel_id"].notna()
            ].copy()
            if not queue_events.empty:
                fig = px.scatter(
                    queue_events,
                    x="sim_time_min",
                    y="queue_length",
                    color="shovel_id",
                    labels={
                        "sim_time_min": "Time (min)",
                        "queue_length": "Queue Length",
                        "shovel_id": "Shovel",
                    },
                    opacity=0.6,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Assignment Counts by Shovel")
            if "shovel_id" in df.columns:
                counts = (
                    df[df["event_type"] == "LOADING_START"]["shovel_id"]
                    .value_counts()
                    .reset_index()
                )
                counts.columns = ["shovel_id", "count"]
                fig = px.bar(
                    counts,
                    x="shovel_id",
                    y="count",
                    labels={"shovel_id": "Shovel", "count": "Loads"},
                    color="count",
                    color_continuous_scale="Purples",
                )
                fig.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        pass

    # ── Equipment status ─────────────────────────────────────────────────
    st.subheader("Equipment Status")
    if enable_disruptions:
        disruption_df = df[
            df["event_type"].isin(["SHOVEL_FAILED", "SHOVEL_REPAIRED", "SHOVEL_REPAIR_START"])
        ]
        if not disruption_df.empty:
            st.dataframe(
                disruption_df[["sim_time_min", "event_type", "shovel_id", "notes"]],
                use_container_width=True,
            )
        else:
            st.info("No disruptions occurred in this run.")
    else:
        st.info("Disruptions disabled — all shovels available throughout.")

    # ── Recent dispatch events ───────────────────────────────────────────
    st.subheader("Recent Dispatch Events (last 20)")
    dispatch_df = df[df["event_type"] == "DISPATCH"].tail(20)
    if not dispatch_df.empty:
        cols = [
            c
            for c in ["sim_time_min", "truck_id", "shovel_id", "dump_id"]
            if c in dispatch_df.columns
        ]
        st.dataframe(dispatch_df[cols], use_container_width=True)

    # ── Full event log ───────────────────────────────────────────────────
    st.subheader("Full Event Log")
    st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "🚛 Run a Scenario": page_run_scenario,
    "📊 Compare Policies": page_compare_policies,
    "🔍 Adaptive Inspector": page_adaptive_inspector,
}

with st.sidebar:
    st.image("https://img.icons8.com/color/96/dump-truck.png", width=80)
    st.title("Digital Twin")
    st.caption("Adaptive Truck-Shovel Dispatch")
    st.divider()
    page = st.radio("Navigate", list(PAGES.keys()))
    st.divider()
    st.caption("Surface Mine Operations")

PAGES[page]()
