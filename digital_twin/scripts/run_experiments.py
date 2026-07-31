"""Multi-replication experiment runner for the Adaptive Truck-Shovel Digital Twin.

Usage:
    python scripts/run_experiments.py \
        --scenario data/scenarios/base_scenario.json \
        --routes data/scenarios/routes.csv \
        --policies fixed shortest_queue \
        --replications 20 \
        --output data/results/base_comparison

Produces:
    - per_replication_kpis.csv
    - aggregated_kpis.csv  (mean + 95% CI for each KPI and policy)
    - policy_comparison.png
    - experiment_config.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from truck_shovel_dt.config import load_scenario
from truck_shovel_dt.metrics import KPICalculator
from truck_shovel_dt.simulation import Sampler, TruckShovelSimulation

# ---------------------------------------------------------------------------
# Single replication
# ---------------------------------------------------------------------------


def run_one_replication(
    scenario_path: str,
    routes_path: str,
    policy: str,
    seed: int,
    enable_disruptions: bool,
) -> dict:
    """Run one replication and return a flat KPI dict."""
    config = load_scenario(scenario_path, routes_path)
    rng = np.random.default_rng(seed)
    sampler = Sampler(config=config, rng=rng)

    model = TruckShovelSimulation(
        config=config,
        sampler=sampler,
        policy=policy,
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

    return {
        "seed": seed,
        "policy": policy,
        "total_production_tonnes": kpis.production.total_production_tonnes,
        "tonnes_per_operating_hour": kpis.production.tonnes_per_operating_hour,
        "completed_trips": kpis.production.completed_trips,
        "mean_cycle_time_min": kpis.cycle.mean_cycle_time_min,
        "mean_shovel_queue_wait_min": kpis.queue.mean_shovel_queue_wait_min,
        "mean_dump_queue_wait_min": kpis.queue.mean_dump_queue_wait_min,
        "mean_truck_utilization": kpis.utilization.mean_truck_utilization,
    }


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------


def confidence_interval_95(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, lower_95ci, upper_95ci)."""
    n = len(values)
    if n < 2:
        mean = values[0] if values else 0.0
        return mean, mean, mean
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    t_crit = stats.t.ppf(0.975, df=n - 1)
    margin = t_crit * std / np.sqrt(n)
    return mean, mean - margin, mean + margin


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

KPI_COLUMNS = [
    "total_production_tonnes",
    "tonnes_per_operating_hour",
    "completed_trips",
    "mean_cycle_time_min",
    "mean_shovel_queue_wait_min",
    "mean_dump_queue_wait_min",
    "mean_truck_utilization",
]


def aggregate_replications(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and 95% CI for each policy and KPI."""
    rows = []
    for policy in df["policy"].unique():
        policy_df = df[df["policy"] == policy]
        row: dict = {"policy": policy, "n_replications": len(policy_df)}
        for kpi in KPI_COLUMNS:
            values = policy_df[kpi].tolist()
            mean, lo, hi = confidence_interval_95(values)
            row[f"{kpi}_mean"] = round(mean, 4)
            row[f"{kpi}_ci_lo"] = round(lo, 4)
            row[f"{kpi}_ci_hi"] = round(hi, 4)
            row[f"{kpi}_std"] = round(float(np.std(values, ddof=1)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_comparison_figure(
    agg_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a policy-comparison bar chart with 95% CI error bars."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figure.")
        return

    kpis_to_plot = [
        ("total_production_tonnes", "Total Production (tonnes)"),
        ("tonnes_per_operating_hour", "Tonnes per Hour"),
        ("mean_cycle_time_min", "Mean Cycle Time (min)"),
        ("mean_shovel_queue_wait_min", "Mean Shovel Queue Wait (min)"),
        ("mean_truck_utilization", "Mean Truck Utilization"),
    ]

    n_plots = len(kpis_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    policies = agg_df["policy"].tolist()
    x = np.arange(len(policies))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, (kpi, label) in zip(axes, kpis_to_plot, strict=True):
        means = agg_df[f"{kpi}_mean"].tolist()
        lo = agg_df[f"{kpi}_ci_lo"].tolist()
        hi = agg_df[f"{kpi}_ci_hi"].tolist()
        yerr_lo = [m - low for m, low in zip(means, lo, strict=True)]
        yerr_hi = [h - m for m, h in zip(means, hi, strict=True)]

        ax.bar(
            x,
            means,
            color=colors[: len(policies)],
            alpha=0.85,
            yerr=[yerr_lo, yerr_hi],
            capsize=5,
            error_kw={"elinewidth": 1.5},
        )
        ax.set_title(label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Policy Comparison — 95% Confidence Intervals", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-replication policy comparison experiments."
    )
    parser.add_argument(
        "--scenario",
        default="data/scenarios/base_scenario.json",
        help="Path to scenario JSON file.",
    )
    parser.add_argument(
        "--routes",
        default="data/scenarios/routes.csv",
        help="Path to routes CSV file.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["fixed", "shortest_queue"],
        choices=["fixed", "shortest_queue", "adaptive_ect"],
        help="Dispatch policies to compare.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=20,
        help="Number of replications per policy.",
    )
    parser.add_argument(
        "--output",
        default="data/results/experiment",
        help="Output directory prefix.",
    )
    parser.add_argument(
        "--disruptions",
        action="store_true",
        default=False,
        help="Enable shovel disruptions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_scenario(args.scenario, args.routes)
    base_seed = config.simulation.seed
    scenario_name = config.scenario_name

    print(f"Scenario  : {scenario_name}")
    print(f"Policies  : {args.policies}")
    print(f"Replications: {args.replications} per policy")
    print(f"Disruptions : {args.disruptions}")
    print(f"Output    : {output_dir}")
    print()

    start_time = time.time()
    all_rows: list[dict] = []

    for policy in args.policies:
        print(f"Running policy: {policy} ...")
        for rep in range(args.replications):
            # Derive unique seed per replication
            seed = base_seed + hash(f"{policy}_{rep}") % (2**31)
            row = run_one_replication(
                scenario_path=args.scenario,
                routes_path=args.routes,
                policy=policy,
                seed=seed,
                enable_disruptions=args.disruptions,
            )
            row["replication"] = rep + 1
            all_rows.append(row)
            print(
                f"  Rep {rep + 1:02d}: trips={row['completed_trips']}, "
                f"production={row['total_production_tonnes']:.0f}t"
            )

    elapsed = time.time() - start_time

    # Save per-replication KPIs
    per_rep_df = pd.DataFrame(all_rows)
    per_rep_path = output_dir / "per_replication_kpis.csv"
    per_rep_df.to_csv(per_rep_path, index=False)
    print(f"\nPer-replication KPIs saved: {per_rep_path}")

    # Aggregate with confidence intervals
    agg_df = aggregate_replications(per_rep_df)
    agg_path = output_dir / "aggregated_kpis.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"Aggregated KPIs saved     : {agg_path}")

    # Policy comparison figure
    fig_path = output_dir / "policy_comparison.png"
    make_comparison_figure(agg_df, fig_path)

    # Experiment metadata
    import subprocess

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    metadata = {
        "scenario": args.scenario,
        "scenario_name": scenario_name,
        "policies": args.policies,
        "replications": args.replications,
        "base_seed": base_seed,
        "disruptions": args.disruptions,
        "elapsed_seconds": round(elapsed, 2),
        "git_commit": git_commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = output_dir / "experiment_config.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Experiment metadata saved : {meta_path}")

    # Print summary table
    print("\n── Summary ──────────────────────────────────────────────────")
    for _, row in agg_df.iterrows():
        print(f"\nPolicy: {row['policy']} (n={row['n_replications']})")
        print(
            f"  Production : {row['total_production_tonnes_mean']:.1f} t "
            f"[{row['total_production_tonnes_ci_lo']:.1f}, "
            f"{row['total_production_tonnes_ci_hi']:.1f}]"
        )
        print(
            f"  t/h        : {row['tonnes_per_operating_hour_mean']:.1f} "
            f"[{row['tonnes_per_operating_hour_ci_lo']:.1f}, "
            f"{row['tonnes_per_operating_hour_ci_hi']:.1f}]"
        )
        print(f"  Cycle time : {row['mean_cycle_time_min_mean']:.2f} min")
        print(f"  Shovel wait: {row['mean_shovel_queue_wait_min_mean']:.3f} min")
    print()


if __name__ == "__main__":
    main()
