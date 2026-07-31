"""Run a single simulation scenario and print the event trace.

Usage:
    python scripts/run_single_scenario.py \
        --scenario data/scenarios/base_scenario.json \
        --routes data/scenarios/routes.csv \
        --policy fixed \
        --duration 60 \
        --deterministic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from truck_shovel_dt.config import load_scenario
from truck_shovel_dt.metrics import KPICalculator
from truck_shovel_dt.simulation import Sampler, TruckShovelSimulation

DETERMINISTIC_VALUES = {
    "empty_travel": 5.0,
    "loading": 4.0,
    "loaded_travel": 7.0,
    "dumping": 1.0,
    "payload": 100.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single simulation scenario.")
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
        "--policy",
        default="fixed",
        choices=["fixed", "shortest_queue", "adaptive_ect"],
        help="Dispatch policy to use.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override simulation duration in minutes.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use fixed durations instead of stochastic sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--output",
        default="data/results",
        help="Directory to save event log and summary.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_scenario(args.scenario, args.routes)

    if args.duration is not None:
        from dataclasses import replace

        sim = replace(config.simulation, duration_minutes=args.duration)
        config = replace(config, simulation=sim)

    seed = args.seed if args.seed is not None else config.simulation.seed
    rng = np.random.default_rng(seed)

    sampler = Sampler(config=config, rng=rng)
    if args.deterministic:
        sampler.set_deterministic(DETERMINISTIC_VALUES)
        print("Running in DETERMINISTIC mode.")
        print(f"Fixed values: {DETERMINISTIC_VALUES}")
    else:
        print(f"Running in STOCHASTIC mode (seed={seed}).")

    print(f"Scenario : {config.scenario_name}")
    print(f"Duration : {config.simulation.duration_minutes} minutes")
    print(f"Policy   : {args.policy}")
    print()

    model = TruckShovelSimulation(
        config=config,
        sampler=sampler,
        policy=args.policy,
    )
    result = model.run()

    # ── KPI calculation ──────────────────────────────────────────────────
    df = result.event_log.to_dataframe()
    calculator = KPICalculator(
        event_log=df,
        warmup_minutes=config.simulation.warmup_minutes,
        simulation_duration_minutes=config.simulation.duration_minutes,
    )
    kpis = calculator.calculate()

    # ── Print trace and summary ──────────────────────────────────────────
    result.event_log.print_trace()

    print("─" * 50)
    print(f"Completed trips      : {kpis.production.completed_trips}")
    print(f"Total production     : {kpis.production.total_production_tonnes:.1f} tonnes")
    print(f"Tonnes per hour      : {kpis.production.tonnes_per_operating_hour:.1f} t/h")
    print(f"Mean cycle time      : {kpis.cycle.mean_cycle_time_min:.1f} min")
    print(f"Mean shovel queue    : {kpis.queue.mean_shovel_queue_wait_min:.2f} min")
    print(f"Mean truck utiliz.   : {kpis.utilization.mean_truck_utilization:.1%}")
    shovel_util = {k: f"{v:.1%}" for k, v in kpis.utilization.shovel_utilization.items()}
    print(f"Shovel utilization   : {shovel_util}")
    print("─" * 50)

    # ── Save outputs ─────────────────────────────────────────────────────
    if not args.no_save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        scenario_slug = config.scenario_name.replace(" ", "_")

        # Event log
        event_log_path = output_dir / f"event_log_{scenario_slug}.csv"
        result.event_log.save(event_log_path)
        print(f"Event log saved  : {event_log_path}")

        # run_summary.json — combines simulation info + KPIs
        summary = {
            "scenario_name": config.scenario_name,
            "policy": args.policy,
            "seed": seed,
            "duration_minutes": config.simulation.duration_minutes,
            "warmup_minutes": config.simulation.warmup_minutes,
            "number_of_trucks": config.fleet.number_of_trucks,
            "kpis": kpis.to_dict(),
            "truck_trip_counts": result.truck_trip_counts,
            "truck_kpi_table": calculator.truck_kpi_table().to_dict(orient="records"),
            "resource_kpi_table": calculator.resource_kpi_table().to_dict(orient="records"),
        }
        summary_path = output_dir / f"run_summary_{scenario_slug}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Run summary saved: {summary_path}")


if __name__ == "__main__":
    main()
