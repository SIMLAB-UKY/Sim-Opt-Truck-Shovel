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
from pathlib import Path

import numpy as np

from truck_shovel_dt.config import load_scenario
from truck_shovel_dt.simulation import MinimalSimulation, Sampler

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

    model = MinimalSimulation(
        config=config,
        sampler=sampler,
        shovel_id=config.shovels[0].id,
        dump_id=config.dumps[0].id,
        policy=args.policy,
    )
    result = model.run()

    result.event_log.print_trace()

    print(f"Completed trips      : {result.completed_trips}")
    print(f"Total production     : {result.total_production_tonnes:.1f} tonnes")
    if result.completed_trips > 0:
        avg = result.total_production_tonnes / result.completed_trips
        print(f"Average payload      : {avg:.1f} tonnes/trip")


if __name__ == "__main__":
    main()
