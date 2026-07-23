"""Simulation engine for the Adaptive Truck-Shovel Digital Twin.

Day 5 scope: multiple trucks, multiple shovels, multiple dumps,
full route matrix, fixed-assignment policy, event log export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import simpy

from truck_shovel_dt.config import ScenarioConfig


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

@dataclass
class EventLog:
    records: list[dict[str, Any]] = field(default_factory=list)

    def log(self, **kwargs: Any) -> None:
        self.records.append(kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save(self, path: str | Path) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def print_trace(self) -> None:
        print(f"\n{'─'*90}")
        print(f"{'Time':>8}  {'Event':<25} {'Truck':<6} {'Origin':<6} {'Dest':<6} {'Queue'}")
        print(f"{'─'*90}")
        for r in self.records:
            print(
                f"{r['sim_time_min']:>8.2f}  "
                f"{r['event_type']:<25} "
                f"{r.get('truck_id', ''):<6} "
                f"{r.get('origin', ''):<6} "
                f"{r.get('destination', ''):<6} "
                f"{r.get('queue_length', '')}"
            )
        print(f"{'─'*90}\n")


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class Sampler:
    def __init__(self, config: ScenarioConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._deterministic = False
        self._fixed: dict[str, float] = {}

    def set_deterministic(self, values: dict[str, float]) -> None:
        self._deterministic = True
        self._fixed = values

    def empty_travel(self, origin: str, destination: str) -> float:
        if self._deterministic:
            return self._fixed["empty_travel"]
        route = self._get_route(origin, destination, "empty")
        base = route.distance_km / route.mean_speed_kph * 60.0
        multiplier = self._rng.triangular(
            self._config.travel_variability.multiplier_min,
            self._config.travel_variability.multiplier_mode,
            self._config.travel_variability.multiplier_max,
        )
        return base * multiplier

    def loaded_travel(self, origin: str, destination: str) -> float:
        if self._deterministic:
            return self._fixed["loaded_travel"]
        route = self._get_route(origin, destination, "loaded")
        base = route.distance_km / route.mean_speed_kph * 60.0
        multiplier = self._rng.triangular(
            self._config.travel_variability.multiplier_min,
            self._config.travel_variability.multiplier_mode,
            self._config.travel_variability.multiplier_max,
        )
        return base * multiplier

    def loading(self, shovel_id: str) -> float:
        if self._deterministic:
            return self._fixed["loading"]
        shovel = next(s for s in self._config.shovels if s.id == shovel_id)
        return float(self._rng.triangular(
            shovel.loading.minimum,
            shovel.loading.mode,
            shovel.loading.maximum,
        ))

    def dumping(self, dump_id: str) -> float:
        if self._deterministic:
            return self._fixed["dumping"]
        dump = next(d for d in self._config.dumps if d.id == dump_id)
        return float(self._rng.triangular(
            dump.dump.minimum,
            dump.dump.mode,
            dump.dump.maximum,
        ))

    def payload(self) -> float:
        if self._deterministic:
            return self._fixed["payload"]
        f = self._config.fleet
        return float(np.clip(
            self._rng.normal(f.payload_mean, f.payload_std),
            f.payload_min,
            f.payload_max,
        ))

    def _get_route(self, origin: str, destination: str, load_state: str):
        for r in self._config.routes:
            if (
                r.origin == origin
                and r.destination == destination
                and r.load_state == load_state
            ):
                return r
        raise ValueError(f"No {load_state} route from {origin} to {destination}")


# ---------------------------------------------------------------------------
# Fixed-assignment policy
# ---------------------------------------------------------------------------

class FixedAssignment:
    """Assign trucks to shovels before the shift based on truck index.

    Truck index % n_shovels determines the assigned shovel.
    The dump is chosen as the one with the shortest loaded travel time
    from the assigned shovel.
    """
    name = "fixed"

    def __init__(self, config: ScenarioConfig) -> None:
        self._shovels = [s.id for s in config.shovels]
        self._dumps = [d.id for d in config.dumps]
        self._routes = config.routes
        self._assignments: dict[str, tuple[str, str]] = {}

    def assign(self, truck_id: str, truck_index: int) -> tuple[str, str]:
        """Return (shovel_id, dump_id) for this truck."""
        shovel_id = self._shovels[truck_index % len(self._shovels)]
        dump_id = self._best_dump(shovel_id)
        self._assignments[truck_id] = (shovel_id, dump_id)
        return shovel_id, dump_id

    def get_assignment(self, truck_id: str) -> tuple[str, str]:
        return self._assignments[truck_id]

    def _best_dump(self, shovel_id: str) -> str:
        best_dump = self._dumps[0]
        best_time = float("inf")
        for r in self._routes:
            if r.origin == shovel_id and r.load_state == "loaded":
                t = r.distance_km / r.mean_speed_kph * 60.0
                if t < best_time:
                    best_time = t
                    best_dump = r.destination
        return best_dump


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    event_log: EventLog
    total_production_tonnes: float
    completed_trips: int
    truck_trip_counts: dict[str, int]
    policy: str = "fixed"

    def save_summary(self, path: str | Path) -> None:
        summary = {
            "policy": self.policy,
            "completed_trips": self.completed_trips,
            "total_production_tonnes": round(self.total_production_tonnes, 2),
            "truck_trip_counts": self.truck_trip_counts,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main simulation model — Day 5: full network
# ---------------------------------------------------------------------------

class TruckShovelSimulation:
    """Multiple trucks, multiple shovels, multiple dumps, full route matrix."""

    def __init__(
        self,
        config: ScenarioConfig,
        sampler: Sampler,
        policy: str = "fixed",
        number_of_trucks: int | None = None,
    ) -> None:
        self._config = config
        self._sampler = sampler
        self._policy_name = policy
        self._n_trucks = (
            number_of_trucks
            if number_of_trucks is not None
            else config.fleet.number_of_trucks
        )
        self._event_log = EventLog()
        self._total_tonnes = 0.0
        self._completed_trips = 0
        self._truck_trip_counts: dict[str, int] = {}

        # Fixed assignment policy
        self._fixed_policy = FixedAssignment(config)

    def run(self) -> SimulationResult:
        env = simpy.Environment()

        # One simpy.Resource per shovel and per dump
        shovel_resources = {
            s.id: simpy.Resource(env, capacity=1)
            for s in self._config.shovels
        }
        dump_resources = {
            d.id: simpy.Resource(env, capacity=1)
            for d in self._config.dumps
        }

        end_time = self._config.simulation.duration_minutes

        for i in range(self._n_trucks):
            truck_id = f"T{i + 1:02d}"
            self._truck_trip_counts[truck_id] = 0
            shovel_id, dump_id = self._fixed_policy.assign(truck_id, i)
            env.process(
                self._truck_process(
                    env=env,
                    truck_id=truck_id,
                    shovel_id=shovel_id,
                    dump_id=dump_id,
                    shovel_resources=shovel_resources,
                    dump_resources=dump_resources,
                    end_time=end_time,
                )
            )

        env.run(until=end_time + 1e-9)

        return SimulationResult(
            event_log=self._event_log,
            total_production_tonnes=self._total_tonnes,
            completed_trips=self._completed_trips,
            truck_trip_counts=dict(self._truck_trip_counts),
            policy=self._policy_name,
        )

    def _log(self, env: simpy.Environment, event_type: str, **kwargs: Any) -> None:
        self._event_log.log(
            sim_time_min=round(env.now, 4),
            event_type=event_type,
            policy=self._policy_name,
            **kwargs,
        )

    def _truck_process(
        self,
        env: simpy.Environment,
        truck_id: str,
        shovel_id: str,
        dump_id: str,
        shovel_resources: dict[str, simpy.Resource],
        dump_resources: dict[str, simpy.Resource],
        end_time: float,
    ):
        current_location = dump_id

        while env.now < end_time:
            # ── 1. Dispatch ──────────────────────────────────────────────
            self._log(env, "DISPATCH",
                truck_id=truck_id,
                origin=current_location,
                destination=shovel_id,
                shovel_id=shovel_id,
                dump_id=dump_id,
            )

            # ── 2. Empty travel ──────────────────────────────────────────
            empty_duration = self._sampler.empty_travel(current_location, shovel_id)
            self._log(env, "EMPTY_TRAVEL_START",
                truck_id=truck_id,
                origin=current_location,
                destination=shovel_id,
                duration_min=round(empty_duration, 4),
            )
            yield env.timeout(empty_duration)
            self._log(env, "EMPTY_TRAVEL_END",
                truck_id=truck_id,
                origin=current_location,
                destination=shovel_id,
            )

            # ── 3. Queue and load ────────────────────────────────────────
            shovel_resource = shovel_resources[shovel_id]
            queue_start = env.now
            self._log(env, "QUEUE_FOR_SHOVEL",
                truck_id=truck_id,
                origin=shovel_id,
                destination=shovel_id,
                shovel_id=shovel_id,
                queue_length=len(shovel_resource.queue),
            )
            with shovel_resource.request() as req:
                yield req
                queue_wait = round(env.now - queue_start, 4)
                loading_duration = self._sampler.loading(shovel_id)
                self._log(env, "LOADING_START",
                    truck_id=truck_id,
                    origin=shovel_id,
                    destination=shovel_id,
                    shovel_id=shovel_id,
                    duration_min=round(loading_duration, 4),
                    queue_wait_min=queue_wait,
                    queue_length=0,
                )
                yield env.timeout(loading_duration)
                self._log(env, "LOADING_END",
                    truck_id=truck_id,
                    origin=shovel_id,
                    destination=dump_id,
                    shovel_id=shovel_id,
                    duration_min=round(loading_duration, 4),
                )

            # ── 4. Loaded travel ─────────────────────────────────────────
            loaded_duration = self._sampler.loaded_travel(shovel_id, dump_id)
            self._log(env, "LOADED_TRAVEL_START",
                truck_id=truck_id,
                origin=shovel_id,
                destination=dump_id,
                duration_min=round(loaded_duration, 4),
            )
            yield env.timeout(loaded_duration)
            self._log(env, "LOADED_TRAVEL_END",
                truck_id=truck_id,
                origin=shovel_id,
                destination=dump_id,
            )

            # ── 5. Queue and dump ────────────────────────────────────────
            dump_resource = dump_resources[dump_id]
            dump_queue_start = env.now
            self._log(env, "QUEUE_FOR_DUMP",
                truck_id=truck_id,
                origin=dump_id,
                destination=dump_id,
                dump_id=dump_id,
                queue_length=len(dump_resource.queue),
            )
            with dump_resource.request() as req:
                yield req
                dump_queue_wait = round(env.now - dump_queue_start, 4)
                dump_duration = self._sampler.dumping(dump_id)
                payload = self._sampler.payload()
                self._log(env, "DUMPING_START",
                    truck_id=truck_id,
                    origin=dump_id,
                    destination=dump_id,
                    dump_id=dump_id,
                    duration_min=round(dump_duration, 4),
                    payload_tonnes=round(payload, 4),
                    queue_wait_min=dump_queue_wait,
                )
                yield env.timeout(dump_duration)

                if env.now <= end_time + 1e-9:
                    self._total_tonnes += payload
                    self._completed_trips += 1
                    self._truck_trip_counts[truck_id] += 1

                self._log(env, "DUMPING_END",
                    truck_id=truck_id,
                    origin=dump_id,
                    destination=dump_id,
                    dump_id=dump_id,
                    payload_tonnes=round(payload, 4),
                )

            current_location = dump_id


# ---------------------------------------------------------------------------
# Backward-compatible alias (Day 3/4 tests use MinimalSimulation)
# ---------------------------------------------------------------------------

class MinimalSimulation(TruckShovelSimulation):
    """Alias kept for backward compatibility with Day 3/4 tests."""

    def __init__(
        self,
        config: ScenarioConfig,
        sampler: Sampler,
        shovel_id: str = "S1",
        dump_id: str = "D1",
        policy: str = "fixed",
        number_of_trucks: int | None = None,
    ) -> None:
        super().__init__(
            config=config,
            sampler=sampler,
            policy=policy,
            number_of_trucks=number_of_trucks,
        )
        # Override fixed assignments to use specified shovel/dump
        self._shovel_id = shovel_id
        self._dump_id = dump_id

    def run(self) -> SimulationResult:
        env = simpy.Environment()
        shovel_resources = {
            s.id: simpy.Resource(env, capacity=1)
            for s in self._config.shovels
        }
        dump_resources = {
            d.id: simpy.Resource(env, capacity=1)
            for d in self._config.dumps
        }
        end_time = self._config.simulation.duration_minutes
        n = (
            self._n_trucks
            if self._n_trucks is not None
            else self._config.fleet.number_of_trucks
        )
        for i in range(n):
            truck_id = f"T{i + 1:02d}"
            self._truck_trip_counts[truck_id] = 0
            env.process(
                self._truck_process(
                    env=env,
                    truck_id=truck_id,
                    shovel_id=self._shovel_id,
                    dump_id=self._dump_id,
                    shovel_resources=shovel_resources,
                    dump_resources=dump_resources,
                    end_time=end_time,
                )
            )
        env.run(until=end_time + 1e-9)
        return SimulationResult(
            event_log=self._event_log,
            total_production_tonnes=self._total_tonnes,
            completed_trips=self._completed_trips,
            truck_trip_counts=dict(self._truck_trip_counts),
            policy=self._policy_name,
        )
