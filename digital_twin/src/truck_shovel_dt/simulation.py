"""Simulation engine for the Adaptive Truck-Shovel Digital Twin.

Day 3 scope: one truck, one shovel, one dump.
Supports deterministic mode (fixed durations) and stochastic mode.
Logs every event to a list of dicts that can be converted to a DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
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

    def print_trace(self) -> None:
        print(f"\n{'─'*70}")
        print(f"{'Time':>8}  {'Event':<25} {'Truck':<8} {'Location'}")
        print(f"{'─'*70}")
        for r in self.records:
            location = r.get("destination") or r.get("origin") or ""
            print(
                f"{r['sim_time_min']:>8.2f}  "
                f"{r['event_type']:<25} "
                f"{r.get('truck_id', ''):<8} "
                f"{location}"
            )
        print(f"{'─'*70}\n")


# ---------------------------------------------------------------------------
# Sampler — deterministic or stochastic
# ---------------------------------------------------------------------------

class Sampler:
    """Draw activity durations from scenario distributions or fixed values."""

    def __init__(self, config: ScenarioConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._deterministic = False
        self._fixed: dict[str, float] = {}

    def set_deterministic(self, values: dict[str, float]) -> None:
        """Switch to deterministic mode with a fixed-value dict.

        Keys: 'empty_travel', 'loading', 'loaded_travel', 'dumping', 'payload'
        """
        self._deterministic = True
        self._fixed = values

    def empty_travel(self, origin: str, destination: str) -> float:
        if self._deterministic:
            return self._fixed["empty_travel"]
        route = self._get_route(origin, destination, "empty")
        base = route.distance_km / route.mean_speed_kph * 60.0
        multiplier = self._rng.triangular(
            self._config.travel_variability.multiplier_min,  # type: ignore[attr-defined]
            self._config.travel_variability.multiplier_mode,  # type: ignore[attr-defined]
            self._config.travel_variability.multiplier_max,  # type: ignore[attr-defined]
        )
        return base * multiplier

    def loaded_travel(self, origin: str, destination: str) -> float:
        if self._deterministic:
            return self._fixed["loaded_travel"]
        route = self._get_route(origin, destination, "loaded")
        base = route.distance_km / route.mean_speed_kph * 60.0
        multiplier = self._rng.triangular(
            self._config.travel_variability.multiplier_min,  # type: ignore[attr-defined]
            self._config.travel_variability.multiplier_mode,  # type: ignore[attr-defined]
            self._config.travel_variability.multiplier_max,  # type: ignore[attr-defined]
        )
        return base * multiplier

    def loading(self, shovel_id: str) -> float:
        if self._deterministic:
            return self._fixed["loading"]
        shovel = next(s for s in self._config.shovels if s.id == shovel_id)
        return float(
            self._rng.triangular(
                shovel.loading.minimum,
                shovel.loading.mode,
                shovel.loading.maximum,
            )
        )

    def dumping(self, dump_id: str) -> float:
        if self._deterministic:
            return self._fixed["dumping"]
        dump = next(d for d in self._config.dumps if d.id == dump_id)
        return float(
            self._rng.triangular(
                dump.dump.minimum,
                dump.dump.mode,
                dump.dump.maximum,
            )
        )

    def payload(self) -> float:
        if self._deterministic:
            return self._fixed["payload"]
        f = self._config.fleet
        return float(
            np.clip(
                self._rng.normal(f.payload_mean, f.payload_std),
                f.payload_min,
                f.payload_max,
            )
        )

    def _get_route(self, origin: str, destination: str, load_state: str):
        for r in self._config.routes:
            if (
                r.origin == origin
                and r.destination == destination
                and r.load_state == load_state
            ):
                return r
        raise ValueError(
            f"No {load_state} route from {origin} to {destination}"
        )


# ---------------------------------------------------------------------------
# Simulation model
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    event_log: EventLog
    total_production_tonnes: float
    completed_trips: int
    policy: str = "fixed"


class MinimalSimulation:
    """One truck, one shovel, one dump — Day 3 scope."""

    def __init__(
        self,
        config: ScenarioConfig,
        sampler: Sampler,
        shovel_id: str = "S1",
        dump_id: str = "D1",
        policy: str = "fixed",
    ) -> None:
        self._config = config
        self._sampler = sampler
        self._shovel_id = shovel_id
        self._dump_id = dump_id
        self._policy = policy
        self._event_log = EventLog()
        self._total_tonnes = 0.0
        self._completed_trips = 0

    def run(self) -> SimulationResult:
        env = simpy.Environment()
        shovel_resource = simpy.Resource(env, capacity=1)
        dump_resource = simpy.Resource(env, capacity=1)

        end_time = self._config.simulation.duration_minutes

        env.process(
            self._truck_process(
                env=env,
                truck_id="T01",
                shovel_resource=shovel_resource,
                dump_resource=dump_resource,
                end_time=end_time,
            )
        )
        env.run(until=end_time + 1e-9)

        return SimulationResult(
            event_log=self._event_log,
            total_production_tonnes=self._total_tonnes,
            completed_trips=self._completed_trips,
            policy=self._policy,
        )

    def _log(self, env: simpy.Environment, event_type: str, **kwargs: Any) -> None:
        self._event_log.log(
            sim_time_min=round(env.now, 4),
            event_type=event_type,
            policy=self._policy,
            **kwargs,
        )

    def _truck_process(
        self,
        env: simpy.Environment,
        truck_id: str,
        shovel_resource: simpy.Resource,
        dump_resource: simpy.Resource,
        end_time: float,
    ):
        current_location = self._dump_id  # trucks start at dump

        while env.now < end_time:
            # ── 1. Dispatch ──────────────────────────────────────────────
            self._log(
                env, "DISPATCH",
                truck_id=truck_id,
                origin=current_location,
                destination=self._shovel_id,
            )

            # ── 2. Empty travel ──────────────────────────────────────────
            empty_duration = self._sampler.empty_travel(
                current_location, self._shovel_id
            )
            self._log(
                env, "EMPTY_TRAVEL_START",
                truck_id=truck_id,
                origin=current_location,
                destination=self._shovel_id,
                duration_min=round(empty_duration, 4),
            )
            yield env.timeout(empty_duration)
            self._log(
                env, "EMPTY_TRAVEL_END",
                truck_id=truck_id,
                origin=current_location,
                destination=self._shovel_id,
            )

            # ── 3. Queue and load ────────────────────────────────────────
            queue_start = env.now
            self._log(
                env, "QUEUE_FOR_SHOVEL",
                truck_id=truck_id,
                origin=self._shovel_id,
                destination=self._shovel_id,
                shovel_id=self._shovel_id,
                queue_length=len(shovel_resource.queue),
            )
            with shovel_resource.request() as req:
                yield req
                queue_wait = env.now - queue_start

                loading_duration = self._sampler.loading(self._shovel_id)
                self._log(
                    env, "LOADING_START",
                    truck_id=truck_id,
                    origin=self._shovel_id,
                    destination=self._shovel_id,
                    shovel_id=self._shovel_id,
                    duration_min=round(loading_duration, 4),
                    queue_length=0,
                )
                yield env.timeout(loading_duration)
                self._log(
                    env, "LOADING_END",
                    truck_id=truck_id,
                    origin=self._shovel_id,
                    destination=self._dump_id,
                    shovel_id=self._shovel_id,
                    duration_min=round(loading_duration, 4),
                )

            # ── 4. Loaded travel ─────────────────────────────────────────
            loaded_duration = self._sampler.loaded_travel(
                self._shovel_id, self._dump_id
            )
            self._log(
                env, "LOADED_TRAVEL_START",
                truck_id=truck_id,
                origin=self._shovel_id,
                destination=self._dump_id,
                duration_min=round(loaded_duration, 4),
            )
            yield env.timeout(loaded_duration)
            self._log(
                env, "LOADED_TRAVEL_END",
                truck_id=truck_id,
                origin=self._shovel_id,
                destination=self._dump_id,
            )

            # ── 5. Queue and dump ────────────────────────────────────────
            dump_queue_start = env.now
            self._log(
                env, "QUEUE_FOR_DUMP",
                truck_id=truck_id,
                origin=self._dump_id,
                destination=self._dump_id,
                dump_id=self._dump_id,
                queue_length=len(dump_resource.queue),
            )
            with dump_resource.request() as req:
                yield req

                dump_duration = self._sampler.dumping(self._dump_id)
                payload = self._sampler.payload()

                self._log(
                    env, "DUMPING_START",
                    truck_id=truck_id,
                    origin=self._dump_id,
                    destination=self._dump_id,
                    dump_id=self._dump_id,
                    duration_min=round(dump_duration, 4),
                    payload_tonnes=round(payload, 4),
                )
                yield env.timeout(dump_duration)

                if env.now <= end_time + 1e-9:
                    self._total_tonnes += payload
                    self._completed_trips += 1

                self._log(
                    env, "DUMPING_END",
                    truck_id=truck_id,
                    origin=self._dump_id,
                    destination=self._dump_id,
                    dump_id=self._dump_id,
                    payload_tonnes=round(payload, 4),
                )

            current_location = self._dump_id
