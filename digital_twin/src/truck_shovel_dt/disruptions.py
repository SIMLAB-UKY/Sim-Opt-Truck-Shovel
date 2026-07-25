"""Disruption processes for the Adaptive Truck-Shovel Digital Twin.

Day 9 scope: shovel failure and repair processes.

Rerouting rule (documented per Section 10.10 of project plan):
- If a shovel fails BEFORE a truck joins its queue, the truck
  immediately requests a new assignment from the dispatcher.
- If a shovel fails WHILE a truck is already waiting in queue,
  the truck is released from the queue and immediately reroutes
  (simplified MVP rule — documented here).

This module provides:
- ShovelAvailability: tracks which shovels are available
- shovel_disruption_process: SimPy generator that triggers failures/repairs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import simpy


# ---------------------------------------------------------------------------
# Availability tracker
# ---------------------------------------------------------------------------

@dataclass
class ShovelAvailability:
    """Tracks failure/repair state for all shovels."""

    shovel_ids: list[str]
    _available: dict[str, bool] = field(init=False)

    def __post_init__(self) -> None:
        self._available = {sid: True for sid in self.shovel_ids}

    def is_available(self, shovel_id: str) -> bool:
        return self._available.get(shovel_id, False)

    def set_failed(self, shovel_id: str) -> None:
        self._available[shovel_id] = False

    def set_repaired(self, shovel_id: str) -> None:
        self._available[shovel_id] = True

    def available_shovels(self) -> list[str]:
        return [sid for sid, avail in self._available.items() if avail]

    def all_available(self) -> bool:
        return all(self._available.values())


# ---------------------------------------------------------------------------
# Disruption process
# ---------------------------------------------------------------------------

def shovel_disruption_process(
    env: simpy.Environment,
    shovel_id: str,
    mtbf_minutes: float,
    repair_min: float,
    repair_mode: float,
    repair_max: float,
    availability: ShovelAvailability,
    rng: np.random.Generator,
    event_log_fn: Callable[..., None],
    end_time: float,
) -> Any:
    """SimPy generator process that models shovel failure and repair cycles.

    Parameters
    ----------
    env           : SimPy environment
    shovel_id     : ID of the shovel this process controls
    mtbf_minutes  : mean time between failures (exponential distribution)
    repair_min    : triangular repair duration minimum
    repair_mode   : triangular repair duration mode
    repair_max    : triangular repair duration maximum
    availability  : shared availability tracker (mutated in place)
    rng           : numpy random generator
    event_log_fn  : callable(sim_time_min, event_type, shovel_id, **kwargs)
    end_time      : simulation end time
    """
    while env.now < end_time:
        # Draw time to next failure
        time_to_failure = rng.exponential(scale=mtbf_minutes)
        yield env.timeout(time_to_failure)

        if env.now >= end_time:
            break

        # Shovel fails
        availability.set_failed(shovel_id)
        event_log_fn(
            sim_time_min=round(env.now, 4),
            event_type="SHOVEL_FAILED",
            shovel_id=shovel_id,
            notes=f"Shovel {shovel_id} failed at t={env.now:.2f}",
        )

        # Draw repair duration
        repair_duration = rng.triangular(repair_min, repair_mode, repair_max)
        event_log_fn(
            sim_time_min=round(env.now, 4),
            event_type="SHOVEL_REPAIR_START",
            shovel_id=shovel_id,
            duration_min=round(repair_duration, 4),
            notes=f"Repair started, estimated {repair_duration:.1f} min",
        )

        yield env.timeout(repair_duration)

        if env.now > end_time:
            break

        # Shovel repaired
        availability.set_repaired(shovel_id)
        event_log_fn(
            sim_time_min=round(env.now, 4),
            event_type="SHOVEL_REPAIRED",
            shovel_id=shovel_id,
            notes=f"Shovel {shovel_id} repaired at t={env.now:.2f}",
        )
