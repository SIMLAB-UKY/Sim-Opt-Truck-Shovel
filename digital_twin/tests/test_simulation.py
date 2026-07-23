"""Tests for the simulation engine — Day 3 scope.

Covers:
- correct event order for a deterministic one-truck cycle
- expected number of completed trips and production in deterministic mode
- hand-validation: 1 truck, 1 shovel, 1 dump, cycle=17 min, duration=51 min
  → exactly 3 completed trips and 300 tonnes (Section 13.2 of project plan)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from truck_shovel_dt.config import load_scenario
from truck_shovel_dt.simulation import MinimalSimulation, Sampler

BASE_SCENARIO = (
    Path(__file__).resolve().parents[1]
    / "data" / "scenarios" / "base_scenario.json"
)
ROUTES = (
    Path(__file__).resolve().parents[1]
    / "data" / "scenarios" / "routes.csv"
)

DETERMINISTIC_VALUES = {
    "empty_travel": 5.0,
    "loading": 4.0,
    "loaded_travel": 7.0,
    "dumping": 1.0,
    "payload": 100.0,
}

EXPECTED_EVENT_ORDER = [
    "DISPATCH",
    "EMPTY_TRAVEL_START",
    "EMPTY_TRAVEL_END",
    "QUEUE_FOR_SHOVEL",
    "LOADING_START",
    "LOADING_END",
    "LOADED_TRAVEL_START",
    "LOADED_TRAVEL_END",
    "QUEUE_FOR_DUMP",
    "DUMPING_START",
    "DUMPING_END",
]


@pytest.fixture
def deterministic_result():
    config = load_scenario(BASE_SCENARIO, ROUTES)
    rng = np.random.default_rng(config.simulation.seed)
    sampler = Sampler(config=config, rng=rng)
    sampler.set_deterministic(DETERMINISTIC_VALUES)

    # Override duration to exactly 51 minutes for hand validation
    from dataclasses import replace
    sim = replace(config.simulation, duration_minutes=51.0)
    config = replace(config, simulation=sim)

    model = MinimalSimulation(
        config=config,
        sampler=sampler,
        shovel_id="S1",
        dump_id="D1",
        policy="fixed",
    )
    return model.run()


def test_deterministic_completed_trips(deterministic_result):
    """Hand check: cycle=17 min, duration=51 min → exactly 3 trips."""
    assert deterministic_result.completed_trips == 3


def test_deterministic_total_production(deterministic_result):
    """Hand check: 3 trips × 100 tonnes = 300 tonnes."""
    assert deterministic_result.total_production_tonnes == pytest.approx(300.0)


def test_event_order_first_cycle(deterministic_result):
    """First cycle must follow the documented state-machine order."""
    events = [
        r["event_type"]
        for r in deterministic_result.event_log.records
    ]
    # Check that all expected events appear in order within the full log
    idx = 0
    for expected in EXPECTED_EVENT_ORDER:
        found = False
        while idx < len(events):
            if events[idx] == expected:
                idx += 1
                found = True
                break
            idx += 1
        assert found, f"Expected event '{expected}' not found in correct order"


def test_event_times_are_nonnegative(deterministic_result):
    for r in deterministic_result.event_log.records:
        assert r["sim_time_min"] >= 0.0


def test_event_times_are_nondecreasing(deterministic_result):
    times = [r["sim_time_min"] for r in deterministic_result.event_log.records]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1], (
            f"Event times not nondecreasing at index {i}: "
            f"{times[i-1]} -> {times[i]}"
        )


def test_cycle_duration_is_correct(deterministic_result):
    """Each complete cycle must take exactly 17 minutes."""
    dispatch_times = [
        r["sim_time_min"]
        for r in deterministic_result.event_log.records
        if r["event_type"] == "DISPATCH"
    ]
    # First dispatch at t=0, second at t=17, third at t=34
    assert dispatch_times[0] == pytest.approx(0.0)
    assert dispatch_times[1] == pytest.approx(17.0)
    assert dispatch_times[2] == pytest.approx(34.0)


def test_stochastic_mode_produces_positive_output():
    """Stochastic mode must complete at least one trip in a full shift."""
    config = load_scenario(BASE_SCENARIO, ROUTES)
    rng = np.random.default_rng(config.simulation.seed)
    sampler = Sampler(config=config, rng=rng)

    model = MinimalSimulation(
        config=config,
        sampler=sampler,
        shovel_id="S1",
        dump_id="D1",
        policy="fixed",
    )
    result = model.run()
    assert result.completed_trips > 0
    assert result.total_production_tonnes > 0
