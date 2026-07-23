"""Tests for the simulation engine — Day 3 and Day 4 scope.

Covers:
- correct event order for a deterministic one-truck cycle
- expected number of completed trips and production in deterministic mode
- hand-validation: 1 truck, cycle=17 min, duration=51 min → 3 trips, 300 tonnes
- multiple trucks: resource capacity (no simultaneous loading)
- queue ordering and waiting time recording
- all trucks complete at least one trip
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
def deterministic_result_1truck():
    """One truck, deterministic, duration=51 min."""
    config = load_scenario(BASE_SCENARIO, ROUTES)
    from dataclasses import replace
    sim = replace(config.simulation, duration_minutes=51.0)
    config = replace(config, simulation=sim)

    rng = np.random.default_rng(config.simulation.seed)
    sampler = Sampler(config=config, rng=rng)
    sampler.set_deterministic(DETERMINISTIC_VALUES)

    model = MinimalSimulation(
        config=config,
        sampler=sampler,
        shovel_id="S1",
        dump_id="D1",
        policy="fixed",
        number_of_trucks=1,
    )
    return model.run()


@pytest.fixture
def deterministic_result_3trucks():
    """Three trucks, deterministic, duration=120 min."""
    config = load_scenario(BASE_SCENARIO, ROUTES)
    from dataclasses import replace
    sim = replace(config.simulation, duration_minutes=120.0)
    config = replace(config, simulation=sim)

    rng = np.random.default_rng(config.simulation.seed)
    sampler = Sampler(config=config, rng=rng)
    sampler.set_deterministic(DETERMINISTIC_VALUES)

    model = MinimalSimulation(
        config=config,
        sampler=sampler,
        shovel_id="S1",
        dump_id="D1",
        policy="fixed",
        number_of_trucks=3,
    )
    return model.run()


# ── Day 3 tests (still valid) ────────────────────────────────────────────────

def test_deterministic_completed_trips(deterministic_result_1truck):
    """Hand check: cycle=17 min, duration=51 min → exactly 3 trips."""
    assert deterministic_result_1truck.completed_trips == 3


def test_deterministic_total_production(deterministic_result_1truck):
    """Hand check: 3 trips × 100 tonnes = 300 tonnes."""
    assert deterministic_result_1truck.total_production_tonnes == pytest.approx(300.0)


def test_event_order_first_cycle(deterministic_result_1truck):
    """First cycle must follow the documented state-machine order."""
    events = [r["event_type"] for r in deterministic_result_1truck.event_log.records]
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


def test_event_times_are_nonnegative(deterministic_result_1truck):
    for r in deterministic_result_1truck.event_log.records:
        assert r["sim_time_min"] >= 0.0


def test_event_times_are_nondecreasing(deterministic_result_1truck):
    times = [r["sim_time_min"] for r in deterministic_result_1truck.event_log.records]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1]


def test_cycle_duration_is_correct(deterministic_result_1truck):
    """Each complete cycle must take exactly 17 minutes."""
    dispatch_times = [
        r["sim_time_min"]
        for r in deterministic_result_1truck.event_log.records
        if r["event_type"] == "DISPATCH" and r.get("truck_id") == "T01"
    ]
    assert dispatch_times[0] == pytest.approx(0.0)
    assert dispatch_times[1] == pytest.approx(17.0)
    assert dispatch_times[2] == pytest.approx(34.0)


# ── Day 4 tests ──────────────────────────────────────────────────────────────

def test_multiple_trucks_are_created(deterministic_result_3trucks):
    """All three trucks must appear in the event log."""
    truck_ids = {
        r["truck_id"]
        for r in deterministic_result_3trucks.event_log.records
        if "truck_id" in r
    }
    assert "T01" in truck_ids
    assert "T02" in truck_ids
    assert "T03" in truck_ids


def test_no_simultaneous_loading(deterministic_result_3trucks):
    """Shovel must never serve two trucks at the same time."""
    loading_intervals: list[tuple[float, float]] = []
    start_time = None

    for r in deterministic_result_3trucks.event_log.records:
        if r["event_type"] == "LOADING_START":
            start_time = r["sim_time_min"]
        elif r["event_type"] == "LOADING_END" and start_time is not None:
            loading_intervals.append((start_time, r["sim_time_min"]))
            start_time = None

    # Check no two intervals overlap
    for i, (s1, e1) in enumerate(loading_intervals):
        for j, (s2, e2) in enumerate(loading_intervals):
            if i >= j:
                continue
            overlap = s1 < e2 and s2 < e1
            assert not overlap, (
                f"Simultaneous loading detected: "
                f"interval {i} [{s1}, {e1}] overlaps interval {j} [{s2}, {e2}]"
            )


def test_queue_length_is_nonnegative(deterministic_result_3trucks):
    """Queue lengths recorded in events must always be nonnegative."""
    for r in deterministic_result_3trucks.event_log.records:
        if "queue_length" in r:
            assert r["queue_length"] >= 0


def test_queue_wait_is_nonnegative(deterministic_result_3trucks):
    """Waiting times must always be nonnegative."""
    for r in deterministic_result_3trucks.event_log.records:
        if "queue_wait_min" in r:
            assert r["queue_wait_min"] >= 0.0


def test_all_trucks_complete_at_least_one_trip(deterministic_result_3trucks):
    """Every truck must complete at least one trip in 120 minutes."""
    for truck_id, count in deterministic_result_3trucks.truck_trip_counts.items():
        assert count >= 1, f"{truck_id} completed no trips"


def test_production_matches_trip_counts(deterministic_result_3trucks):
    """Total production must equal completed_trips × payload."""
    expected = deterministic_result_3trucks.completed_trips * 100.0
    assert deterministic_result_3trucks.total_production_tonnes == pytest.approx(expected)


def test_stochastic_mode_multiple_trucks():
    """Stochastic mode with 6 trucks must complete more trips than 1 truck."""
    config = load_scenario(BASE_SCENARIO, ROUTES)

    rng1 = np.random.default_rng(config.simulation.seed)
    sampler1 = Sampler(config=config, rng=rng1)
    result1 = MinimalSimulation(
        config=config, sampler=sampler1,
        shovel_id="S1", dump_id="D1",
        number_of_trucks=1,
    ).run()

    rng6 = np.random.default_rng(config.simulation.seed)
    sampler6 = Sampler(config=config, rng=rng6)
    result6 = MinimalSimulation(
        config=config, sampler=sampler6,
        shovel_id="S1", dump_id="D1",
        number_of_trucks=6,
    ).run()

    assert result6.completed_trips > result1.completed_trips
