"""Tests for Day 5 — full multi-shovel, multi-dump network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from truck_shovel_dt.config import load_scenario
from truck_shovel_dt.simulation import TruckShovelSimulation, Sampler

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


@pytest.fixture
def full_network_result():
    """6 trucks, 2 shovels, 2 dumps, stochastic, full 480-min shift."""
    config = load_scenario(BASE_SCENARIO, ROUTES)
    rng = np.random.default_rng(config.simulation.seed)
    sampler = Sampler(config=config, rng=rng)
    model = TruckShovelSimulation(config=config, sampler=sampler)
    return model.run()


@pytest.fixture
def deterministic_full_result():
    """6 trucks, 2 shovels, 2 dumps, deterministic, 120-min shift."""
    config = load_scenario(BASE_SCENARIO, ROUTES)
    from dataclasses import replace
    sim = replace(config.simulation, duration_minutes=120.0)
    config = replace(config, simulation=sim)
    rng = np.random.default_rng(config.simulation.seed)
    sampler = Sampler(config=config, rng=rng)
    sampler.set_deterministic(DETERMINISTIC_VALUES)
    model = TruckShovelSimulation(config=config, sampler=sampler)
    return model.run()


def test_full_shift_completes_without_deadlock(full_network_result):
    """Simulation must complete and produce a positive trip count."""
    assert full_network_result.completed_trips > 0
    assert full_network_result.total_production_tonnes > 0


def test_all_six_trucks_appear(full_network_result):
    """All 6 trucks must appear in the event log."""
    truck_ids = {
        r["truck_id"]
        for r in full_network_result.event_log.records
        if "truck_id" in r
    }
    for i in range(1, 7):
        assert f"T{i:02d}" in truck_ids


def test_both_shovels_used(full_network_result):
    """Both shovels must be used for loading."""
    shovel_ids = {
        r["shovel_id"]
        for r in full_network_result.event_log.records
        if r["event_type"] == "LOADING_START"
    }
    assert "S1" in shovel_ids
    assert "S2" in shovel_ids


def test_both_dumps_used(full_network_result):
    """Both dumps must receive at least one truck."""
    dump_ids = {
        r["dump_id"]
        for r in full_network_result.event_log.records
        if r["event_type"] == "DUMPING_START"
    }
    assert "D1" in dump_ids
    assert "D2" in dump_ids


def test_no_simultaneous_loading_per_shovel(full_network_result):
    """Each shovel must serve at most one truck at a time."""
    for shovel_id in ["S1", "S2"]:
        intervals: list[tuple[float, float]] = []
        start = None
        for r in full_network_result.event_log.records:
            if r["event_type"] == "LOADING_START" and r.get("shovel_id") == shovel_id:
                start = r["sim_time_min"]
            elif r["event_type"] == "LOADING_END" and r.get("shovel_id") == shovel_id:
                if start is not None:
                    intervals.append((start, r["sim_time_min"]))
                    start = None

        for i, (s1, e1) in enumerate(intervals):
            for j, (s2, e2) in enumerate(intervals):
                if i >= j:
                    continue
                assert not (s1 < e2 and s2 < e1), (
                    f"Shovel {shovel_id}: simultaneous loading at "
                    f"[{s1},{e1}] and [{s2},{e2}]"
                )


def test_production_equals_sum_of_payloads(full_network_result):
    """Total production must equal sum of all dumped payloads."""
    dumped = sum(
        r["payload_tonnes"]
        for r in full_network_result.event_log.records
        if r["event_type"] == "DUMPING_END"
    )
    assert full_network_result.total_production_tonnes == pytest.approx(dumped, rel=1e-4)


def test_all_trucks_complete_trips_in_full_shift(full_network_result):
    """Every truck must complete at least one trip in a full 480-min shift."""
    for truck_id, count in full_network_result.truck_trip_counts.items():
        assert count >= 1, f"{truck_id} completed no trips"


def test_fixed_assignment_trucks_stay_assigned(deterministic_full_result):
    """In fixed policy, each truck must load only from its assigned shovel."""
    # T01, T03, T05 → S1 (index % 2 == 0)
    # T02, T04, T06 → S2 (index % 2 == 1)
    expected = {
        "T01": "S1", "T02": "S2", "T03": "S1",
        "T04": "S2", "T05": "S1", "T06": "S2",
    }
    for r in deterministic_full_result.event_log.records:
        if r["event_type"] == "LOADING_START":
            truck = r.get("truck_id")
            shovel = r.get("shovel_id")
            if truck in expected:
                assert shovel == expected[truck], (
                    f"{truck} loaded at {shovel} but expected {expected[truck]}"
                )


def test_event_log_to_dataframe(full_network_result):
    """Event log must convert to a DataFrame with required columns."""
    df = full_network_result.event_log.to_dataframe()
    required = {"sim_time_min", "event_type", "policy", "truck_id"}
    assert required.issubset(df.columns)
    assert len(df) > 0


def test_save_and_reload_summary(full_network_result, tmp_path):
    """Summary JSON must save and reload correctly."""
    import json
    path = tmp_path / "summary.json"
    full_network_result.save_summary(path)
    with open(path) as f:
        data = json.load(f)
    assert data["completed_trips"] == full_network_result.completed_trips
    assert data["policy"] == "fixed"
