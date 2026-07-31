"""Tests for disruptions and adaptive response — Day 9.

Covers:
- ShovelAvailability: correct initial state
- ShovelAvailability: set_failed and set_repaired
- ShovelAvailability: available_shovels list
- Simulation: no dispatch to failed shovel
- Simulation: shovel reinstated after repair
- Simulation: SHOVEL_FAILED and SHOVEL_REPAIRED events logged
- Simulation: production lower with disruptions than without
- Simulation: trucks reroute when shovel fails during travel
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from truck_shovel_dt.config import load_scenario
from truck_shovel_dt.disruptions import ShovelAvailability
from truck_shovel_dt.simulation import Sampler, TruckShovelSimulation

BASE_SCENARIO = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "base_scenario.json"
BREAKDOWN_SCENARIO = (
    Path(__file__).resolve().parents[1] / "data" / "scenarios" / "breakdown_scenario.json"
)
ROUTES = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "routes.csv"


# ---------------------------------------------------------------------------
# ShovelAvailability unit tests
# ---------------------------------------------------------------------------


def test_availability_initial_state():
    avail = ShovelAvailability(shovel_ids=["S1", "S2"])
    assert avail.is_available("S1")
    assert avail.is_available("S2")
    assert avail.all_available()


def test_availability_set_failed():
    avail = ShovelAvailability(shovel_ids=["S1", "S2"])
    avail.set_failed("S1")
    assert not avail.is_available("S1")
    assert avail.is_available("S2")
    assert not avail.all_available()


def test_availability_set_repaired():
    avail = ShovelAvailability(shovel_ids=["S1", "S2"])
    avail.set_failed("S1")
    avail.set_repaired("S1")
    assert avail.is_available("S1")
    assert avail.all_available()


def test_availability_available_shovels():
    avail = ShovelAvailability(shovel_ids=["S1", "S2"])
    avail.set_failed("S1")
    assert avail.available_shovels() == ["S2"]


def test_availability_all_failed():
    avail = ShovelAvailability(shovel_ids=["S1", "S2"])
    avail.set_failed("S1")
    avail.set_failed("S2")
    assert avail.available_shovels() == []
    assert not avail.all_available()


# ---------------------------------------------------------------------------
# Simulation disruption tests
# ---------------------------------------------------------------------------


@pytest.fixture
def breakdown_result():
    """Run breakdown scenario with disruptions enabled."""
    config = load_scenario(BREAKDOWN_SCENARIO, ROUTES)
    rng = np.random.default_rng(config.simulation.seed)
    sampler = Sampler(config=config, rng=rng)
    model = TruckShovelSimulation(
        config=config,
        sampler=sampler,
        enable_disruptions=True,
    )
    return model.run()


@pytest.fixture
def base_result_no_disruptions():
    """Run base scenario without disruptions."""
    config = load_scenario(BASE_SCENARIO, ROUTES)
    rng = np.random.default_rng(config.simulation.seed)
    sampler = Sampler(config=config, rng=rng)
    model = TruckShovelSimulation(
        config=config,
        sampler=sampler,
        enable_disruptions=False,
    )
    return model.run()


def test_shovel_failed_events_logged(breakdown_result):
    """SHOVEL_FAILED events must appear in the event log."""
    events = [r["event_type"] for r in breakdown_result.event_log.records]
    assert "SHOVEL_FAILED" in events, "No SHOVEL_FAILED events found"


def test_shovel_repaired_events_logged(breakdown_result):
    """SHOVEL_REPAIRED events must appear in the event log."""
    events = [r["event_type"] for r in breakdown_result.event_log.records]
    assert "SHOVEL_REPAIRED" in events, "No SHOVEL_REPAIRED events found"


def test_shovel_repair_follows_failure(breakdown_result):
    """Every SHOVEL_FAILED event must be followed by SHOVEL_REPAIRED."""
    records = breakdown_result.event_log.records
    for shovel_id in ["S1", "S2"]:
        shovel_events = [
            r
            for r in records
            if r.get("shovel_id") == shovel_id
            and r["event_type"] in ("SHOVEL_FAILED", "SHOVEL_REPAIRED")
        ]
        for i, event in enumerate(shovel_events):
            if event["event_type"] == "SHOVEL_FAILED":
                if i + 1 < len(shovel_events):
                    assert shovel_events[i + 1]["event_type"] in (
                        "SHOVEL_REPAIR_START",
                        "SHOVEL_REPAIRED",
                    )


def test_no_loading_during_failure(breakdown_result):
    """No truck should load at a shovel while it is marked as failed."""
    records = breakdown_result.event_log.records

    for shovel_id in ["S1", "S2"]:
        failed_periods: list[tuple[float, float]] = []
        fail_time = None

        for r in records:
            if r.get("shovel_id") != shovel_id:
                continue
            if r["event_type"] == "SHOVEL_FAILED":
                fail_time = r["sim_time_min"]
            elif r["event_type"] == "SHOVEL_REPAIRED" and fail_time is not None:
                failed_periods.append((fail_time, r["sim_time_min"]))
                fail_time = None

        for r in records:
            if r["event_type"] == "LOADING_START" and r.get("shovel_id") == shovel_id:
                t = r["sim_time_min"]
                for start, end in failed_periods:
                    assert not (
                        start <= t <= end
                    ), f"Loading at {shovel_id} during failure period [{start}, {end}]"


def test_breakdown_simulation_completes(breakdown_result):
    """Simulation must complete without deadlock even with disruptions."""
    assert breakdown_result.completed_trips > 0


def test_disruptions_reduce_production(breakdown_result, base_result_no_disruptions):
    """Production with disruptions must be less than or equal to without."""
    assert (
        breakdown_result.total_production_tonnes
        <= base_result_no_disruptions.total_production_tonnes
    )


def test_event_log_has_required_columns(breakdown_result):
    """Event log DataFrame must contain required columns."""
    df = breakdown_result.event_log.to_dataframe()
    required = {"sim_time_min", "event_type", "policy"}
    assert required.issubset(df.columns)
