"""Tests for the KPI calculation module — Day 6.

Covers:
- production KPIs match event log payloads
- warm-up exclusion works correctly
- utilizations are between 0 and 1
- time accounting consistency
- truck and resource KPI tables have correct structure
- tonnes per operating hour calculation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from truck_shovel_dt.config import load_scenario
from truck_shovel_dt.metrics import KPICalculator
from truck_shovel_dt.simulation import Sampler, TruckShovelSimulation

BASE_SCENARIO = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "base_scenario.json"
ROUTES = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "routes.csv"


@pytest.fixture(scope="module")
def base_result():
    config = load_scenario(BASE_SCENARIO, ROUTES)
    rng = np.random.default_rng(config.simulation.seed)
    sampler = Sampler(config=config, rng=rng)
    return TruckShovelSimulation(config=config, sampler=sampler).run()


@pytest.fixture(scope="module")
def base_config():
    return load_scenario(BASE_SCENARIO, ROUTES)


@pytest.fixture(scope="module")
def calculator(base_result, base_config):
    df = base_result.event_log.to_dataframe()
    return KPICalculator(
        event_log=df,
        warmup_minutes=base_config.simulation.warmup_minutes,
        simulation_duration_minutes=base_config.simulation.duration_minutes,
    )


@pytest.fixture(scope="module")
def kpis(calculator):
    return calculator.calculate()


# ── Production KPIs ───────────────────────────────────────────────────────


def test_production_is_positive(kpis):
    assert kpis.production.total_production_tonnes > 0


def test_completed_trips_is_positive(kpis):
    assert kpis.production.completed_trips > 0


def test_tonnes_per_hour_is_positive(kpis):
    assert kpis.production.tonnes_per_operating_hour > 0


def test_production_consistency(calculator):
    """Total production must equal sum of DUMPING_END payloads."""
    assert calculator.verify_production_consistency()


def test_warmup_exclusion_reduces_trips(base_result, base_config):
    """Post-warmup trip count must be less than total trip count."""
    df = base_result.event_log.to_dataframe()

    calc_no_warmup = KPICalculator(
        event_log=df,
        warmup_minutes=0.0,
        simulation_duration_minutes=base_config.simulation.duration_minutes,
    )
    calc_with_warmup = KPICalculator(
        event_log=df,
        warmup_minutes=base_config.simulation.warmup_minutes,
        simulation_duration_minutes=base_config.simulation.duration_minutes,
    )

    trips_no_warmup = calc_no_warmup.calculate().production.completed_trips
    trips_with_warmup = calc_with_warmup.calculate().production.completed_trips

    assert trips_with_warmup <= trips_no_warmup


# ── Cycle KPIs ────────────────────────────────────────────────────────────


def test_cycle_time_is_positive(kpis):
    assert kpis.cycle.mean_cycle_time_min > 0


def test_cycle_time_min_le_mean_le_max(kpis):
    assert kpis.cycle.min_cycle_time_min <= kpis.cycle.mean_cycle_time_min
    assert kpis.cycle.mean_cycle_time_min <= kpis.cycle.max_cycle_time_min


# ── Queue KPIs ────────────────────────────────────────────────────────────


def test_queue_waits_are_nonnegative(kpis):
    assert kpis.queue.mean_shovel_queue_wait_min >= 0.0
    assert kpis.queue.mean_dump_queue_wait_min >= 0.0


# ── Utilization KPIs ──────────────────────────────────────────────────────


def test_truck_utilizations_between_0_and_1(kpis):
    for truck_id, util in kpis.utilization.truck_utilization.items():
        assert 0.0 <= util <= 1.0, f"{truck_id} utilization {util} out of range"


def test_shovel_utilizations_between_0_and_1(kpis):
    for shovel_id, util in kpis.utilization.shovel_utilization.items():
        assert 0.0 <= util <= 1.0, f"{shovel_id} utilization {util} out of range"


def test_shovel_idle_time_nonnegative(kpis):
    for shovel_id, idle in kpis.utilization.shovel_idle_time_min.items():
        assert idle >= 0.0, f"{shovel_id} idle time {idle} is negative"


def test_time_accounting_consistency(calculator):
    assert calculator.verify_time_accounting()


def test_mean_truck_utilization_between_0_and_1(kpis):
    assert 0.0 <= kpis.utilization.mean_truck_utilization <= 1.0


# ── Tables ────────────────────────────────────────────────────────────────


def test_truck_kpi_table_has_all_trucks(calculator):
    table = calculator.truck_kpi_table()
    assert len(table) == 6
    assert set(table["truck_id"]) == {f"T{i:02d}" for i in range(1, 7)}


def test_truck_kpi_table_columns(calculator):
    table = calculator.truck_kpi_table()
    required = {
        "truck_id",
        "completed_trips",
        "total_tonnes",
        "mean_shovel_wait_min",
        "utilization",
    }
    assert required.issubset(table.columns)


def test_resource_kpi_table_has_both_shovels(calculator):
    table = calculator.resource_kpi_table()
    assert set(table["resource_id"]) == {"S1", "S2"}


def test_resource_kpi_table_columns(calculator):
    table = calculator.resource_kpi_table()
    required = {"resource_id", "resource_type", "completed_loads", "utilization", "idle_time_min"}
    assert required.issubset(table.columns)


# ── to_dict ───────────────────────────────────────────────────────────────


def test_kpi_summary_to_dict(kpis):
    d = kpis.to_dict()
    assert "total_production_tonnes" in d
    assert "tonnes_per_operating_hour" in d
    assert "completed_trips" in d
    assert "mean_cycle_time_min" in d
    assert d["total_production_tonnes"] > 0
