"""Tests for scenario configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from truck_shovel_dt.config import ConfigError, load_scenario

BASE_SCENARIO = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "base_scenario.json"
ROUTES = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "routes.csv"


def test_base_scenario_loads_and_validates():
    config = load_scenario(BASE_SCENARIO, ROUTES)
    assert config.scenario_name == "base_balanced"
    assert config.fleet.number_of_trucks == 6
    assert len(config.shovels) == 2
    assert len(config.dumps) == 2
    assert len(config.routes) == 8


def test_missing_scenario_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_scenario(tmp_path / "does_not_exist.json")


def test_negative_duration_rejected(tmp_path):
    with BASE_SCENARIO.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["simulation"]["duration_minutes"] = -480

    bad_path = tmp_path / "bad_scenario.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="duration_minutes must be positive"):
        load_scenario(bad_path)


def test_warmup_greater_than_duration_rejected(tmp_path):
    with BASE_SCENARIO.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["simulation"]["warmup_minutes"] = 999

    bad_path = tmp_path / "bad_scenario.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="warmup_minutes must be less than"):
        load_scenario(bad_path)


def test_invalid_triangular_bounds_rejected(tmp_path):
    with BASE_SCENARIO.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["shovels"][0]["loading_min"] = 10.0

    bad_path = tmp_path / "bad_scenario.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="triangular distribution"):
        load_scenario(bad_path)


def test_duplicate_shovel_ids_rejected(tmp_path):
    with BASE_SCENARIO.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["shovels"][1]["id"] = raw["shovels"][0]["id"]

    bad_path = tmp_path / "bad_scenario.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="Shovel ids must be unique"):
        load_scenario(bad_path)


def test_empty_shovel_id_rejected(tmp_path):
    with BASE_SCENARIO.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["shovels"][0]["id"] = ""

    bad_path = tmp_path / "bad_scenario.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="Shovel id must not be empty"):
        load_scenario(bad_path)


def test_incomplete_route_matrix_rejected(tmp_path):
    bad_routes = tmp_path / "routes.csv"
    bad_routes.write_text(
        "origin,destination,load_state,distance_km,mean_speed_kph\n"
        "D1,S1,empty,2.4,30\n"
        "D1,S2,empty,3.1,28\n"
        "D2,S1,empty,2.8,29\n"
        "D2,S2,empty,2.0,31\n"
        "S1,D1,loaded,2.4,22\n"
        "S1,D2,loaded,2.8,21\n"
        "S2,D1,loaded,3.1,20\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Missing loaded route"):
        load_scenario(BASE_SCENARIO, bad_routes)


def test_ewma_alpha_out_of_range_rejected(tmp_path):
    with BASE_SCENARIO.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["learning"]["ewma_alpha"] = 1.5

    bad_path = tmp_path / "bad_scenario.json"
    bad_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="ewma_alpha must be in"):
        load_scenario(bad_path)
