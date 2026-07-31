"""Tests for the synthetic base-data generator.

Covers the acceptance check from Section 6.10:
- same seed must produce identical output files;
- samples must respect documented engineering bounds;
- required columns and row counts must be present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_base_data.py"
GENERATED_DIR = REPO_ROOT / "data" / "generated"


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("generate_base_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_base_data"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generated_once():
    module = _load_generator_module()
    module.main()
    return {
        "service": GENERATED_DIR / "service_time_samples.csv",
        "travel": GENERATED_DIR / "travel_time_samples.csv",
        "payload": GENERATED_DIR / "payload_samples.csv",
        "summary": GENERATED_DIR / "generation_summary.json",
    }


def test_output_files_exist(generated_once):
    for path in generated_once.values():
        assert path.exists(), f"Expected output file missing: {path}"


def test_service_time_columns_and_positivity(generated_once):
    df = pd.read_csv(generated_once["service"])
    expected_columns = {
        "sample_id",
        "resource_type",
        "resource_id",
        "activity",
        "duration_minutes",
    }
    assert expected_columns.issubset(df.columns)
    assert (df["duration_minutes"] > 0).all()
    assert len(df) == 2000  # 2 shovels + 2 dumps, 500 samples each


def test_travel_time_columns_and_positivity(generated_once):
    df = pd.read_csv(generated_once["travel"])
    expected_columns = {
        "sample_id",
        "origin",
        "destination",
        "load_state",
        "travel_minutes",
    }
    assert expected_columns.issubset(df.columns)
    assert (df["travel_minutes"] > 0).all()
    assert len(df) == 4000  # 8 routes, 500 samples each


def test_payload_within_documented_bounds(generated_once):
    df = pd.read_csv(generated_once["payload"])
    assert "payload_tonnes" in df.columns
    assert len(df) == 500
    assert (df["payload_tonnes"] >= 82).all()
    assert (df["payload_tonnes"] <= 102).all()


def test_shovel_loading_within_triangular_bounds(generated_once):
    df = pd.read_csv(generated_once["service"])
    s1 = df[(df["resource_id"] == "S1") & (df["activity"] == "loading")]
    s2 = df[(df["resource_id"] == "S2") & (df["activity"] == "loading")]
    assert (s1["duration_minutes"] >= 3.5).all()
    assert (s1["duration_minutes"] <= 6.0).all()
    assert (s2["duration_minutes"] >= 4.0).all()
    assert (s2["duration_minutes"] <= 6.5).all()


def test_reproducibility_same_seed_gives_identical_output(generated_once):
    service_before = pd.read_csv(generated_once["service"])
    travel_before = pd.read_csv(generated_once["travel"])
    payload_before = pd.read_csv(generated_once["payload"])

    module = _load_generator_module()
    module.main()

    service_after = pd.read_csv(generated_once["service"])
    travel_after = pd.read_csv(generated_once["travel"])
    payload_after = pd.read_csv(generated_once["payload"])

    pd.testing.assert_frame_equal(service_before, service_after)
    pd.testing.assert_frame_equal(travel_before, travel_after)
    pd.testing.assert_frame_equal(payload_before, payload_after)


def test_different_seed_changes_samples_but_stays_in_bounds(monkeypatch):
    module = _load_generator_module()
    original = pd.read_csv(GENERATED_DIR / "payload_samples.csv")

    monkeypatch.setattr(module, "SEED", 999999)
    module.main()
    changed = pd.read_csv(GENERATED_DIR / "payload_samples.csv")

    assert not changed["payload_tonnes"].equals(original["payload_tonnes"])
    assert (changed["payload_tonnes"] >= 82).all()
    assert (changed["payload_tonnes"] <= 102).all()

    # restore original seed
    monkeypatch.setattr(module, "SEED", 20260715)
    module.main()
