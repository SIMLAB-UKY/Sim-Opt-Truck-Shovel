"""Tests for EWMA estimators and AdaptiveECT policy — Day 8.

Covers:
- EWMA update equation correctness
- Estimate initialization from scenario values
- Observation counting and reliability flag
- AdaptiveECT: scoring favors lower queue + faster shovel
- AdaptiveECT: excludes unavailable shovels
- AdaptiveECT: switching penalty applied correctly
- AdaptiveECT: score changes when queue or estimate changes
- Behavioral validation: S1 slow → assignments shift to S2
"""

from __future__ import annotations

import pytest

from truck_shovel_dt.estimators import EWMAEstimator, EstimatorRegistry
from truck_shovel_dt.dispatch import (
    AdaptiveECT,
    DumpState,
    RouteState,
    ShovelState,
    SystemState,
    build_system_state,
)


# ---------------------------------------------------------------------------
# EWMAEstimator tests
# ---------------------------------------------------------------------------

def test_ewma_initial_estimate():
    e = EWMAEstimator(name="test", initial_estimate=5.0, alpha=0.2)
    assert e.estimate == pytest.approx(5.0)


def test_ewma_update_equation():
    """mu_hat = alpha * x + (1 - alpha) * mu_hat_prev"""
    e = EWMAEstimator(name="test", initial_estimate=5.0, alpha=0.2)
    before, after = e.update(7.0)
    expected = 0.2 * 7.0 + 0.8 * 5.0
    assert before == pytest.approx(5.0)
    assert after == pytest.approx(expected)


def test_ewma_multiple_updates():
    e = EWMAEstimator(name="test", initial_estimate=4.0, alpha=0.2)
    e.update(6.0)
    e.update(6.0)
    # After 2 updates of 6.0 starting from 4.0:
    # step1: 0.2*6 + 0.8*4 = 4.4
    # step2: 0.2*6 + 0.8*4.4 = 4.72
    assert e.estimate == pytest.approx(4.72)


def test_ewma_observation_count():
    e = EWMAEstimator(name="test", initial_estimate=5.0, alpha=0.2)
    assert e.observation_count == 0
    e.update(5.0)
    assert e.observation_count == 1
    e.update(5.0)
    assert e.observation_count == 2


def test_ewma_reliability_flag():
    e = EWMAEstimator(
        name="test", initial_estimate=5.0, alpha=0.2, minimum_observations=3
    )
    assert not e.is_reliable
    e.update(5.0)
    e.update(5.0)
    assert not e.is_reliable
    e.update(5.0)
    assert e.is_reliable


def test_ewma_reset():
    e = EWMAEstimator(name="test", initial_estimate=5.0, alpha=0.2)
    e.update(10.0)
    e.reset()
    assert e.estimate == pytest.approx(5.0)
    assert e.observation_count == 0


def test_ewma_alpha_one_replaces_estimate():
    """With alpha=1, estimate always equals the last observation."""
    e = EWMAEstimator(name="test", initial_estimate=5.0, alpha=1.0)
    e.update(9.0)
    assert e.estimate == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# EstimatorRegistry tests
# ---------------------------------------------------------------------------

def make_registry() -> EstimatorRegistry:
    reg = EstimatorRegistry(alpha=0.2, minimum_observations=3)
    reg.init_loading("S1", 4.5)
    reg.init_loading("S2", 5.0)
    reg.init_dumping("D1", 1.2)
    reg.init_dumping("D2", 1.5)
    reg.init_empty_travel("D1", "S1", 4.8)
    reg.init_empty_travel("D1", "S2", 6.2)
    reg.init_empty_travel("D2", "S1", 5.6)
    reg.init_empty_travel("D2", "S2", 3.9)
    reg.init_loaded_travel("S1", "D1", 6.5)
    reg.init_loaded_travel("S1", "D2", 7.6)
    reg.init_loaded_travel("S2", "D1", 9.3)
    reg.init_loaded_travel("S2", "D2", 5.2)
    return reg


def test_registry_get_loading():
    reg = make_registry()
    assert reg.get_loading("S1") == pytest.approx(4.5)
    assert reg.get_loading("S2") == pytest.approx(5.0)


def test_registry_update_loading():
    reg = make_registry()
    before, after = reg.update_loading("S1", 6.0)
    assert before == pytest.approx(4.5)
    expected = 0.2 * 6.0 + 0.8 * 4.5
    assert after == pytest.approx(expected)


def test_registry_update_travel():
    reg = make_registry()
    before, after = reg.update_empty_travel("D1", "S1", 7.0)
    assert before == pytest.approx(4.8)


def test_registry_summary_keys():
    reg = make_registry()
    summary = reg.summary()
    assert "loading" in summary
    assert "dumping" in summary
    assert "empty_travel" in summary
    assert "loaded_travel" in summary
    assert "S1" in summary["loading"]


# ---------------------------------------------------------------------------
# AdaptiveECT policy tests
# ---------------------------------------------------------------------------

def make_state_for_ect(
    s1_queue: int = 0,
    s2_queue: int = 0,
    s1_available: bool = True,
    s2_available: bool = True,
    s1_remaining: float = 0.0,
    s2_remaining: float = 0.0,
    current_location: str = "D1",
) -> SystemState:
    routes = [
        RouteState("D1", "S1", "empty", 4.8),
        RouteState("D1", "S2", "empty", 6.2),
        RouteState("D2", "S1", "empty", 5.6),
        RouteState("D2", "S2", "empty", 3.9),
        RouteState("S1", "D1", "loaded", 6.5),
        RouteState("S1", "D2", "loaded", 7.6),
        RouteState("S2", "D1", "loaded", 9.3),
        RouteState("S2", "D2", "loaded", 5.2),
    ]
    return build_system_state(
        current_location=current_location,
        shovel_states=[
            ShovelState("S1", s1_available, s1_queue, s1_remaining, 4.5),
            ShovelState("S2", s2_available, s2_queue, s2_remaining, 5.0),
        ],
        dump_states=[
            DumpState("D1", True, 0, 1.2),
            DumpState("D2", True, 0, 1.5),
        ],
        route_states=routes,
    )


def test_adaptive_ect_selects_lower_score_shovel():
    """S1 faster loading and shorter queue → should be selected."""
    reg = make_registry()
    policy = AdaptiveECT(estimators=reg, switch_penalty_minutes=0.0)
    state = make_state_for_ect(s1_queue=0, s2_queue=2)
    result = policy.choose_assignment(state, "T01")
    assert result.shovel_id == "S1"


def test_adaptive_ect_excludes_unavailable_shovel():
    """S1 unavailable → must select S2."""
    reg = make_registry()
    policy = AdaptiveECT(estimators=reg, switch_penalty_minutes=0.0)
    state = make_state_for_ect(s1_available=False)
    result = policy.choose_assignment(state, "T01")
    assert result.shovel_id == "S2"


def test_adaptive_ect_score_changes_with_queue():
    """Adding trucks to S1 queue must increase S1's score."""
    reg = make_registry()
    policy = AdaptiveECT(estimators=reg, switch_penalty_minutes=0.0)

    state_empty = make_state_for_ect(s1_queue=0, s2_queue=0)
    state_crowded = make_state_for_ect(s1_queue=5, s2_queue=0)

    result_empty = policy.choose_assignment(state_empty, "T01")
    result_crowded = policy.choose_assignment(state_crowded, "T02")

    # With S1 crowded, S2 should be preferred
    assert result_crowded.shovel_id == "S2"


def test_adaptive_ect_switching_penalty_applied():
    """Switching penalty must increase score when changing shovel."""
    reg = make_registry()
    policy_no_penalty = AdaptiveECT(
        estimators=reg,
        switch_penalty_minutes=0.0,
        last_shovel={"T01": "S2"},
    )
    policy_with_penalty = AdaptiveECT(
        estimators=reg,
        switch_penalty_minutes=100.0,  # large penalty to force staying
        last_shovel={"T01": "S2"},
    )
    state = make_state_for_ect(s1_queue=0, s2_queue=0)

    result_no_penalty = policy_no_penalty.choose_assignment(state, "T01")
    result_with_penalty = policy_with_penalty.choose_assignment(state, "T01")

    # With large penalty, truck should stay at S2
    assert result_with_penalty.shovel_id == "S2"


def test_adaptive_ect_explanation_contains_score():
    """Explanation must mention the score."""
    reg = make_registry()
    policy = AdaptiveECT(estimators=reg)
    state = make_state_for_ect()
    result = policy.choose_assignment(state, "T01")
    assert "Score=" in result.explanation or "score" in result.explanation.lower()


def test_adaptive_ect_score_shifts_after_slow_observations():
    """Behavioral validation: many slow S1 observations → assignments shift to S2."""
    reg = make_registry()
    policy = AdaptiveECT(estimators=reg, switch_penalty_minutes=0.0)

    # Feed many slow loading observations to S1
    for _ in range(20):
        reg.update_loading("S1", 12.0)  # very slow

    state = make_state_for_ect(s1_queue=0, s2_queue=0)
    result = policy.choose_assignment(state, "T01")

    # S1 is now much slower → S2 should be selected
    assert result.shovel_id == "S2", (
        f"Expected S2 after slow S1 observations but got {result.shovel_id}. "
        f"S1 loading estimate: {reg.get_loading('S1'):.2f}"
    )


def test_adaptive_ect_last_shovel_tracked():
    """After assignment, last_shovel must be updated."""
    reg = make_registry()
    last_shovel: dict[str, str] = {}
    policy = AdaptiveECT(
        estimators=reg,
        switch_penalty_minutes=0.0,
        last_shovel=last_shovel,
    )
    state = make_state_for_ect()
    result = policy.choose_assignment(state, "T01")
    assert "T01" in last_shovel
    assert last_shovel["T01"] == result.shovel_id
