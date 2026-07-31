"""Tests for dispatch policies — Day 7.

Covers:
- DispatchPolicy protocol compliance
- FixedAssignment: correct pre-shift assignment
- FixedAssignment: fallback when assigned shovel unavailable
- ShortestQueue: selects shovel with fewest waiting trucks
- ShortestQueue: tie-breaking is deterministic (travel time, then ID)
- ShortestQueue: excludes unavailable shovels
- ShortestQueue: selects best dump by loaded travel time
- Acceptance check: S1=3, S2=1 → ShortestQueue selects S2
- Acceptance check: S2 unavailable → ShortestQueue selects S1
"""

from __future__ import annotations

from truck_shovel_dt.dispatch import (
    Assignment,
    DispatchPolicy,
    DumpState,
    FixedAssignment,
    RouteState,
    ShortestQueue,
    ShovelState,
    SystemState,
    build_system_state,
)

# ---------------------------------------------------------------------------
# Helpers — fabricated state snapshots
# ---------------------------------------------------------------------------


def make_routes(
    s1_empty: float = 5.0,
    s2_empty: float = 7.0,
    s1_d1_loaded: float = 8.0,
    s1_d2_loaded: float = 9.0,
    s2_d1_loaded: float = 6.0,
    s2_d2_loaded: float = 7.0,
) -> list[RouteState]:
    return [
        RouteState("D1", "S1", "empty", s1_empty),
        RouteState("D1", "S2", "empty", s2_empty),
        RouteState("D2", "S1", "empty", s1_empty),
        RouteState("D2", "S2", "empty", s2_empty),
        RouteState("S1", "D1", "loaded", s1_d1_loaded),
        RouteState("S1", "D2", "loaded", s1_d2_loaded),
        RouteState("S2", "D1", "loaded", s2_d1_loaded),
        RouteState("S2", "D2", "loaded", s2_d2_loaded),
    ]


def make_state(
    s1_queue: int = 0,
    s2_queue: int = 0,
    s1_available: bool = True,
    s2_available: bool = True,
    s1_remaining: float = 0.0,
    s2_remaining: float = 0.0,
    current_location: str = "D1",
    routes: list[RouteState] | None = None,
) -> SystemState:
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
        route_states=routes or make_routes(),
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_fixed_assignment_implements_protocol():
    assignments = {"T01": ("S1", "D1"), "T02": ("S2", "D2")}
    policy = FixedAssignment(assignments)
    assert isinstance(policy, DispatchPolicy)
    assert policy.name == "fixed"


def test_shortest_queue_implements_protocol():
    policy = ShortestQueue()
    assert isinstance(policy, DispatchPolicy)
    assert policy.name == "shortest_queue"


# ---------------------------------------------------------------------------
# FixedAssignment tests
# ---------------------------------------------------------------------------


def test_fixed_assignment_returns_correct_shovel():
    assignments = {"T01": ("S1", "D1"), "T02": ("S2", "D2")}
    policy = FixedAssignment(assignments)
    state = make_state()
    result = policy.choose_assignment(state, "T01")
    assert result.shovel_id == "S1"
    assert result.dump_id == "D1"


def test_fixed_assignment_returns_assignment_object():
    assignments = {"T01": ("S1", "D1")}
    policy = FixedAssignment(assignments)
    state = make_state()
    result = policy.choose_assignment(state, "T01")
    assert isinstance(result, Assignment)
    assert result.explanation != ""


def test_fixed_assignment_fallback_when_shovel_unavailable():
    assignments = {"T01": ("S1", "D1")}
    policy = FixedAssignment(assignments)
    state = make_state(s1_available=False)
    result = policy.choose_assignment(state, "T01")
    assert result.shovel_id == "S2"
    assert "unavailable" in result.explanation.lower()


def test_fixed_assignment_from_config():
    routes = make_routes()
    policy = FixedAssignment.from_config(
        truck_ids=["T01", "T02", "T03", "T04"],
        shovel_ids=["S1", "S2"],
        dump_ids=["D1", "D2"],
        routes=routes,
    )
    state = make_state(routes=routes)
    r1 = policy.choose_assignment(state, "T01")
    r2 = policy.choose_assignment(state, "T02")
    assert r1.shovel_id == "S1"
    assert r2.shovel_id == "S2"


# ---------------------------------------------------------------------------
# ShortestQueue tests
# ---------------------------------------------------------------------------


def test_shortest_queue_selects_shorter_queue():
    """Acceptance check: S1=3, S2=1 → must select S2."""
    policy = ShortestQueue()
    state = make_state(s1_queue=3, s2_queue=1)
    result = policy.choose_assignment(state, "T01")
    assert result.shovel_id == "S2", f"Expected S2 (queue=1) but got {result.shovel_id}"


def test_shortest_queue_unavailable_shovel_excluded():
    """Acceptance check: S2 unavailable → must select S1."""
    policy = ShortestQueue()
    state = make_state(s1_queue=3, s2_queue=1, s2_available=False)
    result = policy.choose_assignment(state, "T01")
    assert result.shovel_id == "S1", f"S2 is unavailable; expected S1 but got {result.shovel_id}"


def test_shortest_queue_tie_broken_by_travel_time():
    """When queue lengths are equal, shorter travel time wins."""
    policy = ShortestQueue()
    # S1 travel = 5, S2 travel = 7 → S1 should win
    state = make_state(s1_queue=1, s2_queue=1, routes=make_routes(s1_empty=5.0, s2_empty=7.0))
    result = policy.choose_assignment(state, "T01")
    assert result.shovel_id == "S1"


def test_shortest_queue_tie_broken_by_id_when_equal_travel():
    """When queue and travel are equal, lexicographic ID wins (S1 < S2)."""
    policy = ShortestQueue()
    state = make_state(s1_queue=1, s2_queue=1, routes=make_routes(s1_empty=5.0, s2_empty=5.0))
    result = policy.choose_assignment(state, "T01")
    assert result.shovel_id == "S1"


def test_shortest_queue_selects_best_dump():
    """Dump with shortest loaded travel from chosen shovel is selected."""
    policy = ShortestQueue()
    # S1→D1 = 6, S1→D2 = 9 → D1 should be selected
    state = make_state(routes=make_routes(s1_d1_loaded=6.0, s1_d2_loaded=9.0))
    result = policy.choose_assignment(state, "T01")
    if result.shovel_id == "S1":
        assert result.dump_id == "D1"


def test_shortest_queue_explanation_contains_candidates():
    """Explanation must mention candidate shovels."""
    policy = ShortestQueue()
    state = make_state(s1_queue=2, s2_queue=0)
    result = policy.choose_assignment(state, "T01")
    assert "S1" in result.explanation
    assert "S2" in result.explanation


def test_shortest_queue_assignment_is_deterministic():
    """Same state must always produce same assignment."""
    policy = ShortestQueue()
    state = make_state(s1_queue=2, s2_queue=1)
    results = [policy.choose_assignment(state, "T01") for _ in range(10)]
    assert all(r.shovel_id == results[0].shovel_id for r in results)
    assert all(r.dump_id == results[0].dump_id for r in results)


def test_score_is_nonnegative():
    """Assignment score must always be nonnegative."""
    policy = ShortestQueue()
    state = make_state(s1_queue=0, s2_queue=0)
    result = policy.choose_assignment(state, "T01")
    assert result.score >= 0.0
