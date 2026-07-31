"""Dispatch policies for the Adaptive Truck-Shovel Digital Twin.

Day 7 scope: common policy interface, fixed-assignment, and shortest-queue.

All policies expose the same interface so the simulation engine can switch
between them without any changes to simulation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# State snapshot — what the dispatcher sees at decision time
# ---------------------------------------------------------------------------


@dataclass
class ShovelState:
    shovel_id: str
    available: bool  # False if failed or under repair
    queue_length: int  # trucks waiting (not yet being loaded)
    remaining_service_min: float  # estimated time until current truck finishes
    ewma_loading_min: float  # current EWMA loading estimate


@dataclass
class DumpState:
    dump_id: str
    available: bool
    queue_length: int
    ewma_dumping_min: float


@dataclass
class RouteState:
    origin: str
    destination: str
    load_state: str  # 'empty' or 'loaded'
    ewma_travel_min: float  # current EWMA travel estimate


@dataclass
class SystemState:
    shovels: list[ShovelState]
    dumps: list[DumpState]
    routes: list[RouteState]
    current_location: str  # where the truck currently is


@dataclass
class Assignment:
    shovel_id: str
    dump_id: str
    score: float
    explanation: str


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DispatchPolicy(Protocol):
    name: str

    def choose_assignment(self, state: SystemState, truck_id: str) -> Assignment:
        """Return shovel_id, dump_id, score, and explanation."""
        ...


# ---------------------------------------------------------------------------
# Helper: route lookup
# ---------------------------------------------------------------------------


def _get_travel(
    routes: list[RouteState],
    origin: str,
    destination: str,
    load_state: str,
) -> float:
    for r in routes:
        if r.origin == origin and r.destination == destination and r.load_state == load_state:
            return r.ewma_travel_min
    # fallback: return a large penalty if route not found
    return 9999.0


# ---------------------------------------------------------------------------
# Policy 1: Fixed Assignment
# ---------------------------------------------------------------------------


class FixedAssignment:
    """Assign trucks to shovels before the shift begins.

    Truck index % n_shovels determines the assigned shovel.
    The dump is chosen as the one with the shortest loaded travel time
    from the assigned shovel.

    If the assigned shovel is unavailable, fall back to the shovel with
    the shortest queue among available shovels.
    """

    name = "fixed"

    def __init__(self, assignments: dict[str, tuple[str, str]]) -> None:
        """
        Parameters
        ----------
        assignments : dict mapping truck_id → (shovel_id, dump_id)
        """
        self._assignments = assignments

    @classmethod
    def from_config(
        cls,
        truck_ids: list[str],
        shovel_ids: list[str],
        dump_ids: list[str],
        routes: list[RouteState],
    ) -> FixedAssignment:
        """Build pre-shift assignments from config information."""
        assignments: dict[str, tuple[str, str]] = {}
        for i, truck_id in enumerate(truck_ids):
            shovel_id = shovel_ids[i % len(shovel_ids)]
            dump_id = _best_dump_for_shovel(shovel_id, dump_ids, routes)
            assignments[truck_id] = (shovel_id, dump_id)
        return cls(assignments)

    def choose_assignment(self, state: SystemState, truck_id: str) -> Assignment:
        shovel_id, dump_id = self._assignments.get(
            truck_id, (state.shovels[0].shovel_id, state.dumps[0].dump_id)
        )

        # Check if assigned shovel is available
        assigned_shovel = next((s for s in state.shovels if s.shovel_id == shovel_id), None)

        if assigned_shovel is None or not assigned_shovel.available:
            # Fallback: shortest queue among available shovels
            available = [s for s in state.shovels if s.available]
            if not available:
                # No shovel available — return original with explanation
                return Assignment(
                    shovel_id=shovel_id,
                    dump_id=dump_id,
                    score=9999.0,
                    explanation=(
                        f"Truck {truck_id}: assigned shovel {shovel_id} unavailable "
                        f"and no fallback available."
                    ),
                )
            fallback = min(
                available,
                key=lambda s: (s.queue_length, s.shovel_id),
            )
            shovel_id = fallback.shovel_id
            dump_id = _best_dump_for_shovel(
                shovel_id,
                [d.dump_id for d in state.dumps if d.available],
                state.routes,
            )
            explanation = (
                f"Truck {truck_id}: assigned shovel unavailable; "
                f"fallback to {shovel_id} (queue={fallback.queue_length})."
            )
        else:
            explanation = f"Truck {truck_id}: fixed assignment → {shovel_id}, {dump_id}."

        travel = _get_travel(state.routes, state.current_location, shovel_id, "empty")

        return Assignment(
            shovel_id=shovel_id,
            dump_id=dump_id,
            score=travel,
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# Policy 2: Shortest Queue
# ---------------------------------------------------------------------------


class ShortestQueue:
    """Select the available shovel with the fewest waiting trucks.

    Tie-breaking rules (applied in order, fully deterministic):
    1. Shortest empty travel time from the truck's current location.
    2. Lexicographic shovel ID (alphabetical) — ensures reproducibility.

    Dump selection: the dump with the shortest loaded travel time from
    the selected shovel, with ties broken by dump ID.
    """

    name = "shortest_queue"

    def choose_assignment(self, state: SystemState, truck_id: str) -> Assignment:
        available_shovels = [s for s in state.shovels if s.available]

        if not available_shovels:
            # No shovel available
            fallback_shovel = state.shovels[0]
            fallback_dump = state.dumps[0]
            return Assignment(
                shovel_id=fallback_shovel.shovel_id,
                dump_id=fallback_dump.dump_id,
                score=9999.0,
                explanation=f"Truck {truck_id}: no shovel available.",
            )

        # Score each shovel: primary = queue_length,
        # tie-break 1 = empty travel time, tie-break 2 = shovel_id
        def shovel_key(s: ShovelState) -> tuple[int, float, str]:
            travel = _get_travel(state.routes, state.current_location, s.shovel_id, "empty")
            return (s.queue_length, travel, s.shovel_id)

        best_shovel = min(available_shovels, key=shovel_key)
        q, travel, _ = shovel_key(best_shovel)

        # Select dump: shortest loaded travel from chosen shovel
        available_dumps = [d for d in state.dumps if d.available]
        if not available_dumps:
            available_dumps = state.dumps  # fallback if all unavailable

        def dump_key(d: DumpState) -> tuple[float, str]:
            t = _get_travel(state.routes, best_shovel.shovel_id, d.dump_id, "loaded")
            return (t, d.dump_id)

        best_dump = min(available_dumps, key=dump_key)
        loaded_travel, _ = dump_key(best_dump)

        score = travel + best_shovel.remaining_service_min + loaded_travel

        # Build candidate summary for logging
        candidates = []
        for s in available_shovels:
            t = _get_travel(state.routes, state.current_location, s.shovel_id, "empty")
            candidates.append(f"{s.shovel_id}(q={s.queue_length}, t={t:.1f})")

        explanation = (
            f"Truck {truck_id}: shortest-queue → {best_shovel.shovel_id} "
            f"(queue={best_shovel.queue_length}, travel={travel:.2f} min), "
            f"dump={best_dump.dump_id}. "
            f"Candidates: [{', '.join(candidates)}]."
        )

        return Assignment(
            shovel_id=best_shovel.shovel_id,
            dump_id=best_dump.dump_id,
            score=score,
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _best_dump_for_shovel(
    shovel_id: str,
    dump_ids: list[str],
    routes: list[RouteState],
) -> str:
    """Return dump with shortest loaded travel from shovel, ties by dump_id."""
    if not dump_ids:
        return ""
    best = min(
        dump_ids,
        key=lambda d: (
            _get_travel(routes, shovel_id, d, "loaded"),
            d,
        ),
    )
    return best


def build_system_state(
    current_location: str,
    shovel_states: list[ShovelState],
    dump_states: list[DumpState],
    route_states: list[RouteState],
) -> SystemState:
    """Convenience constructor for building a SystemState."""
    return SystemState(
        shovels=shovel_states,
        dumps=dump_states,
        routes=route_states,
        current_location=current_location,
    )


# ---------------------------------------------------------------------------
# Policy 3: Adaptive Estimated Completion Time (ECT)
# ---------------------------------------------------------------------------


class AdaptiveECT:
    """Adaptive dispatch using estimated completion time scoring.

    For truck i and candidate shovel s, the score is:

        Score(i, s) = T_empty(i→s)
                    + R_s + Q_s * T_loading(s)
                    + T_loading(s)
                    + min_d [ T_loaded(s→d) + W_d + T_dumping(d) ]

    where:
        T_empty     = EWMA empty travel estimate
        R_s         = remaining service time at shovel s
        Q_s         = trucks waiting in queue at shovel s
        T_loading   = EWMA loading estimate for shovel s
        T_loaded    = EWMA loaded travel estimate
        W_d         = estimated waiting time at dump d (queue * dumping time)
        T_dumping   = EWMA dumping estimate

    The truck is assigned to the feasible shovel with the lowest score.
    A switching penalty is added when changing from the truck's last shovel.
    """

    name = "adaptive_ect"

    def __init__(
        self,
        estimators: EstimatorRegistry,  # noqa: F821
        switch_penalty_minutes: float = 0.5,
        last_shovel: dict[str, str] | None = None,
    ) -> None:
        self._estimators = estimators
        self._penalty = switch_penalty_minutes
        self._last_shovel: dict[str, str] = last_shovel if last_shovel is not None else {}

    def choose_assignment(self, state: SystemState, truck_id: str) -> Assignment:
        available_shovels = [s for s in state.shovels if s.available]

        if not available_shovels:
            fallback = state.shovels[0]
            fallback_dump = state.dumps[0]
            return Assignment(
                shovel_id=fallback.shovel_id,
                dump_id=fallback_dump.dump_id,
                score=9999.0,
                explanation=f"Truck {truck_id}: no shovel available.",
            )

        best_score = float("inf")
        best_shovel_id = ""
        best_dump_id = ""
        best_components: dict = {}

        for shovel in available_shovels:
            sid = shovel.shovel_id

            # Empty travel estimate
            t_empty = self._estimators.get_empty_travel(state.current_location, sid)

            # Queue and loading estimate
            t_loading = self._estimators.get_loading(sid)
            queue_time = shovel.remaining_service_min + shovel.queue_length * t_loading

            # Best dump estimate
            available_dumps = [d for d in state.dumps if d.available]
            if not available_dumps:
                available_dumps = state.dumps

            best_dump_score = float("inf")
            best_dump = available_dumps[0]
            for dump in available_dumps:
                did = dump.dump_id
                t_loaded = self._estimators.get_loaded_travel(sid, did)
                t_dumping = self._estimators.get_dumping(did)
                w_d = dump.queue_length * t_dumping
                dump_score = t_loaded + w_d + t_dumping
                if dump_score < best_dump_score or (
                    dump_score == best_dump_score and did < best_dump.dump_id
                ):
                    best_dump_score = dump_score
                    best_dump = dump

            # Total score
            score = t_empty + queue_time + t_loading + best_dump_score

            # Switching penalty
            penalty = 0.0
            if truck_id in self._last_shovel and self._last_shovel[truck_id] != sid:
                penalty = self._penalty
            score += penalty

            if score < best_score or (score == best_score and sid < best_shovel_id):
                best_score = score
                best_shovel_id = sid
                best_dump_id = best_dump.dump_id
                best_components = {
                    "t_empty": round(t_empty, 3),
                    "queue_time": round(queue_time, 3),
                    "t_loading": round(t_loading, 3),
                    "best_dump_score": round(best_dump_score, 3),
                    "switch_penalty": round(penalty, 3),
                    "total_score": round(score, 3),
                }

        # Record last assignment
        self._last_shovel[truck_id] = best_shovel_id

        # Build explanation
        component_str = ", ".join(f"{k}={v}" for k, v in best_components.items())
        explanation = (
            f"Truck {truck_id}: adaptive ECT → {best_shovel_id}, {best_dump_id}. "
            f"Score={best_score:.2f} ({component_str})."
        )

        return Assignment(
            shovel_id=best_shovel_id,
            dump_id=best_dump_id,
            score=best_score,
            explanation=explanation,
        )


# Late import to avoid circular dependency
from truck_shovel_dt.estimators import EstimatorRegistry  # noqa: E402
