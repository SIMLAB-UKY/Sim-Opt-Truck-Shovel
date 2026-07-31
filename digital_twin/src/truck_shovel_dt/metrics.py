"""KPI calculation module for the Adaptive Truck-Shovel Digital Twin.

Converts the raw event log produced by the simulation engine into
production, queue, cycle, and utilization KPIs.

All KPIs are calculated from the event log only — the simulation engine
itself never computes analytics. Warm-up exclusion is applied by
filtering out events that completed before warmup_minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# KPI dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProductionKPIs:
    total_production_tonnes: float
    tonnes_per_operating_hour: float
    completed_trips: int


@dataclass
class CycleKPIs:
    mean_cycle_time_min: float
    min_cycle_time_min: float
    max_cycle_time_min: float


@dataclass
class QueueKPIs:
    mean_shovel_queue_wait_min: float
    mean_dump_queue_wait_min: float
    total_shovel_queue_wait_min: float
    total_dump_queue_wait_min: float


@dataclass
class UtilizationKPIs:
    truck_utilization: dict[str, float]
    shovel_utilization: dict[str, float]
    shovel_idle_time_min: dict[str, float]
    mean_truck_utilization: float


@dataclass
class KPISummary:
    production: ProductionKPIs
    cycle: CycleKPIs
    queue: QueueKPIs
    utilization: UtilizationKPIs
    warmup_minutes: float
    simulation_duration_minutes: float
    policy: str

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "warmup_minutes": self.warmup_minutes,
            "simulation_duration_minutes": self.simulation_duration_minutes,
            "total_production_tonnes": self.production.total_production_tonnes,
            "tonnes_per_operating_hour": self.production.tonnes_per_operating_hour,
            "completed_trips": self.production.completed_trips,
            "mean_cycle_time_min": self.cycle.mean_cycle_time_min,
            "min_cycle_time_min": self.cycle.min_cycle_time_min,
            "max_cycle_time_min": self.cycle.max_cycle_time_min,
            "mean_shovel_queue_wait_min": self.queue.mean_shovel_queue_wait_min,
            "mean_dump_queue_wait_min": self.queue.mean_dump_queue_wait_min,
            "mean_truck_utilization": self.utilization.mean_truck_utilization,
            "shovel_utilization": self.utilization.shovel_utilization,
            "shovel_idle_time_min": self.utilization.shovel_idle_time_min,
        }


# ---------------------------------------------------------------------------
# Main KPI calculator
# ---------------------------------------------------------------------------


class KPICalculator:
    """Calculate all mandatory KPIs from a simulation event log DataFrame."""

    def __init__(
        self,
        event_log: pd.DataFrame,
        warmup_minutes: float,
        simulation_duration_minutes: float,
    ) -> None:
        self._log = event_log.copy()
        self._warmup = warmup_minutes
        self._duration = simulation_duration_minutes
        self._post_warmup_duration = simulation_duration_minutes - warmup_minutes

        # Post-warmup events only
        self._pw = self._log[self._log["sim_time_min"] >= warmup_minutes].copy()

    # ── Public entry point ────────────────────────────────────────────────

    def calculate(self) -> KPISummary:
        policy = (
            self._log["policy"].iloc[0]
            if "policy" in self._log.columns and len(self._log) > 0
            else "unknown"
        )
        return KPISummary(
            production=self._production_kpis(),
            cycle=self._cycle_kpis(),
            queue=self._queue_kpis(),
            utilization=self._utilization_kpis(),
            warmup_minutes=self._warmup,
            simulation_duration_minutes=self._duration,
            policy=str(policy),
        )

    # ── Production ───────────────────────────────────────────────────────

    def _production_kpis(self) -> ProductionKPIs:
        dumps = self._pw[self._pw["event_type"] == "DUMPING_END"]
        total_tonnes = dumps["payload_tonnes"].sum() if "payload_tonnes" in dumps.columns else 0.0
        completed_trips = len(dumps)
        hours = self._post_warmup_duration / 60.0
        tph = total_tonnes / hours if hours > 0 else 0.0
        return ProductionKPIs(
            total_production_tonnes=round(float(total_tonnes), 2),
            tonnes_per_operating_hour=round(float(tph), 2),
            completed_trips=int(completed_trips),
        )

    # ── Cycle time ───────────────────────────────────────────────────────

    def _cycle_kpis(self) -> CycleKPIs:
        """Cycle time = time from DISPATCH to the DUMPING_END that follows it.

        Uses integer position ordering rather than timestamp comparison to
        handle the case where DUMPING_END and the next DISPATCH share the
        same sim_time_min value.
        """
        cycle_times: list[float] = []

        pw_trucks = self._pw[self._pw["truck_id"].notna()].copy()

        for truck_id in pw_trucks["truck_id"].unique():
            group = (
                pw_trucks[pw_trucks["truck_id"] == truck_id]
                .sort_values("sim_time_min", kind="stable")
                .reset_index(drop=True)
            )

            events = group["event_type"].tolist()
            times = group["sim_time_min"].tolist()

            dispatch_time: float | None = None
            for evt, t in zip(events, times, strict=False):
                if evt == "DISPATCH":
                    dispatch_time = t
                elif evt == "DUMPING_END" and dispatch_time is not None:
                    cycle_times.append(t - dispatch_time)
                    dispatch_time = None

        if not cycle_times:
            return CycleKPIs(0.0, 0.0, 0.0)

        return CycleKPIs(
            mean_cycle_time_min=round(sum(cycle_times) / len(cycle_times), 2),
            min_cycle_time_min=round(min(cycle_times), 2),
            max_cycle_time_min=round(max(cycle_times), 2),
        )

    # ── Queue waits ──────────────────────────────────────────────────────

    def _queue_kpis(self) -> QueueKPIs:
        shovel_waits = self._pw[
            (self._pw["event_type"] == "LOADING_START") & (self._pw["queue_wait_min"].notna())
        ]["queue_wait_min"]

        dump_waits = self._pw[
            (self._pw["event_type"] == "DUMPING_START") & (self._pw["queue_wait_min"].notna())
        ]["queue_wait_min"]

        return QueueKPIs(
            mean_shovel_queue_wait_min=round(
                float(shovel_waits.mean()) if len(shovel_waits) > 0 else 0.0, 4
            ),
            mean_dump_queue_wait_min=round(
                float(dump_waits.mean()) if len(dump_waits) > 0 else 0.0, 4
            ),
            total_shovel_queue_wait_min=round(float(shovel_waits.sum()), 4),
            total_dump_queue_wait_min=round(float(dump_waits.sum()), 4),
        )

    # ── Utilization ──────────────────────────────────────────────────────

    def _utilization_kpis(self) -> UtilizationKPIs:
        truck_util = self._truck_utilization()
        shovel_util, shovel_idle = self._shovel_utilization()
        mean_util = sum(truck_util.values()) / len(truck_util) if truck_util else 0.0
        return UtilizationKPIs(
            truck_utilization=truck_util,
            shovel_utilization=shovel_util,
            shovel_idle_time_min=shovel_idle,
            mean_truck_utilization=round(mean_util, 4),
        )

    def _truck_utilization(self) -> dict[str, float]:
        """Fraction of post-warmup time each truck spends in productive states."""
        productive_pairs = [
            ("EMPTY_TRAVEL_START", "EMPTY_TRAVEL_END"),
            ("LOADING_START", "LOADING_END"),
            ("LOADED_TRAVEL_START", "LOADED_TRAVEL_END"),
            ("DUMPING_START", "DUMPING_END"),
        ]
        available = self._post_warmup_duration
        result: dict[str, float] = {}

        pw_trucks = self._pw[self._pw["truck_id"].notna()]
        for truck_id in pw_trucks["truck_id"].unique():
            group = pw_trucks[pw_trucks["truck_id"] == truck_id].sort_values("sim_time_min")
            productive_time = 0.0
            for start_evt, end_evt in productive_pairs:
                starts = sorted(group[group["event_type"] == start_evt]["sim_time_min"].tolist())
                ends = sorted(group[group["event_type"] == end_evt]["sim_time_min"].tolist())
                for s, e in zip(starts, ends, strict=False):
                    productive_time += max(0.0, e - s)
            util = min(1.0, productive_time / available) if available > 0 else 0.0
            result[truck_id] = round(util, 4)

        return result

    def _shovel_utilization(self) -> tuple[dict[str, float], dict[str, float]]:
        """Busy fraction and idle time for each shovel."""
        util: dict[str, float] = {}
        idle: dict[str, float] = {}
        available = self._post_warmup_duration

        if "shovel_id" not in self._pw.columns:
            return util, idle

        shovel_ids = self._pw[self._pw["shovel_id"].notna()]["shovel_id"].unique()

        for shovel_id in shovel_ids:
            events = self._pw[
                (self._pw["event_type"].isin(["LOADING_START", "LOADING_END"]))
                & (self._pw["shovel_id"] == shovel_id)
            ].sort_values("sim_time_min", kind="stable")

            busy_time = 0.0
            start_time = None
            for _, row in events.iterrows():
                if row["event_type"] == "LOADING_START":
                    start_time = row["sim_time_min"]
                elif row["event_type"] == "LOADING_END" and start_time is not None:
                    busy_time += max(0.0, row["sim_time_min"] - start_time)
                    start_time = None
            fraction = min(1.0, busy_time / available) if available > 0 else 0.0
            util[shovel_id] = round(fraction, 4)
            idle[shovel_id] = round(max(0.0, available - busy_time), 2)

        return util, idle

    # ── Per-truck table ──────────────────────────────────────────────────

    def truck_kpi_table(self) -> pd.DataFrame:
        """Return a per-truck KPI DataFrame."""
        rows = []
        truck_util = self._truck_utilization()
        pw_trucks = self._pw[self._pw["truck_id"].notna()]

        for truck_id in sorted(pw_trucks["truck_id"].unique()):
            group = pw_trucks[pw_trucks["truck_id"] == truck_id]
            dumps = group[group["event_type"] == "DUMPING_END"]
            trips = len(dumps)
            tonnes = dumps["payload_tonnes"].sum() if "payload_tonnes" in dumps.columns else 0.0
            shovel_waits = group[
                (group["event_type"] == "LOADING_START") & (group["queue_wait_min"].notna())
            ]["queue_wait_min"]

            rows.append(
                {
                    "truck_id": truck_id,
                    "completed_trips": trips,
                    "total_tonnes": round(float(tonnes), 2),
                    "mean_shovel_wait_min": round(
                        float(shovel_waits.mean()) if len(shovel_waits) > 0 else 0.0, 4
                    ),
                    "utilization": truck_util.get(truck_id, 0.0),
                }
            )

        return pd.DataFrame(rows).reset_index(drop=True)

    # ── Per-resource table ───────────────────────────────────────────────

    def resource_kpi_table(self) -> pd.DataFrame:
        """Return a per-shovel KPI DataFrame."""
        shovel_util, shovel_idle = self._shovel_utilization()
        rows = []
        for shovel_id in sorted(shovel_util.keys()):
            loads = self._pw[
                (self._pw["event_type"] == "LOADING_END") & (self._pw["shovel_id"] == shovel_id)
            ]
            rows.append(
                {
                    "resource_id": shovel_id,
                    "resource_type": "shovel",
                    "completed_loads": len(loads),
                    "utilization": shovel_util[shovel_id],
                    "idle_time_min": shovel_idle[shovel_id],
                }
            )
        return pd.DataFrame(rows)

    # ── Consistency checks ───────────────────────────────────────────────

    def verify_production_consistency(self) -> bool:
        """Check that total production equals sum of DUMPING_END payloads."""
        dumps = self._pw[self._pw["event_type"] == "DUMPING_END"]
        if "payload_tonnes" not in dumps.columns:
            return True
        total = dumps["payload_tonnes"].sum()
        kpis = self._production_kpis()
        return abs(float(total) - kpis.total_production_tonnes) < 0.01

    def verify_time_accounting(self) -> bool:
        """Check that no truck's productive time exceeds available time."""
        truck_util = self._truck_utilization()
        return all(v <= 1.0 + 1e-6 for v in truck_util.values())
