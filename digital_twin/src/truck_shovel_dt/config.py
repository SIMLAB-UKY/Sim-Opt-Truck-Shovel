"""Load, validate, and expose scenario configuration.

Raises ConfigError for any missing field, non-positive duration,
invalid triangular distribution, or incomplete route matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class ConfigError(ValueError):
    """Raised when a scenario file is missing, incomplete, or invalid."""


@dataclass(frozen=True)
class Triangular:
    minimum: float
    mode: float
    maximum: float

    def validate(self, name: str) -> None:
        if not (self.minimum <= self.mode <= self.maximum):
            raise ConfigError(
                f"{name}: triangular distribution must satisfy "
                f"min <= mode <= max, got "
                f"({self.minimum}, {self.mode}, {self.maximum})"
            )
        if self.minimum <= 0:
            raise ConfigError(
                f"{name}: triangular minimum must be positive, "
                f"got {self.minimum}"
            )


@dataclass(frozen=True)
class TravelVariability:
    multiplier_min: float
    multiplier_mode: float
    multiplier_max: float

    def validate(self) -> None:
        if not (self.multiplier_min <= self.multiplier_mode <= self.multiplier_max):
            raise ConfigError(
                "travel_variability: must satisfy min <= mode <= max, got "
                f"({self.multiplier_min}, {self.multiplier_mode}, {self.multiplier_max})"
            )
        if self.multiplier_min <= 0:
            raise ConfigError("travel_variability.multiplier_min must be positive")


@dataclass(frozen=True)
class ShovelConfig:
    id: str
    loading: Triangular
    mtbf_minutes: float | None = None
    repair: Triangular | None = None

    def validate(self) -> None:
        if not self.id:
            raise ConfigError("Shovel id must not be empty")
        self.loading.validate(f"shovel[{self.id}].loading")
        if self.mtbf_minutes is not None and self.mtbf_minutes <= 0:
            raise ConfigError(
                f"shovel[{self.id}].mtbf_minutes must be positive, "
                f"got {self.mtbf_minutes}"
            )
        if self.repair is not None:
            self.repair.validate(f"shovel[{self.id}].repair")


@dataclass(frozen=True)
class DumpConfig:
    id: str
    dump: Triangular

    def validate(self) -> None:
        if not self.id:
            raise ConfigError("Dump id must not be empty")
        self.dump.validate(f"dump[{self.id}].dump")


@dataclass(frozen=True)
class FleetConfig:
    number_of_trucks: int
    truck_capacity_tonnes: float
    payload_mean: float
    payload_std: float
    payload_min: float
    payload_max: float

    def validate(self) -> None:
        if self.number_of_trucks <= 0:
            raise ConfigError(
                f"fleet.number_of_trucks must be positive, "
                f"got {self.number_of_trucks}"
            )
        if self.truck_capacity_tonnes <= 0:
            raise ConfigError("fleet.truck_capacity_tonnes must be positive")
        if self.payload_std < 0:
            raise ConfigError("fleet.payload_std must be nonnegative")
        if not (self.payload_min <= self.payload_mean <= self.payload_max):
            raise ConfigError(
                "fleet payload bounds must satisfy "
                "payload_min <= payload_mean <= payload_max"
            )


@dataclass(frozen=True)
class SimulationConfig:
    duration_minutes: float
    warmup_minutes: float
    seed: int

    def validate(self) -> None:
        if self.duration_minutes <= 0:
            raise ConfigError("simulation.duration_minutes must be positive")
        if self.warmup_minutes < 0:
            raise ConfigError("simulation.warmup_minutes must be nonnegative")
        if self.warmup_minutes >= self.duration_minutes:
            raise ConfigError(
                "simulation.warmup_minutes must be less than duration_minutes"
            )


@dataclass(frozen=True)
class LearningConfig:
    ewma_alpha: float
    minimum_observations: int
    switch_penalty_minutes: float

    def validate(self) -> None:
        if not (0.0 < self.ewma_alpha <= 1.0):
            raise ConfigError(
                f"learning.ewma_alpha must be in (0, 1], got {self.ewma_alpha}"
            )
        if self.minimum_observations < 0:
            raise ConfigError(
                "learning.minimum_observations must be nonnegative"
            )
        if self.switch_penalty_minutes < 0:
            raise ConfigError(
                "learning.switch_penalty_minutes must be nonnegative"
            )


@dataclass(frozen=True)
class RouteConfig:
    origin: str
    destination: str
    load_state: str
    distance_km: float
    mean_speed_kph: float

    def validate(self) -> None:
        if self.load_state not in ("empty", "loaded"):
            raise ConfigError(
                f"route {self.origin}->{self.destination}: "
                f"load_state must be 'empty' or 'loaded', got {self.load_state}"
            )
        if self.distance_km <= 0:
            raise ConfigError(
                f"route {self.origin}->{self.destination}: "
                "distance_km must be positive"
            )
        if self.mean_speed_kph <= 0:
            raise ConfigError(
                f"route {self.origin}->{self.destination}: "
                "mean_speed_kph must be positive"
            )


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_name: str
    simulation: SimulationConfig
    learning: LearningConfig
    fleet: FleetConfig
    shovels: list[ShovelConfig]
    dumps: list[DumpConfig]
    routes: list[RouteConfig] = field(default_factory=list)
    travel_variability: TravelVariability = field(
        default_factory=lambda: TravelVariability(0.88, 1.00, 1.18)
    )

    def validate(self) -> None:
        if not self.scenario_name:
            raise ConfigError("scenario_name must not be empty")
        self.simulation.validate()
        self.learning.validate()
        self.fleet.validate()
        self.travel_variability.validate()

        if not self.shovels:
            raise ConfigError("At least one shovel must be defined")
        if not self.dumps:
            raise ConfigError("At least one dump must be defined")

        for shovel in self.shovels:
            shovel.validate()
        for dump in self.dumps:
            dump.validate()

        shovel_ids = {s.id for s in self.shovels}
        dump_ids = {d.id for d in self.dumps}
        if len(shovel_ids) != len(self.shovels):
            raise ConfigError("Shovel ids must be unique")
        if len(dump_ids) != len(self.dumps):
            raise ConfigError("Dump ids must be unique")

        if self.routes:
            self._validate_route_completeness(shovel_ids, dump_ids)

    def _validate_route_completeness(
        self, shovel_ids: set[str], dump_ids: set[str]
    ) -> None:
        for route in self.routes:
            route.validate()

        empty_pairs = {
            (r.origin, r.destination)
            for r in self.routes
            if r.load_state == "empty"
        }
        loaded_pairs = {
            (r.origin, r.destination)
            for r in self.routes
            if r.load_state == "loaded"
        }

        for dump_id in dump_ids:
            for shovel_id in shovel_ids:
                if (dump_id, shovel_id) not in empty_pairs:
                    raise ConfigError(
                        f"Missing empty route from dump {dump_id} "
                        f"to shovel {shovel_id}"
                    )
                if (shovel_id, dump_id) not in loaded_pairs:
                    raise ConfigError(
                        f"Missing loaded route from shovel {shovel_id} "
                        f"to dump {dump_id}"
                    )


def _triangular_from_dict(d: dict, prefix: str) -> Triangular:
    try:
        return Triangular(
            minimum=float(d[f"{prefix}_min"]),
            mode=float(d[f"{prefix}_mode"]),
            maximum=float(d[f"{prefix}_max"]),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required field: {prefix}_* ({exc})") from exc


def load_scenario(
    scenario_path: str | Path,
    routes_path: str | Path | None = None,
) -> ScenarioConfig:
    """Load and validate a scenario JSON file and optional routes.csv."""
    scenario_path = Path(scenario_path)
    if not scenario_path.exists():
        raise ConfigError(f"Scenario file not found: {scenario_path}")

    with scenario_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    try:
        simulation = SimulationConfig(
            duration_minutes=float(raw["simulation"]["duration_minutes"]),
            warmup_minutes=float(raw["simulation"]["warmup_minutes"]),
            seed=int(raw["simulation"]["seed"]),
        )
        learning = LearningConfig(
            ewma_alpha=float(raw["learning"]["ewma_alpha"]),
            minimum_observations=int(raw["learning"]["minimum_observations"]),
            switch_penalty_minutes=float(
                raw["learning"]["switch_penalty_minutes"]
            ),
        )
        fleet = FleetConfig(
            number_of_trucks=int(raw["fleet"]["number_of_trucks"]),
            truck_capacity_tonnes=float(raw["fleet"]["truck_capacity_tonnes"]),
            payload_mean=float(raw["fleet"]["payload_mean"]),
            payload_std=float(raw["fleet"]["payload_std"]),
            payload_min=float(raw["fleet"]["payload_min"]),
            payload_max=float(raw["fleet"]["payload_max"]),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required scenario field: {exc}") from exc

    tv_raw = raw.get("travel_variability", {})
    travel_variability = TravelVariability(
        multiplier_min=float(tv_raw.get("multiplier_min", 0.88)),
        multiplier_mode=float(tv_raw.get("multiplier_mode", 1.00)),
        multiplier_max=float(tv_raw.get("multiplier_max", 1.18)),
    )

    shovels = []
    for s in raw.get("shovels", []):
        repair = None
        if s.get("repair_min") is not None:
            repair = _triangular_from_dict(s, "repair")
        shovels.append(
            ShovelConfig(
                id=s["id"],
                loading=_triangular_from_dict(s, "loading"),
                mtbf_minutes=s.get("mtbf_minutes"),
                repair=repair,
            )
        )

    dumps = [
        DumpConfig(id=d["id"], dump=_triangular_from_dict(d, "dump"))
        for d in raw.get("dumps", [])
    ]

    routes: list[RouteConfig] = []
    if routes_path is not None:
        routes_path = Path(routes_path)
        if not routes_path.exists():
            raise ConfigError(f"Routes file not found: {routes_path}")
        routes_df = pd.read_csv(routes_path)
        required_columns = {
            "origin", "destination", "load_state",
            "distance_km", "mean_speed_kph",
        }
        missing = required_columns - set(routes_df.columns)
        if missing:
            raise ConfigError(f"routes.csv missing required columns: {missing}")
        for row in routes_df.itertuples(index=False):
            routes.append(
                RouteConfig(
                    origin=row.origin,
                    destination=row.destination,
                    load_state=row.load_state,
                    distance_km=float(row.distance_km),
                    mean_speed_kph=float(row.mean_speed_kph),
                )
            )

    config = ScenarioConfig(
        scenario_name=raw.get("scenario_name", ""),
        simulation=simulation,
        learning=learning,
        fleet=fleet,
        shovels=shovels,
        dumps=dumps,
        routes=routes,
        travel_variability=travel_variability,
    )
    config.validate()
    return config
