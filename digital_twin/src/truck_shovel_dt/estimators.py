"""Online EWMA estimators for activity and travel durations.

Each estimator maintains a running mean using exponentially weighted
moving averages (EWMA). Estimates are initialized from scenario
assumptions and updated after every completed activity.

Update equation (Section 2.1.2 of project plan):
    mu_hat_t = alpha * x_t + (1 - alpha) * mu_hat_{t-1}

where x_t is the newest observed duration and alpha is the learning rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Single EWMA estimator
# ---------------------------------------------------------------------------

@dataclass
class EWMAEstimator:
    """Maintains an EWMA estimate for one activity or route."""

    name: str
    initial_estimate: float
    alpha: float
    minimum_observations: int = 3

    _estimate: float = field(init=False)
    _observation_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._estimate = self.initial_estimate

    @property
    def estimate(self) -> float:
        """Current EWMA estimate."""
        return self._estimate

    @property
    def observation_count(self) -> int:
        return self._observation_count

    @property
    def is_reliable(self) -> bool:
        """True when enough observations have been collected."""
        return self._observation_count >= self.minimum_observations

    def update(self, observed_value: float) -> tuple[float, float]:
        """Update the estimate with a new observation.

        Returns
        -------
        (estimate_before, estimate_after)
        """
        before = self._estimate
        self._estimate = self.alpha * observed_value + (1 - self.alpha) * self._estimate
        self._observation_count += 1
        return before, self._estimate

    def reset(self) -> None:
        """Reset to initial estimate and clear observations."""
        self._estimate = self.initial_estimate
        self._observation_count = 0


# ---------------------------------------------------------------------------
# Estimator registry — one per activity/route combination
# ---------------------------------------------------------------------------

@dataclass
class EstimatorRegistry:
    """Manages all EWMA estimators for a simulation run."""

    alpha: float
    minimum_observations: int

    _loading: dict[str, EWMAEstimator] = field(default_factory=dict)
    _dumping: dict[str, EWMAEstimator] = field(default_factory=dict)
    _empty_travel: dict[tuple[str, str], EWMAEstimator] = field(default_factory=dict)
    _loaded_travel: dict[tuple[str, str], EWMAEstimator] = field(default_factory=dict)

    # ── Initialization ───────────────────────────────────────────────────

    def init_loading(self, shovel_id: str, initial_minutes: float) -> None:
        self._loading[shovel_id] = EWMAEstimator(
            name=f"loading_{shovel_id}",
            initial_estimate=initial_minutes,
            alpha=self.alpha,
            minimum_observations=self.minimum_observations,
        )

    def init_dumping(self, dump_id: str, initial_minutes: float) -> None:
        self._dumping[dump_id] = EWMAEstimator(
            name=f"dumping_{dump_id}",
            initial_estimate=initial_minutes,
            alpha=self.alpha,
            minimum_observations=self.minimum_observations,
        )

    def init_empty_travel(
        self, origin: str, destination: str, initial_minutes: float
    ) -> None:
        self._empty_travel[(origin, destination)] = EWMAEstimator(
            name=f"empty_{origin}_{destination}",
            initial_estimate=initial_minutes,
            alpha=self.alpha,
            minimum_observations=self.minimum_observations,
        )

    def init_loaded_travel(
        self, origin: str, destination: str, initial_minutes: float
    ) -> None:
        self._loaded_travel[(origin, destination)] = EWMAEstimator(
            name=f"loaded_{origin}_{destination}",
            initial_estimate=initial_minutes,
            alpha=self.alpha,
            minimum_observations=self.minimum_observations,
        )

    # ── Updates ──────────────────────────────────────────────────────────

    def update_loading(
        self, shovel_id: str, observed: float
    ) -> tuple[float, float]:
        return self._loading[shovel_id].update(observed)

    def update_dumping(
        self, dump_id: str, observed: float
    ) -> tuple[float, float]:
        return self._dumping[dump_id].update(observed)

    def update_empty_travel(
        self, origin: str, destination: str, observed: float
    ) -> tuple[float, float]:
        return self._empty_travel[(origin, destination)].update(observed)

    def update_loaded_travel(
        self, origin: str, destination: str, observed: float
    ) -> tuple[float, float]:
        return self._loaded_travel[(origin, destination)].update(observed)

    # ── Queries ──────────────────────────────────────────────────────────

    def get_loading(self, shovel_id: str) -> float:
        return self._loading[shovel_id].estimate

    def get_dumping(self, dump_id: str) -> float:
        return self._dumping[dump_id].estimate

    def get_empty_travel(self, origin: str, destination: str) -> float:
        return self._empty_travel[(origin, destination)].estimate

    def get_loaded_travel(self, origin: str, destination: str) -> float:
        return self._loaded_travel[(origin, destination)].estimate

    def get_estimator_loading(self, shovel_id: str) -> EWMAEstimator:
        return self._loading[shovel_id]

    def get_estimator_dumping(self, dump_id: str) -> EWMAEstimator:
        return self._dumping[dump_id]

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return all current estimates as a dict for logging/display."""
        result: dict = {"loading": {}, "dumping": {}, "empty_travel": {}, "loaded_travel": {}}
        for sid, e in self._loading.items():
            result["loading"][sid] = {
                "estimate": round(e.estimate, 4),
                "observations": e.observation_count,
                "reliable": e.is_reliable,
            }
        for did, e in self._dumping.items():
            result["dumping"][did] = {
                "estimate": round(e.estimate, 4),
                "observations": e.observation_count,
                "reliable": e.is_reliable,
            }
        for (o, d), e in self._empty_travel.items():
            result["empty_travel"][f"{o}->{d}"] = {
                "estimate": round(e.estimate, 4),
                "observations": e.observation_count,
            }
        for (o, d), e in self._loaded_travel.items():
            result["loaded_travel"][f"{o}->{d}"] = {
                "estimate": round(e.estimate, 4),
                "observations": e.observation_count,
            }
        return result
