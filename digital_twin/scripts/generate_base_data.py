from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "data" / "scenarios"
GENERATED_DIR = ROOT / "data" / "generated"

SEED = 20260715
N_SAMPLES = 500


def clipped_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    """Generate normal samples and clip them to engineering bounds."""
    values = rng.normal(loc=mean, scale=std, size=size)
    return np.clip(values, low, high)


def travel_minutes(
    rng: np.random.Generator,
    distance_km: float,
    speed_kph: float,
    size: int,
) -> np.ndarray:
    """Calculate mean route time and add bounded stochastic variation."""
    base_minutes = distance_km / speed_kph * 60.0
    multipliers = rng.triangular(0.88, 1.00, 1.18, size=size)
    return base_minutes * multipliers


def main() -> None:
    rng = np.random.default_rng(SEED)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    scenario_path = SCENARIO_DIR / "base_scenario.json"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Create {scenario_path} from the project-plan template first.")

    with scenario_path.open("r", encoding="utf-8") as file:
        scenario = json.load(file)

    service_rows: list[dict[str, float | str | int]] = []

    for shovel in scenario["shovels"]:
        samples = rng.triangular(
            shovel["loading_min"],
            shovel["loading_mode"],
            shovel["loading_max"],
            size=N_SAMPLES,
        )
        for sample_id, value in enumerate(samples, start=1):
            service_rows.append(
                {
                    "sample_id": sample_id,
                    "resource_type": "shovel",
                    "resource_id": shovel["id"],
                    "activity": "loading",
                    "duration_minutes": round(float(value), 4),
                }
            )

    for dump in scenario["dumps"]:
        samples = rng.triangular(
            dump["dump_min"],
            dump["dump_mode"],
            dump["dump_max"],
            size=N_SAMPLES,
        )
        for sample_id, value in enumerate(samples, start=1):
            service_rows.append(
                {
                    "sample_id": sample_id,
                    "resource_type": "dump",
                    "resource_id": dump["id"],
                    "activity": "dumping",
                    "duration_minutes": round(float(value), 4),
                }
            )

    service_df = pd.DataFrame(service_rows)
    service_df.to_csv(GENERATED_DIR / "service_time_samples.csv", index=False)

    payload = clipped_normal(
        rng=rng,
        mean=scenario["fleet"]["payload_mean"],
        std=scenario["fleet"]["payload_std"],
        low=scenario["fleet"]["payload_min"],
        high=scenario["fleet"]["payload_max"],
        size=N_SAMPLES,
    )
    pd.DataFrame(
        {
            "sample_id": np.arange(1, N_SAMPLES + 1),
            "payload_tonnes": np.round(payload, 4),
        }
    ).to_csv(GENERATED_DIR / "payload_samples.csv", index=False)

    route_df = pd.read_csv(SCENARIO_DIR / "routes.csv")
    travel_rows: list[dict[str, float | str | int]] = []

    for route in route_df.itertuples(index=False):
        samples = travel_minutes(
            rng=rng,
            distance_km=float(route.distance_km),
            speed_kph=float(route.mean_speed_kph),
            size=N_SAMPLES,
        )
        for sample_id, value in enumerate(samples, start=1):
            travel_rows.append(
                {
                    "sample_id": sample_id,
                    "origin": route.origin,
                    "destination": route.destination,
                    "load_state": route.load_state,
                    "travel_minutes": round(float(value), 4),
                }
            )

    pd.DataFrame(travel_rows).to_csv(GENERATED_DIR / "travel_time_samples.csv", index=False)

    summary = {
        "seed": SEED,
        "samples_per_distribution": N_SAMPLES,
        "service_rows": len(service_rows),
        "travel_rows": len(travel_rows),
        "payload_rows": len(payload),
    }
    with (GENERATED_DIR / "generation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Synthetic base data generated successfully.")
    print(summary)


if __name__ == "__main__":
    main()
