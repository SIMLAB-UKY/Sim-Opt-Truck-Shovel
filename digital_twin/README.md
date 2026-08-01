# Adaptive Dispatch Digital Twin for Surface Mine Truck–Shovel Operations

A modular Python simulation platform that models truck–shovel haulage cycles,
compares static and adaptive dispatch policies, and visualises results through
an interactive Streamlit dashboard.

> **All input data are synthetic engineering assumptions** created for software
> verification and comparative experiments. No values are derived from an
> operating mine.

---

## Why the Dispatch is Adaptive

Static truck assignments cannot react to changing queues, variable travel and
service times, or equipment failures. This digital twin observes the **current
simulated state** at every dispatch decision and selects the shovel–dump pair
with the lowest estimated completion time. It also updates travel and service
time estimates online using an **exponentially weighted moving average (EWMA)**
after every completed activity, and immediately excludes any shovel that has
failed from the feasible assignment set.

Three levels of adaptation are implemented:

| Level | Behaviour |
|---|---|
| State-aware reassignment | Score recomputed from current queues, availability, and EWMA estimates at every dispatch |
| Self-updating estimates | EWMA updates loading, travel, and dumping estimates after every observed activity |
| Disruption response | Failed shovels excluded instantly; restored automatically after repair |

---

## Key Features

- Discrete-event simulation with SimPy (trucks, shovels, dumps, queues, failures)
- Three dispatch policies: Fixed Assignment, Shortest Queue, Adaptive ECT
- Online EWMA estimators initialised from scenario assumptions
- Shovel failure and repair processes (exponential MTBF, triangular repair time)
- Reproducible synthetic data generator with documented engineering assumptions
- Multi-replication experiment runner with 95% confidence intervals
- Streamlit dashboard with three pages: Run, Compare, Inspect
- GitHub Actions CI: ruff lint, black format check, pytest with coverage

---

## Architecture

```
Scenario JSON ──► Simulation Engine ──► State Snapshot ──► Dispatch Policy
                        │                                         │
                Synthetic Sampler                        EWMA Estimator Updates
                        │                                         │
                Event Log CSV ◄─── Truck/Shovel/Dump Events ──────┘
                        │
                KPI Module ──► Experiment Runner ──► Streamlit Dashboard
```

### Software Components

| Module | Responsibility |
|---|---|
| `config.py` | Load, validate, and expose scenario parameters |
| `simulation.py` | SimPy environment, truck/shovel/dump processes |
| `dispatch.py` | Policy interface and concrete policies |
| `estimators.py` | EWMA estimators for service and travel times |
| `disruptions.py` | Shovel failure and repair processes |
| `metrics.py` | KPI calculation from event log |
| `dashboard/app.py` | Streamlit interactive dashboard |

---

## Installation

```bash
git clone https://github.com/a8li6/truck-shovel-cps.git
cd truck-shovel-cps/digital_twin

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Quick Start

### Generate synthetic base data

```bash
python scripts/generate_base_data.py
```

### Run a single scenario

```bash
PYTHONPATH=src python scripts/run_single_scenario.py \
    --scenario data/scenarios/base_scenario.json \
    --routes   data/scenarios/routes.csv \
    --policy   adaptive_ect
```

### Run deterministic hand-validation

```bash
PYTHONPATH=src python scripts/run_single_scenario.py \
    --scenario data/scenarios/base_scenario.json \
    --routes   data/scenarios/routes.csv \
    --policy   fixed \
    --duration 51 \
    --deterministic
# Expected: 3 completed trips, 300 tonnes (cycle = 5+4+7+1 = 17 min)
```

### Launch the dashboard

```bash
PYTHONPATH=src streamlit run dashboard/app.py
```

---

## Scenario Configuration

Scenario files are JSON documents in `data/scenarios/`. Five scenarios are provided:

| File | Description |
|---|---|
| `base_scenario.json` | Balanced baseline — 6 trucks, 2 shovels, 2 dumps |
| `congested_scenario.json` | 8 trucks, slower dumping, higher queue pressure |
| `slow_shovel_scenario.json` | Shovel 2 substantially slower than Shovel 1 |
| `breakdown_scenario.json` | Both shovels fail and repair during the shift |
| `validation_deterministic.json` | All durations fixed; used for hand validation |

Route data is stored separately in `data/scenarios/routes.csv`.

Key configurable parameters:

```json
{
  "simulation": { "duration_minutes": 480, "warmup_minutes": 30, "seed": 20260715 },
  "learning":   { "ewma_alpha": 0.20, "switch_penalty_minutes": 0.50 },
  "fleet":      { "number_of_trucks": 6, "truck_capacity_tonnes": 95 }
}
```

---

## Dispatch Policies

### Policy 1 — Fixed Assignment
Trucks are assigned to shovels before the shift begins (truck index mod number
of shovels). If the assigned shovel is unavailable, the truck falls back to the
shovel with the shortest queue. This is the non-adaptive baseline.

### Policy 2 — Shortest Queue
At every dispatch decision, the controller selects the available shovel with the
fewest waiting trucks. Ties are broken by empty travel time, then by shovel ID
(deterministic). Dump selection uses shortest loaded travel time.

### Policy 3 — Adaptive ECT (Estimated Completion Time)
For truck *i* and candidate shovel *s*, the score is:

```
Score(i, s) = T_empty(i→s)
            + R_s + Q_s × T_loading(s) + T_loading(s)
            + min_d [ T_loaded(s→d) + W_d + T_dumping(d) ]
            + switch_penalty (if changing shovel)
```

All time estimates are updated online using EWMA after every completed
activity. Failed shovels are excluded from the feasible set automatically.

---

## Synthetic-Data Methodology

Every numeric value is a documented engineering assumption, not real mine data.

| Parameter | Distribution | Values |
|---|---|---|
| Shovel 1 loading | Triangular | min=3.5, mode=4.5, max=6.0 min |
| Shovel 2 loading | Triangular | min=4.0, mode=5.0, max=6.5 min |
| Dump 1 service | Triangular | min=0.8, mode=1.2, max=2.0 min |
| Dump 2 service | Triangular | min=1.0, mode=1.5, max=2.4 min |
| Travel variability | Triangular multiplier | min=0.88, mode=1.00, max=1.18 |
| Payload | Truncated Normal | mean=95 t, std=4 t, clip [82, 102] |
| Shovel MTBF | Exponential | mean=240 min |
| Shovel repair | Triangular | min=15, mode=25, max=45 min |

Route mean travel times are derived from `distance / speed × 60`. See
`docs/assumptions.md` for full documentation.

---

## Experiment Reproduction

```bash
PYTHONPATH=src python scripts/run_experiments.py \
    --scenario     data/scenarios/base_scenario.json \
    --routes       data/scenarios/routes.csv \
    --policies     fixed shortest_queue adaptive_ect \
    --replications 20 \
    --output       data/results/base_comparison
```

Outputs saved to `data/results/base_comparison/`:
- `per_replication_kpis.csv` — one row per replication
- `aggregated_kpis.csv` — mean and 95% CI per policy
- `policy_comparison.png` — bar chart with error bars
- `experiment_config.json` — metadata including git commit and timestamp

---

## Dashboard Instructions

```bash
PYTHONPATH=src streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

Three pages are available:

| Page | Description |
|---|---|
| 🚛 Run a Scenario | Configure and run a single simulation; view KPI cards and charts |
| 📊 Compare Policies | Multi-replication policy comparison with box plots |
| 🔍 Adaptive Inspector | EWMA estimates, assignment counts, disruption events |

Sidebar controls: scenario, policy, number of trucks, duration, seed, EWMA α, failures on/off.

---

## Tests and CI

```bash
# Run all tests
PYTHONPATH=src python -m pytest tests/ -v

# Run with coverage
PYTHONPATH=src python -m pytest tests/ --cov=src/truck_shovel_dt --cov-report=term-missing
```

Current coverage: **92%** across 101 tests.

GitHub Actions runs on every push and pull request:
- ruff lint
- black format check
- pytest with coverage (threshold: 80%)

[![Digital Twin CI](https://github.com/a8li6/truck-shovel-cps/actions/workflows/digital-twin-tests.yml/badge.svg)](https://github.com/a8li6/truck-shovel-cps/actions/workflows/digital-twin-tests.yml)

---

## Example Results

Base balanced scenario — 20 replications, no disruptions:

| Policy | Production (t) | 95% CI | t/h | Shovel Queue (min) |
|---|---|---|---|---|
| Fixed | 14,916 | [14,858 – 14,974] | 1,989 | 0.38 |
| Shortest Queue | 14,969 | [14,923 – 15,015] | 1,996 | 0.34 |
| Adaptive ECT | — | — | — | — |

> The confidence intervals overlap in the balanced scenario, which is
> expected: all three policies produce similar results when the system
> is not congested or disrupted. Differences are more pronounced in the
> `congested_scenario` and `breakdown_scenario`, where the adaptive
> policy responds to changing conditions.

---

## Limitations

- All input data are synthetic; results do not represent any real mine.
- Payload does not affect travel speed in the current MVP.
- Road-network routing, grade effects, and fuel consumption are not modelled.
- Truck failures are implemented as an optional extension only.
- The adaptive ECT policy uses a simple scalar score; no reinforcement
  learning or optimisation solver is used (by design).
- The dashboard is a local Streamlit application, not a deployed service.

---

## Future CPS and MQTT Integration

The simulation was designed as the software intelligence layer of a physical
ESP32-based truck–shovel CPS prototype. Planned extensions:

- **MQTT telemetry bridge**: replace synthetic event generation with real
  truck position and state messages from ESP32 sensors.
- **Real-time replay mode**: replay a completed event log at accelerated
  wall-clock speed to simulate live dashboard updates.
- **OR-Tools rolling-horizon policy**: optional Policy 4 for small assignment
  optimisation problems.
- **Docker packaging**: containerise the dashboard for deployment.

---

## Contributors and Roles

| Contributor | Role |
|---|---|
| **Ali Kamelshahroudi** | Primary Software Developer — simulation, dispatch, estimators, experiments, dashboard, tests, CI |
| **Dr. Ali Moradi** | Technical Lead — requirements, modelling assumptions, validation, result interpretation |

See `CONTRIBUTORS.md` for full attribution.

---

## License and Citation

This project is released under the MIT License. See `LICENSE` for details.

If you use this work in research or teaching, please cite:

```
Ali Kamel and Ali Moradi Afrapoli (2026).
Adaptive Dispatch Digital Twin for Surface Mine Truck–Shovel Operations.
GitHub: https://github.com/a8li6/truck-shovel-cps
```
