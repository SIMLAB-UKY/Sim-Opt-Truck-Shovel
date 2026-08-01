# System Architecture

## Logical Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│                                                                 │
│  base_scenario.json ──► config.py ──► ScenarioConfig           │
│  routes.csv         ──►                                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SIMULATION LAYER                           │
│                                                                 │
│  TruckShovelSimulation (SimPy)                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │  Truck   │   │  Shovel  │   │   Dump   │   │ Disruption │  │
│  │ Process  │   │Resource  │   │ Resource │   │  Process   │  │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └─────┬──────┘  │
│       │              │              │               │           │
│       └──────────────┴──────────────┴───────────────┘           │
│                               │                                 │
│                        Event Log CSV                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
               ▼                               ▼
┌──────────────────────┐         ┌─────────────────────────┐
│   DISPATCH LAYER     │         │    ESTIMATOR LAYER       │
│                      │         │                          │
│  DispatchPolicy      │◄────────│  EstimatorRegistry       │
│  ├ FixedAssignment   │         │  ├ EWMAEstimator(loading)│
│  ├ ShortestQueue     │         │  ├ EWMAEstimator(travel) │
│  └ AdaptiveECT  ─────┼────────►│  └ EWMAEstimator(dump)  │
│                      │  update │                          │
└──────────────────────┘         └─────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ANALYTICS LAYER                           │
│                                                                 │
│  KPICalculator                                                  │
│  ├ ProductionKPIs  (tonnes, t/h, trips)                        │
│  ├ CycleKPIs       (mean, min, max cycle time)                 │
│  ├ QueueKPIs       (shovel and dump queue waits)               │
│  └ UtilizationKPIs (truck, shovel utilization)                 │
│                                                                 │
│  run_experiments.py                                             │
│  └ 20× replications → 95% confidence intervals → CSV + PNG     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
│                                                                 │
│  Streamlit Dashboard (dashboard/app.py)                         │
│  ├ Page 1: Run a Scenario   — KPI cards, charts, event log     │
│  ├ Page 2: Compare Policies — box plots, aggregated results    │
│  └ Page 3: Adaptive Inspector — EWMA estimates, assignments    │
└─────────────────────────────────────────────────────────────────┘
```

## Component Dependencies

```
config.py
    └── used by: simulation.py, estimators.py, dispatch.py,
                 metrics.py, all scripts, dashboard

simulation.py
    ├── imports: config.py, disruptions.py
    └── used by: scripts/run_single_scenario.py,
                 scripts/run_experiments.py, dashboard/app.py,
                 all tests

dispatch.py
    ├── imports: estimators.py (AdaptiveECT only)
    └── used by: simulation.py (future integration), tests

estimators.py
    └── used by: dispatch.py, dashboard/app.py, tests

disruptions.py
    ├── imports: simpy, numpy
    └── used by: simulation.py

metrics.py
    ├── imports: pandas
    └── used by: scripts/run_single_scenario.py,
                 scripts/run_experiments.py, dashboard/app.py,
                 tests/test_metrics.py
```

## State Machine

### Truck States
```
AWAITING_DISPATCH
      │
      ▼
EMPTY_TRAVEL ──► (shovel failed on arrival) ──► AWAITING_DISPATCH
      │
      ▼
QUEUE_FOR_SHOVEL ──► (shovel failed while waiting) ──► AWAITING_DISPATCH
      │
      ▼
LOADING
      │
      ▼
LOADED_TRAVEL
      │
      ▼
QUEUE_FOR_DUMP
      │
      ▼
DUMPING
      │
      └──► AWAITING_DISPATCH
```

### Shovel States
```
AVAILABLE_IDLE ◄──── REPAIRED
      │                  ▲
      ▼                  │
AVAILABLE_BUSY        UNDER_REPAIR
      │                  ▲
      └──► FAILED ───────┘
```

## File Structure

```
digital_twin/
├── data/
│   ├── scenarios/          # Scenario JSON files and routes.csv
│   ├── generated/          # Synthetic samples (git-ignored)
│   └── results/            # Experiment outputs (git-ignored)
├── src/truck_shovel_dt/
│   ├── config.py           # Scenario loading and validation
│   ├── simulation.py       # SimPy simulation engine
│   ├── dispatch.py         # Dispatch policies
│   ├── estimators.py       # EWMA estimators
│   ├── disruptions.py      # Failure and repair processes
│   ├── metrics.py          # KPI calculation
│   └── logging_utils.py    # (reserved for future structured logging)
├── scripts/
│   ├── generate_base_data.py
│   ├── run_single_scenario.py
│   └── run_experiments.py
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── tests/                  # pytest test suite (101 tests, 92% coverage)
└── docs/
    ├── architecture.md     # This file
    ├── assumptions.md      # Engineering assumptions
    ├── data_dictionary.md  # Column definitions
    └── validation.md       # Hand checks and consistency checks
```
