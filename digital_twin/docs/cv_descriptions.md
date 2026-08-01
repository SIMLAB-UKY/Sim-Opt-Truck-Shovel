# CV-Ready Project Descriptions

## For Ali Kamelshahroudi (Primary Software Developer)

### Short version (one line)
Designed and implemented a modular Python/SimPy adaptive dispatch digital twin
for surface mine truck–shovel operations with online EWMA estimation, Streamlit
dashboard, and GitHub Actions CI.

### Medium version (three sentences)
Built a reproducible discrete-event simulation platform in Python/SimPy
modelling truck, shovel, dump, queue, and failure/repair processes for surface
mine haulage. Implemented three dispatch policies — fixed assignment, shortest
queue, and adaptive estimated-completion-time — with online EWMA estimators
that update travel and service time estimates from completed activities.
Delivered a Streamlit decision-support dashboard, multi-replication experiment
runner with 95% confidence intervals, 101 automated tests at 92% coverage, and
GitHub Actions continuous integration.

### Full version (for portfolio or application)
**Primary Software Developer — Adaptive Truck–Shovel Digital Twin**
*Python · SimPy · Streamlit · Plotly · pytest · GitHub Actions*

Designed and implemented a modular digital twin of surface mine truck–shovel
haulage operations as part of a supervised research project. Key contributions:

- **Simulation engine**: multi-truck, multi-shovel, multi-dump discrete-event
  simulation using SimPy with stochastic loading, travel, dumping, payload, and
  failure/repair processes; deterministic validation mode for hand-checking.
- **Dispatch policies**: common policy interface supporting fixed assignment,
  shortest-queue, and adaptive estimated-completion-time (ECT) policies;
  state-aware scoring from current queues, availability, and EWMA estimates.
- **Online learning**: EWMA estimator registry initialised from scenario
  assumptions and updated after every completed activity; configurable learning
  rate α and switching penalty.
- **Disruptions**: shovel failure and repair processes with exponential MTBF
  and triangular repair time; immediate rerouting when assigned shovel fails.
- **Reproducible data pipeline**: synthetic scenario generator with documented
  engineering assumptions, fixed random seeds, and automated bounds checking.
- **Experiment platform**: multi-replication runner producing per-replication
  KPIs, aggregated means, 95% confidence intervals, and policy-comparison
  figures.
- **Dashboard**: three-page Streamlit application for scenario configuration,
  multi-policy comparison, and adaptive controller inspection.
- **Quality**: 101 pytest tests at 92% coverage, ruff linting, black
  formatting, and GitHub Actions CI on Python 3.11 and 3.12.

---

## For Dr. Ali Moradi (Technical Lead)

### Short version (one line)
Directed development of a Python/SimPy digital twin for surface mine haulage
integrating adaptive dispatch, online estimation, disruption response, and
Streamlit decision-support tooling.

### Full version
**Technical Lead — Adaptive Truck–Shovel Dispatch Digital Twin**
*Surface Mine Operations Research · Discrete-Event Simulation · Adaptive Control*

Directed the research design and technical execution of a portfolio-grade
Python/SimPy digital twin for surface mine truck–shovel haulage. Defined
operational scope, modelling assumptions, dispatch policy concepts, scenario
design, and validation methodology. Supervised primary developer through
14-day structured execution plan covering simulation engine, online EWMA
parameter estimation, state-aware adaptive dispatching, disruption response,
reproducible experimentation, KPI analytics, and Streamlit decision-support
dashboard. Project integrates with an existing ESP32-based physical
truck–shovel CPS prototype and is designed for future MQTT telemetry
replacement of synthetic event generation.
