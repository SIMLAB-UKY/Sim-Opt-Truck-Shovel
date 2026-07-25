# Validation Document

This document records hand calculations and consistency checks for the
simulation engine and KPI module. It satisfies the Day 6 deliverable
requirement from the project plan (Section 10.7).

## Warm-Up Handling

The event log includes all events from simulation time 0. Final KPI
calculations exclude all events that completed before `warmup_minutes`
(default: 30 minutes). This is implemented by filtering the event log:

```
post_warmup_events = event_log[event_log["sim_time_min"] >= warmup_minutes]
```

The warm-up period removes early transient behavior when the system
starts empty. All production, queue, and utilization KPIs are calculated
from post-warmup events only.

## Hand Validation — Deterministic One-Truck Cycle (Section 13.2)

**Scenario:** `validation_deterministic.json`
- 1 truck, 1 shovel (S1), 1 dump (D1)
- Empty travel: 5 min (fixed)
- Loading: 4 min (fixed)
- Loaded travel: 7 min (fixed)
- Dumping: 1 min (fixed)
- Payload: 100 tonnes (fixed)
- Duration: 51 minutes

**Expected cycle time:** 5 + 4 + 7 + 1 = **17 minutes**

**Expected trips:** floor(51 / 17) = **3 trips**

**Expected production:** 3 × 100 = **300 tonnes**

**Verified by test:** `test_deterministic_completed_trips`,
`test_deterministic_total_production`, `test_cycle_duration_is_correct`

## Consistency Checks

### Production Consistency
Total production must equal the sum of all `payload_tonnes` values in
`DUMPING_END` events after warm-up:

```
sum(DUMPING_END.payload_tonnes) == total_production_tonnes
```

Verified by: `KPICalculator.verify_production_consistency()`
and test: `test_production_consistency`

### Time Accounting
No truck's productive time (travel + loading + dumping) may exceed the
available post-warmup time:

```
truck_utilization[truck_id] <= 1.0  for all trucks
```

Verified by: `KPICalculator.verify_time_accounting()`
and test: `test_time_accounting_consistency`

### Queue Wait Non-Negativity
All recorded queue wait times must be ≥ 0:

```
queue_wait_min >= 0.0  for all LOADING_START and DUMPING_START events
```

Verified by: `test_queue_waits_are_nonnegative`, `test_queue_wait_is_nonnegative`

### Utilization Bounds
All utilization values must be in [0, 1]:

```
0.0 <= truck_utilization[t] <= 1.0  for all trucks t
0.0 <= shovel_utilization[s] <= 1.0  for all shovels s
```

Verified by: `test_truck_utilizations_between_0_and_1`,
`test_shovel_utilizations_between_0_and_1`

## Sanity Checks (Section 13.4)

The following sanity checks were verified manually by running
`TruckShovelSimulation` with the base scenario:

| Check | Expected | Result |
|---|---|---|
| Adding trucks increases production (up to a point) | More trips with 6 vs 1 truck | ✅ Confirmed |
| Increasing all travel times does not increase throughput | Fewer trips with slower routes | ✅ Confirmed |
| Queue waiting times are nonnegative | ≥ 0 always | ✅ Confirmed |
| Utilizations are between 0% and 100% | [0, 1] always | ✅ Confirmed |
| Completed payload matches recorded production | Exact match | ✅ Confirmed |
