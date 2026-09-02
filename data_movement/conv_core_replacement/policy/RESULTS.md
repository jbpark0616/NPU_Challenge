# Full-model compute-balancing policy results

## Current result

The current calibration uses one uninterrupted BOS ResNet50 batch-1 profile
and equal-weight `TRISC-KERNEL` engine cycles. The `H=20` scheduler balances 17
placement domains while preserving all unsupported or non-movable operations
as measured fixed background.

| Policy | Max / mean | Max cycles / inference | CV |
|---|---:|---:|---:|
| Stock | 1.449874 | 3,777,635.0 | 0.422146 |
| Existing Layer3 cyclic | 1.398125 | 3,642,804.2 | 0.350292 |
| Predicted full-model schedule | 1.065910 | 2,777,219.8 | 0.096990 |

Relative to stock, the predicted schedule reduces maximum modeled compute
exposure by 26.47%. Relative to the existing Layer3-only cyclic policy, it
reduces it by 23.75%. Total and mean cycles are invariant because candidate
costs are permutations of the stock logical-core measurements.

The stock partition reconstruction is exact: fixed background plus every
movable domain at its stock placement reproduces the raw full-model per-worker
profile without a cycle difference.

## Solver certificate

The optimizer is a five-stage lexicographic MILP over placement-use counts:

1. minimize maximum cumulative worker cycles;
2. hold that incumbent bound and minimize total absolute deviation;
3. hold both bounds and minimize extra physical-axis segments;
4. hold the first three bounds and minimize used domain-placement programs;
5. hold all prior bounds and minimize cyclic phase switches.

With a 120-second limit per global stage:

- stage 1 found a 55,544,404-cycle incumbent with a 55,523,263 dual lower
  bound;
- the final schedule achieved 55,544,396 period cycles;
- the resulting certified max-load gap is at most 0.0381%;
- stage 2 proved its absolute-deviation optimum at the incumbent max bound;
- stage 3 found segment penalty 208 with lower bound 202, leaving a 2.88% gap;
- stages 2--5 each accepted the preceding incumbent through highspy
  `setSolution()` with `HighsStatus.kOk`;
- global stage 4 found 37 domain-placement programs with lower bound 24;
- global stage 5 found 31 cyclic switches with lower bound 10.

All five stages now search the complete 167-candidate space; the restricted
fallback was not used. Stages 1 and 3--5 reached the 120-second limit, so their
incumbents are global feasible solutions but not proven global optima. The
generated schedule exactly attains segment penalty 208, 37 unique domain
placements, and 31 changes over the cyclic H=20 boundary.

The complete machine-readable certificate is embedded in
`full_model_schedule_h20.json`. Summary tables are written to:

```text
full_model_balance_summary.csv
full_model_stock_vs_optimized_per_worker.csv
full_model_schedule_complexity.csv
```

## Corrected placement-domain boundary

For BLOCK_SHARDED bottlenecks, the movable domain is `Conv2 -> Conv3`, not
`Conv1 -> Conv3`. The request is made at Conv2 and Conv3 consumes the remapped
input; Conv1 executes before that request and remains stock fixed background.
The calibration tables and all reported numbers above use this corrected
boundary.

## What is measured and what is predicted

Measured:

- stock per-worker `TRISC-KERNEL` cycles from the uninterrupted full model;
- exact fixed/movable partition of those stock cycles;
- model topology and stock worker order.

Predicted:

- candidate per-worker cycles obtained by permuting the stock logical-core
  cycle vector onto a legal-slice physical placement;
- H=20 cumulative compute-exposure balance under that cost model.

The schedule remains tagged
`predicted_not_deployable_until_candidate_validation`. It is not evidence that
candidate runtime cycles are placement invariant, and it is not a complete
aging model. SRAM, NIU/NoC, temperature, voltage, energy, and process variation
remain outside the modeled objective.

## Minimum validation gate before deployment

Do not attempt the entire H=20 schedule first. Validate the assumptions in this
order:

1. Profile one HEIGHT_SHARDED Conv2 at stock, boundary-row translation, and an
   interior row gap. Compare its logical-core TRISC vector after inverse
   placement mapping.
2. Profile one 4x4 BLOCK `Conv2 -> Conv3` chain at stock and each resting-column
   class used by the schedule.
3. Profile one 4x2 Layer4 chain for boundary-row translation, an interior row
   gap, and a row-plus-column four-rectangle placement. Explicitly cover the
   two-row exclusion forms present in the candidate table.
4. Reject a candidate class if PCC, normal device shutdown, domain lifetime,
   latency, or inverse-mapped per-core cycles violate the declared tolerance.
5. Regenerate candidates and rerun the optimizer using only validated classes.
6. Only then execute one guarded full H=20 period and compare measured
   cumulative per-worker cycles with the prediction.
