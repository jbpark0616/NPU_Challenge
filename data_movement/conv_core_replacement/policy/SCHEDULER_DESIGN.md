# ResNet50 Aging-Aware Core-Remapping Scheduler Design

## Decision summary

Use a measured-load, placement-domain scheduler rather than a per-operation
round-robin policy.

- Optimize a fixed periodic schedule offline for ResNet50.
- Schedule a compatible `Propagate` chain as one placement domain.
- Enumerate only placements already accepted by the Conv remapping preflight.
- Instantiate the current research model with the defensible per-core compute
  stress proxy, derived from measured TRISC compute cycles rather than
  invocation count.
- Minimize the maximum normalized cumulative load first; minimize remapping
  transition overhead second.
- Start with a 20-inference horizon. It covers five worker columns and four
  worker rows and can be repeated without long-term drift.
- Keep WIDTH_SHARDED and COL_MAJOR / `transpose_mcast` outside this scheduler's
  enabled capability set.

The complete physical-aging objective would include compute datapaths, SRAM,
NIU/NoC activity, temperature, voltage, time, and process variation. None of
those omitted dimensions is assumed to be zero or irrelevant. The current
evaluation instantiates only the compute component because the available
architecture and measurements do not identify defensible SRAM, NIU/NoC, or
thermal aging functions. They must not be folded into the compute score through
arbitrary weights.

The remapping mechanism and scheduler interface remain vector-cost capable so
calibrated subsystem models can be added later. The current research claim is
therefore compute-stress balancing as an aging proxy, not complete lifetime
optimization.

## Why the current cyclic schedule is only a baseline

Layer3 currently rotates the resting column by operation index. This proves
placement exposure and gives invocation-count balance, but the six Conv2
operations do not necessarily have equal cost.

For the 4x4 topology on Eagle-N's 5x4 worker grid, each operation selects one
resting column. If operation `o` has measured cost `w_o`, assigning it to rest
column `r` gives that column a rest credit of `w_o`. Equal operation counts do
not imply equal rest credit. The correct reduced problem is therefore to
partition weighted rest credits across five columns as evenly as possible.

The general ResNet50 problem also includes row choices, partial HEIGHT_SHARDED
prefixes, four-rectangle placements, and boundaries that must exit to stock.
Those cases should use the common load-vector formulation below.

## Scheduling unit: placement domain

A placement domain is the largest consecutive graph region in which the same
logical-to-physical placement can be propagated without an `ExitToStock`.

Examples:

- Layer3 `Conv2 (3x3) -> Conv3 (1x1 Matmul)` is one domain when their logical
  4x4 topology is compatible.
- A residual join, a core-count expansion, a shard-layout change, or an
  unsupported consumer terminates the domain.
- A downsample path is a separate domain unless both branches can use the same
  placement at the join.

The optimizer must choose one candidate for an entire domain. Choosing each
Conv independently would create unnecessary resharding and could request
placements that cannot survive to the consumer.

Each domain record contains:

```text
domain_id
member_ops
stock_input_grid
stock_output_grid
logical_grid_shape
shard_layout
orientation
candidate_placements
exit_boundary
```

## Candidate generation

Generate candidates from the existing production preflight; do not duplicate
placement legality inside the optimizer.

For each domain:

1. Start with its stock placement.
2. Enumerate allowed excluded worker rows and columns.
3. Construct the ordered physical worker grid through `CorePlacement`.
4. Reject candidates outside the compute grid.
5. Reject candidates that change logical shard count or order.
6. Reject candidates outside the enabled ROW_MAJOR placement slice:
   contiguous translation or at most two segments per axis/four Cartesian
   rectangles.
7. Propagate the candidate through all domain members with the lifetime
   planner. Retain it only if every member selects `Propagate` and the declared
   domain exit remains valid.

Store candidates as explicit ordered worker coordinates, not only excluded
axis lists. This avoids ambiguity for partial HEIGHT_SHARDED grids.

## Measurement and aging model boundary

The general scheduler interface admits a subsystem cost vector:

```text
A[s, d, p, c] = modeled stress contribution for subsystem s
                on physical core c when domain d uses placement p
```

The physically complete subsystem set would include compute, SRAM, NIU/NoC,
and thermal/electrical stress. In the current evaluation, only the identifiable
compute projection is populated:

```text
A_compute[d, p, c] = compute_stress_model(
                         measured TRISC cycles,
                         available operation-class information)
```

A programmable core is not a single MAC unit: equal elapsed cycles can exercise
different instruction streams and datapaths. Raw TRISC cycles are therefore an
observable activity input/proxy, not by themselves a transistor-aging law. If
the research has a calibrated compute-aging equation, it belongs inside
`compute_stress_model`; otherwise the result must be named compute-activity or
compute-exposure balance.

Collect:

- TRISC compute cycles per active core;
- domain wall-clock latency;
- reshard/ExitToStock latency;

Scheduler calibration must use a single uninterrupted full-model profile and
the operation report produced by that same run. Do not use the existing
stage-split captures for scheduler weights: those runs were split to satisfy
NPE/DRAM-register collection limits, and their layer boundaries and execution
graph differ from the production full-model execution. Stage-split data remains
useful for its original NoC/DRAM analysis, but it is not interchangeable with
full-model TRISC exposure.

Wall-clock and transition latency are performance constraints/secondary costs,
not aging contributions. Unmodeled BRISC/NCRISC, SRAM, NIU/NoC, temperature,
voltage, and energy dimensions are recorded as model limitations rather than
assigned zero stress.

If per-core profiling is unavailable initially, bootstrap with:

```text
A[d, p, c] = domain_cycles[d] * role_factor[d, p, c]
```

where `role_factor` is 1 for an ordinary active worker. Do not assign an
unmeasured compute-aging multiplier to multicast sender roles: the additional
sender-side NIU/data-movement stress is currently unknown, not absent. A uniform
active mask is acceptable only for the first prototype and must be presented as
invocation exposure, not complete aging balance.

Repeat measurements and use a robust statistic such as the median. Reject a
candidate if its latency or PCC violates the established stock tolerance.

## Optimization model

The implemented offline model first solves aggregate use counts. Let:

- `d` be a placement domain;
- `p` be one feasible placement for `d`;
- `c` be a physical worker core;
- `n[d,p]` be the integer number of phases using placement `p`;
- `A[d,p,c]` be the current compute-stress contribution;
- `B[c]` be existing accumulated modeled compute stress;
- `g[d,p] = (x_segments - 1) + (y_segments - 1)` be the current uncalibrated
  extra-axis-segment proxy;
- `u[d,p]` be 1 when `n[d,p] > 0`;
- `q[d]` be 1 when domain `d` uses at least two placements;
- `v[d,p] = u[d,p] AND q[d]`.

Every domain is assigned exactly `H` times:

```text
sum_p n[d,p] = H
```

Cumulative load is:

```text
L[c] = H * B[c] + sum_{d,p} n[d,p] * A[d,p,c]
```

The objectives are solved lexicographically, not as an arbitrary weighted sum:

```text
Stage 1: minimize z, subject to L[c] <= z for every core c
Stage 2: hold z; minimize sum_c |L[c] - mean(L)|
Stage 3: hold stages 1--2; minimize sum_{d,p} g[d,p] * n[d,p]
Stage 4: hold stages 1--3; minimize sum_{d,p} u[d,p]
Stage 5: hold stages 1--4; minimize sum_{d,p} v[d,p]
```

For a domain using `m` distinct placements, the minimum number of transitions
around a cyclic horizon is zero for `m=1` and `m` for `m>1`. This minimum is
attained by placing all occurrences of each selected placement in one
contiguous block. The `q`/`v` linearization therefore makes the Stage-5
objective exact without introducing `H` copies of every candidate variable.

```text
n[d,p] <= H * u[d,p]
sum_p u[d,p] <= 1 + (|P[d]| - 1) * q[d]
sum_p u[d,p] >= 1 + q[d]
v[d,p] <= u[d,p]
v[d,p] <= q[d]
v[d,p] >= u[d,p] + q[d] - 1
```

The min-max term prevents one hot core from being hidden by a good average.
The segment proxy expresses a preference for simpler placements; it is not a
latency model. Replace or augment `g[d,p]` with measured candidate latency once
those profiles exist. Temperature, SRAM, and NIU/NoC costs must remain separate
until their cross-domain aging models and aggregation weights are justified.

For the initial fixed model, solve this as a small MILP or CP-SAT problem. The
number of domains and legal candidates is small enough that an offline bounded
solution with a reported optimality gap is preferable to a complicated runtime
heuristic. Call it exact only when the solver proves optimality.

## Horizon and periodicity

Use `H=20` initially:

- five phases cover all worker-column choices;
- four phases cover all worker-row choices;
- `lcm(5,4)=20` permits a periodic schedule with no row/column phase drift.

If profiling shows that only column rotation matters for the selected domains,
reduce to `H=5`. Increase the horizon only when the min-max optimum materially
improves; a large horizon increases program-cache and validation cost.

The schedule is deterministic and repeats:

```text
phase = inference_index % H
placement = schedule[phase][domain_id]
```

## Runtime integration

Keep optimization outside the Conv kernels.

1. An offline profiler emits a domain/candidate cost table.
2. A Python scheduler solves and serializes a schedule manifest.
3. The model wrapper looks up the current phase and writes the selected
   `excluded_worker_rows` / `excluded_worker_columns` or explicit placement.
4. Conv's existing planner remains the authority for `Propagate` and
   `ExitToStock`.
5. Program-cache keys continue to include physical placement. Pre-warm only
   placements used by the chosen periodic schedule.

Suggested manifest shape:

```json
{
  "architecture": "Eagle-N",
  "model": "BOS-ResNet50-b1",
  "horizon": 20,
  "metric": "median_per_core_trisc_cycles",
  "phases": [
    {
      "phase": 0,
      "domains": {
        "layer3.module1.conv2_to_conv3": {
          "excluded_rows": [],
          "excluded_columns": [0]
        }
      }
    }
  ]
}
```

The manifest must include a topology/configuration fingerprint so a schedule
cannot be reused for a model or device shape with different placement domains.

## Optional online compute-load extension

Do not begin with an online controller. Add it only if workloads vary and a
reliable cumulative compute-load counter is available.

An online greedy controller can select the candidate minimizing:

```text
score(p) =
    max_c projected_normalized_load[c]
  + alpha * projected_load_variance
  + beta  * transition_cost
```

The offline schedule remains the fallback whenever compute-load telemetry is
missing. Thermal, SRAM, and NIU/NoC feedback are not inputs to the current
controller because their aging transfer functions are unavailable; the
architecture may accept them later as separate modeled dimensions.

## Implementation phases

### Phase 1: inventory and domains

- Extend the existing placement trace to emit all ResNet50 Conv metadata.
- Partition the graph into placement domains using planner results.
- Enumerate and host-validate every legal candidate.
- Produce a machine-readable inventory without running optimization.

### Phase 2: measured costs

- Profile stock and candidate placements domain by domain.
- Record per-core cycles and transition latency.
- Confirm candidate PCC and latency stability.
- Mark candidates invalid rather than assigning an artificial high cost when
  they timeout, deadlock, or violate correctness.

### Phase 3: offline optimizer

- [x] Implement the `H=20` count-based min-max MILP in Python.
- [x] Compare it with stock and the current cyclic baseline.
- [x] Serialize a deterministic schedule, predicted load vectors, and solver
  certificates.
- [x] Add extra-axis-segment, used-placement, and cyclic-switch objectives.
- [x] Use highspy `setSolution()` to pass every incumbent into the next
  lexicographic stage and retain the full candidate space through Stage 5.
- [x] Retain an explicitly labeled restricted-support fallback only as an
  emergency path if a future solver run rejects or loses its incumbent.
- [ ] Increase the stage budgets or use a stronger exact-solver configuration
  if global optimality certificates for stages 1 and 3--5 are required.

### Phase 4: full-model execution

- Pre-warm the selected placements.
- Run at least one entire schedule period.
- Validate every domain placement, final PCC, and normal device shutdown.
- Measure end-to-end latency and program-cache footprint.

### Phase 5: long-run evidence

- Run multiple periods and compare measured cumulative per-core load with the
  optimizer prediction.
- State explicitly that SRAM, NIU/NoC, temperature, voltage, and energy remain
  unmodeled aging dimensions; do not interpret compute balance as total-aging
  balance.

## Acceptance criteria

The optimized schedule is accepted only if:

- full-model PCC remains at the established BOS threshold;
- no candidate leaves its declared placement domain unexpectedly;
- no watchdog or device hang occurs;
- maximum normalized cumulative modeled compute stress is lower than stock;
- maximum normalized load is no worse than the cyclic baseline;
- measured and predicted per-core load agree within a declared tolerance;
- end-to-end latency overhead and program-cache growth are reported;
- the result is reproducible for at least one complete schedule period.

Report both balance and cost. A schedule that perfectly balances modeled
compute stress by adding excessive resharding or latency is not optimal for the
system, and it is not automatically optimal for unmodeled SRAM, NIU/NoC, or
thermal aging.

## First concrete experiment

Build the calibration table from one uninterrupted stock ResNet50 execution:

1. Join `profile_log_device.csv` with the `ops_perf_results` file generated by
   the same full-model run through `GLOBAL CALL COUNT == run host ID`.
2. Extract TRISC 0/1/2 kernel cycles for every operation and physical worker.
3. Map the model's Conv invocation trace to the corresponding device-operation
   sequence and construct every supported placement domain across the full
   model, not only Layer3.
4. Exclude unsupported WIDTH_SHARDED and COL_MAJOR / `transpose_mcast` domains,
   while retaining their measured stock exposure as fixed background load.
5. Convert each movable domain's stock physical-core measurements into an
   ordered logical-core exposure vector.
6. Validate the placement-invariance assumption on representative contiguous
   and segmented remaps before using logical-vector permutation to predict all
   candidate placements.
7. Compare stock, the current unweighted cyclic baseline, and the optimized
   full-model schedule over one complete period.

Layer3 remains a useful validation case because its cyclic full-model execution
already passes, but it is not the scope boundary of the calibration table or
optimizer.
