# NPU Architecture Research

This private repository records architecture analysis, measurement tooling, and
prototype implementation work for programmable-core NPUs. The main thread is
the relationship between physical worker placement, NoC data movement, and
long-term compute-load distribution.

The repository intentionally preserves SDK-facing source paths where build and
import rules depend on them. Research notes, presentation assets, experiments,
and generated evidence are documented separately so that each activity can be
understood without reconstructing the entire development history.

## Research map

| Area | Purpose | Location |
|---|---|---|
| Architecture and aging study | Device structure, profiling, stress modeling, and early simulations | [`study_note/`](study_note/README.md) |
| Data-movement microbenchmarks | NoC0/NoC1, multicast geometry, sender position, and segmented transfers | [`data_movement/one_to_all/`](data_movement/one_to_all/README.md) |
| Programmable Conv remapping | Logical-shard-to-worker translation, placement lifetime, and production integration | [`data_movement/conv_core_replacement/`](data_movement/conv_core_replacement/README.md) |
| MILP placement policy | Candidate generation and lexicographic worker-load balancing | [`data_movement/conv_core_replacement/policy/`](data_movement/conv_core_replacement/policy/README.md) |
| Presentation assets | Editable diagrams, equations, and the presentation script | [`presentation/`](presentation/README.md) |
| SDK integration source | Modified Conv, Matmul, reshard, halo, and tensor-placement paths | [`operations/`](operations/) and [`ttnn/`](ttnn/) |

## Current headline results

- Supported contiguous and segmented worker placements preserve logical shard
  order through an explicit placement-translation layer.
- Disconnected worker regions are decomposed into legal rectangular multicast
  commands with guarded sender handling.
- The offline H=20 MILP predicts a **26.47% reduction in peak modeled TRISC
  exposure**, reducing max/mean worker load from 1.4499 to 1.0659.
- A separately board-validated full-model remapping configuration measured
  **2.049 ms versus 2.057 ms stock** in aggregate device latency, showing no
  measurable overhead for the validated policy slice.

The complete 20-phase MILP schedule has not been claimed as fully deployed on
board. Measured results, predicted results, and unsupported placement classes
are distinguished in the project documentation.

## Reading order

1. [`RESEARCH_LOG.md`](RESEARCH_LOG.md) — chronological activity record.
2. [`data_movement/conv_core_replacement/README.md`](data_movement/conv_core_replacement/README.md) — implementation and validation boundary.
3. [`data_movement/conv_core_replacement/docs/CORE_REMAPPING_DESIGN.md`](data_movement/conv_core_replacement/docs/CORE_REMAPPING_DESIGN.md) — detailed mechanism design.
4. [`data_movement/conv_core_replacement/policy/RESULTS.md`](data_movement/conv_core_replacement/policy/RESULTS.md) — MILP results and limitations.

## Repository policy

This repository is maintained as a private research archive. Raw profiler
captures, container backups, build outputs, duplicate SDK checkouts, and local
virtual environments are retained locally but excluded from Git. Source and
test files retain their original SPDX and attribution headers.
