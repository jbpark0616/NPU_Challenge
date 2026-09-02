# Research Activity Log

This file records the large research steps and points to the corresponding
implementation, evidence, and presentation material. Detailed debugging history
stays in the project-specific documents.

## 2026-07 — Architecture and workload characterization

- Studied the BOS software stack, Tensix worker topology, sharding layouts, and
  NoC routing behavior.
- Built profiling and visualization flows for per-core activity, DRAM traffic,
  operation timelines, and stage-level behavior.
- Established TRISC compute cycles as the observable compute-exposure proxy
  while explicitly separating unmodeled SRAM, NoC/NIU, thermal, voltage, and
  process-variation effects.
- Records: [`study_note/`](study_note/README.md) and
  [`bos_model_copy/resnet50_jbpark/analysis/`](bos_model_copy/resnet50_jbpark/analysis/).

## 2026-08 — NoC multicast characterization

- Added one-to-all tests for sender position, NoC direction, rectangle shape,
  destination count, and disconnected receiver sets.
- Measured that sender position has limited impact on the 5x4 multicast grid,
  while destination count and placement fragmentation affect completion cost.
- Implemented and tested segmented multicast with separate rectangular commands
  and explicit sender-local handling.
- Records: [`data_movement/one_to_all/`](data_movement/one_to_all/README.md) and
  [`data_movement/conv_core_replacement/policy/figures/noc_microbench/`](data_movement/conv_core_replacement/policy/figures/noc_microbench/).

## 2026-08 — Programmable Conv core remapping

- Introduced `CorePlacement` as a translation layer between logical tensor
  shards and selected worker coordinates.
- Extended supported Conv2D paths to use contiguous translations and segmented
  row/column placements without changing logical shard order.
- Added placement-lifetime planning so compatible consumers propagate a
  remapped placement and unsupported boundaries return to stock placement.
- Added host preflight checks and guarded isolated, chained, and full-model
  tests to prevent unsupported placements from reaching the device.
- Records: [`data_movement/conv_core_replacement/`](data_movement/conv_core_replacement/README.md),
  [`operations/conv/conv2d/`](operations/conv/conv2d/), and
  [`data_movement/conv_core_replacement/evidence/`](data_movement/conv_core_replacement/evidence/README.md).

## 2026-08 — MILP placement policy

- Constructed legal placement domains from the runtime compatibility boundary.
- Derived candidate worker-cost vectors from measured stock TRISC cycles.
- Implemented a five-stage lexicographic MILP: minimize peak load, balance all
  workers, reduce segmentation, reduce program count, and reduce phase changes.
- Produced an H=20 feasible schedule predicting a 26.47% peak-load reduction.
- Evaluated placement flexibility and found that up to two segments capture the
  achievable improvement in the current model.
- Records: [`data_movement/conv_core_replacement/policy/`](data_movement/conv_core_replacement/policy/README.md).

## 2026-08 — Board validation and latency

- Completed a guarded BOS ResNet50 remapping configuration with placement and
  PCC checks.
- Compared successfully completed stock and remapped profiler captures.
- Measured aggregate device latency of 2.056535 ms stock and 2.049374 ms for
  the validated remapping configuration, a -0.35% difference.
- Kept the unvalidated full H=20 deployment outside the final measured claim.

## 2026-09 — Research archive cleanup

- Grouped design documents, editable presentation assets, validation evidence,
  and raw local-only artifacts by purpose.
- Added top-level navigation and explicit measured-versus-predicted claim
  boundaries.
- Retained build-sensitive source paths and excluded caches, raw profiler data,
  duplicate SDK worktrees, and container backups from Git.
