# Conv Core Replacement — Current Handoff

Last updated: 2026-08-19 (Asia/Seoul)

This document is a checkpoint, not a substitute for the source tree. The next
session must treat the working-tree files and `git diff` as the authoritative
source. Do not discard, reset, reformat, or broadly rewrite existing changes.

## Immediate objective

Finish full-model validation of the non-transposed ROW_MAJOR BLOCK_SHARDED
2D reuse-mcast Matmul path used by BOS ResNet50's matmul-lowered 1x1 Conv.

COL_MAJOR / `transpose_mcast` is deliberately deferred. WIDTH_SHARDED
segmented execution is also outside the current enabled boundary.

## Hard safety rules

- The device can remain deadlocked after a host process exits. Never run a
  high-risk device test with bare `pytest`.
- Run high-risk tests only through
  `run_bos_resnet50_full_model_guarded.sh`.
- If the watchdog fires, do not submit another device workload. Stop and
  report the log; the board may require a power-cycle.
- A normal Python assertion or host `TT_FATAL` followed by the logged driver
  shutdown is not a device deadlock.
- Do not request a power-cycle without evidence of a device hang.
- Do not run another device test until the newest host test suite passes.
- The established Linux build directory is `build`. Do not invent or switch
  to `build_Release_tracy`, the default target, or the `install` target.

## Enabled production boundary

- ROW_MAJOR BLOCK_SHARDED Conv placements may be contiguous translations or
  Cartesian placements with at most two physical segments per axis, hence at
  most four rectangles.
- ROW_MAJOR HEIGHT_SHARDED ordered-gap support has passed its applicable BOS
  isolated, chained, and full-model slices.
- The Matmul addition is the narrow non-transposed ROW_MAJOR BLOCK_SHARDED
  `MatmulMultiCoreReuseMultiCastProgramConfig` path used by 1x1 Conv lowering.
- Physical worker placement and dense logical shard topology remain distinct.
- Segmented data multicasts are independent (`linked=false`). Data writes must
  complete with `noc_async_write_barrier()` before VALID semaphore
  publication.

## Device validation already completed

- `ConvCoreReplacementSafety.*`: previously 10/10 on the build before the
  newest host test was added.
- Matmul multicast line coverage host test passed.
- Layer4 matmul-lowered 1x1 Conv isolated cases passed:
  - contiguous translation;
  - segmented column;
  - segmented row on logical 4x3;
  - four rectangles on logical 4x3.
- The isolated four-rectangle candidate matched the stock output exactly
  (`baseline_vs_candidate=1.0`) and both matched the golden at approximately
  `0.9999081` PCC.
- Relevant 3x3 BLOCK_SHARDED and HEIGHT_SHARDED isolated, chain, and
  full-model slices passed before opening the Matmul full-model lifetime.

## Matmul changes already present

Relevant files include:

- `operations/matmul/device/matmul_op.cpp`
- `operations/matmul/device/matmul_op_multi_core_reuse_mcast_2d_program_factory.cpp`
- `operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`
- `operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`

Important fixes and behavior:

- Program configs use the dense logical shard dimensions derived from the
  physical Cartesian placement.
- The 2D reuse-mcast factory maps logical shards to selected physical workers.
- Output specs preserve the input's physical placement for the supported
  non-transposed segmented path.
- The Cartesian-placement validation in `matmul_op.cpp` applies only to
  BLOCK_SHARDED tensors. HEIGHT_SHARDED and WIDTH_SHARDED retain the original
  containment validation, allowing BOS's normal ragged 14-core
  HEIGHT_SHARDED prefix.
- The segmented in0 and in1 data multicast helpers use `linked=false`, then a
  data barrier, then semaphore publication. This fixed the earlier device
  deadlock.

## Most recent full-model run

Test:

```text
test_bos_resnet50_layer3_cyclic_column_rest_full_model
```

Observed result:

- The model completed all Conv operations, `avgpool`, and `fc`.
- There was no watchdog timeout or device hang.
- The driver closed normally and DMA pages were unpinned.
- The test failed only at the strengthened Python placement assertion.
- Layer3 Conv2 produced the expected remapped 16-core grid.
- The following matmul-lowered Conv3 received that remapped grid at its public
  input boundary.
- Conv3's public output was nevertheless the stock origin-anchored 4x4 grid.
  For the first scheduled case, expected X was `{1,2,3,4}`, while output X was
  `{0,1,2,3}`.

This proved that the full model was still silently exiting the remapped
Matmul lifetime even though the isolated Matmul path worked.

## Root cause of the latest assertion

BOS Conv3 carries a stock `override_sharding_config`. During Conv input
preparation, `get_conv_padded_input_shape_and_mem_config()` compared the stock
override grid and the already-remapped input grid by physical
`CoreRangeSet` equality. It therefore requested a stock reshard before Matmul
dispatch even when both grids represented the same dense logical 4x4
ROW_MAJOR BLOCK_SHARDED topology.

The first two attempted fixes focused on the override path. The 2026-08-18
17:49 and 17:55 reruns still produced stock Conv3 output. The failing operation
was the first scheduled operation, `layer3.module1.conv3`; Layer3 module1 is the
one module configured with `reshard_if_not_optimal=True`. That path selected its
calculated optimal stock 4x4 grid even though the existing remapped input was a
logically equivalent ROW_MAJOR 4x4 BLOCK_SHARDED placement. The remapping
planner and Matmul factory could execute the placement, but
`reshard_if_not_optimal` was reanchoring it before dispatch.

## Newest patch — not yet confirmed on Linux

The following latest changes exist in the Windows working tree. Their sync to
Linux and build/test status must be confirmed before any device rerun.

1. `operations/conv/conv2d/conv_core_remapping_planner.hpp`
   - Adds `block_sharded_override_matches_remapped_logical_grid()`.
   - It compares the dense logical shard-grid dimensions of the input and
     override physical placements.
   - Adds `select_equivalent_remapped_block_execution_grid()` so the execution
     grid selection itself is shared with the host regression test.

2. `operations/conv/conv2d/conv2d_utils.cpp`
   - When core remapping and segmented execution are enabled, a ROW_MAJOR
     BLOCK_SHARDED stock override may accept an input on a different physical
     grid if the logical shard-grid dimensions match.
   - Scheme and orientation checks remain in place.
   - Non-equivalent logical grids still request resharding.
   - Carries the acceptance decision through the later override branch. If
     another reason still requires resharding, the reshard now retains the
     equivalent remapped input grid instead of forcing the stock override grid.
   - Applies the same selection to the `reshard_if_not_optimal` branch. A
     calculated stock grid with the same logical ROW_MAJOR BLOCK_SHARDED shape
     no longer replaces an executable remapped physical placement.

3. `data_movement/conv_core_replacement/test_conv_core_replacement.cpp`
   - Adds
     `EquivalentStockGridRetainsRemappedBlockExecutionPlacement`.
   - It accepts both translated and column-gap 4x4 placements against a stock
     4x4 override and rejects a smaller 4x3 placement.
   - It also checks execution-grid selection: accepted translated/column-gap
     inputs remain selected, while a smaller grid or disabled acceptance falls
     back to stock.

4. `data_movement/conv_core_replacement/test_bos_resnet50_baseline.py`
   - Adds
     `test_bos_resnet50_layer4_translated_conv1_reshard_if_not_optimal_isolated`.
   - This is the short device regression to run before the full model. It
     exercises a translated ROW_MAJOR 4x4 1x1 Matmul-lowered Conv with
     `reshard_if_not_optimal=True`, checks the output remains translated, and
     compares stock/translated output and PyTorch PCC.

5. `data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh`
   - Allows the new isolated regression filter through the existing watchdog.

Validation completed on Linux/device after this patch:

- Host `ConvCoreReplacementSafety.*`: **11/11 passed**, including
  `EquivalentStockGridRetainsRemappedBlockExecutionPlacement`.
- `test_bos_resnet50_layer4_translated_conv1_reshard_if_not_optimal_isolated`:
  **passed** with stock/translated PCC `1.0`; both candidates had
  `0.9999081043514484` PCC against PyTorch.
- The isolated device test closed the driver normally and unpinned DMA pages.
- `test_bos_resnet50_layer3_cyclic_column_rest_full_model`: **passed** on
  2026-08-18 18:17 with final ResNet50 PCC `0.999603608935384`.
- The full-model run completed in 53.68 seconds, closed the driver normally,
  and unpinned DMA pages.
- ROW_MAJOR, non-transposed 2D reuse-mcast Matmul core-remapping full-model
  validation is complete. COL_MAJOR / `transpose_mcast` remains deferred.

`git diff --check` passed for these files in the Windows workspace.

## Exact next steps

First confirm the three newest files above are mirrored to their established
locations under Linux `/tt-metal`. Then, from `/tt-metal`, run only:

```bash
cmake --build build --target ttnn -j"$(nproc)"
```

Build the host test target:

```bash
cmake --build build --target unit_tests_conv_core_replacement -j"$(nproc)"
```

Run the host-only safety suite:

```bash
./build/test/ttnn/unit_tests_conv_core_replacement \
  --gtest_filter='ConvCoreReplacementSafety.*'
```

Expected result: **11 tests passed**. If the count is still 10, the newest test
file was not mirrored or rebuilt. Do not run the device test in that state.

After 11/11 passes, run the guarded isolated regression first:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_translated_conv1_reshard_if_not_optimal_isolated \
  --clean-device-confirmed
```

Only after that test passes, rerun the guarded full-model test:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer3_cyclic_column_rest_full_model \
  --clean-device-confirmed
```

## Expected success conditions for the next full-model run

For every `layer3.moduleN.conv2` selected by the cyclic schedule:

- Conv2 output workers equal all workers in the 5x4 grid except the scheduled
  resting column.
- Conv2 output has 16 cores.
- The following `layer3.moduleN.conv3` input has the same workers.
- Conv3 output also has the same workers and 16 cores, proving the Matmul
  executed on the remapped placement rather than resharding to stock first.
- The exposure counts remain balanced (`[4,5,5,5,5]` when sorted).
- Final BOS model PCC validation passes.
- The device closes normally.

## First-turn verification for the next session

Before editing or running anything, the next session must:

1. Read this file completely.
2. Inspect `git diff` without modifying the working tree.
3. Open and verify the three newest patch files listed above.
4. Restate the latest full-model failure and why the board is currently clean.
5. State the exact host-only build/test commands, including `build` and the
   expected 11-test count.
6. Confirm that COL_MAJOR / `transpose_mcast` remains deferred.

Stop immediately if the next session suggests `build_Release_tracy`, the
`install` target, a bare high-risk `pytest`, an unnecessary power-cycle, or
claims that COL_MAJOR support is complete.
