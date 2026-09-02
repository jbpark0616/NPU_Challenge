# Conv Core Replacement Integration Test

This directory is the integration boundary between the validated segmented
multicast microbenchmarks and the production TTNN Conv path.

Detailed Conv2D structure and the core-replacement insertion points are tracked
in `docs/CONV2D_CORE_REPLACEMENT_NOTES.md`.

The current remapping lifetime, logical-to-worker mapping, segmented multicast
protocol, sharding-specific halo behavior, safety boundary, and validation
status are recorded in `docs/CORE_REMAPPING_DESIGN.md`. That document is the
authoritative status snapshot for the ongoing segmented-placement work.

Production `conv_core_placement.hpp` owns the first replacement boundary;
the local `core_placement.hpp` is now only a compatibility import for its
host tests:

```text
ShardCoord -> WorkerCoord -> (existing TT-Metal mapping) -> NocCoord
```

Its unit tests are host-only and are part of `unit_tests_conv_core_replacement`.

After the individual HEIGHT_SHARDED slices have passed, run their consolidated
regression sequence from the repository root. It rebuilds and runs the complete
host safety binary first, checks the Python and shell syntax, then launches nine
device tests as independent guarded pytest processes. Any nonzero result stops
the sequence before another workload is submitted:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_height_regression_guarded.sh \
  --clean-device-confirmed
```

Each ordered worker axis is split into contiguous intervals carrying
`logical_start`, `worker_start`, and `length`. Their Cartesian product produces
one, two, four, or more rectangular route units without crossing excluded
workers. The activation reader and program factory consume this metadata for
the guarded ROW_MAJOR segmented path. Single-column, single-row, combined
row-and-column, and boundary-origin translation cases have passed their guarded
device validation slices.

The placement planner accepts multiple excluded worker rows and columns. It
selects the first remaining ordered coordinates, so excluding boundary axes
naturally produces a multi-cell shift while excluding interior axes produces
gaps. Out-of-range coordinates, duplicates, and insufficient replacement
capacity fail on the host before any device work is submitted. The original
single-row/single-column overload remains available for compatibility.

`conv_core_remapping_planner.hpp` is the production lifetime-policy boundary.
It receives explicit remapping state plus the input placement and the unchanged
stock output topology. A same-or-smaller output propagates the ordered logical
prefix only when the next Conv implementation can execute the current placement.
Growth along either grid dimension, or entry into an operation such as a
matmul-lowered 1x1 Conv that cannot execute a segmented grid, returns
`ExitToStock` and requests an input reshard to the origin before dispatch. The
Conv preparation path applies that plan. A caller can now request the validated
segmented input through `Conv2dConfig`:

```python
ttnn.Conv2dConfig(
    enable_core_remapping=True,
    excluded_worker_columns=[2],
)
```

For the enabled 3x3 slice, Conv preparation reshards a stock input internally,
propagates the segmented output, and exits to stock when the following operation
cannot execute that placement. A consumer of an already-remapped tensor carries
the original stock grid explicitly in
`Conv2dConfig.core_remapping_input_stock_grid`. HEIGHT lifetime is therefore
decided by comparing the current grid with declared provenance, not by guessing
the stock row-scan width from the remapped grid's origin or bounding box.

The `ConvCoreReplacementSafety` tests exercise the production Conv placement
logic without opening a device. Origin-based and translated contiguous
rectangles are accepted, output placement preserves the translated origin, and
explicit exclusions are accepted when the result has at most two segments per
axis. Placements requiring three segments on either axis and out-of-bounds
placements fail before enqueue.

## Current safety status

The contiguous translated production path has passed isolated 1x1 and 3x3
Conv tests, the downsample `ExitToStock` transition, and the translated layer4
full-model test. The first segmented placement has also passed its isolated
guarded device test: input upload/readback completed, segmented Halo completed,
the Conv program completed, the output retained the ordered physical grid, and
stock-versus-segmented PCC was 1.0.

The placement lifetime has also passed a guarded two-Conv BOS chain. Layer4
module1 Conv2 completed on the segmented 16-core grid and retained that grid in
its output. The following matmul-lowered Conv3 recognized the segmented input,
resharded it before dispatch, and completed on the stock contiguous 4x2 grid.
The final chained output matched the PyTorch golden with PCC
`0.9998673514960605`.

The config-driven full BOS model has also passed. Only layer4 module1 Conv2
requested `excluded_worker_columns=[2]`: its public input remained the stock
16-core tensor, Conv preparation performed the internal segmented reshard, its
output used the ordered segmented 16-core grid, and Conv3 returned to the stock
contiguous 8-core grid. Final model PCC was `0.999603608935384`. The unmodified
default-disabled environment baseline was rerun successfully on the same build.

The row-gap path has separately passed on a logical 4x3 placement using physical
worker Y coordinates `{0,1,3}`. Both an explicit segmented grid and
`excluded_worker_rows=[2]` matched the stock Conv2 result. Its segmented 12-core
output also exited through reshard to the stock 8-core Conv3 placement and
matched the PyTorch golden. Eagle-N has only four worker rows, so the logical
4x4 full-model Conv2 cannot preserve all 16 cores while skipping a row; the
full-model validation therefore remains the column-gap case.

The first device attempt deadlocked because the segmented Conv path combined
sender-containing rectangles with `noc_async_write_multicast_loopback_src`.
The validated protocol now performs one sender-local copy and emits only
remote-only multicast rectangles. Each sender's rectangles are checked on the
host to exclude the sender and to contain exactly `logical_width - 1`
destinations. This matches the previously device-tested segmented one-to-all
protocol.

All high-risk device tests remain opt-in and must use the guarded runner. A
shell timeout can stop the host process, but it cannot guarantee recovery of a
deadlocked device program.

## What the production path currently assumes

Contiguous BLOCK_SHARDED rectangles may be translated. The factory resolves
logical shard coordinates through `CorePlacement` instead of subtracting an
origin from absolute worker coordinates.

The enabled narrow production slice is a non-transposed ROW_MAJOR block-sharded
3x3 Conv whose identical input/output placements have at most two contiguous
segments per physical worker axis, hence at most four Cartesian rectangles. A
column gap segments activation multicast; a row gap segments weight, bias, and
semaphore multicast. Boundary exclusions may leave one translated segment.
COL_MAJOR segmented grids and block placements producing more than four
rectangles remain outside the enabled production boundary. HEIGHT_SHARDED has
a device-validated ROW_MAJOR 3x3 ordered-gap path for BOS column-gap, row-gap,
combined row-and-column four-rectangle, and boundary-origin translation
placements. Isolated correctness, the four-rectangle Conv2-to-Conv3 chain, and
the applicable full-model lifetimes passed. The following matmul-lowered Conv3
consumes explicit stock-grid provenance before `ExitToStock`.

WIDTH_SHARDED exclusions remain deliberately host-rejected. Its current
activation kernel derives logical rank from `x + grid_width * y`, launches on
the full bounding rectangle, and performs one loopback multicast across that
rectangle. Supporting gaps therefore requires an ordered worker-rank runtime
map plus segmented loopback data/semaphore multicast; planner-only placement
selection would still route through excluded workers and is not enabled.

The first HEIGHT_SHARDED device attempt timed out before a completion boundary.
Static review found that the first stock/remapped discriminator incorrectly used
the device-wide Blackhole worker-grid width, although BOS intentionally uses a
smaller 5-wide 14-core topology. A later boundary translation also proved that
the remapped bounding box cannot uniquely recover the original stock topology.
The final path carries that stock grid explicitly across the Conv boundary. The guarded test
also logs input synchronization/readback and Conv submit/complete separately so
the next clean-device run can localize a failure without another ambiguous log.

The clean-device rerun proved the stock Conv and both input roundtrips, then
timed out only while synchronizing the submitted segmented Conv. Static review
then found two translation defects. First, the HEIGHT helper emitted one
rectangle per horizontal run instead of merging adjacent runs into maximal
rectangles; it also removed the sender from the physical geometry even though
non-loopback multicast safely ignores self. The production receiver set
therefore used seven commands instead of preserving the planner's exact two
physical bands. Second, the 1D weight sender emitted those
independent rectangles with `linked=true`, unlike the already device-tested
segmented activation protocol. The helper now merges identical adjacent-row
runs and restores the sender-containing band while subtracting self from its
destination count (seven to two rectangles for this placement). The sender
issues each resulting rectangle as an independent unlinked multicast before
the existing data barrier and semaphore publication. The guarded rerun
completed with stock/golden and segmented/golden PCC `0.9999153194491188` and
stock/segmented PCC `1.0`, followed by a clean driver shutdown.

All segmented shapes outside that predicate remain fail-fast in both preflight
and the program factory. TensorSpec and BufferDistributionSpec provide only
the generic Cartesian GRID_2D representation and deterministic page ordering;
they do not opt another operation into segmented execution. The Conv call site
enables its capability only for non-matmul 3x3 Conv, and the factory repeats
the stricter placement checks before enqueue.

## Known-good BOS execution shape

`bos_model_copy/resnet50` prepares the input before Conv instead of relying on
an arbitrary NCHW tensor and a forced grid:

1. fold the 224x224 RGB input on the host into a 115x115, 16-channel form;
2. call `determine_parallel_config` for a height-sharded input on the device's
   5x4 compute grid;
3. run the first Conv with an explicit height-sharded `Conv2dConfig`;
4. use layer-specific reshards, including a 14-core height-sharded grid and a
   contiguous 4x4 block-sharded grid.

The standalone runner uses `l1_small_size=32768`; its pytest fixture uses
`l1_small_size=24576`. The experimental C++ test currently uses 16384, so it is
not yet a faithful reproduction of either known model path.

## Safe progression

1. Keep unsupported segmented grids rejected before device enqueue.
2. Open only the ROW_MAJOR column-gap predicate required by the first test.
3. Confirm ordered input upload/readback before launching Conv.
4. Run an isolated guarded 3x3 Conv on stock and two-segment placements.
5. Compare stock/golden, segmented/golden, and stock/segmented outputs.
6. Keep COL_MAJOR and any unvalidated sharding-specific path behind host
   preflight until its isolated guarded test passes.

Build the host test from `/tt-metal` with:

```bash
cmake --build build --target unit_tests_conv_core_replacement
```

To list it without executing device work:

```bash
./build/test/ttnn/unit_tests_conv_core_replacement --gtest_list_tests
```

## BOS baseline tests

`test_bos_resnet50_baseline.py` contains explicitly gated tests for:

- the unmodified full BOS ResNet50 correctness path;
- the folded-input stem Conv in isolation, checked against a PyTorch golden;
- the stem with its known grid explicitly selected;
- layer4 module1's matmul-lowered 1x1 Conv in isolation on a translated 4x4 grid;
- layer4 module1's 3x3 Conv in isolation on a translated 4x4 grid;
- layer4 module1's 3x3 Conv in isolation on an ordered 4x4 logical grid whose
  physical worker placement skips one interior column;
- a layer4 module1 Conv2-to-Conv3 chain that preserves the segmented 16-core
  Conv2 output and exits to the stock contiguous 8-core placement before the
  matmul-lowered Conv3;
- layer4 module1's downsample transition from a translated 4x4 input to the
  stock 5x4 output, validating deterministic remapping-lifetime exit;
- full ResNet50 with layer4's 4x4 `CORE_SET_16` translated right by one worker;
- full ResNet50 with a config-driven skipped column at layer1 module1 Conv2,
  asserting its 14-core HEIGHT_SHARDED stock-to-segmented transition and the
  following Conv3 `ExitToStock` boundary;
- full ResNet50 with a config-driven single skipped column at layer4 module1
  Conv2, including its internal input reshard and Conv3 `ExitToStock` boundary.

The full-model tests and translated-Conv experiment require both
`RUN_CONV_CORE_REPLACEMENT_DEVICE_TESTS=1` and
`RUN_CONV_CORE_REPLACEMENT_HIGH_RISK_TESTS=1`. Do not invoke them with a bare
pytest command. The guarded runner requires an explicit clean-device
confirmation and places pytest under an external watchdog. The watchdog can
release the host terminal, but it still cannot recover a deadlocked board.

After a clean device reset, run the full known-good path first:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  environment_baseline --clean-device-confirmed
```

Run the isolated stem only if that passes:

```bash
RUN_CONV_CORE_REPLACEMENT_DEVICE_TESTS=1 pytest -v -s \
  tests/tt_metal/tt_metal/data_movement/conv_core_replacement/test_bos_resnet50_baseline.py \
  -k stem_conv_baseline
```

Without the environment variable, all device tests skip before opening the device.

After rebuilding on a clean device, validate the first HEIGHT_SHARDED slice in isolation:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer1_height_segmented_column_conv2_isolated --clean-device-confirmed
```

After the isolated test passes, validate its full-model remapping lifetime:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer1_height_segmented_column_conv2_full_model --clean-device-confirmed
```

The final HEIGHT_SHARDED placement slice excludes boundary column 0. This
translates the logical sender from worker `(0,0)` to `(1,0)` and validates a
nonzero physical origin before any full-model attempt:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer1_height_boundary_translation_conv2_isolated --clean-device-confirmed
```

After the isolated boundary translation passes, validate that the translated
Conv2 output propagates into Conv3, exits to the canonical stock grid, and
finishes the full model:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer1_height_boundary_translation_conv2_full_model --clean-device-confirmed
```

The next HEIGHT_SHARDED slice skips physical worker row 2 while preserving the
same 14 logical shards. Rebuild and pass the targeted host tests before running
this isolated case:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer1_height_segmented_row_conv2_isolated --clean-device-confirmed
```

After that isolated row-gap case passes, validate its full-model lifetime:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer1_height_segmented_row_conv2_full_model --clean-device-confirmed
```

A simultaneous row-plus-column exclusion cannot retain the BOS model's 14
logical shards on a 5x4 worker grid: only 12 workers remain. The same real
layer1 Conv2 shape is therefore validated in isolation with a 12-core stock
baseline and a four-rectangle placement:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer1_height_four_rectangle_conv2_isolated --clean-device-confirmed
```

This checks physical X `{0,1,3,4}` and Y `{0,1,3}` without claiming that the
12-core placement is the unchanged BOS full-model topology.

After the isolated case passes, validate the 12-core remapping lifetime through
the following matmul-lowered Conv3. Its stock optimizer selects a canonical
11-core `(5+5+1)` output, so this validates a 12-core segmented to 11-core stock
transition:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer1_height_four_rectangle_conv2_to_conv3_exits_remapping --clean-device-confirmed
```

Next, test the translated BLOCK_SHARDED 1x1 bottleneck Conv:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_translated_conv1_isolated --clean-device-confirmed
```

Both the origin and translated cases use the normal matmul lowering. The
translated test verifies that the matmul program placement, output shard grid,
sender IDs, and multicast coordinates all preserve the input worker-grid
origin.

Then test the translated 3x3 Conv. These isolated tests are the primary
production-path validation and must pass before another translated full-model
run:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_translated_conv2_isolated --clean-device-confirmed
```

Only after the rebuilt host tests pass, run the first segmented production
slice. It compares stock and ordered column-gap placements, verifies input
upload/readback before Conv, and checks output grid order and PCC after Conv:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_segmented_column_conv2_isolated --clean-device-confirmed
```

Then verify placement lifetime across the real Conv2-to-Conv3 boundary:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_segmented_conv2_to_conv3_exits_remapping --clean-device-confirmed
```

The independently validated row-gap slice uses a logical 4x3 Conv2 placement,
because the 5x4 Eagle-N worker grid has no spare row for the logical 4x4 case:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_segmented_row_conv2_isolated --clean-device-confirmed

bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_segmented_row_conv2_to_conv3_exits_remapping --clean-device-confirmed
```

After rebuilding and passing the host suite, validate one row and one column
exclusion together. On Eagle-N's 5x4 grid this uses a logical 4x3 placement and
produces two X segments times two Y segments, or four Cartesian rectangles:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_row_and_column_exclusions_conv2_isolated --clean-device-confirmed

bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_four_rectangle_conv2_to_conv3_exits_remapping --clean-device-confirmed
```

A timeout still requires a board reset before submitting another workload.

Finally, run the config-driven segmented full model:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_segmented_column_conv2_full_model --clean-device-confirmed
```

The real layer4 module2 Conv2 uses a stock 4x2 block-sharded grid, so it can
preserve all eight logical shards while skipping one physical row and column.
Run the four-rectangle full-model path separately:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_four_rectangle_conv2_full_model --clean-device-confirmed
```

After the baseline passes, the translated layer4 experiment must also use the
guarded runner:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  layer4_contiguous_core_set_16_translation --clean-device-confirmed
```

The translated full-model test synchronizes immediately before and after every
Conv.  Its first missing `CONV_SYNC ... phase=complete` line identifies the
operation whose queued device work did not finish; it is diagnostic and is not
the first test to run while changing Conv placement logic.
