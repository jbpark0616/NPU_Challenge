# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Known-good BOS ResNet50 baselines for Conv core-replacement work.

These tests deliberately reuse the model's existing preprocessing and Conv
configuration.  They are gated because killing a host process cannot recover a
device program that has deadlocked.
"""

import csv
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
import models.bos_model.resnet50.tt.ttnn_functional_resnet50 as functional_resnet50
from models.bos_model.resnet50.tests.resnet50_test_infra import create_test_infra
from models.bos_model.resnet50.tt.Conv import convNxN
from tests.ttnn.utils_for_testing import check_with_pcc


DEVICE_TEST_GATE = "RUN_CONV_CORE_REPLACEMENT_DEVICE_TESTS"
HIGH_RISK_TEST_GATE = "RUN_CONV_CORE_REPLACEMENT_HIGH_RISK_TESTS"
MILP_PHASE_ENV = "CONV_CORE_MILP_PHASE"
MILP_MANIFEST_ENV = "CONV_CORE_MILP_MANIFEST"
MILP_SYNC_EACH_CONV_ENV = "CONV_CORE_MILP_SYNC_EACH_CONV"
DEFAULT_MILP_MANIFEST = (
    Path(__file__).resolve().parent
    / "policy"
    / "data"
    / "experiments"
    / "single_axis_split"
    / "full_model_schedule_h20.json"
)

# A skip inside the test body is too late because pytest resolves the `device`
# fixture first.  The module-level marker prevents device creation unless the
# user explicitly opts in before collection.
pytestmark = pytest.mark.skipif(
    os.environ.get(DEVICE_TEST_GATE) != "1",
    reason=f"set {DEVICE_TEST_GATE}=1 to run Conv core-replacement device tests",
)

# Experimental translated Conv and full-model failures can leave a device
# program deadlocked after the host process exits. Keep these tests behind a
# second gate so known-good isolated tests remain independently runnable. Use
# the guarded shell runner below, which also places pytest under an external
# watchdog; pytest's in-process signal timeout cannot interrupt every blocking
# C++/driver call.
high_risk_device_test = pytest.mark.skipif(
    os.environ.get(HIGH_RISK_TEST_GATE) != "1",
    reason=(
        f"set {HIGH_RISK_TEST_GATE}=1 only after a clean device reset, "
        "and run through run_bos_resnet50_full_model_guarded.sh"
    ),
)


def create_bos_test_infra(device):
    return create_test_infra(
        device=device,
        batch_size=1,
        act_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        use_pretrained_weight=False,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=None,
    )


def core_grid_signature(grid):
    """Structural representation; CoreRangeSet Python objects lack value equality."""

    return sorted((r.start.x, r.start.y, r.end.x, r.end.y) for r in grid.ranges())


def core_grid_ordered_signature(grid):
    """Range sequence used by TT-Metal when assigning logical tensor shards."""

    return [(r.start.x, r.start.y, r.end.x, r.end.y) for r in grid.ranges()]


def make_bos_height_sharded_stock_grid(device, num_cores):
    """Build a canonical ROW_MAJOR HEIGHT_SHARDED prefix in BOS's 5-wide topology."""

    compute_grid = device.compute_with_storage_grid_size()
    assert compute_grid.x >= 5 and compute_grid.y >= 4
    full_rows, remainder = divmod(num_cores, 5)
    ranges = []
    if full_rows:
        ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, full_rows - 1))
        )
    if remainder:
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, full_rows),
                ttnn.CoreCoord(remainder - 1, full_rows),
            )
        )
    stock_grid = ttnn.CoreRangeSet(ranges)
    assert stock_grid.num_cores() == num_cores
    return stock_grid


def make_bos_height_sharded_column_gap_grids(device):
    """Return BOS's stock 14-core prefix and its ordered column-gap translation."""

    stock_grid = make_bos_height_sharded_stock_grid(device, 14)
    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for x, y in (
                [(x, y) for y in range(3) for x in (0, 1, 3, 4)]
                + [(0, 3), (1, 3)]
            )
        ]
    )
    assert stock_grid.num_cores() == segmented_grid.num_cores() == 14
    return stock_grid, segmented_grid


def make_bos_height_sharded_boundary_translation_grids(device):
    """Translate BOS's 14-core HEIGHT_SHARDED placement right by excluding column 0."""

    stock_grid = make_bos_height_sharded_stock_grid(device, 14)
    translated_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for x, y in (
                [(x, y) for y in range(3) for x in (1, 2, 3, 4)]
                + [(1, 3), (2, 3)]
            )
        ]
    )
    assert stock_grid.num_cores() == translated_grid.num_cores() == 14
    return stock_grid, translated_grid


def make_bos_height_sharded_row_gap_grids(device):
    """Return BOS's stock 14-core prefix with physical worker row 2 skipped."""

    stock_grid, _ = make_bos_height_sharded_column_gap_grids(device)
    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for x, y in (
                [(x, y) for y in (0, 1) for x in range(5)]
                + [(x, 3) for x in range(4)]
            )
        ]
    )
    assert stock_grid.num_cores() == segmented_grid.num_cores() == 14
    return stock_grid, segmented_grid


def make_bos_height_sharded_four_rectangle_grids(device):
    """Return a 12-core stock prefix and a row-plus-column-gap translation."""

    stock_grid = make_bos_height_sharded_stock_grid(device, 12)
    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in (0, 1, 3)
            for x in (0, 1, 3, 4)
        ]
    )
    assert stock_grid.num_cores() == segmented_grid.num_cores() == 12
    return stock_grid, segmented_grid


def tensor_placement_signature(tensor):
    """Return the placement fields that determine which workers participate in a sharded tensor."""

    memory_config = tensor.memory_config()
    shard_spec = memory_config.shard_spec
    if shard_spec is None:
        return str(memory_config.memory_layout), None, 0
    return (
        str(memory_config.memory_layout),
        core_grid_signature(shard_spec.grid),
        shard_spec.grid.num_cores(),
    )


def resnet50_conv_invocation_names():
    names = ["stem.conv1"]
    for layer_name, block_count in (("layer1", 3), ("layer2", 4), ("layer3", 6), ("layer4", 3)):
        for module_index in range(1, block_count + 1):
            prefix = f"{layer_name}.module{module_index}"
            names.extend((f"{prefix}.conv1", f"{prefix}.conv2", f"{prefix}.conv3"))
            if module_index == 1:
                names.append(f"{prefix}.downsample")
    return names


def layer3_conv2_column_rest_schedule(inference_index=0):
    """Rotate the resting worker column across Layer3's six 3x3 Convs.

    Eagle-N exposes five compute-worker columns while the production Layer3
    BLOCK_SHARDED Conv2 topology uses four. Advancing the phase once per
    inference makes every physical column rest equally over five inferences.
    This is a placement-exposure policy; measured TRISC cycles are evaluated
    separately because the six Conv operations need not have identical cost.
    """

    worker_columns = tuple(range(5))
    op_names = tuple(f"layer3.module{module_index}.conv2" for module_index in range(1, 7))
    phase = (inference_index * len(op_names)) % len(worker_columns)
    return {
        op_name: [worker_columns[(phase + op_index) % len(worker_columns)]]
        for op_index, op_name in enumerate(op_names)
    }


def worker_coordinates_from_grid_signature(signature):
    """Expand a structural CoreRange signature into physical worker coordinates."""

    return {
        (x, y)
        for start_x, start_y, end_x, end_y in signature
        for y in range(start_y, end_y + 1)
        for x in range(start_x, end_x + 1)
    }


def load_milp_full_model_phase(manifest_path, phase):
    """Load one deploy-validation phase and its expected physical workers."""

    manifest_path = Path(manifest_path)
    # utf-8-sig accepts both ordinary UTF-8 and files copied from Windows
    # tools that prepend a UTF-8 BOM.
    with manifest_path.open("r", encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    assert manifest["model"] == "BOS-ResNet50-b1"
    assert manifest["architecture"] == "Eagle-N-5x4-workers"
    horizon = int(manifest["horizon"])
    assert 0 <= phase < horizon

    phase_records = {int(record["phase"]): record for record in manifest["phases"]}
    assert set(phase_records) == set(range(horizon))
    domains = phase_records[phase]["domains"]
    # HEIGHT_SHARDED 1x1 downsample Conv2d is not an executable explicit
    # remapping path. It intentionally remains stock and is absent here.
    assert len(domains) == 16
    assert "layer2.module1.downsample" not in domains

    candidates_path = manifest_path.with_name("full_model_domain_candidates.csv")
    with candidates_path.open("r", encoding="utf-8-sig", newline="") as stream:
        candidates = {
            (row["domain_id"], row["placement_id"]): row
            for row in csv.DictReader(stream)
        }

    rows_by_op = {}
    columns_by_op = {}
    expected_workers_by_entry_op = {}
    consumer_op_by_entry_op = {}
    exit_to_stock_op_by_entry_op = {}
    placement_ids_by_domain = {}
    for domain_id, request in domains.items():
        placement_id = request["placement_id"]
        candidate = candidates[(domain_id, placement_id)]
        assert json.loads(candidate["excluded_rows"]) == request["excluded_rows"]
        assert json.loads(candidate["excluded_columns"]) == request["excluded_columns"]

        if domain_id.endswith(".conv2"):
            entry_op = domain_id
            consumer_op = f"{domain_id.rsplit('.', 1)[0]}.conv3"
            exit_to_stock_op_by_entry_op[entry_op] = consumer_op
        elif domain_id.endswith(".conv2_to_conv3"):
            entry_op = domain_id.removesuffix("_to_conv3")
            consumer_op = f"{entry_op.rsplit('.', 1)[0]}.conv3"
        else:
            raise AssertionError(f"Unsupported MILP placement domain: {domain_id}")

        rows_by_op[entry_op] = list(request["excluded_rows"])
        columns_by_op[entry_op] = list(request["excluded_columns"])
        expected_workers_by_entry_op[entry_op] = {
            tuple(worker) for worker in json.loads(candidate["ordered_entry_workers"])
        }
        if consumer_op is not None:
            consumer_op_by_entry_op[entry_op] = consumer_op
        placement_ids_by_domain[domain_id] = placement_id

    return {
        "manifest": manifest,
        "rows_by_op": rows_by_op,
        "columns_by_op": columns_by_op,
        "expected_workers_by_entry_op": expected_workers_by_entry_op,
        "consumer_op_by_entry_op": consumer_op_by_entry_op,
        "exit_to_stock_op_by_entry_op": exit_to_stock_op_by_entry_op,
        "placement_ids_by_domain": placement_ids_by_domain,
    }


def install_conv_sync_tracing(
    monkeypatch,
    *,
    enable_core_remapping=False,
    placement_trace=None,
    excluded_worker_columns_by_op=None,
    excluded_worker_rows_by_op=None,
    input_stock_grids_by_op=None,
    capture_input_stock_grid_for_exit_by_op=None,
    synchronize_each_conv=True,
):
    """Force queue completion around each model Conv for deadlock localization."""

    original_conv = functional_resnet50.convNxN
    invocation = 0
    invocation_names = resnet50_conv_invocation_names()
    captured_input_stock_grids_by_op = {}

    def synchronized_conv(*args, **kwargs):
        nonlocal invocation
        invocation += 1
        op_name = (
            invocation_names[invocation - 1]
            if invocation <= len(invocation_names)
            else f"unexpected_conv_{invocation}"
        )
        device = args[1]
        input_tensor = args[0]
        input_height = args[2]
        input_width = args[3]
        in_channels = args[4]
        out_channels = args[5]
        kernel_size = args[14] if len(args) > 14 else kwargs.get("kernel_size")
        stride = args[15] if len(args) > 15 else kwargs.get("stride", (1, 1))
        input_layout, input_grid, input_cores = tensor_placement_signature(input_tensor)
        logger.info(
            "CONV_PLACEMENT invocation={} op={} phase=input hw={}x{} kernel={} stride={} channels={}->{} "
            "layout={} cores={} grid={}",
            invocation,
            op_name,
            input_height,
            input_width,
            kernel_size,
            stride,
            in_channels,
            out_channels,
            input_layout,
            input_cores,
            input_grid,
        )
        if synchronize_each_conv:
            ttnn.synchronize_device(device)
        if capture_input_stock_grid_for_exit_by_op and op_name in capture_input_stock_grid_for_exit_by_op:
            shard_spec = input_tensor.memory_config().shard_spec
            assert shard_spec is not None
            exit_op = capture_input_stock_grid_for_exit_by_op[op_name]
            captured_input_stock_grids_by_op[exit_op] = shard_spec.grid
        if enable_core_remapping:
            args[9].enable_core_remapping = True
        if excluded_worker_columns_by_op is not None:
            args[9].excluded_worker_columns = excluded_worker_columns_by_op.get(op_name, [])
        if excluded_worker_rows_by_op is not None:
            args[9].excluded_worker_rows = excluded_worker_rows_by_op.get(op_name, [])
        if input_stock_grids_by_op and op_name in input_stock_grids_by_op:
            args[9].core_remapping_input_stock_grid = input_stock_grids_by_op[op_name]
        elif op_name in captured_input_stock_grids_by_op:
            args[9].core_remapping_input_stock_grid = captured_input_stock_grids_by_op[op_name]
        result = original_conv(*args, **kwargs)
        output_tensor = result[0]
        output_layout, output_grid, output_cores = tensor_placement_signature(output_tensor)
        logger.info(
            "CONV_PLACEMENT invocation={} op={} phase=output kernel={} stride={} channels={}->{} "
            "layout={} cores={} grid={}",
            invocation,
            op_name,
            kernel_size,
            stride,
            in_channels,
            out_channels,
            output_layout,
            output_cores,
            output_grid,
        )
        if placement_trace is not None:
            placement_trace.append(
                {
                    "op": op_name,
                    "input_layout": input_layout,
                    "input_grid": input_grid,
                    "input_cores": input_cores,
                    "output_layout": output_layout,
                    "output_grid": output_grid,
                    "output_cores": output_cores,
                }
            )
        if synchronize_each_conv:
            ttnn.synchronize_device(device)
            logger.info("CONV_SYNC invocation={} phase=complete", invocation)
        return result

    monkeypatch.setattr(functional_resnet50, "convNxN", synchronized_conv)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_unmodified_bos_resnet50_environment_baseline(device, monkeypatch):
    """Trace every Conv placement while proving the unmodified BOS model path completes."""

    install_conv_sync_tracing(monkeypatch)
    test_infra = create_bos_test_infra(device)
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)

    test_infra.run()
    pcc_passed, pcc_message = test_infra.validate()
    assert pcc_passed, pcc_message


def run_bos_resnet50_stem_conv(device, *, explicit_grid_override):
    grid_mode = "explicit-same-grid" if explicit_grid_override else "automatic-grid"
    logger.info(
        "CONV_CORE_REPLACEMENT_STEM mode={} step=1 creating BOS test infrastructure",
        grid_mode,
    )
    test_infra = create_bos_test_infra(device)
    model = test_infra.ttnn_resnet50_model
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)

    assert input_mem_config.shard_spec is not None
    selected_grid = input_mem_config.shard_spec.grid
    if explicit_grid_override:
        model.conv1_config.core_grid = selected_grid
        model.conv1_config.override_sharding_config = True

    logger.info(
        "CONV_CORE_REPLACEMENT_STEM mode={} step=2 building folded-input golden "
        "input_shape={} weight_shape={} worker_cores={} worker_ranges={}",
        grid_mode,
        tuple(tt_inputs_host.shape),
        tuple(model.conv1_weight_tensor.shape),
        selected_grid.num_cores(),
        core_grid_signature(selected_grid),
    )
    folded_torch = ttnn.to_torch(tt_inputs_host).to(torch.float32)
    weight_torch = ttnn.to_torch(model.conv1_weight_tensor).to(torch.float32)
    bias_torch = ttnn.to_torch(model.conv1_bias_tensor).reshape(-1).to(torch.float32)
    golden = torch.nn.functional.conv2d(
        folded_torch.permute(0, 3, 1, 2),
        weight_torch,
        bias_torch,
        stride=model.conv1_stride,
        padding=model.conv1_padding,
    )
    golden = torch.relu(golden)

    logger.info(
        "CONV_CORE_REPLACEMENT_STEM mode={} step=3 uploading known-good height-sharded input",
        grid_mode,
    )
    stem_input = tt_inputs_host.to(device, input_mem_config)
    assert stem_input.memory_config().shard_spec is not None
    uploaded_grid = stem_input.memory_config().shard_spec.grid
    assert uploaded_grid.num_cores() == selected_grid.num_cores()
    assert core_grid_signature(uploaded_grid) == core_grid_signature(selected_grid)
    n, h, w, c = stem_input.shape
    stem_input = ttnn.reshape(stem_input, (1, 1, n * h * w, c))
    logger.info(
        "CONV_CORE_REPLACEMENT_STEM mode={} step=4 launching stock BOS stem conv",
        grid_mode,
    )
    stem_output, [output_height, output_width], _ = convNxN(
        stem_input,
        device,
        model.conv1_input_height,
        model.conv1_input_width,
        model.conv1_input_channels,
        model.conv1_output_channels,
        model.batch_size,
        model.conv1_weight_tensor,
        model.conv1_bias_tensor,
        model.conv1_config,
        model.conv1_compute_config,
        model.model_config["ACTIVATIONS_DTYPE"],
        True,
        True,
        model.conv1_kernel_size,
        model.conv1_stride,
        model.conv1_padding,
    )
    logger.info(
        "CONV_CORE_REPLACEMENT_STEM mode={} step=5 conv returned output={}x{}; reading output",
        grid_mode,
        output_height,
        output_width,
    )

    assert (output_height, output_width) == (
        model.conv1_output_height,
        model.conv1_output_width,
    )
    output_torch = ttnn.to_torch(stem_output, device=device).to(torch.float32)
    assert output_torch.shape[-1] == model.conv1_output_channels
    expected_rows = model.batch_size * output_height * output_width
    output_rows = output_torch.reshape(-1, model.conv1_output_channels)
    assert output_rows.shape[0] >= expected_rows
    output_torch = output_rows[:expected_rows].reshape(
        model.batch_size,
        output_height,
        output_width,
        model.conv1_output_channels,
    )
    output_torch = output_torch.permute(0, 3, 1, 2)

    pcc_passed, pcc_message = check_with_pcc(golden, output_torch, pcc=0.99)
    logger.info(
        "CONV_CORE_REPLACEMENT_STEM mode={} step=6 correctness {}",
        grid_mode,
        pcc_message,
    )
    assert pcc_passed, pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bos_resnet50_stem_conv_baseline(device):
    """Run the BOS stem Conv with its automatic grid and compare with PyTorch."""

    run_bos_resnet50_stem_conv(device, explicit_grid_override=False)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bos_resnet50_stem_conv_explicit_same_grid(device):
    """Override Conv with the exact grid already selected for the BOS input."""

    run_bos_resnet50_stem_conv(device, explicit_grid_override=True)


def run_bos_resnet50_layer4_conv1_placements(
    device, placements, logical_grid_shape=(4, 4), *, reshard_if_not_optimal=False
):
    """Compare layer4 module1's production matmul-lowered 1x1 Conv placements."""

    compute_grid = device.compute_with_storage_grid_size()
    assert compute_grid.x >= 5 and compute_grid.y >= 4
    assert len(placements) >= 2
    logical_grid_x, logical_grid_y = logical_grid_shape
    stock_grid = placements[0][1]
    assert stock_grid.num_cores() == logical_grid_x * logical_grid_y

    test_infra = create_bos_test_infra(device)
    model = test_infra.ttnn_resnet50_model
    layer4_module1 = model.layer_component["layer4"]["module"][0]

    batch_size, input_height, input_width = 1, 14, 14
    in_channels = layer4_module1.conv1_input_channels
    out_channels = layer4_module1.conv1_output_channels
    torch_input = torch.randn(
        (batch_size, 1, input_height * input_width, in_channels), dtype=torch.bfloat16
    )
    host_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
    )
    quantized_input = ttnn.to_torch(host_input).to(torch.float32)
    weight = ttnn.to_torch(layer4_module1.conv1_weight_tensor).to(torch.float32)
    bias = ttnn.to_torch(layer4_module1.conv1_bias_tensor).reshape(-1).to(torch.float32)
    golden = torch.nn.functional.conv2d(
        quantized_input.reshape(batch_size, input_height, input_width, in_channels).permute(0, 3, 1, 2),
        weight,
        bias,
    )
    golden = torch.relu(golden).permute(0, 2, 3, 1).reshape(-1, out_channels)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=model.model_config["MATH_FIDELITY"],
        packer_l1_acc=True,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=model.model_config["WEIGHTS_DTYPE"],
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        reshard_if_not_optimal=reshard_if_not_optimal,
        enable_core_remapping=True,
    )

    def run_conv_at_grid(grid, placement):
        assert grid.num_cores() == logical_grid_x * logical_grid_y
        flattened_height = batch_size * input_height * input_width
        padded_height = (
            (flattened_height + logical_grid_y * 32 - 1) // (logical_grid_y * 32)
        ) * (logical_grid_y * 32)
        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid,
                (padded_height // logical_grid_y, in_channels // logical_grid_x),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        device_input = host_input.to(device, input_memory_config)
        uploaded_memory_config = device_input.memory_config()
        assert uploaded_memory_config.shard_spec is not None
        uploaded_grid = uploaded_memory_config.shard_spec.grid
        logger.info(
            "LAYER4_CONV1_ISOLATED placement={} phase=input-uploaded requested_grid={} "
            "uploaded_grid={} shape={} channels={}->{}",
            placement,
            core_grid_signature(grid),
            core_grid_signature(uploaded_grid),
            tuple(torch_input.shape),
            in_channels,
            out_channels,
        )
        assert core_grid_signature(uploaded_grid) == core_grid_signature(grid)
        ttnn.synchronize_device(device)
        uploaded_input = ttnn.to_torch(device_input, device=device).to(torch.float32)
        input_roundtrip_passed, input_roundtrip_message = check_with_pcc(
            quantized_input.reshape(-1, in_channels),
            uploaded_input.reshape(-1, uploaded_input.shape[-1])[: input_height * input_width, :in_channels],
            pcc=0.999,
        )
        logger.info(
            "LAYER4_CONV1_ISOLATED placement={} input_roundtrip={}",
            placement,
            input_roundtrip_message,
        )
        assert input_roundtrip_passed, f"{placement} input upload/readback failed: {input_roundtrip_message}"

        conv_config.core_remapping_input_stock_grid = (
            None if placement == placements[0][0] else stock_grid
        )
        output, [output_height, output_width], _ = functional_resnet50.conv1x1(
            device_input,
            device,
            input_height,
            input_width,
            in_channels,
            out_channels,
            batch_size,
            layer4_module1.conv1_weight_tensor,
            layer4_module1.conv1_bias_tensor,
            conv_config,
            compute_config,
            model.model_config["ACTIVATIONS_DTYPE"],
            True,
            True,
        )
        logger.info("LAYER4_CONV1_ISOLATED placement={} phase=conv-submitted", placement)
        ttnn.synchronize_device(device)
        logger.info("LAYER4_CONV1_ISOLATED placement={} phase=conv-complete", placement)

        assert (output_height, output_width) == (input_height, input_width)
        output_memory_config = output.memory_config()
        assert output_memory_config.shard_spec is not None
        output_grid = output_memory_config.shard_spec.grid
        logger.info(
            "LAYER4_CONV1_ISOLATED placement={} output_grid={} output_shape={}",
            placement,
            core_grid_signature(output_grid),
            tuple(output.shape),
        )
        assert core_grid_signature(output_grid) == core_grid_signature(grid)
        output_torch = ttnn.to_torch(output, device=device).to(torch.float32)
        return output_torch.reshape(-1, output_torch.shape[-1])[
            : input_height * input_width, :out_channels
        ]

    outputs = {}
    for index, (placement, grid) in enumerate(placements):
        if index:
            # Physical placement and multicast routes are compiled program state.
            device.clear_program_cache()
        outputs[placement] = run_conv_at_grid(grid, placement)

    baseline_name = placements[0][0]
    baseline_output = outputs[baseline_name]
    baseline_golden_passed, baseline_golden_message = check_with_pcc(golden, baseline_output, pcc=0.99)
    assert baseline_golden_passed, baseline_golden_message
    for placement, _ in placements[1:]:
        candidate_golden_passed, candidate_golden_message = check_with_pcc(
            golden, outputs[placement], pcc=0.99
        )
        placement_passed, placement_message = check_with_pcc(
            baseline_output, outputs[placement], pcc=0.999
        )
        logger.info(
            "LAYER4_CONV1_ISOLATED baseline={} candidate={} baseline_vs_golden={} "
            "candidate_vs_golden={} baseline_vs_candidate={}",
            baseline_name,
            placement,
            baseline_golden_message,
            candidate_golden_message,
            placement_message,
        )
        assert candidate_golden_passed, candidate_golden_message
        assert placement_passed, placement_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_translated_conv1_isolated(device):
    """Compare the 1x1 Conv at stock and translated contiguous placements."""

    stock_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    shifted_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(4, 3))})
    run_bos_resnet50_layer4_conv1_placements(
        device, [("stock", stock_grid), ("translated", shifted_grid)]
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_translated_conv1_reshard_if_not_optimal_isolated(device):
    """Keep an equivalent translated Matmul placement when optimal resharding is enabled."""

    stock_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    shifted_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(4, 3))})
    run_bos_resnet50_layer4_conv1_placements(
        device,
        [("stock", stock_grid), ("translated", shifted_grid)],
        reshard_if_not_optimal=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_segmented_column_conv1_isolated(device):
    """Split only the activation row-multicast axis of a production 1x1 Conv."""

    stock_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in range(4)
            for x in (0, 1, 3, 4)
        ]
    )
    run_bos_resnet50_layer4_conv1_placements(
        device, [("stock", stock_grid), ("segmented-column", segmented_grid)]
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_segmented_row_conv1_isolated(device):
    """Split only the weight column-multicast axis of a production 1x1 Conv."""

    stock_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 2))})
    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in (0, 1, 3)
            for x in range(4)
        ]
    )
    run_bos_resnet50_layer4_conv1_placements(
        device,
        [("stock-4x3", stock_grid), ("segmented-row", segmented_grid)],
        logical_grid_shape=(4, 3),
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_four_rectangle_conv1_isolated(device):
    """Exercise both segmented Matmul multicast axes on a production 1x1 Conv."""

    stock_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 2))})
    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in (0, 1, 3)
            for x in (0, 1, 3, 4)
        ]
    )
    assert segmented_grid.num_cores() == 12
    assert core_grid_ordered_signature(segmented_grid) == [
        (x, y, x, y) for y in (0, 1, 3) for x in (0, 1, 3, 4)
    ]
    run_bos_resnet50_layer4_conv1_placements(
        device,
        [("stock-4x3", stock_grid), ("four-rectangle", segmented_grid)],
        logical_grid_shape=(4, 3),
    )


def run_bos_resnet50_layer4_conv2_placements(
    device, placements, logical_grid_shape=(4, 4), placement_requests=None
):
    """Compare layer4 module1's 3x3 Conv across explicit worker placements."""

    compute_grid = device.compute_with_storage_grid_size()
    assert compute_grid.x >= 5 and compute_grid.y >= 4
    assert len(placements) >= 2
    placement_requests = placement_requests or {}

    test_infra = create_bos_test_infra(device)
    model = test_infra.ttnn_resnet50_model
    layer4_module1 = model.layer_component["layer4"]["module"][0]

    batch_size, input_height, input_width = 1, 14, 14
    in_channels = layer4_module1.conv2_input_channels
    out_channels = layer4_module1.conv2_output_channels
    torch_input = torch.randn(
        (batch_size, 1, input_height * input_width, in_channels), dtype=torch.bfloat16
    )
    host_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
    )
    quantized_input = ttnn.to_torch(host_input).to(torch.float32)
    weight = ttnn.to_torch(layer4_module1.conv2_weight_tensor).to(torch.float32)
    bias = ttnn.to_torch(layer4_module1.conv2_bias_tensor).reshape(-1).to(torch.float32)
    golden = torch.nn.functional.conv2d(
        quantized_input.reshape(batch_size, input_height, input_width, in_channels).permute(0, 3, 1, 2),
        weight,
        bias,
        stride=(layer4_module1.stride, layer4_module1.stride),
        padding=(1, 1),
    )
    golden = torch.relu(golden).permute(0, 2, 3, 1).reshape(-1, out_channels)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=model.model_config["MATH_FIDELITY"],
        packer_l1_acc=True,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=model.model_config["WEIGHTS_DTYPE"],
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=False,
        reallocate_halo_output=False,
        act_block_h_override=0,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        reshard_if_not_optimal=False,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        force_split_reader=False,
        enable_core_remapping=True,
    )

    logical_grid_x, logical_grid_y = logical_grid_shape
    tile_height = 32
    flattened_height = batch_size * input_height * input_width
    height_alignment = logical_grid_y * tile_height
    padded_height = ((flattened_height + height_alignment - 1) // height_alignment) * height_alignment
    assert in_channels % logical_grid_x == 0
    input_shard_shape = (padded_height // logical_grid_y, in_channels // logical_grid_x)

    def run_conv_at_grid(grid, placement):
        request = placement_requests.get(placement, {})
        conv_config.excluded_worker_rows = request.get("excluded_worker_rows", [])
        conv_config.excluded_worker_columns = request.get("excluded_worker_columns", [])
        expected_output_grid = request.get("expected_output_grid", grid)
        # create_sharded_memory_config_ infers BLOCK_SHARDED dimensions from
        # the physical bounding box and rejects gaps. Build the ShardSpec from
        # the dense logical topology so a skipped physical row or column does
        # not become a fictitious extra tensor-shard dimension.
        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )
        device_input = host_input.to(device, input_memory_config)
        logger.info(
            "LAYER4_CONV2_ISOLATED placement={} phase=input-uploaded grid={} excluded_rows={} "
            "excluded_columns={} shape={} channels={}->{}",
            placement,
            core_grid_signature(grid),
            conv_config.excluded_worker_rows,
            conv_config.excluded_worker_columns,
            tuple(torch_input.shape),
            in_channels,
            out_channels,
        )
        ttnn.synchronize_device(device)
        logger.info("LAYER4_CONV2_ISOLATED placement={} phase=input-synchronized", placement)
        uploaded_input = ttnn.to_torch(device_input, device=device).to(torch.float32)
        input_roundtrip_passed, input_roundtrip_message = check_with_pcc(
            quantized_input.reshape(-1, in_channels),
            uploaded_input.reshape(-1, uploaded_input.shape[-1])[: input_height * input_width, :in_channels],
            pcc=0.999,
        )
        logger.info(
            "LAYER4_CONV2_ISOLATED placement={} input_roundtrip={}",
            placement,
            input_roundtrip_message,
        )
        assert input_roundtrip_passed, f"{placement} input upload/readback failed: {input_roundtrip_message}"

        output, [output_height, output_width], _ = functional_resnet50.conv3x3(
            device_input,
            device,
            input_height,
            input_width,
            in_channels,
            out_channels,
            batch_size,
            layer4_module1.conv2_weight_tensor,
            layer4_module1.conv2_bias_tensor,
            conv_config,
            compute_config,
            model.model_config["ACTIVATIONS_DTYPE"],
            True,
            True,
            stride=(layer4_module1.stride, layer4_module1.stride),
        )
        logger.info("LAYER4_CONV2_ISOLATED placement={} phase=conv-submitted", placement)
        ttnn.synchronize_device(device)
        logger.info("LAYER4_CONV2_ISOLATED placement={} phase=conv-complete", placement)

        assert (output_height, output_width) == (7, 7)
        output_memory_config = output.memory_config()
        assert output_memory_config.shard_spec is not None
        logger.info(
            "LAYER4_CONV2_ISOLATED placement={} output_grid={} output_shape={}",
            placement,
            core_grid_signature(output_memory_config.shard_spec.grid),
            tuple(output.shape),
        )
        assert core_grid_signature(output_memory_config.shard_spec.grid) == core_grid_signature(expected_output_grid)
        if len(expected_output_grid.ranges()) > 1:
            assert core_grid_ordered_signature(
                output_memory_config.shard_spec.grid
            ) == core_grid_ordered_signature(expected_output_grid)
        output_torch = ttnn.to_torch(output, device=device).to(torch.float32)
        return output_torch.reshape(-1, output_torch.shape[-1])[
            : output_height * output_width, :out_channels
        ]

    outputs = {}
    for index, (placement, grid) in enumerate(placements):
        if index:
            # Placement and segmented NoC routes are compiled program state.
            device.clear_program_cache()
        outputs[placement] = run_conv_at_grid(grid, placement)

    baseline_name = placements[0][0]
    baseline_output = outputs[baseline_name]
    baseline_golden_passed, baseline_golden_message = check_with_pcc(golden, baseline_output, pcc=0.99)
    for placement, _ in placements[1:]:
        candidate_golden_passed, candidate_golden_message = check_with_pcc(
            golden, outputs[placement], pcc=0.99
        )
        placement_passed, placement_message = check_with_pcc(
            baseline_output, outputs[placement], pcc=0.999
        )
        logger.info(
            "LAYER4_CONV2_ISOLATED baseline={} candidate={} baseline_vs_golden={} "
            "candidate_vs_golden={} baseline_vs_candidate={}",
            baseline_name,
            placement,
            baseline_golden_message,
            candidate_golden_message,
            placement_message,
        )
        assert baseline_golden_passed == candidate_golden_passed, (
            f"Placement {placement} changed whether Conv agrees with the synthetic golden: "
            f"baseline={baseline_golden_message}, candidate={candidate_golden_message}"
        )
        assert placement_passed, placement_message


def run_bos_resnet50_layer1_height_conv2_placements(
    device, *, segmented_grid, segmented_placement, excluded_rows, excluded_columns
):
    """Compare stock and remapped placements on the production HEIGHT_SHARDED Conv2."""

    num_cores = segmented_grid.num_cores()
    stock_grid = make_bos_height_sharded_stock_grid(device, num_cores)

    test_infra = create_bos_test_infra(device)
    model = test_infra.ttnn_resnet50_model
    layer1_module1 = model.layer_component["layer1"]["module"][0]
    batch_size, input_height, input_width = 1, 56, 56
    in_channels = layer1_module1.conv2_input_channels
    out_channels = layer1_module1.conv2_output_channels
    assert in_channels == out_channels == 64

    torch_input = torch.randn(
        (batch_size, 1, input_height * input_width, in_channels), dtype=torch.bfloat16
    )
    host_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    quantized_input = ttnn.to_torch(host_input).to(torch.float32)
    weight = ttnn.to_torch(layer1_module1.conv2_weight_tensor).to(torch.float32)
    bias = ttnn.to_torch(layer1_module1.conv2_bias_tensor).reshape(-1).to(torch.float32)
    golden = torch.nn.functional.conv2d(
        quantized_input.reshape(batch_size, input_height, input_width, in_channels).permute(0, 3, 1, 2),
        weight,
        bias,
        padding=(1, 1),
    )
    golden = torch.relu(golden).permute(0, 2, 3, 1).reshape(-1, out_channels)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=model.model_config["MATH_FIDELITY"],
        packer_l1_acc=True,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=model.model_config["WEIGHTS_DTYPE"],
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=False,
        reallocate_halo_output=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=False,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        force_split_reader=True,
        enable_core_remapping=True,
    )
    flattened_height = batch_size * input_height * input_width
    height_alignment = num_cores * 32
    padded_height = ((flattened_height + height_alignment - 1) // height_alignment) * height_alignment
    input_shard_shape = (padded_height // num_cores, in_channels)

    def run_at_grid(input_grid, expected_output_grid, placement, requested_rows, requested_columns):
        conv_config.excluded_worker_rows = requested_rows
        conv_config.excluded_worker_columns = requested_columns
        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(input_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )
        device_input = host_input.to(device, input_memory_config)
        logger.info(
            "LAYER1_HEIGHT_CONV2_ISOLATED placement={} phase=input-uploaded grid={} "
            "excluded_rows={} excluded_columns={}",
            placement,
            core_grid_ordered_signature(input_grid),
            requested_rows,
            requested_columns,
        )
        ttnn.synchronize_device(device)
        logger.info(
            "LAYER1_HEIGHT_CONV2_ISOLATED placement={} phase=input-synchronized",
            placement,
        )
        uploaded_input = ttnn.to_torch(device_input, device=device).to(torch.float32)
        input_roundtrip_passed, input_roundtrip_message = check_with_pcc(
            quantized_input.reshape(-1, in_channels),
            uploaded_input.reshape(-1, uploaded_input.shape[-1])[
                : input_height * input_width, :in_channels
            ],
            pcc=0.999,
        )
        logger.info(
            "LAYER1_HEIGHT_CONV2_ISOLATED placement={} input_roundtrip={}",
            placement,
            input_roundtrip_message,
        )
        assert input_roundtrip_passed, (
            f"{placement} input upload/readback failed: {input_roundtrip_message}"
        )

        output, [output_height, output_width], _ = functional_resnet50.conv3x3(
            device_input,
            device,
            input_height,
            input_width,
            in_channels,
            out_channels,
            batch_size,
            layer1_module1.conv2_weight_tensor,
            layer1_module1.conv2_bias_tensor,
            conv_config,
            compute_config,
            model.model_config["ACTIVATIONS_DTYPE"],
            True,
            True,
            stride=(1, 1),
        )
        logger.info(
            "LAYER1_HEIGHT_CONV2_ISOLATED placement={} phase=conv-submitted",
            placement,
        )
        ttnn.synchronize_device(device)
        logger.info(
            "LAYER1_HEIGHT_CONV2_ISOLATED placement={} phase=conv-complete",
            placement,
        )
        assert (output_height, output_width) == (input_height, input_width)
        assert output.memory_config().shard_spec is not None
        output_grid = output.memory_config().shard_spec.grid
        assert core_grid_signature(output_grid) == core_grid_signature(expected_output_grid)
        assert core_grid_ordered_signature(output_grid) == core_grid_ordered_signature(expected_output_grid)
        output_torch = ttnn.to_torch(output, device=device).to(torch.float32)
        return output_torch.reshape(-1, output_torch.shape[-1])[: input_height * input_width, :out_channels]

    stock_output = run_at_grid(stock_grid, stock_grid, "stock", [], [])
    device.clear_program_cache()
    segmented_output = run_at_grid(
        stock_grid,
        segmented_grid,
        segmented_placement,
        excluded_rows,
        excluded_columns,
    )
    stock_golden_passed, stock_golden_message = check_with_pcc(golden, stock_output, pcc=0.99)
    segmented_golden_passed, segmented_golden_message = check_with_pcc(golden, segmented_output, pcc=0.99)
    placement_passed, placement_message = check_with_pcc(stock_output, segmented_output, pcc=0.999)
    logger.info(
        "LAYER1_HEIGHT_CONV2_ISOLATED stock_vs_golden={} segmented_vs_golden={} stock_vs_segmented={}",
        stock_golden_message,
        segmented_golden_message,
        placement_message,
    )
    assert stock_golden_passed == segmented_golden_passed
    assert placement_passed, placement_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer1_height_segmented_column_conv2_isolated(device):
    """Skip one worker column on the production 14-core HEIGHT_SHARDED Conv2 path."""

    _, segmented_grid = make_bos_height_sharded_column_gap_grids(device)
    run_bos_resnet50_layer1_height_conv2_placements(
        device,
        segmented_grid=segmented_grid,
        segmented_placement="configured-column-gap",
        excluded_rows=[],
        excluded_columns=[2],
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer1_height_boundary_translation_conv2_isolated(device):
    """Translate the production 14-core HEIGHT_SHARDED Conv2 right by one column."""

    _, translated_grid = make_bos_height_sharded_boundary_translation_grids(device)
    run_bos_resnet50_layer1_height_conv2_placements(
        device,
        segmented_grid=translated_grid,
        segmented_placement="configured-boundary-translation",
        excluded_rows=[],
        excluded_columns=[0],
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer1_height_segmented_row_conv2_isolated(device):
    """Skip worker row 2 on the production 14-core HEIGHT_SHARDED Conv2 path."""

    _, segmented_grid = make_bos_height_sharded_row_gap_grids(device)
    run_bos_resnet50_layer1_height_conv2_placements(
        device,
        segmented_grid=segmented_grid,
        segmented_placement="configured-row-gap",
        excluded_rows=[2],
        excluded_columns=[],
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer1_height_four_rectangle_conv2_isolated(device):
    """Skip one worker row and column on a 12-core HEIGHT_SHARDED Conv2 path."""

    _, segmented_grid = make_bos_height_sharded_four_rectangle_grids(device)
    run_bos_resnet50_layer1_height_conv2_placements(
        device,
        segmented_grid=segmented_grid,
        segmented_placement="configured-four-rectangles",
        excluded_rows=[2],
        excluded_columns=[2],
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer1_height_four_rectangle_conv2_to_conv3_exits_remapping(device):
    """Run a 12-core four-rectangle Conv2 and exit to stock at Conv3."""

    stock_grid, segmented_grid = make_bos_height_sharded_four_rectangle_grids(device)
    stock_conv3_output_grid = make_bos_height_sharded_stock_grid(device, 11)
    test_infra = create_bos_test_infra(device)
    model = test_infra.ttnn_resnet50_model
    layer1_module1 = model.layer_component["layer1"]["module"][0]

    batch_size, input_height, input_width = 1, 56, 56
    conv2_in_channels = layer1_module1.conv2_input_channels
    conv2_out_channels = layer1_module1.conv2_output_channels
    conv3_in_channels = layer1_module1.conv3_input_channels
    conv3_out_channels = layer1_module1.conv3_output_channels
    assert conv2_in_channels == conv2_out_channels == conv3_in_channels == 64

    torch_input = torch.randn(
        (batch_size, 1, input_height * input_width, conv2_in_channels),
        dtype=torch.bfloat16,
    )
    host_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    quantized_input = ttnn.to_torch(host_input).to(torch.float32)

    conv2_weight = ttnn.to_torch(layer1_module1.conv2_weight_tensor).to(torch.float32)
    conv2_bias = ttnn.to_torch(layer1_module1.conv2_bias_tensor).reshape(-1).to(torch.float32)
    conv3_weight = ttnn.to_torch(layer1_module1.conv3_weight_tensor).to(torch.float32)
    conv3_bias = ttnn.to_torch(layer1_module1.conv3_bias_tensor).reshape(-1).to(torch.float32)
    golden = torch.nn.functional.conv2d(
        quantized_input.reshape(batch_size, input_height, input_width, conv2_in_channels).permute(0, 3, 1, 2),
        conv2_weight,
        conv2_bias,
        padding=(1, 1),
    )
    golden = torch.relu(golden)
    golden = torch.nn.functional.conv2d(golden, conv3_weight, conv3_bias)
    golden = golden.permute(0, 2, 3, 1).reshape(-1, conv3_out_channels)

    num_cores = stock_grid.num_cores()
    flattened_height = batch_size * input_height * input_width
    height_alignment = num_cores * 32
    padded_height = ((flattened_height + height_alignment - 1) // height_alignment) * height_alignment
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            stock_grid,
            (padded_height // num_cores, conv2_in_channels),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    device_input = host_input.to(device, input_memory_config)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=model.model_config["MATH_FIDELITY"],
        packer_l1_acc=True,
    )
    conv2_config = ttnn.Conv2dConfig(
        weights_dtype=model.model_config["WEIGHTS_DTYPE"],
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=False,
        reallocate_halo_output=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=False,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        force_split_reader=True,
        enable_core_remapping=True,
    )
    conv2_config.excluded_worker_rows = [2]
    conv2_config.excluded_worker_columns = [2]
    conv3_config = ttnn.Conv2dConfig(
        weights_dtype=model.model_config["WEIGHTS_DTYPE"],
        deallocate_activation=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=False,
        enable_core_remapping=True,
        core_remapping_input_stock_grid=stock_grid,
    )

    logger.info(
        "LAYER1_HEIGHT_FOUR_RECTANGLE_CHAIN phase=conv2-input stock_grid={} expected_segmented_grid={}",
        core_grid_signature(stock_grid),
        core_grid_signature(segmented_grid),
    )
    ttnn.synchronize_device(device)
    conv2_output, [conv2_output_height, conv2_output_width], _ = functional_resnet50.conv3x3(
        device_input,
        device,
        input_height,
        input_width,
        conv2_in_channels,
        conv2_out_channels,
        batch_size,
        layer1_module1.conv2_weight_tensor,
        layer1_module1.conv2_bias_tensor,
        conv2_config,
        compute_config,
        model.model_config["ACTIVATIONS_DTYPE"],
        True,
        True,
        stride=(1, 1),
    )
    ttnn.synchronize_device(device)
    assert (conv2_output_height, conv2_output_width) == (input_height, input_width)
    assert conv2_output.memory_config().shard_spec is not None
    conv2_output_grid = conv2_output.memory_config().shard_spec.grid
    assert core_grid_signature(conv2_output_grid) == core_grid_signature(segmented_grid)
    assert core_grid_ordered_signature(conv2_output_grid) == core_grid_ordered_signature(segmented_grid)
    logger.info(
        "LAYER1_HEIGHT_FOUR_RECTANGLE_CHAIN phase=conv2-complete output_grid={}",
        core_grid_ordered_signature(conv2_output_grid),
    )

    conv3_output, [conv3_output_height, conv3_output_width], _ = functional_resnet50.conv1x1(
        conv2_output,
        device,
        conv2_output_height,
        conv2_output_width,
        conv3_in_channels,
        conv3_out_channels,
        batch_size,
        layer1_module1.conv3_weight_tensor,
        layer1_module1.conv3_bias_tensor,
        conv3_config,
        compute_config,
        model.model_config["ACTIVATIONS_DTYPE"],
        True,
        True,
        stride=(1, 1),
    )
    logger.info("LAYER1_HEIGHT_FOUR_RECTANGLE_CHAIN phase=conv3-submitted")
    ttnn.synchronize_device(device)
    logger.info("LAYER1_HEIGHT_FOUR_RECTANGLE_CHAIN phase=conv3-complete")
    assert (conv3_output_height, conv3_output_width) == (input_height, input_width)
    assert conv3_output.memory_config().shard_spec is not None
    conv3_output_grid = conv3_output.memory_config().shard_spec.grid
    assert core_grid_signature(conv3_output_grid) == core_grid_signature(stock_conv3_output_grid)
    assert core_grid_ordered_signature(conv3_output_grid) == core_grid_ordered_signature(
        stock_conv3_output_grid
    )
    logger.info(
        "LAYER1_HEIGHT_FOUR_RECTANGLE_CHAIN phase=stock-exit output_grid={} output_cores={}",
        core_grid_ordered_signature(conv3_output_grid),
        conv3_output_grid.num_cores(),
    )

    output_torch = ttnn.to_torch(conv3_output, device=device).to(torch.float32)
    output_torch = output_torch.reshape(-1, output_torch.shape[-1])[
        : input_height * input_width, :conv3_out_channels
    ]
    pcc_passed, pcc_message = check_with_pcc(golden, output_torch, pcc=0.99)
    logger.info("LAYER1_HEIGHT_FOUR_RECTANGLE_CHAIN correctness={}", pcc_message)
    assert pcc_passed, pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_translated_conv2_isolated(device):
    """Compare layer4 module1's 3x3 Conv at stock and translated origins."""

    stock_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))]
    )
    shifted_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(4, 3))]
    )
    run_bos_resnet50_layer4_conv2_placements(
        device, [("stock", stock_grid), ("translated", shifted_grid)]
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_segmented_column_conv2_isolated(device):
    """Skip one worker column while preserving logical ROW_MAJOR shard order."""

    stock_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))]
    )
    # A list (not a set) is intentional: CoreRangeSet's vector constructor
    # preserves this logical ROW_MAJOR shard order across the physical gap.
    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in range(4)
            for x in (0, 1, 3, 4)
        ]
    )
    assert segmented_grid.num_cores() == 16
    expected_logical_order = [(x, y, x, y) for y in range(4) for x in (0, 1, 3, 4)]
    assert core_grid_ordered_signature(segmented_grid) == expected_logical_order
    run_bos_resnet50_layer4_conv2_placements(
        device, [("stock", stock_grid), ("segmented-column", segmented_grid)]
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_segmented_row_conv2_isolated(device):
    """Compare explicit and config-selected row gaps on a logical 4x3 placement."""

    stock_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 2))]
    )
    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in (0, 1, 3)
            for x in range(4)
        ]
    )
    assert segmented_grid.num_cores() == 12
    expected_logical_order = [(x, y, x, y) for y in (0, 1, 3) for x in range(4)]
    assert core_grid_ordered_signature(segmented_grid) == expected_logical_order
    run_bos_resnet50_layer4_conv2_placements(
        device,
        [
            ("stock-4x3", stock_grid),
            ("segmented-row", segmented_grid),
            ("configured-row", stock_grid),
        ],
        logical_grid_shape=(4, 3),
        placement_requests={
            "configured-row": {
                "excluded_worker_rows": [2],
                "expected_output_grid": segmented_grid,
            }
        },
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_row_and_column_exclusions_conv2_isolated(device):
    """Skip one interior row and column, producing four Cartesian rectangles."""

    stock_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 2))]
    )
    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in (0, 1, 3)
            for x in (0, 1, 3, 4)
        ]
    )
    assert segmented_grid.num_cores() == 12
    run_bos_resnet50_layer4_conv2_placements(
        device,
        [
            ("stock-4x3", stock_grid),
            ("segmented-four-rectangles", segmented_grid),
            ("configured-four-rectangles", stock_grid),
        ],
        logical_grid_shape=(4, 3),
        placement_requests={
            "configured-four-rectangles": {
                "excluded_worker_rows": [2],
                "excluded_worker_columns": [2],
                "expected_output_grid": segmented_grid,
            }
        },
    )


def run_bos_resnet50_layer4_segmented_conv2_to_conv3_chain(
    device, *, segmented_grid, logical_grid_shape, stock_conv3_output_grid
):
    """Propagate a segmented 3x3 output, then exit to stock at the following 1x1 Conv."""

    compute_grid = device.compute_with_storage_grid_size()
    assert compute_grid.x >= 5 and compute_grid.y >= 4

    test_infra = create_bos_test_infra(device)
    model = test_infra.ttnn_resnet50_model
    layer4_module1 = model.layer_component["layer4"]["module"][0]

    batch_size, input_height, input_width = 1, 14, 14
    conv2_in_channels = layer4_module1.conv2_input_channels
    conv2_out_channels = layer4_module1.conv2_output_channels
    conv3_in_channels = layer4_module1.conv3_input_channels
    conv3_out_channels = layer4_module1.conv3_output_channels
    assert conv2_out_channels == conv3_in_channels

    torch_input = torch.randn(
        (batch_size, 1, input_height * input_width, conv2_in_channels), dtype=torch.bfloat16
    )
    host_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    quantized_input = ttnn.to_torch(host_input).to(torch.float32)

    conv2_weight = ttnn.to_torch(layer4_module1.conv2_weight_tensor).to(torch.float32)
    conv2_bias = ttnn.to_torch(layer4_module1.conv2_bias_tensor).reshape(-1).to(torch.float32)
    conv3_weight = ttnn.to_torch(layer4_module1.conv3_weight_tensor).to(torch.float32)
    conv3_bias = ttnn.to_torch(layer4_module1.conv3_bias_tensor).reshape(-1).to(torch.float32)
    golden = torch.nn.functional.conv2d(
        quantized_input.reshape(batch_size, input_height, input_width, conv2_in_channels).permute(0, 3, 1, 2),
        conv2_weight,
        conv2_bias,
        stride=(layer4_module1.stride, layer4_module1.stride),
        padding=(1, 1),
    )
    golden = torch.relu(golden)
    golden = torch.nn.functional.conv2d(golden, conv3_weight, conv3_bias)
    golden = golden.permute(0, 2, 3, 1).reshape(-1, conv3_out_channels)

    logical_grid_x, logical_grid_y = logical_grid_shape
    flattened_height = batch_size * input_height * input_width
    height_alignment = logical_grid_y * 32
    padded_height = ((flattened_height + height_alignment - 1) // height_alignment) * height_alignment
    assert conv2_in_channels % logical_grid_x == 0
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            segmented_grid,
            (padded_height // logical_grid_y, conv2_in_channels // logical_grid_x),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    device_input = host_input.to(device, input_memory_config)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=model.model_config["MATH_FIDELITY"],
        packer_l1_acc=True,
    )
    conv2_config = ttnn.Conv2dConfig(
        weights_dtype=model.model_config["WEIGHTS_DTYPE"],
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=False,
        reallocate_halo_output=False,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        reshard_if_not_optimal=False,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        force_split_reader=False,
        enable_core_remapping=True,
    )
    conv3_config = ttnn.Conv2dConfig(
        weights_dtype=model.model_config["WEIGHTS_DTYPE"],
        deallocate_activation=False,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        reshard_if_not_optimal=False,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        force_split_reader=False,
        enable_core_remapping=True,
    )

    logger.info(
        "LAYER4_SEGMENTED_CHAIN phase=conv2-input grid={} channels={}->{}",
        core_grid_signature(device_input.memory_config().shard_spec.grid),
        conv2_in_channels,
        conv2_out_channels,
    )
    ttnn.synchronize_device(device)
    conv2_output, [conv2_output_height, conv2_output_width], _ = functional_resnet50.conv3x3(
        device_input,
        device,
        input_height,
        input_width,
        conv2_in_channels,
        conv2_out_channels,
        batch_size,
        layer4_module1.conv2_weight_tensor,
        layer4_module1.conv2_bias_tensor,
        conv2_config,
        compute_config,
        model.model_config["ACTIVATIONS_DTYPE"],
        True,
        True,
        stride=(layer4_module1.stride, layer4_module1.stride),
    )
    ttnn.synchronize_device(device)
    assert (conv2_output_height, conv2_output_width) == (7, 7)
    assert conv2_output.memory_config().shard_spec is not None
    conv2_output_grid = conv2_output.memory_config().shard_spec.grid
    logger.info(
        "LAYER4_SEGMENTED_CHAIN phase=conv2-complete output_grid={} output_cores={}",
        core_grid_signature(conv2_output_grid),
        conv2_output_grid.num_cores(),
    )
    assert core_grid_signature(conv2_output_grid) == core_grid_signature(segmented_grid)
    assert core_grid_ordered_signature(conv2_output_grid) == core_grid_ordered_signature(segmented_grid)

    logger.info(
        "LAYER4_SEGMENTED_CHAIN phase=conv3-input segmented_grid={} channels={}->{}",
        core_grid_signature(conv2_output_grid),
        conv3_in_channels,
        conv3_out_channels,
    )
    conv3_output, [conv3_output_height, conv3_output_width], _ = functional_resnet50.conv1x1(
        conv2_output,
        device,
        conv2_output_height,
        conv2_output_width,
        conv3_in_channels,
        conv3_out_channels,
        batch_size,
        layer4_module1.conv3_weight_tensor,
        layer4_module1.conv3_bias_tensor,
        conv3_config,
        compute_config,
        model.model_config["ACTIVATIONS_DTYPE"],
        True,
        True,
        stride=(1, 1),
    )
    logger.info("LAYER4_SEGMENTED_CHAIN phase=conv3-submitted")
    ttnn.synchronize_device(device)
    logger.info("LAYER4_SEGMENTED_CHAIN phase=conv3-complete")

    assert (conv3_output_height, conv3_output_width) == (7, 7)
    assert conv3_output.memory_config().shard_spec is not None
    conv3_output_grid = conv3_output.memory_config().shard_spec.grid
    logger.info(
        "LAYER4_SEGMENTED_CHAIN phase=stock-exit output_grid={} output_cores={}",
        core_grid_signature(conv3_output_grid),
        conv3_output_grid.num_cores(),
    )
    assert core_grid_signature(conv3_output_grid) == core_grid_signature(stock_conv3_output_grid)
    assert conv3_output_grid.num_cores() == 8

    output_torch = ttnn.to_torch(conv3_output, device=device).to(torch.float32)
    output_torch = output_torch.reshape(-1, output_torch.shape[-1])[
        : conv3_output_height * conv3_output_width, :conv3_out_channels
    ]
    pcc_passed, pcc_message = check_with_pcc(golden, output_torch, pcc=0.99)
    logger.info("LAYER4_SEGMENTED_CHAIN correctness={}", pcc_message)
    assert pcc_passed, pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_segmented_conv2_to_conv3_exits_remapping(device):
    """Propagate a column-gap Conv2 output, then exit to stock at Conv3."""

    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in range(4)
            for x in (0, 1, 3, 4)
        ]
    )
    stock_conv3_output_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))]
    )
    run_bos_resnet50_layer4_segmented_conv2_to_conv3_chain(
        device,
        segmented_grid=segmented_grid,
        logical_grid_shape=(4, 4),
        stock_conv3_output_grid=stock_conv3_output_grid,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_segmented_row_conv2_to_conv3_exits_remapping(device):
    """Propagate a row-gap Conv2 output, then exit to stock at Conv3."""

    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in (0, 1, 3)
            for x in range(4)
        ]
    )
    stock_conv3_output_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))]
    )
    run_bos_resnet50_layer4_segmented_conv2_to_conv3_chain(
        device,
        segmented_grid=segmented_grid,
        logical_grid_shape=(4, 3),
        stock_conv3_output_grid=stock_conv3_output_grid,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_four_rectangle_conv2_to_conv3_exits_remapping(device):
    """Propagate a row-and-column-gap Conv2 output, then exit to stock at Conv3."""

    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in (0, 1, 3)
            for x in (0, 1, 3, 4)
        ]
    )
    assert segmented_grid.num_cores() == 12
    expected_logical_order = [
        (x, y, x, y) for y in (0, 1, 3) for x in (0, 1, 3, 4)
    ]
    assert core_grid_ordered_signature(segmented_grid) == expected_logical_order

    stock_conv3_output_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))]
    )
    run_bos_resnet50_layer4_segmented_conv2_to_conv3_chain(
        device,
        segmented_grid=segmented_grid,
        logical_grid_shape=(4, 3),
        stock_conv3_output_grid=stock_conv3_output_grid,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_translated_downsample_exits_remapping(device):
    """Verify that a translated 16-core input safely exits remapping before a stock 20-core output."""

    compute_grid = device.compute_with_storage_grid_size()
    assert compute_grid.x >= 5 and compute_grid.y >= 4
    shifted_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(4, 3))}
    )
    stock_output_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}
    )

    test_infra = create_bos_test_infra(device)
    model = test_infra.ttnn_resnet50_model
    layer4_module1 = model.layer_component["layer4"]["module"][0]

    batch_size, input_height, input_width = 1, 14, 14
    in_channels = layer4_module1.ds_conv_input_channels
    out_channels = layer4_module1.ds_conv_output_channels
    torch_input = torch.randn(
        (batch_size, 1, input_height * input_width, in_channels), dtype=torch.bfloat16
    )
    host_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
    )
    quantized_input = ttnn.to_torch(host_input).to(torch.float32)
    weight = ttnn.to_torch(layer4_module1.ds_conv_weight_tensor).to(torch.float32)
    bias = ttnn.to_torch(layer4_module1.ds_conv_bias_tensor).reshape(-1).to(torch.float32)
    golden = torch.nn.functional.conv2d(
        quantized_input.reshape(batch_size, input_height, input_width, in_channels).permute(0, 3, 1, 2),
        weight,
        bias,
        stride=(layer4_module1.stride, layer4_module1.stride),
    )
    golden = golden.permute(0, 2, 3, 1).reshape(-1, out_channels)

    input_memory_config = ttnn.create_sharded_memory_config_(
        ttnn.Shape([batch_size, input_height, input_width, in_channels]),
        shifted_grid,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
    )
    device_input = host_input.to(device, input_memory_config)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=model.model_config["MATH_FIDELITY"],
        packer_l1_acc=True,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=model.model_config["WEIGHTS_DTYPE"],
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        deallocate_activation=True,
        reallocate_halo_output=True,
        reshard_if_not_optimal=False,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        force_split_reader=False,
        enable_core_remapping=True,
    )

    logger.info(
        "LAYER4_DOWNSAMPLE_EXIT phase=before input_grid={} channels={}->{}",
        core_grid_signature(device_input.memory_config().shard_spec.grid),
        in_channels,
        out_channels,
    )
    ttnn.synchronize_device(device)
    output, [output_height, output_width], _ = functional_resnet50.conv1x1(
        device_input,
        device,
        input_height,
        input_width,
        in_channels,
        out_channels,
        batch_size,
        layer4_module1.ds_conv_weight_tensor,
        layer4_module1.ds_conv_bias_tensor,
        conv_config,
        compute_config,
        model.model_config["ACTIVATIONS_DTYPE"],
        True,
        True,
        stride=(layer4_module1.stride, layer4_module1.stride),
    )
    logger.info("LAYER4_DOWNSAMPLE_EXIT phase=submitted")
    ttnn.synchronize_device(device)
    logger.info("LAYER4_DOWNSAMPLE_EXIT phase=complete")

    assert (output_height, output_width) == (7, 7)
    assert output.memory_config().shard_spec is not None
    output_grid = output.memory_config().shard_spec.grid
    logger.info(
        "LAYER4_DOWNSAMPLE_EXIT output_grid={} output_cores={}",
        core_grid_signature(output_grid),
        output_grid.num_cores(),
    )
    assert core_grid_signature(output_grid) == core_grid_signature(stock_output_grid)
    assert output_grid.num_cores() == 20

    output_torch = ttnn.to_torch(output, device=device).to(torch.float32)
    output_torch = output_torch.reshape(-1, output_torch.shape[-1])[: output_height * output_width, :out_channels]
    pcc_passed, pcc_message = check_with_pcc(golden, output_torch, pcc=0.99)
    logger.info("LAYER4_DOWNSAMPLE_EXIT correctness={}", pcc_message)
    assert pcc_passed, pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_contiguous_core_set_16_translation(device, monkeypatch):
    """Move BOS layer4's 4x4 block-sharded grid right by one worker."""

    compute_grid = device.compute_with_storage_grid_size()
    assert compute_grid.x >= 5 and compute_grid.y >= 4
    shifted_core_set_16 = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(4, 3))}
    )
    monkeypatch.setattr(functional_resnet50, "CORE_SET_16", shifted_core_set_16)
    install_conv_sync_tracing(monkeypatch, enable_core_remapping=True)
    assert core_grid_signature(functional_resnet50.CORE_SET_16) == [(1, 0, 4, 3)]
    logger.info(
        "CONV_CORE_REPLACEMENT_LAYER4 step=1 CORE_SET_16 translated "
        "from worker range (0,0)->(3,3) to (1,0)->(4,3)"
    )

    test_infra = create_bos_test_infra(device)
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    logger.info("CONV_CORE_REPLACEMENT_LAYER4 step=2 running full BOS ResNet50")
    test_infra.run()
    pcc_passed, pcc_message = test_infra.validate()
    logger.info(
        "CONV_CORE_REPLACEMENT_LAYER4 step=3 correctness {}",
        pcc_message,
    )
    assert pcc_passed, pcc_message


def run_bos_resnet50_layer1_height_segmented_conv2_full_model(
    device,
    monkeypatch,
    *,
    segmented_grid,
    segmented_placement,
    excluded_rows,
    excluded_columns,
):
    """Validate a layer1 Conv2 placement and its Conv3 ExitToStock lifetime."""

    num_cores = 14
    stock_grid, _ = make_bos_height_sharded_column_gap_grids(device)
    stock_signature = core_grid_signature(stock_grid)
    segmented_signature = core_grid_signature(segmented_grid)

    placement_trace = []
    install_conv_sync_tracing(
        monkeypatch,
        enable_core_remapping=True,
        placement_trace=placement_trace,
        excluded_worker_columns_by_op={"layer1.module1.conv2": excluded_columns},
        excluded_worker_rows_by_op={"layer1.module1.conv2": excluded_rows},
        input_stock_grids_by_op={"layer1.module1.conv3": stock_grid},
    )
    logger.info(
        "CONV_CORE_REPLACEMENT_LAYER1_HEIGHT_SEGMENTED step=1 requesting "
        "layer1.module1.conv2 placement={} excluded_rows={} excluded_columns={} expected_grid={}",
        segmented_placement,
        excluded_rows,
        excluded_columns,
        segmented_signature,
    )

    test_infra = create_bos_test_infra(device)
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    logger.info("CONV_CORE_REPLACEMENT_LAYER1_HEIGHT_SEGMENTED step=2 running full BOS ResNet50")
    test_infra.run()

    conv2_placement = next(
        record for record in placement_trace if record["op"] == "layer1.module1.conv2"
    )
    assert conv2_placement["input_grid"] == stock_signature
    assert conv2_placement["input_cores"] == num_cores
    assert conv2_placement["output_grid"] == segmented_signature
    assert conv2_placement["output_cores"] == num_cores

    conv3_placement = next(
        record for record in placement_trace if record["op"] == "layer1.module1.conv3"
    )
    assert conv3_placement["input_grid"] == segmented_signature
    assert conv3_placement["input_cores"] == num_cores
    assert conv3_placement["output_grid"] == stock_signature
    assert conv3_placement["output_cores"] == num_cores

    pcc_passed, pcc_message = test_infra.validate()
    logger.info(
        "CONV_CORE_REPLACEMENT_LAYER1_HEIGHT_SEGMENTED step=3 correctness {}",
        pcc_message,
    )
    assert pcc_passed, pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer1_height_segmented_column_conv2_full_model(device, monkeypatch):
    """Run a column-gap Conv2, exit to stock at Conv3, and finish ResNet50."""

    _, segmented_grid = make_bos_height_sharded_column_gap_grids(device)
    run_bos_resnet50_layer1_height_segmented_conv2_full_model(
        device,
        monkeypatch,
        segmented_grid=segmented_grid,
        segmented_placement="configured-column-gap",
        excluded_rows=[],
        excluded_columns=[2],
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer1_height_boundary_translation_conv2_full_model(device, monkeypatch):
    """Translate Conv2's sender origin, exit to stock at Conv3, and finish ResNet50."""

    _, translated_grid = make_bos_height_sharded_boundary_translation_grids(device)
    run_bos_resnet50_layer1_height_segmented_conv2_full_model(
        device,
        monkeypatch,
        segmented_grid=translated_grid,
        segmented_placement="configured-boundary-translation",
        excluded_rows=[],
        excluded_columns=[0],
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer1_height_segmented_row_conv2_full_model(device, monkeypatch):
    """Run a row-gap Conv2, exit to stock at Conv3, and finish ResNet50."""

    _, segmented_grid = make_bos_height_sharded_row_gap_grids(device)
    run_bos_resnet50_layer1_height_segmented_conv2_full_model(
        device,
        monkeypatch,
        segmented_grid=segmented_grid,
        segmented_placement="configured-row-gap",
        excluded_rows=[2],
        excluded_columns=[],
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_segmented_column_conv2_full_model(device, monkeypatch):
    """Request a segmented grid at layer4 module1 Conv2 and finish the full model.

    Conv2 reshards its stock API input to the requested segmented placement
    internally and exposes that placement on its output. The following 1x1
    Conv3 must end the segmented remapping lifetime and return to its stock
    contiguous output grid before the residual path is joined.
    """

    compute_grid = device.compute_with_storage_grid_size()
    assert compute_grid.x >= 5 and compute_grid.y >= 4
    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in range(4)
            for x in (0, 1, 3, 4)
        ]
    )
    expected_segmented_order = [
        (x, y, x, y) for y in range(4) for x in (0, 1, 3, 4)
    ]
    assert core_grid_ordered_signature(segmented_grid) == expected_segmented_order
    segmented_signature = core_grid_signature(segmented_grid)
    stock_conv2_input_signature = [(0, 0, 3, 3)]
    stock_conv3_output_signature = [(0, 0, 3, 1)]

    placement_trace = []
    install_conv_sync_tracing(
        monkeypatch,
        enable_core_remapping=True,
        placement_trace=placement_trace,
        excluded_worker_columns_by_op={"layer4.module1.conv2": [2]},
    )
    logger.info(
        "CONV_CORE_REPLACEMENT_LAYER4_SEGMENTED step=1 requesting Conv2 excluded_column=2 expected_grid={}",
        segmented_signature,
    )

    test_infra = create_bos_test_infra(device)
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    logger.info("CONV_CORE_REPLACEMENT_LAYER4_SEGMENTED step=2 running full BOS ResNet50")
    test_infra.run()

    conv2_placement = next(
        record for record in placement_trace if record["op"] == "layer4.module1.conv2"
    )
    # Tracing observes the tensor at the public Conv boundary; the requested
    # stock-to-segmented reshard happens inside Conv preparation.
    assert conv2_placement["input_grid"] == stock_conv2_input_signature
    assert conv2_placement["input_cores"] == 16
    assert conv2_placement["output_grid"] == segmented_signature
    assert conv2_placement["output_cores"] == 16

    conv3_placement = next(
        record for record in placement_trace if record["op"] == "layer4.module1.conv3"
    )
    assert conv3_placement["input_grid"] == segmented_signature
    assert conv3_placement["input_cores"] == 16
    assert conv3_placement["output_grid"] == stock_conv3_output_signature
    assert conv3_placement["output_cores"] == 8

    pcc_passed, pcc_message = test_infra.validate()
    logger.info(
        "CONV_CORE_REPLACEMENT_LAYER4_SEGMENTED step=3 correctness {}",
        pcc_message,
    )
    assert pcc_passed, pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer4_four_rectangle_conv2_full_model(device, monkeypatch):
    """Use a four-rectangle placement on the real 8-core layer4 module2 Conv2."""

    compute_grid = device.compute_with_storage_grid_size()
    assert compute_grid.x >= 5 and compute_grid.y >= 4
    segmented_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for y in (0, 2)
            for x in (0, 1, 3, 4)
        ]
    )
    expected_segmented_order = [
        (x, y, x, y) for y in (0, 2) for x in (0, 1, 3, 4)
    ]
    assert segmented_grid.num_cores() == 8
    assert core_grid_ordered_signature(segmented_grid) == expected_segmented_order
    segmented_signature = core_grid_signature(segmented_grid)
    stock_signature = [(0, 0, 3, 1)]

    placement_trace = []
    install_conv_sync_tracing(
        monkeypatch,
        enable_core_remapping=True,
        placement_trace=placement_trace,
        excluded_worker_columns_by_op={"layer4.module2.conv2": [2]},
        excluded_worker_rows_by_op={"layer4.module2.conv2": [1]},
    )
    logger.info(
        "CONV_CORE_REPLACEMENT_LAYER4_FOUR_RECTANGLES step=1 requesting "
        "layer4.module2.conv2 excluded_column=2 excluded_row=1 expected_grid={}",
        segmented_signature,
    )

    test_infra = create_bos_test_infra(device)
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    logger.info("CONV_CORE_REPLACEMENT_LAYER4_FOUR_RECTANGLES step=2 running full BOS ResNet50")
    test_infra.run()

    conv2_placement = next(
        record for record in placement_trace if record["op"] == "layer4.module2.conv2"
    )
    assert conv2_placement["input_grid"] == stock_signature
    assert conv2_placement["input_cores"] == 8
    assert conv2_placement["output_grid"] == segmented_signature
    assert conv2_placement["output_cores"] == 8

    conv3_placement = next(
        record for record in placement_trace if record["op"] == "layer4.module2.conv3"
    )
    assert conv3_placement["input_grid"] == segmented_signature
    assert conv3_placement["input_cores"] == 8
    assert conv3_placement["output_grid"] == stock_signature
    assert conv3_placement["output_cores"] == 8

    pcc_passed, pcc_message = test_infra.validate()
    logger.info(
        "CONV_CORE_REPLACEMENT_LAYER4_FOUR_RECTANGLES step=3 correctness {}",
        pcc_message,
    )
    assert pcc_passed, pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_layer3_cyclic_column_rest_full_model(device, monkeypatch):
    """Rotate Layer3 Conv2's resting column and complete the real model.

    The six 4x4 BLOCK_SHARDED Conv2 operations are placed on a five-column
    worker grid. A different column rests at each operation; advancing the
    schedule phase per inference produces exact five-inference placement balance.
    """

    compute_grid = device.compute_with_storage_grid_size()
    assert compute_grid.x >= 5 and compute_grid.y >= 4

    five_inference_schedules = [
        layer3_conv2_column_rest_schedule(inference_index)
        for inference_index in range(5)
    ]
    for op_name in layer3_conv2_column_rest_schedule():
        assert {
            schedule_for_inference[op_name][0]
            for schedule_for_inference in five_inference_schedules
        } == set(range(5))

    inference_index = int(os.environ.get("CONV_CORE_ROTATION_INFERENCE_INDEX", "0"))
    assert 0 <= inference_index < 5
    schedule = layer3_conv2_column_rest_schedule(inference_index=inference_index)
    placement_trace = []
    install_conv_sync_tracing(
        monkeypatch,
        enable_core_remapping=True,
        placement_trace=placement_trace,
        excluded_worker_columns_by_op=schedule,
    )
    logger.info(
        "LAYER3_COLUMN_REST step=1 inference_index={} schedule={}",
        inference_index,
        schedule,
    )

    test_infra = create_bos_test_infra(device)
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    logger.info("LAYER3_COLUMN_REST step=2 running full BOS ResNet50")
    test_infra.run()

    active_counts = {column: 0 for column in range(5)}
    for op_name, excluded_columns in schedule.items():
        record = next(record for record in placement_trace if record["op"] == op_name)
        output_workers = worker_coordinates_from_grid_signature(record["output_grid"])
        excluded_column = excluded_columns[0]
        expected_workers = {
            (x, y) for y in range(4) for x in range(5) if x != excluded_column
        }
        assert output_workers == expected_workers
        assert record["output_cores"] == 16
        for column in range(5):
            if column != excluded_column:
                active_counts[column] += 1

        module_index = int(op_name.split("module", 1)[1].split(".", 1)[0])
        conv3_record = next(
            record
            for record in placement_trace
            if record["op"] == f"layer3.module{module_index}.conv3"
        )
        assert worker_coordinates_from_grid_signature(conv3_record["input_grid"]) == expected_workers
        # Conv3 is a matmul-lowered 1x1 Conv. Requiring the same segmented grid on its output proves that the
        # ROW_MAJOR 2D reuse-mcast Matmul executed on the remapped placement instead of silently exiting to stock.
        assert worker_coordinates_from_grid_signature(conv3_record["output_grid"]) == expected_workers
        assert conv3_record["output_cores"] == 16

    baseline_active_counts = {0: 6, 1: 6, 2: 6, 3: 6, 4: 0}
    assert sorted(active_counts.values()) == [4, 5, 5, 5, 5]
    baseline_max_over_mean = max(baseline_active_counts.values()) / (
        sum(baseline_active_counts.values()) / len(baseline_active_counts)
    )
    rotated_max_over_mean = max(active_counts.values()) / (
        sum(active_counts.values()) / len(active_counts)
    )
    logger.info(
        "LAYER3_COLUMN_REST exposure baseline={} rotated={} max_over_mean={:.6f}->{:.6f}",
        baseline_active_counts,
        active_counts,
        baseline_max_over_mean,
        rotated_max_over_mean,
    )
    assert rotated_max_over_mean < baseline_max_over_mean

    pcc_passed, pcc_message = test_infra.validate()
    logger.info("LAYER3_COLUMN_REST step=3 correctness {}", pcc_message)
    assert pcc_passed, pcc_message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@high_risk_device_test
def test_bos_resnet50_milp_phase_full_model(device, monkeypatch):
    """Run one H=20 MILP phase through the full model and validate placement/PCC."""

    compute_grid = device.compute_with_storage_grid_size()
    assert compute_grid.x >= 5 and compute_grid.y >= 4
    manifest_path = Path(os.environ.get(MILP_MANIFEST_ENV, DEFAULT_MILP_MANIFEST))
    phase = int(os.environ.get(MILP_PHASE_ENV, "0"))
    synchronize_each_conv = os.environ.get(MILP_SYNC_EACH_CONV_ENV, "1") == "1"
    policy = load_milp_full_model_phase(manifest_path, phase)
    logger.info(
        "MILP_FULL_MODEL step=1 phase={}/{} manifest={} status={} placements={}",
        phase,
        policy["manifest"]["horizon"],
        manifest_path,
        policy["manifest"]["status"],
        policy["placement_ids_by_domain"],
    )

    placement_trace = []
    install_conv_sync_tracing(
        monkeypatch,
        enable_core_remapping=True,
        placement_trace=placement_trace,
        excluded_worker_columns_by_op=policy["columns_by_op"],
        excluded_worker_rows_by_op=policy["rows_by_op"],
        capture_input_stock_grid_for_exit_by_op=policy[
            "exit_to_stock_op_by_entry_op"
        ],
        synchronize_each_conv=synchronize_each_conv,
    )

    test_infra = create_bos_test_infra(device)
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    logger.info("MILP_FULL_MODEL step=2 running phase={} full BOS ResNet50", phase)
    test_infra.run()

    records_by_op = {record["op"]: record for record in placement_trace}
    for entry_op, expected_workers in policy["expected_workers_by_entry_op"].items():
        entry_record = records_by_op[entry_op]
        actual_output_workers = worker_coordinates_from_grid_signature(
            entry_record["output_grid"]
        )
        assert actual_output_workers == expected_workers, (
            f"phase {phase} {entry_op} output placement mismatch: "
            f"expected={sorted(expected_workers)}, actual={sorted(actual_output_workers)}"
        )

        consumer_op = policy["consumer_op_by_entry_op"].get(entry_op)
        if consumer_op is not None:
            consumer_record = records_by_op[consumer_op]
            actual_input_workers = worker_coordinates_from_grid_signature(
                consumer_record["input_grid"]
            )
            assert actual_input_workers == expected_workers, (
                f"phase {phase} {consumer_op} did not consume the remapped placement: "
                f"expected={sorted(expected_workers)}, actual={sorted(actual_input_workers)}"
            )
            actual_consumer_output_workers = worker_coordinates_from_grid_signature(
                consumer_record["output_grid"]
            )
            if entry_op in policy["exit_to_stock_op_by_entry_op"]:
                expected_consumer_output_workers = worker_coordinates_from_grid_signature(
                    entry_record["input_grid"]
                )
            else:
                expected_consumer_output_workers = expected_workers
            assert actual_consumer_output_workers == expected_consumer_output_workers, (
                f"phase {phase} {consumer_op} output lifetime mismatch: "
                f"expected={sorted(expected_consumer_output_workers)}, "
                f"actual={sorted(actual_consumer_output_workers)}"
            )

    pcc_passed, pcc_message = test_infra.validate()
    logger.info(
        "MILP_FULL_MODEL step=3 phase={} correctness {} validated_domains={}",
        phase,
        pcc_message,
        len(policy["expected_workers_by_entry_op"]),
    )
    assert pcc_passed, pcc_message
