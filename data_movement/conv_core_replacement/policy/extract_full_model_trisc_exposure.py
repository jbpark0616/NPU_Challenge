#!/usr/bin/env python3
"""Extract per-operation, per-worker TRISC kernel exposure from one full-model run.

The device profiler emits nested TRISC-FW and TRISC-KERNEL zones.  Scheduler
calibration intentionally uses only TRISC-KERNEL intervals so firmware and
kernel time are not counted twice.  The resulting metric is an equal-weight
sum of TRISC 0/1/2 engine cycles; it is a compute-pipeline activity proxy, not
a transistor-aging or MAC-utilization measurement.

The device log and ops-performance CSV must come from the same uninterrupted
full-model run.  Stage-split captures are rejected because their graph and
layer boundaries differ from the production full-model execution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


TRISC_TYPES = ("TRISC_0", "TRISC_1", "TRISC_2")
KERNEL_ZONE = "TRISC-KERNEL"

# Eagle-N exposes a logical 5x4 worker grid.  Device-profiler coordinates use
# physical NoC rows, where row 3 is DRAM and logical worker row 3 is physical 4.
EAGLE_N_PHYSICAL_TO_LOGICAL_Y = {0: 0, 1: 1, 2: 2, 4: 3}


def resnet50_compute_op_names() -> list[str]:
    """Names for the 53 Conv calls followed by the final classifier Matmul."""
    names = ["stem.conv1"]
    for layer_name, block_count in (
        ("layer1", 3),
        ("layer2", 4),
        ("layer3", 6),
        ("layer4", 3),
    ):
        for module_index in range(1, block_count + 1):
            prefix = f"{layer_name}.module{module_index}"
            names.extend(
                (f"{prefix}.conv1", f"{prefix}.conv2", f"{prefix}.conv3")
            )
            if module_index == 1:
                names.append(f"{prefix}.downsample")
    names.append("head.classifier")
    assert len(names) == 54
    return names


def _normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _reject_stage_split(path: Path) -> None:
    if any(part.lower().startswith("stage_split_") for part in path.parts):
        raise ValueError(
            f"Refusing stage-split scheduler calibration input: {path}. "
            "Use an uninterrupted full-model profile and its matching ops report."
        )


def load_device_events(path: Path) -> tuple[pd.DataFrame, str]:
    _reject_stage_split(path)
    with path.open("r", encoding="utf-8") as stream:
        metadata = stream.readline().strip()
    events = _normalized_columns(pd.read_csv(path, skiprows=1))
    required = {
        "core_x",
        "core_y",
        "RISC processor type",
        "timer_id",
        "run host ID",
        "zone name",
        "type",
        "time[cycles since reset]",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Device profile is missing columns: {sorted(missing)}")
    return events, metadata


def load_ops(path: Path) -> pd.DataFrame:
    _reject_stage_split(path)
    ops = _normalized_columns(pd.read_csv(path))
    required = {"GLOBAL CALL COUNT", "OP CODE", "OP TYPE"}
    missing = required.difference(ops.columns)
    if missing:
        raise ValueError(f"Ops report is missing columns: {sorted(missing)}")
    if ops["GLOBAL CALL COUNT"].duplicated().any():
        duplicates = sorted(
            ops.loc[ops["GLOBAL CALL COUNT"].duplicated(), "GLOBAL CALL COUNT"]
            .astype(int)
            .unique()
        )
        raise ValueError(f"Ops report contains duplicate GLOBAL CALL COUNT values: {duplicates}")
    return ops


def annotate_resnet50_compute_sequence(ops: pd.DataFrame) -> pd.DataFrame:
    """Attach model names to the Conv2d/Matmul sequence of full inference(s).

    A BOS ResNet50 inference emits exactly 20 Conv2d device operations and 34
    Matmul operations.  The first 53 correspond to model Conv invocations; the
    final Matmul is the classifier.  Other device operations remain unlabeled
    here and are retained as fixed background exposure until placement-domain
    attribution is available.
    """
    ops = ops.copy()
    ops["model_op_name"] = pd.NA
    ops["model_op_index"] = pd.NA
    ops["model_inference_index"] = pd.NA
    compute_indices = ops.index[ops["OP CODE"].isin(("Conv2d", "Matmul"))].tolist()
    names = resnet50_compute_op_names()
    if len(compute_indices) % len(names) != 0:
        raise ValueError(
            "Full-model compute sequence is incomplete: expected a multiple of "
            f"{len(names)} Conv2d/Matmul operations, found {len(compute_indices)}"
        )
    for sequence_index, row_index in enumerate(compute_indices):
        ops.at[row_index, "model_op_name"] = names[sequence_index % len(names)]
        ops.at[row_index, "model_op_index"] = sequence_index % len(names)
        ops.at[row_index, "model_inference_index"] = sequence_index // len(names)
    ops["model_inference_index"] = ops["model_inference_index"].astype("Int64")
    ops["model_op_index"] = ops["model_op_index"].astype("Int64")
    return ops


def _regex_group(value: object, pattern: str) -> object:
    match = re.search(pattern, str(value))
    return match.group(1) if match else pd.NA


def annotate_topology(ops: pd.DataFrame) -> pd.DataFrame:
    """Extract topology fields without deciding placement-domain legality."""
    ops = ops.copy()
    attributes = ops["ATTRIBUTES"].fillna("")
    ops["shard_layout"] = attributes.map(
        lambda value: _regex_group(value, r"memory_layout=TensorMemoryLayout::([A-Z_]+)")
    )
    ops["shard_orientation"] = attributes.map(
        lambda value: _regex_group(value, r"orientation=ShardOrientation::([A-Z_]+)")
    )
    ops["stock_grid"] = attributes.map(
        lambda value: _regex_group(value, r"shard_spec=ShardSpec\(grid=(\{.*?\});shape=")
    )

    def grid_dimension(value: object, axis: str) -> object:
        patterns = (
            rf"compute_with_storage_grid_size=\(x=\d+;y=\d+\)",
            rf"grid_size=\(x=\d+;y=\d+\)",
        )
        for pattern in patterns:
            match = re.search(pattern, str(value))
            if match:
                coordinate = re.search(rf"{axis}=(\d+)", match.group(0))
                return int(coordinate.group(1)) if coordinate else pd.NA
        return pd.NA

    ops["logical_grid_x"] = attributes.map(lambda value: grid_dimension(value, "x")).astype(
        "Int64"
    )
    ops["logical_grid_y"] = attributes.map(lambda value: grid_dimension(value, "y")).astype(
        "Int64"
    )
    # Current reports spell the boolean either transpose_mcast=0 or
    # transpose_mcast_: 'false'. Build the nullable integer column in one
    # conversion instead of assigning integers into pandas' string dtype.
    transpose_mcast = attributes.str.extract(r"transpose_mcast=([01])", expand=False)
    transpose_mcast = transpose_mcast.mask(
        attributes.str.contains("transpose_mcast_': 'false", regex=False), "0"
    )
    transpose_mcast = transpose_mcast.mask(
        attributes.str.contains("transpose_mcast_': 'true", regex=False), "1"
    )
    ops["transpose_mcast"] = pd.to_numeric(
        transpose_mcast, errors="coerce"
    ).astype("Int64")

    ops["remap_scope_status"] = "candidate_requires_planner_validation"
    ops.loc[ops["shard_layout"] == "WIDTH_SHARDED", "remap_scope_status"] = (
        "fixed_unsupported_width_sharded"
    )
    ops.loc[ops["shard_orientation"] == "COL_MAJOR", "remap_scope_status"] = (
        "fixed_unsupported_col_major"
    )
    ops.loc[ops["transpose_mcast"] == 1, "remap_scope_status"] = (
        "fixed_unsupported_transpose_mcast"
    )
    return ops


def pair_trisc_kernel_intervals(events: pd.DataFrame) -> pd.DataFrame:
    kernel = events[
        events["RISC processor type"].isin(TRISC_TYPES)
        & (events["zone name"] == KERNEL_ZONE)
        & events["type"].isin(("ZONE_START", "ZONE_END"))
    ].copy()
    if kernel.empty:
        raise ValueError(f"No {KERNEL_ZONE} events were found for TRISC 0/1/2")

    key = [
        "core_x",
        "core_y",
        "RISC processor type",
        "timer_id",
        "run host ID",
        "zone name",
    ]
    time_column = "time[cycles since reset]"
    kernel = kernel.sort_values(key + [time_column])

    starts = kernel[kernel["type"] == "ZONE_START"].copy()
    ends = kernel[kernel["type"] == "ZONE_END"].copy()
    starts["occurrence"] = starts.groupby(key).cumcount()
    ends["occurrence"] = ends.groupby(key).cumcount()

    merge_key = key + ["occurrence"]
    paired = starts[merge_key + [time_column]].merge(
        ends[merge_key + [time_column]],
        on=merge_key,
        how="outer",
        suffixes=("_start", "_end"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = paired[paired["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(
            f"Found {len(unmatched)} unmatched {KERNEL_ZONE} start/end events; "
            "the profile may be truncated"
        )

    paired["cycles"] = paired[f"{time_column}_end"] - paired[f"{time_column}_start"]
    non_positive = paired[paired["cycles"] <= 0]
    if not non_positive.empty:
        raise ValueError(f"Found {len(non_positive)} non-positive TRISC kernel intervals")
    return paired.drop(columns="_merge")


def aggregate_per_worker(intervals: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        intervals.groupby(
            ["run host ID", "core_x", "core_y", "RISC processor type"],
            as_index=False,
        )["cycles"]
        .sum()
        .pivot(
            index=["run host ID", "core_x", "core_y"],
            columns="RISC processor type",
            values="cycles",
        )
        .fillna(0)
        .reset_index()
    )
    grouped.columns.name = None
    for risc in TRISC_TYPES:
        if risc not in grouped:
            grouped[risc] = 0

    grouped = grouped.rename(
        columns={
            "run host ID": "run_host_id",
            "core_x": "profiler_core_x",
            "core_y": "profiler_core_y",
            "TRISC_0": "trisc_0_cycles",
            "TRISC_1": "trisc_1_cycles",
            "TRISC_2": "trisc_2_cycles",
        }
    )
    grouped["worker_x"] = grouped["profiler_core_x"]
    grouped["worker_y"] = grouped["profiler_core_y"].map(EAGLE_N_PHYSICAL_TO_LOGICAL_Y)
    grouped["is_eagle_n_compute_worker"] = grouped["worker_y"].notna() & grouped[
        "worker_x"
    ].between(0, 4)
    grouped["worker_y"] = grouped["worker_y"].astype("Int64")
    cycle_columns = [f"trisc_{index}_cycles" for index in range(3)]
    grouped["trisc_engine_cycles"] = grouped[cycle_columns].sum(axis=1)
    grouped["active_trisc_count"] = (grouped[cycle_columns] > 0).sum(axis=1)
    return grouped


def join_operation_metadata(per_worker: pd.DataFrame, ops: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "GLOBAL CALL COUNT",
        "OP CODE",
        "OP TYPE",
        "CORE COUNT",
        "MATH FIDELITY",
        "PARALLELIZATION STRATEGY",
        "ATTRIBUTES",
        "model_op_name",
        "model_op_index",
        "model_inference_index",
    ]
    metadata_columns = [column for column in metadata_columns if column in ops.columns]
    metadata = ops[metadata_columns].rename(
        columns={
            "GLOBAL CALL COUNT": "run_host_id",
            "OP CODE": "op_code",
            "OP TYPE": "op_type",
            "CORE COUNT": "reported_core_count",
            "MATH FIDELITY": "math_fidelity",
            "PARALLELIZATION STRATEGY": "parallelization_strategy",
            "ATTRIBUTES": "attributes",
        }
    )
    joined = per_worker.merge(
        metadata, on="run_host_id", how="left", validate="many_to_one", indicator=True
    )
    missing = joined.loc[joined["_merge"] != "both", "run_host_id"].unique()
    if len(missing):
        raise ValueError(
            "Device profile and ops report are not from the same run; missing operation "
            f"metadata for run host IDs {sorted(int(value) for value in missing)[:20]}"
        )
    return joined.drop(columns="_merge")


def build_operation_summary(per_worker: pd.DataFrame) -> pd.DataFrame:
    compute = per_worker[per_worker["is_eagle_n_compute_worker"]].copy()
    summary = (
        compute.groupby(["run_host_id", "op_code", "op_type"], as_index=False)
        .agg(
            measured_worker_count=("trisc_engine_cycles", "size"),
            total_trisc_engine_cycles=("trisc_engine_cycles", "sum"),
            min_worker_trisc_engine_cycles=("trisc_engine_cycles", "min"),
            max_worker_trisc_engine_cycles=("trisc_engine_cycles", "max"),
            mean_worker_trisc_engine_cycles=("trisc_engine_cycles", "mean"),
        )
        .sort_values("run_host_id")
    )
    return summary


def build_model_compute_costs(per_worker: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_inference_index",
        "model_op_name",
        "model_op_index",
        "run_host_id",
        "op_code",
        "worker_x",
        "worker_y",
        "trisc_0_cycles",
        "trisc_1_cycles",
        "trisc_2_cycles",
        "trisc_engine_cycles",
    ]
    return per_worker[
        per_worker["model_op_name"].notna()
        & per_worker["is_eagle_n_compute_worker"]
    ][columns].sort_values(
        ["model_inference_index", "run_host_id", "worker_y", "worker_x"]
    )


def build_model_compute_inventory(ops: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    inventory_columns = [
        "model_inference_index",
        "model_op_index",
        "model_op_name",
        "GLOBAL CALL COUNT",
        "OP CODE",
        "CORE COUNT",
        "shard_layout",
        "shard_orientation",
        "logical_grid_x",
        "logical_grid_y",
        "transpose_mcast",
        "stock_grid",
        "remap_scope_status",
    ]
    inventory_columns = [column for column in inventory_columns if column in ops.columns]
    inventory = ops[ops["model_op_name"].notna()][inventory_columns].rename(
        columns={
            "GLOBAL CALL COUNT": "run_host_id",
            "OP CODE": "op_code",
            "CORE COUNT": "reported_core_count",
        }
    )
    return inventory.merge(
        summary,
        on=["run_host_id", "op_code"],
        how="left",
        validate="one_to_one",
    ).sort_values(["model_inference_index", "model_op_index"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        help=(
            "Directory from one uninterrupted full-model profiling run. It must "
            "contain profile_log_device.csv and exactly one ops_perf_results_*.csv."
        ),
    )
    parser.add_argument("--device-csv", type=Path)
    parser.add_argument("--ops-csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.report_dir is not None:
        if args.device_csv is not None or args.ops_csv is not None:
            parser.error("--report-dir cannot be combined with --device-csv/--ops-csv")
        _reject_stage_split(args.report_dir)
        args.device_csv = args.report_dir / "profile_log_device.csv"
        ops_candidates = sorted(args.report_dir.glob("ops_perf_results_*.csv"))
        if len(ops_candidates) != 1:
            parser.error(
                "--report-dir must contain exactly one ops_perf_results_*.csv; "
                f"found {len(ops_candidates)}"
            )
        args.ops_csv = ops_candidates[0]
    elif args.device_csv is None or args.ops_csv is None:
        parser.error("provide --report-dir or both --device-csv and --ops-csv")
    return args


def main() -> None:
    args = parse_args()
    events, metadata = load_device_events(args.device_csv)
    ops = annotate_topology(annotate_resnet50_compute_sequence(load_ops(args.ops_csv)))
    intervals = pair_trisc_kernel_intervals(events)
    per_worker = join_operation_metadata(aggregate_per_worker(intervals), ops)
    summary = build_operation_summary(per_worker)
    model_compute_costs = build_model_compute_costs(per_worker)
    model_inventory = build_model_compute_inventory(ops, summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_worker_path = args.output_dir / "full_model_trisc_per_op_per_worker.csv"
    summary_path = args.output_dir / "full_model_trisc_op_summary.csv"
    model_costs_path = args.output_dir / "full_model_trisc_model_compute_costs.csv"
    inventory_path = args.output_dir / "full_model_compute_inventory.csv"
    per_worker.sort_values(
        ["run_host_id", "profiler_core_y", "profiler_core_x"]
    ).to_csv(per_worker_path, index=False)
    summary.to_csv(summary_path, index=False)
    model_compute_costs.to_csv(model_costs_path, index=False)
    model_inventory.to_csv(inventory_path, index=False)

    print(f"Device metadata: {metadata}")
    print(f"Paired TRISC kernel intervals: {len(intervals):,}")
    print(f"Operations with TRISC kernel activity: {summary['run_host_id'].nunique():,}")
    print(f"Wrote: {per_worker_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {model_costs_path}")
    print(f"Wrote: {inventory_path}")


if __name__ == "__main__":
    main()
