#!/usr/bin/env python3
"""Build ResNet50 placement domains from the full-model TRISC calibration.

This is a model-specific graph partitioner, not a replacement for the C++
placement planner. It identifies boundaries at unsupported HEIGHT_SHARDED
Matmul execution, residual joins, core-count expansion, and the WIDTH_SHARDED
head. Every movable candidate must still pass the production planner and the
declared validation status before it can enter an optimized schedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_MODEL_OPS = 54
BLOCK_DOMAIN_LAYERS = {"layer3", "layer4"}


def module_prefix(op_name: str) -> str:
    parts = op_name.split(".")
    if len(parts) < 3 or not parts[0].startswith("layer") or not parts[1].startswith("module"):
        raise ValueError(f"Operation does not have a ResNet module prefix: {op_name}")
    return ".".join(parts[:2])


def ordered_workers(costs: pd.DataFrame) -> pd.DataFrame:
    return costs.sort_values(["worker_y", "worker_x"]).reset_index(drop=True)


def worker_topology(costs: pd.DataFrame) -> dict[str, object]:
    ordered = ordered_workers(costs)
    worker_x = sorted(int(value) for value in ordered["worker_x"].unique())
    worker_y = sorted(int(value) for value in ordered["worker_y"].unique())
    return {
        "active_cores": len(ordered),
        "worker_x": worker_x,
        "worker_y": worker_y,
        "grid_x": len(worker_x),
        "grid_y": len(worker_y),
    }


def make_domain(
    domain_id: str,
    kind: str,
    members: list[str],
    inventory_by_name: dict[str, dict[str, object]],
    costs_by_name: dict[str, pd.DataFrame],
    exit_boundary: str,
    validation_status: str,
) -> dict[str, object]:
    member_records = [inventory_by_name[name] for name in members]
    layouts = {record["shard_layout"] for record in member_records}
    orientations = {record["shard_orientation"] for record in member_records}
    if len(layouts) != 1 or len(orientations) != 1:
        raise ValueError(f"Domain {domain_id} mixes layouts or orientations")

    topologies = [worker_topology(costs_by_name[name]) for name in members]
    if next(iter(layouts)) == "BLOCK_SHARDED":
        for previous, current in zip(topologies, topologies[1:]):
            if current["grid_x"] > previous["grid_x"] or current["grid_y"] > previous["grid_y"]:
                raise ValueError(f"BLOCK_SHARDED domain {domain_id} expands within the chain")

    return {
        "domain_id": domain_id,
        "kind": kind,
        "shard_layout": next(iter(layouts)),
        "shard_orientation": next(iter(orientations)),
        "members": members,
        "member_topologies": topologies,
        "entry_active_cores": topologies[0]["active_cores"],
        "entry_grid_x": topologies[0]["grid_x"],
        "entry_grid_y": topologies[0]["grid_y"],
        "exit_boundary": exit_boundary,
        "validation_status": validation_status,
    }


def partition_domains(
    inventory: pd.DataFrame, costs: pd.DataFrame
) -> tuple[list[dict[str, object]], dict[str, str]]:
    if len(inventory) != EXPECTED_MODEL_OPS or inventory["model_op_name"].nunique() != EXPECTED_MODEL_OPS:
        raise ValueError(
            f"Expected {EXPECTED_MODEL_OPS} unique model operations, found "
            f"{len(inventory)} rows/{inventory['model_op_name'].nunique()} names"
        )
    inventory = inventory.sort_values("model_op_index")
    inventory_by_name = inventory.set_index("model_op_name").to_dict("index")
    costs_by_name = {
        name: ordered_workers(group)
        for name, group in costs.groupby("model_op_name", sort=False)
    }

    domains: list[dict[str, object]] = []
    movable_ops: set[str] = set()

    # HEIGHT_SHARDED 3x3 Conv2 can execute a requested placement, but the
    # following 1x1 Matmul cannot consume that remapped HEIGHT placement and
    # therefore exits to stock. Each Conv2 is its own placement domain.
    height_conv2 = inventory[
        (inventory["shard_layout"] == "HEIGHT_SHARDED")
        & inventory["model_op_name"].str.endswith(".conv2")
        & (inventory["op_code"] == "Conv2d")
    ]
    for name in height_conv2["model_op_name"]:
        domain = make_domain(
            domain_id=name,
            kind="height_conv2_singleton",
            members=[name],
            inventory_by_name=inventory_by_name,
            costs_by_name=costs_by_name,
            exit_boundary="following_height_sharded_1x1_matmul_exits_to_stock",
            validation_status="full_model_validated_path_class",
        )
        domains.append(domain)
        movable_ops.add(name)

    # A BLOCK_SHARDED Conv2 starts the requested placement and its following
    # Conv3 consumes that remapped input. Conv1 runs before the request point
    # and therefore remains fixed stock background. The residual join
    # terminates each Conv2 -> Conv3 placement domain.
    block_ops = inventory[
        inventory["shard_layout"].eq("BLOCK_SHARDED")
        & ~inventory["model_op_name"].str.endswith(".downsample")
    ]
    block_modules: list[str] = []
    for name in block_ops["model_op_name"]:
        prefix = module_prefix(name)
        if prefix.split(".")[0] in BLOCK_DOMAIN_LAYERS and prefix not in block_modules:
            block_modules.append(prefix)
    for prefix in block_modules:
        members = [f"{prefix}.conv2", f"{prefix}.conv3"]
        missing = [name for name in members if name not in inventory_by_name]
        if missing:
            raise ValueError(f"Incomplete block domain {prefix}: missing {missing}")
        domain = make_domain(
            domain_id=f"{prefix}.conv2_to_conv3",
            kind="block_conv2_to_conv3_chain",
            members=members,
            inventory_by_name=inventory_by_name,
            costs_by_name=costs_by_name,
            exit_boundary="residual_join",
            validation_status="mechanisms_validated_full_chain_pending",
        )
        domains.append(domain)
        movable_ops.update(members)

    fixed_reasons: dict[str, str] = {}
    for record in inventory.to_dict("records"):
        name = record["model_op_name"]
        if name in movable_ops:
            continue
        if record["shard_layout"] == "WIDTH_SHARDED":
            reason = "unsupported_width_sharded"
        elif name == "stem.conv1":
            reason = "uses_all_20_workers_no_alternative_placement"
        elif record["shard_layout"] == "HEIGHT_SHARDED" and record["op_code"] == "Matmul":
            reason = "height_sharded_1x1_matmul_cannot_execute_remapped_input"
        elif name.endswith(".downsample"):
            reason = "downsample_remapping_not_in_enabled_execution_paths"
        else:
            reason = "not_in_enabled_placement_domain"
        fixed_reasons[name] = reason

    covered = movable_ops | set(fixed_reasons)
    expected = set(inventory["model_op_name"])
    if covered != expected:
        raise ValueError(f"Domain partition did not cover model ops: {sorted(expected - covered)}")
    return domains, fixed_reasons


def flatten_domains(domains: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for domain in domains:
        rows.append(
            {
                "domain_id": domain["domain_id"],
                "kind": domain["kind"],
                "shard_layout": domain["shard_layout"],
                "shard_orientation": domain["shard_orientation"],
                "member_count": len(domain["members"]),
                "member_ops": ";".join(domain["members"]),
                "entry_active_cores": domain["entry_active_cores"],
                "entry_grid_x": domain["entry_grid_x"],
                "entry_grid_y": domain["entry_grid_y"],
                "exit_boundary": domain["exit_boundary"],
                "validation_status": domain["validation_status"],
            }
        )
    return pd.DataFrame(rows)


def build_domain_member_costs(
    domains: list[dict[str, object]], costs: pd.DataFrame
) -> pd.DataFrame:
    op_to_domain = {
        op_name: domain["domain_id"]
        for domain in domains
        for op_name in domain["members"]
    }
    movable = costs[costs["model_op_name"].isin(op_to_domain)].copy()
    movable["domain_id"] = movable["model_op_name"].map(op_to_domain)
    movable = movable.sort_values(["domain_id", "model_op_index", "worker_y", "worker_x"])
    movable["logical_core_index"] = movable.groupby("model_op_name").cumcount()
    columns = [
        "domain_id",
        "model_op_index",
        "model_op_name",
        "op_code",
        "logical_core_index",
        "worker_x",
        "worker_y",
        "trisc_0_cycles",
        "trisc_1_cycles",
        "trisc_2_cycles",
        "trisc_engine_cycles",
    ]
    return movable[columns]


def build_fixed_background(
    all_per_worker: pd.DataFrame, fixed_reasons: dict[str, str]
) -> pd.DataFrame:
    model_names = all_per_worker["model_op_name"].fillna("")
    fixed = all_per_worker[
        model_names.eq("") | model_names.isin(fixed_reasons)
    ].copy()
    fixed = fixed[fixed["is_eagle_n_compute_worker"].astype(str).str.lower() == "true"]
    cycle_columns = ["trisc_0_cycles", "trisc_1_cycles", "trisc_2_cycles", "trisc_engine_cycles"]
    result = (
        fixed.groupby(["worker_x", "worker_y"], as_index=False)[cycle_columns]
        .sum()
        .sort_values(["worker_y", "worker_x"])
    )
    all_workers = pd.DataFrame(
        [(x, y) for y in range(4) for x in range(5)],
        columns=["worker_x", "worker_y"],
    )
    return all_workers.merge(result, on=["worker_x", "worker_y"], how="left").fillna(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = pd.read_csv(args.calibration_dir / "full_model_compute_inventory.csv")
    costs = pd.read_csv(args.calibration_dir / "full_model_trisc_model_compute_costs.csv")
    all_per_worker = pd.read_csv(args.calibration_dir / "full_model_trisc_per_op_per_worker.csv")

    domains, fixed_reasons = partition_domains(inventory, costs)
    domain_table = flatten_domains(domains)
    member_costs = build_domain_member_costs(domains, costs)
    fixed_background = build_fixed_background(all_per_worker, fixed_reasons)
    fixed_operations = pd.DataFrame(
        [
            {"model_op_name": name, "fixed_reason": reason}
            for name, reason in fixed_reasons.items()
        ]
    )

    domain_table.to_csv(args.calibration_dir / "full_model_placement_domains.csv", index=False)
    member_costs.to_csv(args.calibration_dir / "full_model_domain_member_costs.csv", index=False)
    fixed_background.to_csv(
        args.calibration_dir / "full_model_fixed_background_per_worker.csv", index=False
    )
    fixed_operations.to_csv(args.calibration_dir / "full_model_fixed_operations.csv", index=False)
    with (args.calibration_dir / "full_model_placement_domains.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(domains, stream, indent=2)
        stream.write("\n")

    movable_ops = sum(len(domain["members"]) for domain in domains)
    print(f"Movable placement domains: {len(domains)}")
    print(f"Movable model operations: {movable_ops}")
    print(f"Fixed model operations: {len(fixed_reasons)}")
    print(f"Fixed background worker rows: {len(fixed_background)}")


if __name__ == "__main__":
    main()
