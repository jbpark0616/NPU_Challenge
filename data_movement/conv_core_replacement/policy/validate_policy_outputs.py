#!/usr/bin/env python3
"""Validate cross-file consistency of generated full-model policy outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GRID_X = 5
GRID_Y = 4
WORKERS = [(x, y) for y in range(GRID_Y) for x in range(GRID_X)]
WORKER_INDEX = {worker: index for index, worker in enumerate(WORKERS)}


def vector_from_rows(rows: pd.DataFrame, value_column: str) -> np.ndarray:
    vector = np.zeros(len(WORKERS), dtype=np.int64)
    for row in rows.itertuples(index=False):
        worker = (int(row.worker_x), int(row.worker_y))
        vector[WORKER_INDEX[worker]] += int(getattr(row, value_column))
    return vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help="Optional schedule/candidate directory; missing inputs fall back to calibration-dir.",
    )
    parser.add_argument("--horizon", type=int, default=20)
    return parser.parse_args()


def policy_file(calibration_dir: Path, experiment_dir: Path | None, name: str) -> Path:
    if experiment_dir is not None:
        candidate = experiment_dir / name
        if candidate.is_file():
            return candidate
    return calibration_dir / name


def main() -> None:
    args = parse_args()
    directory = args.calibration_dir
    inventory = pd.read_csv(directory / "full_model_compute_inventory.csv")
    domains = pd.read_csv(directory / "full_model_placement_domains.csv")
    member_costs = pd.read_csv(directory / "full_model_domain_member_costs.csv")
    fixed = pd.read_csv(
        policy_file(
            directory, args.experiment_dir, "full_model_fixed_background_per_worker.csv"
        )
    )
    candidates = pd.read_csv(
        policy_file(directory, args.experiment_dir, "full_model_domain_candidates.csv")
    )
    candidate_costs = pd.read_csv(
        policy_file(directory, args.experiment_dir, "full_model_candidate_worker_costs.csv")
    )
    raw = pd.read_csv(
        policy_file(directory, args.experiment_dir, "full_model_trisc_per_op_per_worker.csv")
    )
    schedule = pd.read_csv(
        policy_file(
            directory,
            args.experiment_dir,
            f"full_model_schedule_h{args.horizon}.csv",
        )
    )
    comparison = pd.read_csv(
        policy_file(
            directory,
            args.experiment_dir,
            "full_model_stock_vs_optimized_per_worker.csv",
        )
    )
    with policy_file(
        directory,
        args.experiment_dir,
        f"full_model_schedule_h{args.horizon}.json",
    ).open(
        "r", encoding="utf-8"
    ) as stream:
        manifest = json.load(stream)

    if len(inventory) != 54 or inventory["model_op_name"].nunique() != 54:
        raise ValueError("Expected exactly 54 mapped ResNet50 compute operations")
    if len(domains) != 16 or domains["domain_id"].nunique() != 16:
        raise ValueError("Expected exactly 16 executable placement domains")
    if int(domains["member_count"].sum()) != 25:
        raise ValueError("Expected 25 movable model operations")
    if (domains["shard_layout"] == "WIDTH_SHARDED").any():
        raise ValueError("WIDTH_SHARDED operation entered the movable domain set")
    if candidates.groupby("domain_id")["placement_id"].apply(
        lambda values: (values == "stock").sum()
    ).ne(1).any():
        raise ValueError("Every domain must have exactly one stock candidate")
    required_candidate_columns = {
        "x_segments",
        "y_segments",
        "extra_axis_segments",
    }
    if not required_candidate_columns.issubset(candidates.columns):
        raise ValueError("Candidate segment metadata is missing")
    if not (
        candidates["extra_axis_segments"]
        == candidates["x_segments"] + candidates["y_segments"] - 2
    ).all():
        raise ValueError("Candidate segment penalty is inconsistent")

    fixed_vector = vector_from_rows(fixed, "trisc_engine_cycles")
    stock_vector = fixed_vector.copy()
    for domain in domains["domain_id"]:
        rows = candidate_costs[
            (candidate_costs["domain_id"] == domain)
            & (candidate_costs["placement_id"] == "stock")
        ]
        stock_vector += vector_from_rows(rows, "trisc_engine_cycles")
    measured_rows = raw[
        raw["is_eagle_n_compute_worker"].astype(str).str.lower() == "true"
    ]
    measured_vector = vector_from_rows(measured_rows, "trisc_engine_cycles")
    if not np.array_equal(stock_vector, measured_vector):
        raise ValueError("Stock fixed/movable partition does not reconstruct raw cycles")

    expected_pairs = {
        (phase, domain)
        for phase in range(args.horizon)
        for domain in domains["domain_id"]
    }
    actual_pairs = list(zip(schedule["phase"], schedule["domain_id"]))
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError("Schedule must contain one row per phase/domain pair")

    candidate_lookup = candidates.set_index(["domain_id", "placement_id"])
    optimized_period = fixed_vector * args.horizon
    schedule_segment_penalty = 0
    for row in schedule.itertuples(index=False):
        key = (row.domain_id, row.placement_id)
        if key not in candidate_lookup.index:
            raise ValueError(f"Schedule references unknown candidate {key}")
        candidate = candidate_lookup.loc[key]
        schedule_segment_penalty += int(candidate["extra_axis_segments"])
        if json.loads(row.excluded_rows) != json.loads(candidate["excluded_rows"]):
            raise ValueError(f"Excluded-row mismatch for {key}")
        if json.loads(row.excluded_columns) != json.loads(candidate["excluded_columns"]):
            raise ValueError(f"Excluded-column mismatch for {key}")
        rows = candidate_costs[
            (candidate_costs["domain_id"] == row.domain_id)
            & (candidate_costs["placement_id"] == row.placement_id)
        ]
        optimized_period += vector_from_rows(rows, "trisc_engine_cycles")

    comparison = comparison.sort_values(["worker_y", "worker_x"])
    expected_optimized = optimized_period / args.horizon
    reported_optimized = comparison["optimized_cycles_per_inference"].to_numpy()
    if not np.allclose(expected_optimized, reported_optimized, rtol=0, atol=1e-9):
        raise ValueError("Schedule costs do not reproduce the reported optimized loads")

    if manifest["horizon"] != args.horizon:
        raise ValueError("Manifest horizon does not match the validated schedule")
    if manifest["status"] != "predicted_not_deployable_until_candidate_validation":
        raise ValueError("Manifest must retain its pre-validation status")
    if manifest["cost_model"] != "stock_logical_core_permutation_unvalidated":
        raise ValueError("Manifest cost-model warning is missing")
    if manifest["solver"] not in {
        "highspy_warm_started_bounded_lexicographic_milp",
        "gurobi_warm_started_bounded_lexicographic_milp",
    }:
        raise ValueError("Manifest must identify a supported warm-started solver")

    schedule_switches = 0
    for domain in domains["domain_id"]:
        domain_schedule = schedule[schedule["domain_id"] == domain].set_index("phase")
        for phase in range(args.horizon):
            next_phase = (phase + 1) % args.horizon
            schedule_switches += (
                domain_schedule.loc[phase, "placement_id"]
                != domain_schedule.loc[next_phase, "placement_id"]
            )
    solver_objectives = manifest["solver_objectives"]
    if not np.isclose(
        schedule_segment_penalty,
        solver_objectives["stage3_extra_axis_segment_penalty"],
        rtol=0,
        # Solver-reported integer objectives can retain a few micro-units of
        # floating-point tolerance even when the reconstructed count is exact.
        atol=1e-5,
    ):
        raise ValueError("Schedule does not reproduce the reported segment penalty")
    if schedule_switches != solver_objectives["minimum_cyclic_phase_switches"]:
        raise ValueError("Schedule does not attain the reported cyclic-switch minimum")
    used_domain_placements = schedule[["domain_id", "placement_id"]].drop_duplicates()
    if len(used_domain_placements) != int(solver_objectives["used_domain_placements"]):
        raise ValueError("Schedule does not reproduce the reported placement count")

    # Ensure member-cost coverage is neither stale nor duplicated.
    expected_members = set(
        op_name
        for member_list in domains["member_ops"]
        for op_name in member_list.split(";")
    )
    if set(member_costs["model_op_name"]) != expected_members:
        raise ValueError("Domain member-cost table does not match domain membership")

    print("Policy output validation: PASS")
    print("Stock reconstruction: exact")
    print(f"Schedule coverage: {len(schedule)} rows ({args.horizon} x {len(domains)})")
    print(f"Candidate references: {len(candidates)} valid candidates")
    print("Predicted optimized load reconstruction: exact")
    print(f"Extra-axis segment penalty: {schedule_segment_penalty}")
    print(f"Unique domain placements: {len(used_domain_placements)}")
    print(f"Cyclic phase switches: {schedule_switches}")


if __name__ == "__main__":
    main()
