#!/usr/bin/env python3
"""Enumerate legal-slice placement candidates and predict worker costs.

Candidate costs permute stock-measured logical-core TRISC exposure onto the
selected physical workers. This placement-invariance assumption must be checked
with representative remapped profiles before an optimized policy is accepted.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import pandas as pd


COMPUTE_GRID_X = 5
COMPUTE_GRID_Y = 4


def segment_count(values: tuple[int, ...]) -> int:
    if not values:
        return 0
    return 1 + sum(current != previous + 1 for previous, current in zip(values, values[1:]))


def placement_id(excluded_rows: tuple[int, ...], excluded_columns: tuple[int, ...]) -> str:
    if not excluded_rows and not excluded_columns:
        return "stock"
    row_part = "none" if not excluded_rows else "-".join(map(str, excluded_rows))
    column_part = "none" if not excluded_columns else "-".join(map(str, excluded_columns))
    return f"exclude_r{row_part}_c{column_part}"


def block_candidates(grid_x: int, grid_y: int) -> list[dict[str, object]]:
    candidates = []
    for worker_x in combinations(range(COMPUTE_GRID_X), grid_x):
        if segment_count(worker_x) > 2:
            continue
        for worker_y in combinations(range(COMPUTE_GRID_Y), grid_y):
            if segment_count(worker_y) > 2:
                continue
            # Exclusions describe only coordinates skipped while selecting the
            # logical axes. Unused coordinates beyond the last selected axis
            # are not explicit exclusions; e.g. x=(0,1,2,3) is stock 4-wide,
            # not "exclude column 4".
            excluded_rows = tuple(y for y in range(worker_y[-1] + 1) if y not in worker_y)
            excluded_columns = tuple(x for x in range(worker_x[-1] + 1) if x not in worker_x)
            workers = [(x, y) for y in worker_y for x in worker_x]
            candidates.append(
                {
                    "placement_id": placement_id(excluded_rows, excluded_columns),
                    "excluded_rows": excluded_rows,
                    "excluded_columns": excluded_columns,
                    "worker_x": worker_x,
                    "worker_y": worker_y,
                    "ordered_entry_workers": workers,
                }
            )
    return sorted(candidates, key=lambda candidate: candidate["placement_id"] != "stock")


def height_candidates(num_cores: int, row_scan_width: int) -> list[dict[str, object]]:
    # The enabled/validated policy slice requests at most one excluded row and
    # one excluded column. Deduplication removes exclusions outside the prefix
    # that produce the stock worker order.
    row_choices = [()] + [(row,) for row in range(COMPUTE_GRID_Y)]
    column_choices = [()] + [(column,) for column in range(row_scan_width)]
    unique: dict[tuple[tuple[int, int], ...], dict[str, object]] = {}
    for excluded_rows in row_choices:
        for excluded_columns in column_choices:
            workers = []
            for y in range(COMPUTE_GRID_Y):
                if y in excluded_rows:
                    continue
                for x in range(row_scan_width):
                    if x in excluded_columns:
                        continue
                    workers.append((x, y))
                    if len(workers) == num_cores:
                        break
                if len(workers) == num_cores:
                    break
            if len(workers) != num_cores:
                continue
            worker_key = tuple(workers)
            candidate = {
                "placement_id": placement_id(excluded_rows, excluded_columns),
                "excluded_rows": excluded_rows,
                "excluded_columns": excluded_columns,
                "worker_x": tuple(sorted({x for x, _ in workers})),
                "worker_y": tuple(sorted({y for _, y in workers})),
                "ordered_entry_workers": workers,
            }
            previous = unique.get(worker_key)
            if previous is None or candidate["placement_id"] == "stock":
                unique[worker_key] = candidate
    return sorted(unique.values(), key=lambda candidate: candidate["placement_id"] != "stock")


def map_domain_costs(
    domain: dict[str, object],
    candidate: dict[str, object],
    member_costs: pd.DataFrame,
) -> list[dict[str, object]]:
    physical_costs: dict[tuple[int, int], dict[str, int]] = {}
    for topology, op_name in zip(domain["member_topologies"], domain["members"]):
        logical = member_costs[member_costs["model_op_name"] == op_name].sort_values(
            "logical_core_index"
        )
        if domain["shard_layout"] == "HEIGHT_SHARDED":
            physical_workers = candidate["ordered_entry_workers"][: len(logical)]
        else:
            selected_x = candidate["worker_x"][: int(topology["grid_x"])]
            selected_y = candidate["worker_y"][: int(topology["grid_y"])]
            physical_workers = [(x, y) for y in selected_y for x in selected_x]
        if len(physical_workers) != len(logical):
            raise ValueError(
                f"Candidate {candidate['placement_id']} cannot map {op_name}: "
                f"{len(physical_workers)} workers for {len(logical)} logical costs"
            )
        for (_, cost), worker in zip(logical.iterrows(), physical_workers):
            totals = physical_costs.setdefault(
                worker,
                {
                    "trisc_0_cycles": 0,
                    "trisc_1_cycles": 0,
                    "trisc_2_cycles": 0,
                    "trisc_engine_cycles": 0,
                },
            )
            for column in totals:
                totals[column] += int(cost[column])

    return [
        {
            "domain_id": domain["domain_id"],
            "placement_id": candidate["placement_id"],
            "worker_x": worker[0],
            "worker_y": worker[1],
            **cycles,
        }
        for worker, cycles in sorted(physical_costs.items(), key=lambda item: (item[0][1], item[0][0]))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (args.calibration_dir / "full_model_placement_domains.json").open(
        "r", encoding="utf-8"
    ) as stream:
        domains = json.load(stream)
    member_costs = pd.read_csv(args.calibration_dir / "full_model_domain_member_costs.csv")

    candidate_rows = []
    cost_rows = []
    for domain in domains:
        if domain["shard_layout"] == "HEIGHT_SHARDED":
            candidates = height_candidates(
                int(domain["entry_active_cores"]), int(domain["entry_grid_x"])
            )
        elif domain["shard_layout"] == "BLOCK_SHARDED":
            candidates = block_candidates(
                int(domain["entry_grid_x"]), int(domain["entry_grid_y"])
            )
        else:
            raise ValueError(f"Unsupported movable domain layout: {domain['shard_layout']}")

        stock_total = None
        for candidate in candidates:
            mapped_costs = map_domain_costs(domain, candidate, member_costs)
            candidate_total = sum(row["trisc_engine_cycles"] for row in mapped_costs)
            if candidate["placement_id"] == "stock":
                stock_total = candidate_total
            candidate_rows.append(
                {
                    "domain_id": domain["domain_id"],
                    "placement_id": candidate["placement_id"],
                    "excluded_rows": json.dumps(candidate["excluded_rows"]),
                    "excluded_columns": json.dumps(candidate["excluded_columns"]),
                    "ordered_entry_workers": json.dumps(candidate["ordered_entry_workers"]),
                    "active_entry_cores": len(candidate["ordered_entry_workers"]),
                    "x_segments": segment_count(candidate["worker_x"]),
                    "y_segments": segment_count(candidate["worker_y"]),
                    "extra_axis_segments": (
                        segment_count(candidate["worker_x"])
                        + segment_count(candidate["worker_y"])
                        - 2
                    ),
                    "predicted_total_trisc_engine_cycles": candidate_total,
                    "cost_model": "stock_logical_core_permutation_unvalidated",
                    "validation_status": domain["validation_status"],
                }
            )
            cost_rows.extend(mapped_costs)
        if stock_total is None:
            raise ValueError(f"Domain {domain['domain_id']} has no stock candidate")
        totals = {
            row["predicted_total_trisc_engine_cycles"]
            for row in candidate_rows
            if row["domain_id"] == domain["domain_id"]
        }
        if totals != {stock_total}:
            raise ValueError(f"Candidate mapping changed total cost for {domain['domain_id']}")

    candidates_frame = pd.DataFrame(candidate_rows)
    costs_frame = pd.DataFrame(cost_rows)
    candidates_frame.to_csv(
        args.calibration_dir / "full_model_domain_candidates.csv", index=False
    )
    costs_frame.to_csv(
        args.calibration_dir / "full_model_candidate_worker_costs.csv", index=False
    )
    print(f"Domains with candidates: {candidates_frame['domain_id'].nunique()}")
    print(f"Placement candidates: {len(candidates_frame)}")
    print(f"Predicted candidate-worker cost rows: {len(costs_frame)}")


if __name__ == "__main__":
    main()
