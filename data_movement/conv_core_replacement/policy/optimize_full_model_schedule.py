#!/usr/bin/env python3
"""Optimize a predicted periodic core-remapping schedule for ResNet50.

The solver uses a bounded lexicographic mixed-integer model over candidate-use
counts. It minimizes maximum cumulative worker exposure, total absolute
deviation, extra physical-axis segments, the number of domain-placement
programs used, and finally the minimum achievable cyclic phase switches. Every
stage records its optimality certificate or remaining gap. The result is a
research prediction until candidate cycle permutation and pending full-chain
placements are validated on device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import highspy
import gurobipy as gp
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix


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


def load_candidate_vectors(
    candidates: pd.DataFrame, worker_costs: pd.DataFrame
) -> tuple[
    list[str],
    dict[str, list[str]],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], int],
]:
    domains = list(dict.fromkeys(candidates["domain_id"]))
    placements: dict[str, list[str]] = {}
    vectors: dict[tuple[str, str], np.ndarray] = {}
    segment_penalties: dict[tuple[str, str], int] = {}
    for domain in domains:
        domain_candidates = candidates[candidates["domain_id"] == domain]
        placement_ids = list(domain_candidates["placement_id"])
        if placement_ids.count("stock") != 1:
            raise ValueError(f"Domain {domain} must have exactly one stock candidate")
        placements[domain] = placement_ids
        for placement in placement_ids:
            rows = worker_costs[
                (worker_costs["domain_id"] == domain)
                & (worker_costs["placement_id"] == placement)
            ]
            vectors[(domain, placement)] = vector_from_rows(rows, "trisc_engine_cycles")
            candidate = domain_candidates[
                domain_candidates["placement_id"] == placement
            ].iloc[0]
            segment_penalties[(domain, placement)] = int(
                candidate["extra_axis_segments"]
            )
    return domains, placements, vectors, segment_penalties


def verify_stock_reconstruction(
    fixed: np.ndarray,
    domains: list[str],
    vectors: dict[tuple[str, str], np.ndarray],
    all_per_worker: pd.DataFrame,
) -> np.ndarray:
    reconstructed = fixed.copy()
    for domain in domains:
        reconstructed += vectors[(domain, "stock")]
    compute_rows = all_per_worker[
        all_per_worker["is_eagle_n_compute_worker"].astype(str).str.lower() == "true"
    ]
    measured = vector_from_rows(compute_rows, "trisc_engine_cycles")
    if not np.array_equal(reconstructed, measured):
        differences = {
            str(worker): int(reconstructed[index] - measured[index])
            for index, worker in enumerate(WORKERS)
            if reconstructed[index] != measured[index]
        }
        raise ValueError(f"Stock partition does not reconstruct the full-model profile: {differences}")
    return measured


def optimize_exact(
    fixed: np.ndarray,
    domains: list[str],
    placements: dict[str, list[str]],
    vectors: dict[tuple[str, str], np.ndarray],
    segment_penalties: dict[tuple[str, str], int],
    horizon: int,
    time_limit: float,
    initial_counts: dict[tuple[str, str], int] | None = None,
    solver_output: bool = False,
    start_stage: int = 1,
    stop_after_stage: int = 5,
    prior_stage_certificates: dict[str, dict[str, object]] | None = None,
    solver_backend: str = "highspy",
) -> tuple[dict[tuple[int, str], str], np.ndarray, dict[str, object]]:
    keys = [(domain, placement) for domain in domains for placement in placements[domain]]
    candidate_count = len(keys)
    worker_count = len(WORKERS)
    domain_count = len(domains)
    z_index = candidate_count
    deviation_start = z_index + 1
    used_start = deviation_start + worker_count
    multi_start = used_start + candidate_count
    switch_start = multi_start + domain_count
    variable_count = switch_start + candidate_count
    domain_indices = {domain: index for index, domain in enumerate(domains)}

    costs = np.column_stack([vectors[key] for key in keys]).astype(np.float64)
    fixed_period = fixed.astype(np.float64) * horizon
    total_period_cycles = float(fixed_period.sum())
    for domain in domains:
        total_period_cycles += float(vectors[(domain, "stock")].sum() * horizon)
    mean_period_load = total_period_cycles / worker_count

    # A fresh experiment has no saved H-phase schedule yet.  The all-stock
    # assignment is always feasible and provides a deterministic MIP start;
    # saved counts, when present, still take precedence.
    if initial_counts is None:
        initial_counts = {
            (domain, placement): horizon if placement == "stock" else 0
            for domain in domains
            for placement in placements[domain]
        }

    initial_warm_start: np.ndarray | None = None
    if initial_counts is not None:
        initial_count_vector = np.asarray(
            [initial_counts.get(key, 0) for key in keys], dtype=np.float64
        )
        for domain in domains:
            domain_total = sum(
                initial_counts.get((domain, placement), 0)
                for placement in placements[domain]
            )
            if domain_total != horizon:
                raise ValueError(
                    f"Initial schedule count for {domain} is {domain_total}, expected {horizon}"
                )
        initial_loads = fixed_period + costs @ initial_count_vector
        initial_warm_start = np.zeros(variable_count, dtype=np.float64)
        initial_warm_start[:candidate_count] = initial_count_vector
        initial_warm_start[z_index] = initial_loads.max()
        initial_warm_start[deviation_start:used_start] = np.abs(
            initial_loads - mean_period_load
        )

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for domain in domains:
        row = np.zeros(variable_count)
        for index, (candidate_domain, _) in enumerate(keys):
            if candidate_domain == domain:
                row[index] = 1
        rows.append(row)
        lower.append(float(horizon))
        upper.append(float(horizon))

    for worker_index in range(worker_count):
        # fixed + candidate load <= z
        row = np.zeros(variable_count)
        row[:candidate_count] = costs[worker_index]
        row[z_index] = -1
        rows.append(row)
        lower.append(-np.inf)
        upper.append(-fixed_period[worker_index])

        # Absolute deviation from the invariant period mean.
        positive = np.zeros(variable_count)
        positive[:candidate_count] = costs[worker_index]
        positive[deviation_start + worker_index] = -1
        rows.append(positive)
        lower.append(-np.inf)
        upper.append(mean_period_load - fixed_period[worker_index])

        negative = np.zeros(variable_count)
        negative[:candidate_count] = -costs[worker_index]
        negative[deviation_start + worker_index] = -1
        rows.append(negative)
        lower.append(-np.inf)
        upper.append(fixed_period[worker_index] - mean_period_load)

    usage_constraints: list[tuple[np.ndarray, float, float]] = []
    switch_constraints: list[tuple[np.ndarray, float, float]] = []

    for candidate_index in range(candidate_count):
        # A candidate is used if and only if its count is nonzero once the
        # used variables are binary.
        row = np.zeros(variable_count)
        row[candidate_index] = 1
        row[used_start + candidate_index] = -horizon
        usage_constraints.append((row, -np.inf, 0.0))

        # Binary u is exactly the indicator n > 0, not merely an upper-bound
        # helper. This tightens the Stage-4/5 relaxations substantially.
        row = np.zeros(variable_count)
        row[used_start + candidate_index] = 1
        row[candidate_index] = -1
        usage_constraints.append((row, -np.inf, 0.0))

    for domain in domains:
        domain_candidate_indices = [
            index
            for index, (candidate_domain, _) in enumerate(keys)
            if candidate_domain == domain
        ]
        multi_index = multi_start + domain_indices[domain]

        # q_d == 1 iff domain d uses at least two placements.
        row = np.zeros(variable_count)
        row[[used_start + index for index in domain_candidate_indices]] = 1
        row[multi_index] = -(len(domain_candidate_indices) - 1)
        switch_constraints.append((row, -np.inf, 1.0))

        row = np.zeros(variable_count)
        row[[used_start + index for index in domain_candidate_indices]] = -1
        row[multi_index] = 1
        switch_constraints.append((row, -np.inf, -1.0))

        for candidate_index in domain_candidate_indices:
            transition_index = switch_start + candidate_index
            used_index = used_start + candidate_index

            # v_dp = u_dp AND q_d. With each used placement placed in one
            # contiguous cyclic block, sum(v_dp) is the exact minimum number
            # of phase transitions for that domain.
            row = np.zeros(variable_count)
            row[transition_index] = 1
            row[used_index] = -1
            switch_constraints.append((row, -np.inf, 0.0))

            row = np.zeros(variable_count)
            row[transition_index] = 1
            row[multi_index] = -1
            switch_constraints.append((row, -np.inf, 0.0))

            row = np.zeros(variable_count)
            row[transition_index] = -1
            row[used_index] = 1
            row[multi_index] = 1
            switch_constraints.append((row, -np.inf, 1.0))

    base_matrix = np.vstack(rows)
    base_lower = np.asarray(lower)
    base_upper = np.asarray(upper)
    variable_lower = np.zeros(variable_count)
    variable_upper = np.full(variable_count, np.inf)
    variable_upper[:candidate_count] = horizon
    variable_upper[used_start:variable_count] = 1
    base_integrality = np.zeros(variable_count)
    base_integrality[:candidate_count] = 1
    stage_certificates: dict[str, dict[str, object]] = dict(
        prior_stage_certificates or {}
    )

    def complete_warm_start(
        incumbent: np.ndarray | None,
        *,
        binary_used: bool,
        binary_switch: bool,
    ) -> np.ndarray | None:
        if incumbent is None:
            return None
        warm = np.asarray(incumbent, dtype=np.float64).copy()
        counts = np.rint(warm[:candidate_count]).astype(int)
        if binary_used or binary_switch:
            used = (counts > 0).astype(np.float64)
            warm[used_start:multi_start] = used
        if binary_switch:
            for domain in domains:
                candidate_indices = [
                    index
                    for index, (candidate_domain, _) in enumerate(keys)
                    if candidate_domain == domain
                ]
                is_multi = float(np.count_nonzero(counts[candidate_indices]) > 1)
                warm[multi_start + domain_indices[domain]] = is_multi
                for candidate_index in candidate_indices:
                    warm[switch_start + candidate_index] = (
                        warm[used_start + candidate_index] * is_multi
                    )
        return warm

    def solve(
        stage_name: str,
        objective: np.ndarray,
        extra_rows: list[tuple[np.ndarray, float]],
        *,
        binary_used: bool = False,
        binary_switch: bool = False,
        allow_no_solution: bool = False,
        warm_start: np.ndarray | None = None,
    ) -> object:
        optional_constraints: list[tuple[np.ndarray, float, float]] = []
        if binary_used or binary_switch:
            optional_constraints.extend(usage_constraints)
        if binary_switch:
            optional_constraints.extend(switch_constraints)
        optional_constraints.extend(
            (row, -np.inf, bound) for row, bound in extra_rows
        )
        matrix = base_matrix
        constraint_lower = base_lower
        constraint_upper = base_upper
        if optional_constraints:
            matrix = np.vstack(
                [base_matrix, *[row for row, _, _ in optional_constraints]]
            )
            constraint_lower = np.concatenate(
                [base_lower, np.asarray([low for _, low, _ in optional_constraints])]
            )
            constraint_upper = np.concatenate(
                [base_upper, np.asarray([high for _, _, high in optional_constraints])]
            )
        integrality = base_integrality.copy()
        if binary_used:
            integrality[used_start:multi_start] = 1
        if binary_switch:
            integrality[used_start:variable_count] = 1
        completed_warm_start = complete_warm_start(
            warm_start,
            binary_used=binary_used,
            binary_switch=binary_switch,
        )
        if solver_backend == "gurobi":
            solver = gp.Model(stage_name)
            solver.Params.OutputFlag = int(solver_output)
            solver.Params.TimeLimit = time_limit
            solver.Params.MIPGap = 0.0
            solver.Params.MIPGapAbs = 0.0
            variable_types = np.where(
                integrality != 0, gp.GRB.INTEGER, gp.GRB.CONTINUOUS
            ).tolist()
            variables = solver.addMVar(
                variable_count,
                lb=variable_lower,
                ub=variable_upper,
                vtype=variable_types,
                name="x",
            )
            sparse_matrix = csc_matrix(matrix)
            equality = np.isfinite(constraint_lower) & np.isfinite(
                constraint_upper
            ) & np.isclose(constraint_lower, constraint_upper, rtol=0, atol=1e-9)
            lower_only = np.isfinite(constraint_lower) & ~equality
            upper_only = np.isfinite(constraint_upper) & ~equality
            if equality.any():
                solver.addMConstr(
                    sparse_matrix[equality],
                    variables,
                    "=",
                    constraint_upper[equality],
                )
            if lower_only.any():
                solver.addMConstr(
                    sparse_matrix[lower_only],
                    variables,
                    ">",
                    constraint_lower[lower_only],
                )
            if upper_only.any():
                solver.addMConstr(
                    sparse_matrix[upper_only],
                    variables,
                    "<",
                    constraint_upper[upper_only],
                )
            solver.setObjective(objective @ variables, gp.GRB.MINIMIZE)
            warm_start_status = None
            if completed_warm_start is not None:
                variables.Start = completed_warm_start
                warm_start_status = "Gurobi MIP start assigned"
            solver.optimize()
            has_solution = solver.SolCount > 0
            x = np.asarray(variables.X, dtype=np.float64) if has_solution else None
            result = SimpleNamespace(
                x=x,
                fun=float(objective @ x) if x is not None else None,
                success=solver.Status == gp.GRB.OPTIMAL,
                status=int(solver.Status),
                message={
                    gp.GRB.OPTIMAL: "Optimal",
                    gp.GRB.TIME_LIMIT: "Time limit reached",
                    gp.GRB.INFEASIBLE: "Infeasible",
                }.get(solver.Status, f"Gurobi status {solver.Status}"),
                mip_dual_bound=float(solver.ObjBound),
                mip_gap=float(solver.MIPGap) if has_solution else np.inf,
                mip_node_count=int(solver.NodeCount),
                warm_start_submitted=completed_warm_start is not None,
                warm_start_status=warm_start_status,
            )
        else:
            sparse_matrix = csc_matrix(matrix)
            lp = highspy.HighsLp()
            lp.num_col_ = variable_count
            lp.num_row_ = matrix.shape[0]
            lp.col_cost_ = np.asarray(objective, dtype=np.float64)
            lp.col_lower_ = variable_lower
            lp.col_upper_ = variable_upper
            lp.row_lower_ = constraint_lower
            lp.row_upper_ = constraint_upper
            lp.integrality_ = [
                highspy.HighsVarType.kInteger
                if value
                else highspy.HighsVarType.kContinuous
                for value in integrality
            ]
            lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
            lp.a_matrix_.num_col_ = variable_count
            lp.a_matrix_.num_row_ = matrix.shape[0]
            lp.a_matrix_.start_ = sparse_matrix.indptr.astype(np.int32)
            lp.a_matrix_.index_ = sparse_matrix.indices.astype(np.int32)
            lp.a_matrix_.value_ = sparse_matrix.data.astype(np.float64)

            solver = highspy.Highs()
            solver.setOptionValue("time_limit", time_limit)
            solver.setOptionValue("output_flag", solver_output)
            solver.setOptionValue("mip_rel_gap", 0.0)
            solver.setOptionValue("mip_abs_gap", 0.0)
            solver.passModel(lp)
            warm_start_status = None
            if completed_warm_start is not None:
                warm_start_status = solver.setSolution(
                    variable_count,
                    np.arange(variable_count, dtype=np.int32),
                    completed_warm_start,
                )
            solver.run()
            model_status = solver.getModelStatus()
            solution = solver.getSolution()
            info = solver.getInfo()
            has_solution = bool(solution.value_valid)
            x = np.asarray(solution.col_value, dtype=np.float64) if has_solution else None
            result = SimpleNamespace(
                x=x,
                fun=float(objective @ x) if x is not None else None,
                success=model_status == highspy.HighsModelStatus.kOptimal,
                status=int(model_status),
                message=solver.modelStatusToString(model_status),
                mip_dual_bound=float(info.mip_dual_bound),
                mip_gap=float(info.mip_gap),
                mip_node_count=int(info.mip_node_count),
                warm_start_submitted=completed_warm_start is not None,
                warm_start_status=(
                    str(warm_start_status) if warm_start_status is not None else None
                ),
            )
        if result.x is None or result.fun is None:
            stage_certificates[stage_name] = {
                "proven_optimal": False,
                "feasible_incumbent": False,
                "status": int(result.status),
                "message": str(result.message),
            }
            if allow_no_solution:
                return result
            raise RuntimeError(
                "MILP did not find a feasible solution: "
                f"{result.message}; incumbent={result.fun}; "
                f"dual_bound={getattr(result, 'mip_dual_bound', None)}; "
                f"gap={getattr(result, 'mip_gap', None)}; "
                f"nodes={getattr(result, 'mip_node_count', None)}"
            )
        stage_certificates[stage_name] = {
            "proven_optimal": bool(result.success),
            "feasible_incumbent": True,
            "status": int(result.status),
            "message": str(result.message),
            "incumbent_objective": float(result.fun),
            "dual_bound": float(getattr(result, "mip_dual_bound", result.fun)),
            "relative_gap": float(getattr(result, "mip_gap", 0.0)),
            "node_count": int(getattr(result, "mip_node_count", 0)),
            "warm_start_submitted": result.warm_start_submitted,
            "warm_start_status": result.warm_start_status,
        }
        print(
            f"{stage_name}: status={result.message} incumbent={result.fun} "
            f"bound={result.mip_dual_bound} gap={result.mip_gap:.6%} "
            f"warm_start={result.warm_start_status}",
            flush=True,
        )
        return result

    if initial_warm_start is None:
        raise ValueError("A valid existing H-phase schedule is required for staged optimization")

    current_result = SimpleNamespace(x=initial_warm_start)
    placement_search_scope = "global"
    max_row = np.zeros(variable_count)
    max_row[z_index] = 1
    deviation_row = np.zeros(variable_count)
    deviation_row[deviation_start:used_start] = 1
    segment_row = np.zeros(variable_count)
    segment_row[:candidate_count] = np.asarray(
        [segment_penalties[key] for key in keys], dtype=np.float64
    )
    used_row = np.zeros(variable_count)
    used_row[used_start:multi_start] = 1

    if start_stage <= 1 <= stop_after_stage:
        objective = np.zeros(variable_count)
        objective[z_index] = 1
        current_result = solve(
            "minimize_max_load", objective, [], warm_start=current_result.x
        )
    current_counts = np.rint(current_result.x[:candidate_count]).astype(int)
    current_loads = fixed_period + costs @ current_counts
    optimal_max = float(current_loads.max())
    current_result.x[z_index] = optimal_max

    if stop_after_stage >= 2:
        current_result.x[deviation_start:used_start] = np.abs(
            current_loads - mean_period_load
        )
        if start_stage <= 2:
            objective = np.zeros(variable_count)
            objective[deviation_start:used_start] = 1
            current_result = solve(
                "minimize_absolute_deviation",
                objective,
                [(max_row, optimal_max + 1e-6)],
                warm_start=current_result.x,
            )
        optimal_deviation = float(deviation_row @ current_result.x)
    else:
        optimal_deviation = float(
            np.abs(current_loads - mean_period_load).sum()
        )

    if stop_after_stage >= 3:
        if start_stage <= 3:
            current_result = solve(
                "minimize_extra_axis_segments",
                segment_row,
                [
                    (max_row, optimal_max + 1e-5),
                    (deviation_row, optimal_deviation + 1e-5),
                ],
                warm_start=current_result.x,
            )
        optimal_segment_penalty = float(segment_row @ current_result.x)
    else:
        optimal_segment_penalty = float(segment_row @ current_result.x)

    if stop_after_stage >= 4:
        current_result = SimpleNamespace(
            x=complete_warm_start(
                current_result.x, binary_used=True, binary_switch=False
            )
        )
        if start_stage <= 4:
            objective = np.zeros(variable_count)
            objective[used_start:multi_start] = 1
            current_result = solve(
                "minimize_used_placements",
                objective,
                [
                    (max_row, optimal_max + 1e-5),
                    (deviation_row, optimal_deviation + 1e-5),
                    (segment_row, optimal_segment_penalty + 1e-5),
                ],
                binary_used=True,
                warm_start=current_result.x,
            )
        optimal_used_placements = float(used_row @ current_result.x)
    else:
        current_counts = np.rint(current_result.x[:candidate_count]).astype(int)
        optimal_used_placements = float(np.count_nonzero(current_counts))

    if stop_after_stage >= 5:
        current_result = SimpleNamespace(
            x=complete_warm_start(
                current_result.x, binary_used=True, binary_switch=True
            )
        )
        if start_stage <= 5:
            switch_objective = np.zeros(variable_count)
            switch_objective[switch_start:variable_count] = 1
            current_result = solve(
                "minimize_cyclic_phase_switches",
                switch_objective,
                [
                    (max_row, optimal_max + 1e-5),
                    (deviation_row, optimal_deviation + 1e-5),
                    (segment_row, optimal_segment_penalty + 1e-5),
                    (used_row, optimal_used_placements + 1e-5),
                ],
                binary_switch=True,
                warm_start=current_result.x,
            )

    selected_result = current_result
    counts = np.rint(selected_result.x[:candidate_count]).astype(int)
    for domain in domains:
        domain_total = sum(
            counts[index]
            for index, (candidate_domain, _) in enumerate(keys)
            if candidate_domain == domain
        )
        if domain_total != horizon:
            raise RuntimeError(f"MILP count reconstruction failed for {domain}")

    assignments: dict[tuple[int, str], str] = {}
    for domain in domains:
        expanded: list[str] = []
        for index, (candidate_domain, placement) in enumerate(keys):
            if candidate_domain == domain:
                expanded.extend([placement] * counts[index])
        for phase, placement in enumerate(expanded):
            assignments[(phase, domain)] = placement

    loads = fixed * horizon
    for index, key in enumerate(keys):
        loads += vectors[key] * counts[index]
    achieved_max = int(loads.max())
    if achieved_max > optimal_max + 1e-4:
        raise RuntimeError("MILP load reconstruction violates the stage-1 bound")
    stage1_certificate = stage_certificates.get("minimize_max_load", {})
    stage1_bound = float(stage1_certificate.get("dual_bound", achieved_max))
    certified_gap = (achieved_max - stage1_bound) / achieved_max
    used_per_domain = {
        domain: int(
            np.count_nonzero(
                [
                    counts[index]
                    for index, (candidate_domain, _) in enumerate(keys)
                    if candidate_domain == domain
                ]
            )
        )
        for domain in domains
    }
    minimum_cyclic_switches = sum(
        used_count if used_count > 1 else 0
        for used_count in used_per_domain.values()
    )
    solver_stats: dict[str, object] = {
        "stage1_incumbent_max_period_cycles": optimal_max,
        "achieved_max_period_cycles": achieved_max,
        "certified_max_load_relative_gap": certified_gap,
        "stage2_incumbent_total_absolute_deviation": optimal_deviation,
        "stage3_extra_axis_segment_penalty": float(segment_row @ selected_result.x),
        "used_domain_placements": float(np.count_nonzero(counts)),
        "minimum_cyclic_phase_switches": int(minimum_cyclic_switches),
        "segment_stage_completed": stop_after_stage >= 3,
        "placement_count_stage_completed": stop_after_stage >= 4,
        "phase_switch_stage_completed": stop_after_stage >= 5,
        "placement_and_switch_search_scope": placement_search_scope,
        "completed_through_stage": stop_after_stage,
        "all_stages_proven_optimal": all(
            certificate["proven_optimal"]
            for certificate in stage_certificates.values()
        ) and stop_after_stage == 5,
        "stage_certificates": stage_certificates,
    }
    return assignments, loads, solver_stats


def period_load(
    fixed: np.ndarray,
    domains: list[str],
    vectors: dict[tuple[str, str], np.ndarray],
    assignments: dict[tuple[int, str], str],
    horizon: int,
) -> np.ndarray:
    loads = fixed * horizon
    for phase in range(horizon):
        for domain in domains:
            loads += vectors[(domain, assignments[(phase, domain)])]
    return loads


def layer3_cyclic_assignments(
    domains: list[str],
    placements: dict[str, list[str]],
    horizon: int,
) -> dict[tuple[int, str], str]:
    """Reproduce the existing six-op Layer3 resting-column schedule."""

    assignments = {
        (phase, domain): "stock" for phase in range(horizon) for domain in domains
    }
    for phase in range(horizon):
        schedule_phase = (phase * 6) % GRID_X
        for module_offset in range(6):
            domain = f"layer3.module{module_offset + 1}.conv2_to_conv3"
            if domain not in placements:
                raise ValueError(f"Missing Layer3 cyclic domain: {domain}")
            resting_column = (schedule_phase + module_offset) % GRID_X
            # Excluding the unused column to the right of the stock 4x4 grid
            # is physically identical to the stock candidate.
            placement = (
                "stock"
                if resting_column == GRID_X - 1
                else f"exclude_rnone_c{resting_column}"
            )
            # Restricted candidate experiments intentionally remove some of
            # the placements used by this comparison policy.  Keep that
            # phase at stock instead of aborting after the actual optimized
            # MILP has already completed.
            if placement in placements[domain]:
                assignments[(phase, domain)] = placement
    return assignments


def assignment_stats(
    name: str,
    assignments: dict[tuple[int, str], str],
    domains: list[str],
    horizon: int,
) -> dict[str, object]:
    used = {(domain, assignments[(phase, domain)]) for phase in range(horizon) for domain in domains}
    switches = 0
    for domain in domains:
        for phase in range(horizon):
            next_phase = (phase + 1) % horizon
            switches += assignments[(phase, domain)] != assignments[(next_phase, domain)]
    return {
        "policy": name,
        "unique_domain_placements": len(used),
        "cyclic_phase_switches": switches,
    }


def metrics(name: str, per_inference_load: np.ndarray) -> dict[str, object]:
    mean = float(per_inference_load.mean())
    return {
        "policy": name,
        "min_worker_cycles": float(per_inference_load.min()),
        "max_worker_cycles": float(per_inference_load.max()),
        "mean_worker_cycles": mean,
        "max_over_mean": float(per_inference_load.max() / mean),
        "max_over_min": float(per_inference_load.max() / per_inference_load.min()),
        "coefficient_of_variation": float(per_inference_load.std() / mean),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--solver-time-limit", type=float, default=30.0)
    parser.add_argument("--start-stage", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--stop-after-stage", type=int, choices=range(1, 6), default=5)
    parser.add_argument(
        "--solver-backend", choices=("highspy", "gurobi"), default="highspy"
    )
    parser.add_argument(
        "--solver-output",
        action="store_true",
        help="stream HiGHS progress logs (useful for long foreground runs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.start_stage > args.stop_after_stage:
        raise ValueError("--start-stage must not exceed --stop-after-stage")
    calibration = args.calibration_dir
    candidates = pd.read_csv(calibration / "full_model_domain_candidates.csv")
    worker_costs = pd.read_csv(calibration / "full_model_candidate_worker_costs.csv")
    fixed_rows = pd.read_csv(calibration / "full_model_fixed_background_per_worker.csv")
    all_per_worker = pd.read_csv(calibration / "full_model_trisc_per_op_per_worker.csv")

    fixed = vector_from_rows(fixed_rows, "trisc_engine_cycles")
    domains, placements, vectors, segment_penalties = load_candidate_vectors(
        candidates, worker_costs
    )
    measured_stock = verify_stock_reconstruction(
        fixed, domains, vectors, all_per_worker
    )
    initial_counts = None
    prior_stage_certificates: dict[str, dict[str, object]] = {}
    existing_schedule_path = calibration / f"full_model_schedule_h{args.horizon}.csv"
    if existing_schedule_path.exists():
        existing_schedule = pd.read_csv(existing_schedule_path)
        expected_pairs = {
            (phase, domain)
            for phase in range(args.horizon)
            for domain in domains
        }
        actual_pairs = set(
            zip(existing_schedule["phase"], existing_schedule["domain_id"])
        )
        if len(existing_schedule) == len(expected_pairs) and actual_pairs == expected_pairs:
            initial_counts = {
                (domain, placement): int(count)
                for (domain, placement), count in existing_schedule.groupby(
                    ["domain_id", "placement_id"]
                ).size().items()
            }
            unknown_candidates = set(initial_counts) - set(vectors)
            if unknown_candidates:
                raise ValueError(
                    f"Existing schedule contains unknown candidates: {unknown_candidates}"
                )
            existing_manifest_path = (
                calibration / f"full_model_schedule_h{args.horizon}.json"
            )
            if existing_manifest_path.exists() and args.start_stage > 1:
                with existing_manifest_path.open("r", encoding="utf-8") as stream:
                    existing_manifest = json.load(stream)
                prior_stage_certificates = existing_manifest.get(
                    "solver_objectives", {}
                ).get("stage_certificates", {})
                stage_names = [
                    "minimize_max_load",
                    "minimize_absolute_deviation",
                    "minimize_extra_axis_segments",
                    "minimize_used_placements",
                    "minimize_cyclic_phase_switches",
                ]
                prior_stage_certificates = {
                    name: certificate
                    for name, certificate in prior_stage_certificates.items()
                    if name in stage_names[: args.start_stage - 1]
                }
    assignments, optimized_period_load, solver_stats = optimize_exact(
        fixed,
        domains,
        placements,
        vectors,
        segment_penalties,
        args.horizon,
        args.solver_time_limit,
        initial_counts,
        args.solver_output,
        args.start_stage,
        args.stop_after_stage,
        prior_stage_certificates,
        args.solver_backend,
    )

    stock_assignments = {
        (phase, domain): "stock"
        for phase in range(args.horizon)
        for domain in domains
    }
    cyclic_assignments = layer3_cyclic_assignments(
        domains, placements, args.horizon
    )

    stock_period_load = measured_stock * args.horizon
    cyclic_period_load = period_load(
        fixed, domains, vectors, cyclic_assignments, args.horizon
    )
    stock_average = stock_period_load / args.horizon
    cyclic_average = cyclic_period_load / args.horizon
    optimized_average = optimized_period_load / args.horizon
    summary = pd.DataFrame(
        [
            metrics("stock", stock_average),
            metrics("current_layer3_cyclic", cyclic_average),
            metrics("predicted_optimized", optimized_average),
        ]
    )
    summary.to_csv(calibration / "full_model_balance_summary.csv", index=False)

    per_worker = pd.DataFrame(
        [
            {
                "worker_x": worker[0],
                "worker_y": worker[1],
                "stock_cycles_per_inference": stock_average[index],
                "layer3_cyclic_cycles_per_inference": cyclic_average[index],
                "optimized_cycles_per_inference": optimized_average[index],
                "delta_cycles_per_inference": optimized_average[index] - stock_average[index],
            }
            for index, worker in enumerate(WORKERS)
        ]
    )
    per_worker.to_csv(calibration / "full_model_stock_vs_optimized_per_worker.csv", index=False)

    schedule_stats = pd.DataFrame(
        [
            assignment_stats("stock", stock_assignments, domains, args.horizon),
            assignment_stats(
                "current_layer3_cyclic", cyclic_assignments, domains, args.horizon
            ),
            assignment_stats(
                "predicted_optimized", assignments, domains, args.horizon
            ),
        ]
    )
    schedule_stats.to_csv(
        calibration / "full_model_schedule_complexity.csv", index=False
    )

    candidate_lookup = candidates.set_index(["domain_id", "placement_id"])
    schedule_rows = []
    manifest_phases = []
    for phase in range(args.horizon):
        phase_domains = {}
        for domain in domains:
            placement = assignments[(phase, domain)]
            candidate = candidate_lookup.loc[(domain, placement)]
            excluded_rows = json.loads(candidate["excluded_rows"])
            excluded_columns = json.loads(candidate["excluded_columns"])
            schedule_rows.append(
                {
                    "phase": phase,
                    "domain_id": domain,
                    "placement_id": placement,
                    "excluded_rows": json.dumps(excluded_rows),
                    "excluded_columns": json.dumps(excluded_columns),
                }
            )
            phase_domains[domain] = {
                "placement_id": placement,
                "excluded_rows": excluded_rows,
                "excluded_columns": excluded_columns,
            }
        manifest_phases.append({"phase": phase, "domains": phase_domains})
    pd.DataFrame(schedule_rows).to_csv(
        calibration / f"full_model_schedule_h{args.horizon}.csv", index=False
    )

    manifest = {
        "model": "BOS-ResNet50-b1",
        "architecture": "Eagle-N-5x4-workers",
        "horizon": args.horizon,
        "status": "predicted_not_deployable_until_candidate_validation",
        "metric": "equal_weight_trisc_kernel_engine_cycles",
        "cost_model": "stock_logical_core_permutation_unvalidated",
        "solver": f"{args.solver_backend}_warm_started_bounded_lexicographic_milp",
        "solver_proven_optimal": solver_stats["all_stages_proven_optimal"],
        "solver_objectives": solver_stats,
        "phases": manifest_phases,
    }
    with (calibration / f"full_model_schedule_h{args.horizon}.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    stock_metrics = summary.iloc[0]
    cyclic_metrics = summary.iloc[1]
    optimized_metrics = summary.iloc[2]
    print("Stock reconstruction: exact")
    print(f"Domains optimized: {len(domains)} over H={args.horizon}")
    max_certificate = solver_stats["stage_certificates"]["minimize_max_load"]
    print(
        "MILP max-load certificate: "
        f"optimal={max_certificate['proven_optimal']} "
        f"gap={max_certificate['relative_gap']:.6%}"
    )
    print(
        "max/mean: "
        f"{stock_metrics['max_over_mean']:.6f} -> "
        f"{cyclic_metrics['max_over_mean']:.6f} -> "
        f"{optimized_metrics['max_over_mean']:.6f}"
    )
    print(
        "max worker cycles/inference: "
        f"{stock_metrics['max_worker_cycles']:.1f} -> "
        f"{optimized_metrics['max_worker_cycles']:.1f}"
    )
    print(
        "complexity objectives: "
        f"segments={solver_stats['stage3_extra_axis_segment_penalty']:.0f} "
        f"placements={solver_stats['used_domain_placements']:.0f} "
        f"cyclic_switches={solver_stats['minimum_cyclic_phase_switches']} "
        f"scope={solver_stats['placement_and_switch_search_scope']}"
    )


if __name__ == "__main__":
    main()
