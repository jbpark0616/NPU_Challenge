#!/usr/bin/env python3
"""Aggregate H-phase full-model profiles into measured MILP worker exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extract_full_model_trisc_exposure import (
    aggregate_per_worker,
    annotate_resnet50_compute_sequence,
    annotate_topology,
    join_operation_metadata,
    load_device_events,
    load_ops,
    pair_trisc_kernel_intervals,
)


POLICY_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    POLICY_DIR
    / "data"
    / "experiments"
    / "single_axis_split"
    / "full_model_schedule_h20.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--prediction-csv",
        type=Path,
        help="Defaults to full_model_stock_vs_optimized_per_worker.csv beside the manifest.",
    )
    parser.add_argument(
        "--stock-report-dir",
        type=Path,
        help="Optional stock full-model report used for latency comparison.",
    )
    return parser.parse_args()


def one_ops_csv(directory: Path) -> Path:
    candidates = sorted(directory.glob("ops_perf_results_*.csv"))
    if len(candidates) != 1:
        raise ValueError(
            f"{directory} must contain exactly one ops_perf_results_*.csv; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def latency_metrics(ops: pd.DataFrame) -> dict[str, float]:
    host_start = pd.to_numeric(ops["HOST START TS"], errors="coerce")
    host_end = pd.to_numeric(ops["HOST END TS"], errors="coerce")
    op_to_op = pd.to_numeric(ops["OP TO OP LATENCY [ns]"], errors="coerce")
    kernel = pd.to_numeric(ops["DEVICE KERNEL DURATION [ns]"], errors="coerce")
    return {
        "host_profile_span_ns": float(host_end.max() - host_start.min()),
        "sum_op_to_op_latency_ns": float(op_to_op.sum()),
        "sum_device_kernel_duration_ns": float(kernel.sum()),
    }


def load_phase_profile(phase_dir: Path, phase: int) -> tuple[pd.DataFrame, dict[str, float]]:
    device_csv = phase_dir / "profile_log_device.csv"
    ops_csv = one_ops_csv(phase_dir)
    events, _ = load_device_events(device_csv)
    ops = annotate_topology(annotate_resnet50_compute_sequence(load_ops(ops_csv)))
    inference_count = int(ops["model_inference_index"].dropna().nunique())
    if inference_count != 1:
        raise ValueError(
            f"Expected one uninterrupted inference in {phase_dir}, found {inference_count}"
        )
    intervals = pair_trisc_kernel_intervals(events)
    per_worker = join_operation_metadata(aggregate_per_worker(intervals), ops)
    compute = per_worker[per_worker["is_eagle_n_compute_worker"]].copy()
    totals = (
        compute.groupby(["worker_x", "worker_y"], as_index=False)[
            [
                "trisc_0_cycles",
                "trisc_1_cycles",
                "trisc_2_cycles",
                "trisc_engine_cycles",
            ]
        ]
        .sum()
        .sort_values(["worker_y", "worker_x"])
    )
    if len(totals) != 20:
        raise ValueError(f"Expected 20 Eagle-N workers in {phase_dir}, found {len(totals)}")
    totals.insert(0, "phase", phase)
    return totals, latency_metrics(ops)


def coefficient_of_variation(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0) / np.mean(values))


def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    horizon = int(manifest["horizon"])
    prediction_csv = args.prediction_csv or args.manifest.with_name(
        "full_model_stock_vs_optimized_per_worker.csv"
    )

    phase_frames = []
    latency_rows = []
    for phase in range(horizon):
        phase_dir = args.period_dir / f"phase_{phase:02d}"
        if not phase_dir.is_dir():
            raise ValueError(f"Missing phase profile directory: {phase_dir}")
        totals, latency = load_phase_profile(phase_dir, phase)
        phase_frames.append(totals)
        latency_rows.append({"phase": phase, **latency})

    per_phase = pd.concat(phase_frames, ignore_index=True)
    measured = (
        per_phase.groupby(["worker_x", "worker_y"], as_index=False)[
            [
                "trisc_0_cycles",
                "trisc_1_cycles",
                "trisc_2_cycles",
                "trisc_engine_cycles",
            ]
        ]
        .sum()
        .rename(
            columns={
                "trisc_0_cycles": "measured_period_trisc_0_cycles",
                "trisc_1_cycles": "measured_period_trisc_1_cycles",
                "trisc_2_cycles": "measured_period_trisc_2_cycles",
                "trisc_engine_cycles": "measured_period_trisc_engine_cycles",
            }
        )
    )
    measured["measured_cycles_per_inference"] = (
        measured["measured_period_trisc_engine_cycles"] / horizon
    )
    measured["measured_share_percent"] = 100.0 * measured[
        "measured_period_trisc_engine_cycles"
    ] / measured["measured_period_trisc_engine_cycles"].sum()

    prediction = pd.read_csv(prediction_csv)
    comparison = measured.merge(
        prediction,
        on=["worker_x", "worker_y"],
        how="left",
        validate="one_to_one",
    )
    comparison["prediction_error_cycles_per_inference"] = (
        comparison["measured_cycles_per_inference"]
        - comparison["optimized_cycles_per_inference"]
    )

    measured_values = comparison["measured_cycles_per_inference"].to_numpy(float)
    stock_values = comparison["stock_cycles_per_inference"].to_numpy(float)
    predicted_values = comparison["optimized_cycles_per_inference"].to_numpy(float)
    measured_max = float(measured_values.max())
    stock_max = float(stock_values.max())
    predicted_max = float(predicted_values.max())
    summary_rows = [
        {"metric": "horizon", "value": horizon},
        {"metric": "stock_max_cycles_per_inference", "value": stock_max},
        {"metric": "predicted_max_cycles_per_inference", "value": predicted_max},
        {"metric": "measured_max_cycles_per_inference", "value": measured_max},
        {
            "metric": "measured_peak_reduction_vs_stock_percent",
            "value": 100.0 * (stock_max - measured_max) / stock_max,
        },
        {
            "metric": "predicted_peak_reduction_vs_stock_percent",
            "value": 100.0 * (stock_max - predicted_max) / stock_max,
        },
        {"metric": "stock_worker_cv", "value": coefficient_of_variation(stock_values)},
        {
            "metric": "predicted_worker_cv",
            "value": coefficient_of_variation(predicted_values),
        },
        {
            "metric": "measured_worker_cv",
            "value": coefficient_of_variation(measured_values),
        },
        {
            "metric": "prediction_rmse_cycles_per_inference",
            "value": float(np.sqrt(np.mean((measured_values - predicted_values) ** 2))),
        },
    ]

    latency = pd.DataFrame(latency_rows)
    if args.stock_report_dir is not None:
        stock_ops = load_ops(one_ops_csv(args.stock_report_dir))
        stock_latency = latency_metrics(stock_ops)
        for metric, stock_value in stock_latency.items():
            measured_value = float(latency[metric].mean())
            summary_rows.extend(
                [
                    {"metric": f"stock_{metric}", "value": stock_value},
                    {"metric": f"measured_mean_{metric}", "value": measured_value},
                    {
                        "metric": f"measured_{metric}_overhead_percent",
                        "value": 100.0 * (measured_value - stock_value) / stock_value,
                    },
                ]
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_phase.to_csv(args.output_dir / "milp_period_per_phase_per_worker.csv", index=False)
    comparison.to_csv(args.output_dir / "milp_period_per_worker_comparison.csv", index=False)
    latency.to_csv(args.output_dir / "milp_period_per_phase_latency.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "milp_period_measurement_summary.csv", index=False
    )

    print(f"Measured H={horizon} peak cycles/inference: {measured_max:.1f}")
    print(
        "Measured peak reduction vs stock: "
        f"{100.0 * (stock_max - measured_max) / stock_max:.4f}%"
    )
    print(f"Measured worker CV: {coefficient_of_variation(measured_values):.6f}")
    print(f"Wrote measurement tables to: {args.output_dir}")


if __name__ == "__main__":
    main()
