#!/usr/bin/env python3
"""Compare validated stock/remapped device logs without an ops report."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re


TRISC_TYPES = {"TRISC_0", "TRISC_1", "TRISC_2"}
KERNEL_ZONE = "TRISC-KERNEL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-root",
        type=Path,
        required=True,
        help="Directory containing stock/ and remapped/ profile_log_device.csv files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_kernel_events(path: Path) -> tuple[float, list[dict[str, str]]]:
    if not path.is_file():
        raise ValueError(f"Missing device profile: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        metadata = stream.readline().strip()
        frequency = re.search(r"CHIP_FREQ\[MHz\]:\s*([0-9.]+)", metadata)
        if not frequency:
            raise ValueError(f"Missing CHIP_FREQ[MHz] metadata in {path}")
        reader = csv.DictReader(stream, skipinitialspace=True)
        events = [
            row
            for row in reader
            if row["zone name"].strip() == KERNEL_ZONE
            and row["RISC processor type"].strip() in TRISC_TYPES
            and row["type"].strip() in {"ZONE_START", "ZONE_END"}
        ]
    if not events:
        raise ValueError(f"No {KERNEL_ZONE} events in {path}")
    return float(frequency.group(1)), events


def summarize(path: Path, mode: str) -> tuple[dict[str, float | int | str], dict[int, int]]:
    frequency_mhz, events = load_kernel_events(path)
    intervals: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)
    run_events: dict[int, list[tuple[int, str]]] = defaultdict(list)

    for event in events:
        run_host_id = int(event["run host ID"])
        timestamp = int(event["time[cycles since reset]"])
        event_type = event["type"].strip()
        key = (
            event["core_x"],
            event["core_y"],
            event["RISC processor type"].strip(),
            event["timer_id"],
            event["run host ID"],
        )
        intervals[key].append((timestamp, event_type))
        run_events[run_host_id].append((timestamp, event_type))

    trisc_engine_cycles = 0
    interval_count = 0
    for key, group in intervals.items():
        ordered = sorted(group)
        if len(ordered) % 2:
            raise ValueError(f"Unmatched kernel event group in {path}: {key}")
        for index in range(0, len(ordered), 2):
            start, start_type = ordered[index]
            end, end_type = ordered[index + 1]
            if start_type != "ZONE_START" or end_type != "ZONE_END" or end <= start:
                raise ValueError(f"Invalid kernel interval in {path}: {key}")
            trisc_engine_cycles += end - start
            interval_count += 1

    critical_spans: dict[int, int] = {}
    for run_host_id, group in run_events.items():
        starts = [timestamp for timestamp, kind in group if kind == "ZONE_START"]
        ends = [timestamp for timestamp, kind in group if kind == "ZONE_END"]
        if not starts or len(starts) != len(ends):
            raise ValueError(f"Unmatched run-host events in {path}: {run_host_id}")
        critical_spans[run_host_id] = max(ends) - min(starts)

    critical_span_cycles = sum(critical_spans.values())
    critical_span_ns = critical_span_cycles * 1000.0 / frequency_mhz
    return (
        {
            "mode": mode,
            "chip_frequency_mhz": frequency_mhz,
            "run_host_count": len(critical_spans),
            "trisc_interval_count": interval_count,
            "trisc_engine_cycles": trisc_engine_cycles,
            "critical_span_cycles": critical_span_cycles,
            "critical_span_ns": critical_span_ns,
        },
        critical_spans,
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    stock, stock_spans = summarize(
        args.profile_root / "stock" / "profile_log_device.csv", "stock"
    )
    remapped, remapped_spans = summarize(
        args.profile_root / "remapped" / "profile_log_device.csv", "remapped"
    )
    if stock["chip_frequency_mhz"] != remapped["chip_frequency_mhz"]:
        raise ValueError("Stock and remapped profiles use different device frequencies")
    if stock["run_host_count"] != remapped["run_host_count"]:
        raise ValueError("Stock and remapped profiles contain different operation counts")
    if stock["trisc_interval_count"] != remapped["trisc_interval_count"]:
        raise ValueError("Stock and remapped profiles contain different TRISC interval counts")

    raw_rows = [stock, remapped]
    comparison_rows = []
    for metric in ("trisc_engine_cycles", "critical_span_cycles", "critical_span_ns"):
        stock_value = float(stock[metric])
        remapped_value = float(remapped[metric])
        comparison_rows.append(
            {
                "metric": metric,
                "stock": stock_value,
                "remapped": remapped_value,
                "delta": remapped_value - stock_value,
                "delta_percent": 100.0 * (remapped_value - stock_value) / stock_value,
            }
        )

    per_run_rows = []
    for run_host_id in sorted(set(stock_spans) | set(remapped_spans)):
        stock_cycles = stock_spans.get(run_host_id)
        remapped_cycles = remapped_spans.get(run_host_id)
        per_run_rows.append(
            {
                "run_host_id": run_host_id,
                "stock_critical_span_cycles": stock_cycles,
                "remapped_critical_span_cycles": remapped_cycles,
                "delta_cycles": (
                    remapped_cycles - stock_cycles
                    if stock_cycles is not None and remapped_cycles is not None
                    else ""
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "validated_latency_raw_summary.csv",
        raw_rows,
        list(stock),
    )
    write_csv(
        args.output_dir / "validated_latency_comparison.csv",
        comparison_rows,
        ["metric", "stock", "remapped", "delta", "delta_percent"],
    )
    write_csv(
        args.output_dir / "validated_latency_by_run_host_id.csv",
        per_run_rows,
        [
            "run_host_id",
            "stock_critical_span_cycles",
            "remapped_critical_span_cycles",
            "delta_cycles",
        ],
    )

    comparison = {row["metric"]: row for row in comparison_rows}
    print(
        "TRISC critical-span delta: "
        f"{comparison['critical_span_cycles']['delta_percent']:+.4f}%"
    )
    print(
        "TRISC engine-cycle delta: "
        f"{comparison['trisc_engine_cycles']['delta_percent']:+.4f}%"
    )
    print(f"Wrote comparison tables to: {args.output_dir}")


if __name__ == "__main__":
    main()
