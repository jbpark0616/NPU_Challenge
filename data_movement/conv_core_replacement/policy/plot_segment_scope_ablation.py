#!/usr/bin/env python3
"""Plot the placement-segmentation ablation used to select the policy scope."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


POLICY_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = POLICY_DIR / "data"
DEFAULT_OUTPUT_DIR = POLICY_DIR / "figures"
TIMES_REGULAR = Path("C:/Windows/Fonts/times.ttf")
TIMES_BOLD = Path("C:/Windows/Fonts/timesbd.ttf")


def optimized_peak(path: Path) -> tuple[float, float, float]:
    summary = pd.read_csv(path)
    stock_row = summary.loc[summary["policy"] == "stock"].iloc[0]
    stock = stock_row["max_worker_cycles"]
    mean = stock_row["mean_worker_cycles"]
    optimized = summary.loc[
        summary["policy"] == "predicted_optimized", "max_worker_cycles"
    ].item()
    return float(stock), float(optimized), float(mean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_files = [
        args.data_dir
        / "experiments"
        / "contiguous_only"
        / "full_model_balance_summary.csv",
        args.data_dir
        / "experiments"
        / "single_axis_split"
        / "full_model_balance_summary.csv",
        args.data_dir
        / "full_model_baseline"
        / "output"
        / "full_model_balance_summary.csv",
    ]
    labels = [
        "1 segment",
        "Up to 2 segments",
        "Up to 4 segments",
    ]

    peaks = []
    stock_peaks = []
    stock_means = []
    for result_file in result_files:
        stock, peak, mean = optimized_peak(result_file)
        stock_peaks.append(stock)
        stock_means.append(mean)
        peaks.append(peak)
    if max(stock_peaks) != min(stock_peaks):
        raise ValueError(f"Stock peaks differ across experiments: {stock_peaks}")
    if max(stock_means) != min(stock_means):
        raise ValueError(f"Stock means differ across experiments: {stock_means}")

    stock_peak = stock_peaks[0]
    stock_mean = stock_means[0]
    reductions = [100.0 * (stock_peak - peak) / stock_peak for peak in peaks]
    ideal_reduction = 100.0 * (stock_peak - stock_mean) / stock_peak
    for font_path in (TIMES_REGULAR, TIMES_BOLD):
        if font_path.exists():
            font_manager.fontManager.addfont(font_path)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axis = plt.subplots(figsize=(5.4, 2.65))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.94, bottom=0.30)
    x_values = [0, 1, 2]
    axis.plot(
        x_values,
        reductions,
        color="#4472C4",
        linewidth=1.6,
        marker="o",
        markersize=4.5,
        markerfacecolor="#4472C4",
        markeredgecolor="#4472C4",
    )

    axis.set_ylabel("Peak load reduction")
    axis.set_xlabel("Allowed placement segmentation")
    axis.set_xticks(x_values, labels)
    axis.set_xlim(-0.15, 2.15)
    axis.set_ylim(0, 35)
    y_ticks = list(range(0, 36, 5))
    axis.set_yticks(y_ticks, [f"{value}%" for value in y_ticks])
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.axhline(
        ideal_reduction,
        color="#ED7D31",
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )
    axis.text(
        2.13,
        ideal_reduction + 0.55,
        f"Ideal balance limit ({ideal_reduction:.1f}%)",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#C65911",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "segment_scope_ablation.png"
    svg_path = args.output_dir / "segment_scope_ablation.svg"
    pdf_path = args.output_dir / "segment_scope_ablation.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Stock peak: {stock_peak:.1f} cycles/inference")
    print(f"Ideal balance limit: {ideal_reduction:.6f}%")
    for label, peak, reduction in zip(labels, peaks, reductions):
        print(f"{label.replace(chr(10), ' ')}: peak={peak:.1f}, reduction={reduction:.6f}%")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
