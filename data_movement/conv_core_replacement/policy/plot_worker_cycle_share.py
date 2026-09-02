#!/usr/bin/env python3
"""Compare each worker's share of full-model TRISC cycles before and after remapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


POLICY_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    POLICY_DIR
    / "data"
    / "full_model_baseline"
    / "output"
    / "full_model_stock_vs_optimized_per_worker.csv"
)
DEFAULT_OUTPUT_DIR = POLICY_DIR / "figures"
TIMES_REGULAR = Path("C:/Windows/Fonts/times.ttf")
TIMES_BOLD = Path("C:/Windows/Fonts/timesbd.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input).sort_values(["worker_y", "worker_x"])
    if len(data) != 20:
        raise ValueError(f"Expected 20 workers, found {len(data)}")

    stock_cycles = data["stock_cycles_per_inference"].astype(float)
    optimized_cycles = data["optimized_cycles_per_inference"].astype(float)
    stock_total = float(stock_cycles.sum())
    optimized_total = float(optimized_cycles.sum())
    if not np.isclose(stock_total, optimized_total, rtol=0.0, atol=1e-3):
        raise ValueError(
            "Total compute work changed: "
            f"stock={stock_total}, optimized={optimized_total}"
        )

    stock_share = 100.0 * stock_cycles / stock_total
    optimized_share = 100.0 * optimized_cycles / optimized_total
    labels = [f"{x},{y}" for x, y in zip(data["worker_x"], data["worker_y"])]
    ideal_share = 100.0 / len(data)

    for font_path in (TIMES_REGULAR, TIMES_BOLD):
        if font_path.exists():
            font_manager.fontManager.addfont(font_path)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axis = plt.subplots(figsize=(7.2, 2.8))
    fig.subplots_adjust(left=0.095, right=0.99, top=0.88, bottom=0.25)
    x_values = np.arange(len(data))
    bar_width = 0.38

    stock_bars = axis.bar(
        x_values - bar_width / 2,
        stock_share,
        width=bar_width,
        color="#D55E00",
        label="Stock",
    )
    optimized_bars = axis.bar(
        x_values + bar_width / 2,
        optimized_share,
        width=bar_width,
        color="#0072B2",
        label="Optimized",
    )

    # Keep the full 20-worker redistribution visible, but annotate only the
    # quantity optimized by the primary min-max objective: the worst worker.
    stock_peak_index = int(np.argmax(stock_share.to_numpy()))
    optimized_peak_index = int(np.argmax(optimized_share.to_numpy()))
    stock_peak = float(stock_share.iloc[stock_peak_index])
    optimized_peak = float(optimized_share.iloc[optimized_peak_index])
    peak_reduction = 100.0 * (stock_peak - optimized_peak) / stock_peak

    stock_bars[stock_peak_index].set_edgecolor("#7A3500")
    stock_bars[stock_peak_index].set_linewidth(1.0)
    optimized_bars[optimized_peak_index].set_edgecolor("#004C78")
    optimized_bars[optimized_peak_index].set_linewidth(1.0)

    axis.annotate(
        f"Stock peak  {stock_peak:.2f}%",
        xy=(x_values[stock_peak_index] - bar_width / 2, stock_peak),
        xytext=(x_values[stock_peak_index] - 1.25, 7.65),
        ha="center",
        va="bottom",
        color="#A34700",
        fontsize=8,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": "#D55E00", "linewidth": 0.9},
    )
    axis.annotate(
        f"Predicted peak  {optimized_peak:.2f}%  (−{peak_reduction:.2f}%)",
        xy=(x_values[optimized_peak_index] + bar_width / 2, optimized_peak),
        xytext=(x_values[optimized_peak_index] + 0.9, 6.0),
        ha="center",
        va="bottom",
        color="#005F96",
        fontsize=8,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": "#0072B2", "linewidth": 0.9},
    )
    equal_line = axis.axhline(
        ideal_share,
        color="#333333",
        linewidth=1.2,
        linestyle="--",
        label="Equal share (5%)",
        zorder=3,
    )

    for boundary in (4.5, 9.5, 14.5):
        axis.axvline(boundary, color="#D9D9D9", linewidth=0.7, zorder=0)

    axis.set_ylabel("Share of total TRISC-KERNEL cycles")
    axis.set_xlabel("Worker coordinate (x,y)")
    axis.set_xticks(x_values, labels)
    axis.set_xlim(-0.65, len(data) - 0.35)
    axis.set_ylim(0.0, 8.0)
    axis.set_yticks(np.arange(0.0, 8.1, 1.0), [f"{value:.0f}%" for value in np.arange(0.0, 8.1, 1.0)])
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        [stock_bars, optimized_bars, equal_line],
        ["Stock", "Predicted optimized", "Equal share (5%)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.8,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "worker_cycle_share_comparison.png"
    svg_path = args.output_dir / "worker_cycle_share_comparison.svg"
    pdf_path = args.output_dir / "worker_cycle_share_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Stock total: {stock_total:.1f} cycles/inference")
    print(f"Optimized total: {optimized_total:.1f} cycles/inference")
    print(f"Stock share range: {stock_share.min():.6f}% to {stock_share.max():.6f}%")
    print(
        "Optimized share range: "
        f"{optimized_share.min():.6f}% to {optimized_share.max():.6f}%"
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
