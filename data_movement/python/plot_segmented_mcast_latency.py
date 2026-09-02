"""Create a publication-style latency figure from segmented multicast profiler data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "blackhole" / "profile_log_device.csv"
OUT = ROOT / "data" / "blackhole" / "segmented_mcast_latency_paper.png"
SUMMARY_OUT = ROOT / "data" / "blackhole" / "segmented_mcast_latency_summary.csv"

CONFIGS = [
    ("full_5x4", 1),
    ("skip_row_0", 1), ("skip_row_1", 2), ("skip_row_2", 2), ("skip_row_3", 1),
    ("skip_col_0", 1), ("skip_col_1", 2), ("skip_col_2", 2),
    ("skip_col_3", 2), ("skip_col_4", 1),
]


def zone_durations(frame: pd.DataFrame, source_suffix: str) -> pd.DataFrame:
    zone = frame[
        frame["source file"].str.endswith(source_suffix, na=False)
        & frame["type"].isin(["ZONE_START", "ZONE_END"])
    ].copy()
    keys = ["run host ID", "core_x", "core_y", "RISC processor type", "zone name"]
    pivot = zone.pivot_table(index=keys, columns="type", values="time[cycles since reset]", aggfunc="first")
    pivot["cycles"] = pivot["ZONE_END"] - pivot["ZONE_START"]
    return pivot.reset_index()[["run host ID", "core_x", "core_y", "cycles"]]


def main() -> None:
    raw = pd.read_csv(LOG, skiprows=1, skipinitialspace=True)
    raw["run host ID"] = raw["run host ID"].astype(np.int64)
    raw["time[cycles since reset]"] = raw["time[cycles since reset]"].astype(np.int64)
    sender = zone_durations(raw, "segmented_multicast_sender.cpp")
    receiver = zone_durations(raw, "segmented_multicast_receiver.cpp")

    sender_runs = np.sort(sender["run host ID"].unique())
    if len(sender_runs) != len(CONFIGS) * 16:
        raise RuntimeError(f"Expected 160 sender runs, found {len(sender_runs)}. Is this the fresh segmented profiler log?")
    run_to_config = {
        run: CONFIGS[index // 16][0]
        for index, run in enumerate(sender_runs)
    }
    sender["config"] = sender["run host ID"].map(run_to_config)
    receiver["config"] = receiver["run host ID"].map(run_to_config)
    if receiver["config"].isna().any():
        raise RuntimeError("Receiver run IDs do not match sender run IDs.")

    sender_stats = sender.groupby("config")["cycles"].agg(["mean", "min", "max"]).reindex([name for name, _ in CONFIGS])
    receiver_stats = receiver.groupby("config")["cycles"].agg(["mean", "min", "max", "count"]).reindex([name for name, _ in CONFIGS])
    summary = sender_stats.add_prefix("sender_").join(receiver_stats.add_prefix("receiver_"))
    summary["segments"] = [count for _, count in CONFIGS]
    summary.to_csv(SUMMARY_OUT, index_label="config")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure, (sender_ax, receiver_ax) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    figure.set_facecolor("white")
    figure.suptitle("Segmented multicast latency on Blackhole 5×4", y=0.985, fontsize=16, fontweight="bold")
    figure.text(0.5, 0.946, "NoC0 · 64 B payload · non-loopback · 16 repetitions per configuration", ha="center", color="#4b5563")

    names = [name for name, _ in CONFIGS]
    segments = np.array([count for _, count in CONFIGS])
    x = np.arange(len(names))
    colors = np.where(segments == 2, "#d97706", "#1769aa")
    markers = np.where(segments == 2, "s", "o")

    for index, name in enumerate(names):
        stats = sender_stats.loc[name]
        sender_ax.errorbar(
            x[index], stats["mean"],
            yerr=[[stats["mean"] - stats["min"]], [stats["max"] - stats["mean"]]],
            fmt=markers[index], color=colors[index], capsize=4, markersize=7, linewidth=1.8,
        )
        sender_ax.annotate(f"{stats['mean']:.0f}", (x[index], stats["mean"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    sender_ax.set_title("Sender total: all data segments + one barrier + semaphore segments", loc="left", fontweight="bold")
    sender_ax.set_ylabel("Cycles")
    sender_ax.grid(axis="y", color="#d1d5db", linewidth=0.8)

    receiver_samples = [receiver.loc[receiver["config"] == name, "cycles"].to_numpy() for name in names]
    boxes = receiver_ax.boxplot(receiver_samples, positions=x, widths=0.52, patch_artist=True, showfliers=False, medianprops={"color": "#111827", "linewidth": 1.2})
    for index, box in enumerate(boxes["boxes"]):
        box.set_facecolor(colors[index])
        box.set_alpha(0.28)
        box.set_edgecolor(colors[index])
        box.set_linewidth(1.4)
    for index, name in enumerate(names):
        mean = receiver_stats.loc[name, "mean"]
        receiver_ax.scatter(x[index], mean, marker=markers[index], color=colors[index], s=42, zorder=3)
    receiver_ax.set_title("Active receiver semaphore wait distribution", loc="left", fontweight="bold")
    receiver_ax.set_ylabel("Cycles")
    receiver_ax.set_xticks(x, names, rotation=28, ha="right")
    receiver_ax.grid(axis="y", color="#d1d5db", linewidth=0.8)

    legend_handles = [
        plt.Line2D([0], [0], color="#1769aa", marker="o", linestyle="", label="1 segment (baseline / edge rest)"),
        plt.Line2D([0], [0], color="#d97706", marker="s", linestyle="", label="2 segments (middle rest)"),
    ]
    sender_ax.legend(handles=legend_handles, loc="upper left", frameon=False, ncol=2)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"Wrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
