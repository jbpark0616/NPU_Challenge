"""
BOS ResNet50 op fusion ratio 시각화 (baseline + exp3 두 CSV 모두)

CSV 소스:
  - ops_perf_results_2026_07_08_05_15_43.csv                 (BOS baseline, 10 iter)
  - ops_perf_results_resnet50_exp3_2026_07_20_14_20_32.csv   (exp3, 1 iter)

출력 (prefix 로 구분):
  {prefix}_fusion_donut.png       — compute op 중 fused 비율 (도넛)
  {prefix}_fusion_breakdown.png   — 전체 op inventory
  {prefix}_fusion_timeline.png    — 시간 축 3-panel

스타일: time_window 폴더의 fig_*.png 톤 참조.
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_SOURCES = [
    {
        "path": os.path.join(SCRIPT_DIR, "ops_perf_results_2026_07_08_05_15_43.csv"),
        "prefix": "baseline",
        "label": "BOS baseline (07_08, 10 iter)",
    },
    {
        "path": os.path.join(SCRIPT_DIR, "ops_perf_results_resnet50_exp3_2026_07_20_14_20_32.csv"),
        "prefix": "exp3",
        "label": "exp3 (07_20, 1 iter)",
    },
]

COMPUTE_OPS = {"Conv2d", "Matmul", "BinaryDeviceOperation", "Pool2D", "Reduce"}

COLOR_FUSED = "tab:red"
COLOR_UNFUSED = "tab:blue"
COLOR_DM = "tab:orange"
COLOR_OTHER = "lightgray"


def categorize(op_code, attrs):
    has_relu = "RELU" in (attrs or "").upper()
    if op_code in COMPUTE_OPS:
        if op_code in ("Pool2D", "Reduce"):
            return "other"
        return "fused" if has_relu else "unfused"
    return "data_movement"


def parse_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            op = r["OP CODE"]
            attrs = r["ATTRIBUTES"]
            cat = categorize(op, attrs)
            try:
                dur = float(r["DEVICE KERNEL DURATION [ns]"] or 0)
            except ValueError:
                dur = 0.0
            try:
                gcc = int(r["GLOBAL CALL COUNT"] or 0)
            except ValueError:
                gcc = 0
            rows.append({"op": op, "cat": cat, "dur_ns": dur, "gcc": gcc})
    return rows


# ─────────────────────────────────────────────────────────
# Figure 1: Donut (compute only)
# ─────────────────────────────────────────────────────────
def draw_donut(rows, out_path, source_label):
    counts = defaultdict(int)
    for r in rows:
        if r["cat"] in ("fused", "unfused"):
            counts[r["cat"]] += 1
        elif r["op"] in ("Pool2D", "Reduce"):
            counts["other"] += 1

    values = [counts["fused"], counts["unfused"], counts["other"]]
    total = sum(values)
    labels = [
        f"fused with ReLU  ({counts['fused']})",
        f"unfused linear  ({counts['unfused']})",
        f"other  ({counts['other']})",
    ]
    colors = [COLOR_FUSED, COLOR_UNFUSED, COLOR_OTHER]

    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, _ = ax.pie(
        values, colors=colors, startangle=90,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.5),
        counterclock=False,
    )
    ax.set(aspect="equal")

    pct = counts["fused"] / total * 100
    ax.text(0, 0.08, f"{pct:.0f}%", ha="center", va="center",
            fontsize=44, color=COLOR_FUSED)
    ax.text(0, -0.15, "fused with ReLU", ha="center", va="center",
            fontsize=12, color="dimgray")

    ax.set_title(
        "BOS ResNet50 — compute op fusion breakdown",
        fontsize=12, loc="left", pad=10,
    )
    ax.legend(
        wedges, labels,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=10,
    )
    fig.text(
        0.5, 0.02,
        f"source: {source_label} ; compute ops only ({total} of {len(rows)} total, "
        f"data movement excluded)",
        ha="center", fontsize=8, style="italic", color="gray",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────
# Figure 2: Breakdown (전체 op inventory)
# ─────────────────────────────────────────────────────────
def draw_breakdown(rows, out_path, source_label):
    groups = {
        "Conv2d\n(3×3 conv)": ["Conv2d"],
        "Matmul\n(1×1 conv, FC)": ["Matmul"],
        "BinaryOp\n(residual add)": ["BinaryDeviceOperation"],
        "Pool / Reduce": ["Pool2D", "Reduce"],
        "Data movement\n(Halo, Reshard,\nTilize, ...)": [
            "HaloDeviceOperation", "ReshardDeviceOperation",
            "UntilizeWithUnpadding", "MoveDeviceOperation",
            "TilizeWithValPadding", "ShardedToInterleavedDeviceOperation",
            "InterleavedToShardedDeviceOperation", "Tilize",
        ],
    }

    fused_vals, unfused_vals, dm_vals = [], [], []
    for grp_name, ops in groups.items():
        f = u = dm = 0
        for r in rows:
            if r["op"] in ops:
                if r["cat"] == "fused":
                    f += 1
                elif r["cat"] == "unfused":
                    u += 1
                elif r["cat"] == "data_movement":
                    dm += 1
                elif r["cat"] == "other":
                    u += 1
        fused_vals.append(f)
        unfused_vals.append(u)
        dm_vals.append(dm)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_pos = list(range(len(groups)))
    ax.bar(x_pos, fused_vals, color=COLOR_FUSED, label="fused with ReLU",
           edgecolor="white", linewidth=0.8)
    ax.bar(x_pos, unfused_vals, bottom=fused_vals, color=COLOR_UNFUSED,
           label="unfused (compute)", edgecolor="white", linewidth=0.8)
    bottoms_dm = [f + u for f, u in zip(fused_vals, unfused_vals)]
    ax.bar(x_pos, dm_vals, bottom=bottoms_dm, color=COLOR_DM,
           label="data movement", edgecolor="white", linewidth=0.8)

    y_max = max(f + u + dm for f, u, dm in zip(fused_vals, unfused_vals, dm_vals))
    label_offset = y_max * 0.02

    for i in range(len(groups)):
        f, u, dm = fused_vals[i], unfused_vals[i], dm_vals[i]
        if f > 0:
            ax.text(i, f / 2, str(f), ha="center", va="center",
                    fontsize=10, color="white")
        if u > 0:
            ax.text(i, f + u / 2, str(u), ha="center", va="center",
                    fontsize=10, color="white")
        if dm > 0:
            ax.text(i, f + u + dm / 2, str(dm), ha="center", va="center",
                    fontsize=10, color="white")
        total = f + u + dm
        ax.text(i, total + label_offset, f"total = {total}", ha="center",
                va="bottom", fontsize=9, color="dimgray")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(groups.keys()), fontsize=9)
    ax.set_ylabel("number of op instances", fontsize=10)
    ax.set_title(
        "BOS ResNet50 — full op inventory (compute + data movement)",
        fontsize=12, loc="left", pad=10,
    )
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    tot_all = sum(fused_vals) + sum(unfused_vals) + sum(dm_vals)
    tot_fused = sum(fused_vals)
    tot_compute = sum(fused_vals) + sum(unfused_vals)
    fig.text(
        0.5, 0.02,
        f"source: {source_label} ; total {tot_all} ops "
        f"({tot_compute} compute, {sum(dm_vals)} data movement) ; "
        f"fusion ratio in compute = {tot_fused / tot_compute * 100:.1f}%",
        ha="center", fontsize=8, style="italic", color="gray",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────
# Figure 3: Timeline
# ─────────────────────────────────────────────────────────
def draw_timeline(rows, out_path, source_label):
    rows_sorted = sorted(rows, key=lambda r: r["gcc"])

    t_start, t_end = [], []
    t = 0.0
    for r in rows_sorted:
        d = r["dur_ns"] / 1000.0
        t_start.append(t)
        t += d
        t_end.append(t)
    total_us = t

    cat_color = {
        "fused": COLOR_FUSED,
        "unfused": COLOR_UNFUSED,
        "other": "lightgray",
        "data_movement": COLOR_DM,
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 8),
                              gridspec_kw=dict(height_ratios=[1.2, 2.5, 2.5]))

    # Panel 1: strip
    ax = axes[0]
    for r, s, e in zip(rows_sorted, t_start, t_end):
        ax.axvspan(s, e, color=cat_color[r["cat"]], alpha=0.9)
    ax.set_xlim(0, total_us)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(
        "Op timeline — colour = category (fused / unfused / data-movement / other)",
        fontsize=11, loc="left", pad=6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=COLOR_FUSED, label="fused with ReLU"),
        mpatches.Patch(color=COLOR_UNFUSED, label="unfused (compute)"),
        mpatches.Patch(color=COLOR_DM, label="data movement"),
        mpatches.Patch(color="lightgray", label="other"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=8, frameon=False, ncol=4,
              bbox_to_anchor=(1.0, 1.5))

    # Panel 2: cumulative
    ax = axes[1]
    ts = [0.0]
    running = {"fused": 0, "unfused": 0, "other": 0, "data_movement": 0}
    cum = {k: [0] for k in running}
    for r, e in zip(rows_sorted, t_end):
        running[r["cat"]] += 1
        ts.append(e)
        for k in cum:
            cum[k].append(running[k])
    ax.plot(ts, cum["fused"], color=COLOR_FUSED, label="fused", linewidth=1.8)
    ax.plot(ts, cum["unfused"], color=COLOR_UNFUSED, label="unfused", linewidth=1.8)
    ax.plot(ts, cum["data_movement"], color=COLOR_DM, label="data movement", linewidth=1.8)
    ax.plot(ts, cum["other"], color="gray", label="other",
            linewidth=1.2, linestyle="--")
    ax.set_xlim(0, total_us)
    ax.set_ylabel("cumulative op count", fontsize=10)
    ax.set_title("Cumulative op count by category", fontsize=11, loc="left", pad=6)
    ax.grid(True, linestyle="-", linewidth=0.4, color="lightgray", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    # Panel 3: per-op duration
    ax = axes[2]
    for r, s in zip(rows_sorted, t_start):
        d_us = r["dur_ns"] / 1000.0
        ax.bar(s + d_us / 2, d_us, width=max(d_us, 0.3),
               color=cat_color[r["cat"]], alpha=0.85, edgecolor="none")
    ax.set_xlim(0, total_us)
    ax.set_yscale("log")
    ax.set_ylabel("op duration (µs, log)", fontsize=10)
    ax.set_xlabel("t (µs)", fontsize=10)
    ax.set_title("Per-op duration", fontsize=11, loc="left", pad=6)
    ax.grid(True, linestyle="-", linewidth=0.4, color="lightgray",
            alpha=0.7, which="both")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_all = len(rows_sorted)
    n_fused = sum(1 for r in rows_sorted if r["cat"] == "fused")
    n_unfused = sum(1 for r in rows_sorted if r["cat"] == "unfused")
    n_dm = sum(1 for r in rows_sorted if r["cat"] == "data_movement")
    fused_time = sum(r["dur_ns"] for r in rows_sorted if r["cat"] == "fused") / 1000
    total_time = sum(r["dur_ns"] for r in rows_sorted) / 1000
    fig.text(
        0.5, 0.005,
        f"source: {source_label} ; sequential timeline reconstructed from DEVICE KERNEL DURATION; "
        f"{n_all} ops (fused {n_fused}, unfused {n_unfused}, data-movement {n_dm}); "
        f"fused time share = {fused_time / total_time * 100:.1f}%",
        ha="center", fontsize=8, style="italic", color="gray",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────
def print_summary(rows, source_label):
    print(f"\n=== {source_label} ===")
    by_op = defaultdict(lambda: {"fused": 0, "unfused": 0, "data_movement": 0, "other": 0})
    for r in rows:
        by_op[r["op"]][r["cat"]] += 1
    for op in sorted(by_op.keys()):
        c = by_op[op]
        print(f"  {op:40s} fused={c['fused']:4d} unfused={c['unfused']:4d} "
              f"dm={c['data_movement']:4d} other={c['other']:4d}")
    n_fused = sum(1 for r in rows if r["cat"] == "fused")
    n_unfused = sum(1 for r in rows if r["cat"] == "unfused")
    n_dm = sum(1 for r in rows if r["cat"] == "data_movement")
    n_other = sum(1 for r in rows if r["cat"] == "other")
    fused_time = sum(r["dur_ns"] for r in rows if r["cat"] == "fused")
    total_time = sum(r["dur_ns"] for r in rows)
    print(f"  Total ops                     : {len(rows)}")
    print(f"  Fused                         : {n_fused}")
    print(f"  Unfused compute               : {n_unfused}")
    print(f"  Data movement                 : {n_dm}")
    print(f"  Other                         : {n_other}")
    print(f"  Fusion ratio (compute count)  : "
          f"{n_fused / (n_fused + n_unfused + n_other) * 100:.1f}%")
    print(f"  Fusion time share (all ops)   : "
          f"{fused_time / total_time * 100:.1f}%")


def main():
    for src in CSV_SOURCES:
        if not os.path.exists(src["path"]):
            print(f"Skip (not found): {src['path']}")
            continue
        rows = parse_csv(src["path"])
        print_summary(rows, src["label"])

        prefix = src["prefix"]
        draw_donut(rows, os.path.join(SCRIPT_DIR, f"{prefix}_fusion_donut.png"), src["label"])
        draw_breakdown(rows, os.path.join(SCRIPT_DIR, f"{prefix}_fusion_breakdown.png"), src["label"])
        draw_timeline(rows, os.path.join(SCRIPT_DIR, f"{prefix}_fusion_timeline.png"), src["label"])


if __name__ == "__main__":
    main()
