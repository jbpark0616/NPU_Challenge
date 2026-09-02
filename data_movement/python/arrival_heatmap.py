#!/usr/bin/env python3
"""Per-receiver arrival timing heatmap from tracy profile log.

Parses profile_log_device.csv (raw tracy device log) and produces:
- Per-config PNG heatmap of receiver wait cycles on the grid
- Combined comparison figure (all configs side by side)
- Summary CSV with per-core durations

Usage:
    python arrival_heatmap.py [--log CSV] [--out DIR]
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# 5x4 sweep × 2 NoCs: run_id 0..19 = NoC0 sender sweep, 20..39 = NoC1 sender sweep
def _decode(run_id):
    idx = run_id // 1024
    noc = idx // 20
    pos = idx % 20
    sx = pos % 5
    sy = pos // 5
    return sx, sy, noc


def config_label(run_id):
    sx, sy, noc = _decode(run_id)
    return f"5x4 sender=({sx},{sy}) NoC{noc}"


# Compute-only 5x4 grid. Blackhole physical y=3 is DRAM row (no compute).
# Physical y ∈ {0,1,2,4} → display y ∈ {0,1,2,3} (compact, DRAM 제거).
GRID_X = 5
GRID_Y = 4


def phys_to_display_y(py):
    """Physical y coord → compute-only display y (DRAM row y=3 제거)."""
    if py == 3:
        return None  # DRAM row, drop
    return py if py < 3 else py - 1


def parse_receiver_durations(log_path):
    """Return {run_host_id: {(x, y): duration_cycles}} for RISCV1 zones."""
    starts = {}  # (run_id, core) -> start_time
    result = defaultdict(dict)

    with open(log_path, "r", newline="") as f:
        # skip 2 header lines
        f.readline()
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            core_x = int(row[1])
            core_y = int(row[2])
            time_cyc = int(row[5])
            run_id = int(row[7])
            zone_name = row[8].strip()
            zone_type = row[9].strip()
            if zone_name != "RISCV1":
                continue
            key = (run_id, core_x, core_y)
            if zone_type == "ZONE_START":
                starts[key] = time_cyc
            elif zone_type == "ZONE_END":
                if key in starts:
                    dur = time_cyc - starts[key]
                    result[run_id][(core_x, core_y)] = dur
                    del starts[key]
    return result


def plot_single_config(run_id, per_core, out_dir, sender_pos=None):
    label = config_label(run_id)

    grid = np.full((GRID_Y, GRID_X), np.nan)
    for (x, y), dur in per_core.items():
        dy = phys_to_display_y(y)
        if dy is None:
            continue  # DRAM row skip
        if 0 <= x < GRID_X and 0 <= dy < GRID_Y:
            grid[dy, x] = dur

    fig, ax = plt.subplots(figsize=(6, 5))
    valid = grid[~np.isnan(grid)]
    if len(valid) == 0:
        plt.close(fig)
        return None
    # Reds cmap: 진할수록 큰 값 (white=low, dark red=high)
    im = ax.imshow(grid, cmap="Reds", vmin=valid.min(), vmax=valid.max())
    plt.colorbar(im, ax=ax, label="Receiver wait (cycles)")

    # Cell text — 진한 셀엔 흰색, 옅은 셀엔 검은색
    threshold = (valid.min() + valid.max()) / 2
    for y in range(GRID_Y):
        for x in range(GRID_X):
            val = grid[y, x]
            if np.isnan(val):
                ax.text(x, y, "·", ha="center", va="center", color="gray")
            else:
                ax.text(x, y, f"{int(val)}", ha="center", va="center",
                        color="white" if val > threshold else "black",
                        fontsize=8)

    # Mark sender (진한 파랑 테두리로 눈에 띄게)
    if sender_pos:
        sx, sy = sender_pos
        ax.add_patch(plt.Rectangle((sx - 0.5, sy - 0.5), 1, 1, fill=False,
                                    edgecolor="blue", linewidth=3))

    ax.set_title(label)
    ax.set_xlabel("physical x")
    ax.set_ylabel("physical y")
    ax.set_xticks(range(GRID_X))
    ax.set_yticks(range(GRID_Y))
    # imshow 기본 origin='upper' → (0,0) 좌상단. invert_yaxis 안 함.

    out_path = os.path.join(out_dir, f"arrival_run{run_id:04d}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_grid_comparison(all_runs, out_dir):
    """Side-by-side heatmap for all runs — shared color scale."""
    n = len(all_runs)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes).ravel()

    all_vals = [v for run in all_runs.values() for v in run.values()]
    vmin, vmax = min(all_vals), max(all_vals)

    for idx, (run_id, per_core) in enumerate(sorted(all_runs.items())):
        ax = axes[idx]
        grid = np.full((GRID_Y, GRID_X), np.nan)
        for (x, y), dur in per_core.items():
            dy = phys_to_display_y(y)
            if dy is None:
                continue
            grid[dy, x] = dur
        im = ax.imshow(grid, cmap="Reds", vmin=vmin, vmax=vmax)
        sx, sy, _ = _decode(run_id)
        ax.add_patch(plt.Rectangle((sx - 0.5, sy - 0.5), 1, 1, fill=False,
                                    edgecolor="blue", linewidth=2))
        ax.set_title(config_label(run_id), fontsize=8)
        ax.set_xticks(range(GRID_X))
        ax.set_yticks(range(GRID_Y))

    for idx in range(len(all_runs), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Per-receiver arrival wait (cycles) — all configs, shared scale", fontsize=12)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="cycles")
    out_path = os.path.join(out_dir, "arrival_all_configs.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_speedup_grid(all_runs, target_noc, baseline_pos=(0, 0)):
    """Return (speedup_grid, baseline_latency) for one NoC."""
    medians = {}
    for rid, per_core in all_runs.items():
        sx, sy, noc = _decode(rid)
        if noc != target_noc:
            continue
        medians[(sx, sy)] = float(np.median(list(per_core.values())))
    if baseline_pos not in medians:
        return None, None
    baseline = medians[baseline_pos]
    speedup = np.full((GRID_Y, GRID_X), np.nan)
    for (sx, sy), med in medians.items():
        if 0 <= sx < GRID_X and 0 <= sy < GRID_Y:
            speedup[sy, sx] = (baseline - med) / baseline * 100
    return speedup, baseline


def _plot_speedup_heatmap(speedup, title, out_path, cmap="RdBu", vmax=None):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(7, 5.5))
    valid = speedup[~np.isnan(speedup)]
    if vmax is None:
        vmax = max(abs(valid.min()), abs(valid.max()))

    im = ax.imshow(speedup, cmap=cmap, vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label="Speedup [%]")

    for y in range(GRID_Y):
        for x in range(GRID_X):
            v = speedup[y, x]
            if np.isnan(v):
                continue
            color = "white" if abs(v) > vmax * 0.6 else "black"
            ax.text(x, y, f"{v:+.2f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("sender x")
    ax.set_ylabel("sender y")
    ax.set_xticks(range(GRID_X))
    ax.set_yticks(range(GRID_Y))
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    plt.style.use("default")
    plt.rcParams["font.family"] = "sans-serif"


def plot_speedup_maps(all_runs, out_dir):
    """3개 speedup 맵 생성:
    - NoC0 vs NoC0 (0,0) baseline
    - NoC1 vs NoC1 (0,0) baseline
    - NoC1 - NoC0 diff (양수 = NoC1 이 그 sender 에 유리)
    """
    outputs = []
    sp0, base0 = _build_speedup_grid(all_runs, 0)
    sp1, base1 = _build_speedup_grid(all_runs, 1)

    if sp0 is not None:
        # 공통 vmax 로 두 개 같은 스케일
        vmax = max(np.nanmax(np.abs(sp0)),
                   np.nanmax(np.abs(sp1)) if sp1 is not None else 0)
        p = os.path.join(out_dir, "speedup_noc0_vs_00.png")
        _plot_speedup_heatmap(
            sp0,
            f"NoC0 sender speedup vs (0,0) baseline\nbaseline = {base0:.0f} cycles",
            p, vmax=vmax)
        outputs.append(p)
        if sp1 is not None:
            p = os.path.join(out_dir, "speedup_noc1_vs_00.png")
            _plot_speedup_heatmap(
                sp1,
                f"NoC1 sender speedup vs (0,0) baseline\nbaseline = {base1:.0f} cycles",
                p, vmax=vmax)
            outputs.append(p)

            # Cross-NoC diff: NoC1 - NoC0 (양수 = 그 sender 에서 NoC1 이 더 빠름)
            diff = sp1 - sp0
            p = os.path.join(out_dir, "speedup_noc1_minus_noc0.png")
            _plot_speedup_heatmap(
                diff,
                "NoC preference per sender (NoC1 speedup − NoC0 speedup)\n"
                "+ = NoC1 유리, − = NoC0 유리",
                p, cmap="PuOr_r")
            outputs.append(p)
    return outputs


def dump_summary_csv(all_runs, out_dir):
    """Long CSV: run_id, config_label, core_x, core_y, duration_cycles."""
    out_path = os.path.join(out_dir, "arrival_per_core.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_host_id", "config", "core_x", "core_y", "duration_cycles"])
        for run_id in sorted(all_runs):
            label = config_label(run_id)
            for (x, y), dur in sorted(all_runs[run_id].items()):
                w.writerow([run_id, label, x, y, dur])
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="C:/NPU_Challenge/data_movement/data/blackhole/profile_log_device.csv")
    parser.add_argument("--out", default="C:/NPU_Challenge/data_movement/data/blackhole/arrival")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    all_runs = parse_receiver_durations(args.log)
    print(f"Parsed {len(all_runs)} configs, {sum(len(v) for v in all_runs.values())} receiver cores")

    # Individual heatmaps — sender 위치 파란 테두리 표시
    for run_id, per_core in sorted(all_runs.items()):
        n = len(per_core)
        durs = list(per_core.values())
        sx, sy, _ = _decode(run_id)
        print(f"  Run {run_id:5d} ({config_label(run_id)}): "
              f"{n} cores, min={min(durs)}, max={max(durs)}, spread={max(durs)-min(durs)}")
        plot_single_config(run_id, per_core, args.out, sender_pos=(sx, sy))

    # Combined comparison
    combo = plot_grid_comparison(all_runs, args.out)
    print(f"Wrote combined figure: {combo}")

    # Speedup maps (NoC0, NoC1, cross-NoC diff)
    for p in plot_speedup_maps(all_runs, args.out):
        print(f"Wrote speedup map: {p}")

    # Per-core CSV
    csv_path = dump_summary_csv(all_runs, args.out)
    print(f"Wrote per-core CSV: {csv_path}")


if __name__ == "__main__":
    main()
