"""Time-window aggregation for verifier-placement analysis.

Divide the 4.1 ms (2.66 M cycle) timeline into N windows.
For each window, aggregate:
  - Per-link byte demand
  - Per-NIU (cell × {NOC0/1_SRC/SINK}) byte demand
  - Per-cell activity (source count, dest count, total bytes)

Output CSVs + heatmap PNGs.

Bin choice: N = 100 windows, ~27 k cycles/bin ≈ 41 μs. Op count ≈ 140 → ~1.4 op/bin.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from route_simulator import (
    GRID_COLS, GRID_ROWS, ALL_LINK_TYPES,
    route,
)
from arrow_flow import (
    CMAP, CELL_COMPUTE, CELL_SINK, SINK_ROW, SINK_COL,
)


DEFAULT_N_WINDOWS = 100


# ============================================================
# Load per_txn.csv
# ============================================================
def load_per_txn(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "stage": int(row["stage"]),
                "run_host_id": int(row["run_host_id"]),
                "is_boundary": int(row["is_boundary"]),
                "sx": int(row["sx"]), "sy": int(row["sy"]),
                "dx": int(row["dx"]), "dy": int(row["dy"]),
                "num_bytes": int(row["num_bytes"]),
                "global_cycle": int(row["global_cycle"]),
                "type": row["type"],
                "noc": row["noc"],
                "op_global_start": int(row["op_global_start"]),
                "op_cycles": int(row["op_cycles"]),
            })
    return rows


def stage_boundaries(txns: list[dict]) -> dict[int, int]:
    starts: dict[int, int] = {}
    for tx in txns:
        starts.setdefault(tx["stage"], tx["op_global_start"])
    return starts


def is_in_grid(x, y):
    return 0 <= x < GRID_COLS and 0 <= y < GRID_ROWS


# ============================================================
# Aggregation
# ============================================================
def aggregate(txns: list[dict], n_windows: int, total_cycles: int,
              include_boundary: bool) -> dict:
    win_size = (total_cycles + n_windows - 1) // n_windows

    # window × link → bytes
    link_by_win = np.zeros((n_windows, GRID_ROWS * GRID_COLS * 4), dtype=np.float64)
    # window × (cell × 4 NIU types) → bytes
    niu_by_win = np.zeros((n_windows, GRID_ROWS * GRID_COLS * 4), dtype=np.float64)
    # window × cell → outgoing / incoming bytes
    cell_out_by_win = np.zeros((n_windows, GRID_ROWS * GRID_COLS), dtype=np.float64)
    cell_in_by_win = np.zeros((n_windows, GRID_ROWS * GRID_COLS), dtype=np.float64)

    def link_idx(r, c, direction):
        d_idx = ALL_LINK_TYPES.index(direction)
        return (r * GRID_COLS + c) * 4 + d_idx

    NIU_TYPES = ("NOC0_SRC", "NOC0_SINK", "NOC1_SRC", "NOC1_SINK")
    def niu_idx(r, c, niu_type):
        t_idx = NIU_TYPES.index(niu_type)
        return (r * GRID_COLS + c) * 4 + t_idx

    def cell_idx(r, c):
        return r * GRID_COLS + c

    for tx in txns:
        if not include_boundary and tx["is_boundary"]:
            continue
        if not (is_in_grid(tx["sx"], tx["sy"]) and is_in_grid(tx["dx"], tx["dy"])):
            continue

        w = min(n_windows - 1, tx["global_cycle"] // win_size)
        nb = tx["num_bytes"]
        noc = tx["noc"]

        # NIU
        if noc == "NOC_0":
            src_niu, sink_niu = "NOC0_SRC", "NOC0_SINK"
        elif noc == "NOC_1":
            src_niu, sink_niu = "NOC1_SRC", "NOC1_SINK"
        else:
            src_niu = sink_niu = None
        if src_niu:
            niu_by_win[w, niu_idx(tx["sy"], tx["sx"], src_niu)] += nb
            niu_by_win[w, niu_idx(tx["dy"], tx["dx"], sink_niu)] += nb

        # cell activity
        cell_out_by_win[w, cell_idx(tx["sy"], tx["sx"])] += nb
        cell_in_by_win[w, cell_idx(tx["dy"], tx["dx"])] += nb

        # link (hop-by-hop route)
        r = route(tx["sy"], tx["sx"], tx["dy"], tx["dx"], noc=noc)
        for link in r:
            link_by_win[w, link_idx(*link)] += nb

    return {
        "win_size": win_size,
        "n_windows": n_windows,
        "total_cycles": total_cycles,
        "link_by_win": link_by_win,
        "niu_by_win": niu_by_win,
        "cell_out_by_win": cell_out_by_win,
        "cell_in_by_win": cell_in_by_win,
        "NIU_TYPES": NIU_TYPES,
    }


# ============================================================
# CSV writers
# ============================================================
def link_meta():
    """(link_idx, row, col, direction) 목록"""
    meta = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            for d in ALL_LINK_TYPES:
                meta.append((r, c, d))
    return meta


def niu_meta(NIU_TYPES):
    meta = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            for t in NIU_TYPES:
                meta.append((r, c, t))
    return meta


def cell_meta():
    return [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def write_matrix_csv(mat: np.ndarray, meta, header_prefix, path: Path,
                      col_prefix="win"):
    n_win = mat.shape[0]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header_prefix) + [f"{col_prefix}_{i:03d}" for i in range(n_win)])
        for i, m in enumerate(meta):
            row = list(m) + [int(mat[wi, i]) for wi in range(n_win)]
            w.writerow(row)


# ============================================================
# Heatmap drawing
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def draw_link_timeline(agg, path, noc_filter, stage_starts, title):
    lm = link_meta()
    keep_idx = [i for i, m in enumerate(lm) if m[2].startswith(noc_filter)]
    keep_meta = [lm[i] for i in keep_idx]
    mat = agg["link_by_win"][:, keep_idx].T  # (link, window)

    # log for visibility
    display = np.log10(mat + 1)

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(display, aspect="auto", cmap=CMAP,
                    interpolation="nearest", origin="upper")
    ax.set_xlabel(f"time window  ({agg['win_size']} cycles / bin = "
                  f"{agg['win_size'] / 650:.1f} μs)", fontsize=10)
    ax.set_ylabel(f"link  (row, col, direction), {noc_filter} only", fontsize=10)
    ax.set_title(title, fontsize=12, loc="left")

    yticks = list(range(0, len(keep_meta), max(1, len(keep_meta) // 30)))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"({keep_meta[i][0]},{keep_meta[i][1]}) {keep_meta[i][2]}"
                        for i in yticks], fontsize=6.5, family="monospace")

    # stage boundaries
    for st, cyc in sorted(stage_starts.items()):
        wb = cyc / agg["win_size"]
        ax.axvline(wb, color="#00b8d4", alpha=0.6, linewidth=0.7,
                    linestyle="--")
        ax.text(wb, -1, f"S{st}", color="#00b8d4", fontsize=8,
                ha="left", va="bottom")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("log10(bytes + 1)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def draw_niu_timeline(agg, path, noc_filter, stage_starts, title):
    nm = niu_meta(agg["NIU_TYPES"])
    keep_idx = [i for i, m in enumerate(nm) if m[2].startswith(noc_filter)]
    keep_meta = [nm[i] for i in keep_idx]
    mat = agg["niu_by_win"][:, keep_idx].T
    display = np.log10(mat + 1)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(display, aspect="auto", cmap=CMAP,
                    interpolation="nearest", origin="upper")
    ax.set_xlabel(f"time window  ({agg['win_size']} cycles / bin = "
                  f"{agg['win_size'] / 650:.1f} μs)", fontsize=10)
    ax.set_ylabel(f"NIU  (row, col, type), {noc_filter} only", fontsize=10)
    ax.set_title(title, fontsize=12, loc="left")

    yticks = list(range(len(keep_meta)))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"({keep_meta[i][0]},{keep_meta[i][1]}) {keep_meta[i][2]}"
                        for i in yticks], fontsize=6, family="monospace")

    for st, cyc in sorted(stage_starts.items()):
        wb = cyc / agg["win_size"]
        ax.axvline(wb, color="#00b8d4", alpha=0.6, linewidth=0.7, linestyle="--")
        ax.text(wb, -0.5, f"S{st}", color="#00b8d4", fontsize=8,
                ha="left", va="bottom")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("log10(bytes + 1)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def draw_cell_activity(agg, path, stage_starts, title):
    cm = cell_meta()
    total = agg["cell_out_by_win"] + agg["cell_in_by_win"]
    mat = total.T
    display = np.log10(mat + 1)

    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(display, aspect="auto", cmap=CMAP,
                    interpolation="nearest", origin="upper")
    ax.set_xlabel(f"time window  ({agg['win_size']} cycles / bin = "
                  f"{agg['win_size'] / 650:.1f} μs)", fontsize=10)
    ax.set_ylabel("cell (row, col) — total in+out bytes", fontsize=10)
    ax.set_title(title, fontsize=12, loc="left")

    yticks = list(range(len(cm)))
    ax.set_yticks(yticks)
    labels = []
    for r, c in cm:
        role = "sink" if (r == SINK_ROW or c == SINK_COL) else "compute"
        labels.append(f"({r},{c}) {role}")
    ax.set_yticklabels(labels, fontsize=6, family="monospace")

    for st, cyc in sorted(stage_starts.items()):
        wb = cyc / agg["win_size"]
        ax.axvline(wb, color="#00b8d4", alpha=0.6, linewidth=0.7, linestyle="--")
        ax.text(wb, -0.5, f"S{st}", color="#00b8d4", fontsize=8,
                ha="left", va="bottom")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("log10(bytes + 1)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


# ============================================================
# Idle scorecard
# ============================================================
def compute_idle_scorecard(agg) -> dict:
    """cell/link 별 idle 통계."""
    n_win = agg["n_windows"]
    cell_activity = agg["cell_out_by_win"] + agg["cell_in_by_win"]  # (win, cell)

    # per cell
    cell_rows = []
    for i, (r, c) in enumerate(cell_meta()):
        active = cell_activity[:, i] > 0
        idle_pct = (1 - active.mean()) * 100
        # longest idle burst
        max_burst = 0
        cur = 0
        for a in active:
            if not a:
                cur += 1
                max_burst = max(max_burst, cur)
            else:
                cur = 0
        cell_rows.append({
            "row": r, "col": c,
            "role": "sink" if (r == SINK_ROW or c == SINK_COL) else "compute",
            "idle_pct": round(idle_pct, 1),
            "active_windows": int(active.sum()),
            "n_windows": n_win,
            "longest_idle_burst": max_burst,
            "longest_idle_us": round(max_burst * agg["win_size"] / 650, 1),
        })

    # per link (aggregate for both directions per cell)
    link_activity = agg["link_by_win"]  # (win, link_idx)
    lm = link_meta()
    link_rows = []
    for i, (r, c, d) in enumerate(lm):
        active = link_activity[:, i] > 0
        idle_pct = (1 - active.mean()) * 100
        max_burst = 0
        cur = 0
        for a in active:
            if not a:
                cur += 1
                max_burst = max(max_burst, cur)
            else:
                cur = 0
        link_rows.append({
            "row": r, "col": c, "direction": d,
            "idle_pct": round(idle_pct, 1),
            "active_windows": int(active.sum()),
            "n_windows": n_win,
            "longest_idle_burst": max_burst,
            "total_bytes": int(link_activity[:, i].sum()),
        })

    return {"cells": cell_rows, "links": link_rows}


def write_scorecard(sc: dict, out_dir: Path):
    with open(out_dir / "idle_scorecard_cells.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sc["cells"][0].keys()))
        w.writeheader()
        for r in sc["cells"]:
            w.writerow(r)
    with open(out_dir / "idle_scorecard_links.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sc["links"][0].keys()))
        w.writeheader()
        for r in sc["links"]:
            w.writerow(r)


def print_scorecard_summary(sc: dict):
    # compute cores 만 랭킹 (verifier 후보)
    compute = [r for r in sc["cells"] if r["role"] == "compute"]
    ranked = sorted(compute, key=lambda r: -r["idle_pct"])
    print("\n=== compute cores ranked by idle % (verifier candidates) ===")
    hdr = (f"{'coord':>7}  {'idle%':>6}  {'active_w':>8}  "
           f"{'longest_idle':>12}  {'idle_μs':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in ranked[:10]:
        print(f"({r['row']},{r['col']})  "
              f"{r['idle_pct']:>5.1f}%  {r['active_windows']:>8}  "
              f"{r['longest_idle_burst']:>12}  {r['longest_idle_us']:>8}")

    print("\n=== bottom 4 compute cores (avoid) ===")
    for r in ranked[-4:]:
        print(f"({r['row']},{r['col']})  "
              f"{r['idle_pct']:>5.1f}%  {r['active_windows']:>8}  "
              f"{r['longest_idle_burst']:>12}  {r['longest_idle_us']:>8}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    p.add_argument("--n-windows", type=int, default=DEFAULT_N_WINDOWS)
    args = p.parse_args()

    txns = load_per_txn(args.out / "per_txn.csv")
    total_cycles = max(tx["global_cycle"] for tx in txns) + 1
    stage_starts = stage_boundaries(txns)
    print(f"[loaded] {len(txns)} txns, total cycles {total_cycles}, "
          f"windows={args.n_windows}")

    # real ops only
    agg_real = aggregate(txns, args.n_windows, total_cycles,
                          include_boundary=False)
    print(f"[agg]    real ops only, win_size={agg_real['win_size']} cycles "
          f"= {agg_real['win_size']/650:.1f} μs")

    # save CSVs
    tw_dir = args.out / "time_window"
    tw_dir.mkdir(exist_ok=True)
    write_matrix_csv(agg_real["link_by_win"], link_meta(),
                      ("row", "col", "direction"),
                      tw_dir / "link_by_window_real.csv")
    write_matrix_csv(agg_real["niu_by_win"], niu_meta(agg_real["NIU_TYPES"]),
                      ("row", "col", "niu_type"),
                      tw_dir / "niu_by_window_real.csv")
    write_matrix_csv(agg_real["cell_out_by_win"], cell_meta(),
                      ("row", "col"),
                      tw_dir / "cell_out_by_window_real.csv")
    write_matrix_csv(agg_real["cell_in_by_win"], cell_meta(),
                      ("row", "col"),
                      tw_dir / "cell_in_by_window_real.csv")
    print(f"[saved] time_window/ CSVs (4 matrices)")

    # heatmaps
    for noc in ("NOC0", "NOC1"):
        draw_link_timeline(
            agg_real, tw_dir / f"link_timeline_real_{noc.lower()}.png",
            noc, stage_starts,
            title=f"Link demand over time — real ops, {noc} only",
        )
        print(f"[saved] link_timeline_real_{noc.lower()}.png")

        draw_niu_timeline(
            agg_real, tw_dir / f"niu_timeline_real_{noc.lower()}.png",
            noc, stage_starts,
            title=f"NIU demand over time — real ops, {noc} only",
        )
        print(f"[saved] niu_timeline_real_{noc.lower()}.png")

    draw_cell_activity(
        agg_real, tw_dir / "cell_activity_timeline_real.png",
        stage_starts,
        title="Per-cell total activity (in+out bytes) over time — real ops",
    )
    print(f"[saved] cell_activity_timeline_real.png")

    # scorecard
    sc = compute_idle_scorecard(agg_real)
    write_scorecard(sc, tw_dir)
    print_scorecard_summary(sc)
    print(f"\n[saved] idle_scorecard_cells.csv, idle_scorecard_links.csv")


if __name__ == "__main__":
    main()
