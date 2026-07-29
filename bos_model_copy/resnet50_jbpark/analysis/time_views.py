"""Time-domain views from per_txn.csv (뷰 A, C).

A. per_core_timeline.png  — X: global cycle bin, Y: 20 compute cores, color: byte-rate
C. dram_endpoint_timeline.png — X: global cycle bin, Y: y=3 6 cells, color: incoming byte-rate

READ semantic 처리:
    Raw trace 의 sx/sy 는 initiator NIU. READ 시 실제 데이터는 (dx,dy)→(sx,sy) 로 흐름.
    - 뷰 A "core activity" = sx 기준 (initiator 관점 traffic). Compute 코어가 얼마나 바쁜지 반영.
    - 뷰 C "DRAM endpoint load" = READ 는 dx 기준 (physical data source), WRITE 는 dx 기준 (physical target).
      즉 dx/dy 로 endpoint traffic 집계.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# translated 좌표 확정 사실 (5-op 교차검증)
COMPUTE_COORDS: list[tuple[int, int]] = [
    (x, y) for y in (0, 1, 2, 4) for x in range(5)
]  # 20 cores, y=3 skip, x=5 skip

DRAM_ROW = 3
DRAM_CELLS: list[tuple[int, int]] = [(x, DRAM_ROW) for x in range(6)]


def load_per_txn(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "stage": int(row["stage"]),
                "is_boundary": int(row["is_boundary"]),
                "sx": int(row["sx"]), "sy": int(row["sy"]),
                "dx": int(row["dx"]), "dy": int(row["dy"]),
                "num_bytes": int(row["num_bytes"]),
                "global_cycle": int(row["global_cycle"]),
                "type": row["type"],
                "op_global_start": int(row["op_global_start"]),
                "op_cycles": int(row["op_cycles"]),
            })
    return rows


def stage_boundaries(txns: list[dict]) -> dict[int, int]:
    """stage -> 첫 op 의 global start cycle"""
    starts: dict[int, int] = {}
    for tx in txns:
        starts.setdefault(tx["stage"], tx["op_global_start"])
    return starts


# ============================================================
# View A: per-core injection timeline
# ============================================================
def build_core_timeline(
    txns: list[dict],
    total_cycles: int,
    bin_size: int,
    include_boundary: bool,
) -> np.ndarray:
    """(20 cores, n_bins) byte-rate matrix. Initiator (sx,sy) 기준."""
    n_bins = (total_cycles + bin_size - 1) // bin_size
    coord_to_row = {c: i for i, c in enumerate(COMPUTE_COORDS)}
    mat = np.zeros((len(COMPUTE_COORDS), n_bins), dtype=np.float64)
    for tx in txns:
        if not include_boundary and tx["is_boundary"]:
            continue
        coord = (tx["sx"], tx["sy"])
        if coord not in coord_to_row:
            continue
        b = tx["global_cycle"] // bin_size
        if 0 <= b < n_bins:
            mat[coord_to_row[coord], b] += tx["num_bytes"]
    # byte-rate (MB per cycle-bin equivalent)
    return mat


def draw_core_timeline(
    mat: np.ndarray,
    stage_starts: dict[int, int],
    bin_size: int,
    path: Path,
    title_suffix: str,
) -> None:
    n_cores, n_bins = mat.shape
    fig, ax = plt.subplots(figsize=(16, 8))
    # log-scale-ish visibility: bytes / bin
    im = ax.imshow(
        mat / 1024,  # bytes -> KB per bin
        aspect="auto",
        cmap="magma",
        origin="upper",
        interpolation="nearest",
    )
    ax.set_xlabel(f"global cycle bin  ({bin_size} cycles / bin)")
    ax.set_ylabel("compute core (x,y translated)")
    ax.set_yticks(range(n_cores))
    ax.set_yticklabels([f"({c[0]},{c[1]})" for c in COMPUTE_COORDS], fontsize=8)
    ax.set_title(f"Per-core injection timeline{title_suffix}\n"
                 f"20 compute cores × time. Bin = {bin_size} cycles = {bin_size / 650:.1f} μs")
    # stage boundaries
    for st, cyc in sorted(stage_starts.items()):
        b = cyc / bin_size
        ax.axvline(b, color="cyan", alpha=0.5, linewidth=0.8, linestyle="--")
        ax.text(b, -0.6, f"S{st}", color="cyan", fontsize=9, ha="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("bytes/bin (KB)")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ============================================================
# View C: DRAM endpoint timeline (dx/dy 기준)
# ============================================================
def build_dram_timeline(
    txns: list[dict],
    total_cycles: int,
    bin_size: int,
    include_boundary: bool,
) -> np.ndarray:
    """(6 dram-row cells, n_bins) byte-rate matrix. Target (dx,dy) 기준."""
    n_bins = (total_cycles + bin_size - 1) // bin_size
    coord_to_row = {c: i for i, c in enumerate(DRAM_CELLS)}
    mat = np.zeros((len(DRAM_CELLS), n_bins), dtype=np.float64)
    for tx in txns:
        if not include_boundary and tx["is_boundary"]:
            continue
        coord = (tx["dx"], tx["dy"])
        if coord not in coord_to_row:
            continue
        b = tx["global_cycle"] // bin_size
        if 0 <= b < n_bins:
            mat[coord_to_row[coord], b] += tx["num_bytes"]
    return mat


def draw_dram_timeline(
    mat: np.ndarray,
    stage_starts: dict[int, int],
    bin_size: int,
    path: Path,
    title_suffix: str,
) -> None:
    n_cells, n_bins = mat.shape
    fig, ax = plt.subplots(figsize=(16, 4.5))
    im = ax.imshow(
        mat / 1024,
        aspect="auto",
        cmap="viridis",
        origin="upper",
        interpolation="nearest",
    )
    ax.set_xlabel(f"global cycle bin  ({bin_size} cycles / bin)")
    ax.set_ylabel("DRAM-row endpoint")
    ax.set_yticks(range(n_cells))
    ax.set_yticklabels(
        [f"({c[0]},{c[1]})" for c in DRAM_CELLS],
        fontsize=8,
    )
    ax.set_title(f"y=3 row endpoint incoming traffic (target dx/dy){title_suffix}\n"
                 f"6 cells × time. Per-cell identity unresolved (memory row aggregate).")
    for st, cyc in sorted(stage_starts.items()):
        b = cyc / bin_size
        ax.axvline(b, color="red", alpha=0.5, linewidth=0.8, linestyle="--")
        ax.text(b, -0.4, f"S{st}", color="red", fontsize=9, ha="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("bytes/bin (KB)")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    p.add_argument("--bin-size", type=int, default=2000,
                   help="cycles per bin. 650 MHz -> 2000 cycles ≈ 3.1 μs")
    args = p.parse_args()

    txns = load_per_txn(args.out / "per_txn.csv")
    total_cycles = max(tx["global_cycle"] for tx in txns) + 1
    stage_starts = stage_boundaries(txns)
    print(f"[loaded] {len(txns)} txns, total cycles {total_cycles}, bin size {args.bin_size}")

    # A — real ops only
    mat_a_real = build_core_timeline(txns, total_cycles, args.bin_size,
                                      include_boundary=False)
    draw_core_timeline(mat_a_real, stage_starts, args.bin_size,
                        args.out / "per_core_timeline_real.png",
                        title_suffix=" — real ops only")
    print(f"[A] per_core_timeline_real.png "
          f"(active: {(mat_a_real > 0).sum()} / {mat_a_real.size} cells)")

    # A — all (real + boundary)
    mat_a_all = build_core_timeline(txns, total_cycles, args.bin_size,
                                     include_boundary=True)
    draw_core_timeline(mat_a_all, stage_starts, args.bin_size,
                        args.out / "per_core_timeline_all.png",
                        title_suffix=" — real + boundary ops")
    print(f"[A] per_core_timeline_all.png "
          f"(active: {(mat_a_all > 0).sum()} / {mat_a_all.size} cells)")

    # C — real ops only
    mat_c_real = build_dram_timeline(txns, total_cycles, args.bin_size,
                                      include_boundary=False)
    draw_dram_timeline(mat_c_real, stage_starts, args.bin_size,
                        args.out / "dram_endpoint_timeline_real.png",
                        title_suffix=" — real ops only")
    print(f"[C] dram_endpoint_timeline_real.png "
          f"(peak endpoint: {mat_c_real.sum(axis=1).argmax()} "
          f"= {DRAM_CELLS[mat_c_real.sum(axis=1).argmax()]})")

    # C — all
    mat_c_all = build_dram_timeline(txns, total_cycles, args.bin_size,
                                     include_boundary=True)
    draw_dram_timeline(mat_c_all, stage_starts, args.bin_size,
                        args.out / "dram_endpoint_timeline_all.png",
                        title_suffix=" — real + boundary ops")
    print(f"[C] dram_endpoint_timeline_all.png")


if __name__ == "__main__":
    main()
