"""시간축 slice heatmaps — op 진행 따라 코어별 NoC 부하 변화 시각화.

세 종류 출력:
1. per_stage_heatmaps.png     : stage 0~4 각각의 injected/consumed (5행 × 2열)
2. cumulative_progression.png : stage 누적 (stem → +layer1 → +layer2 → ...)
3. per_op_top_N_heatmaps.png  : real ops 중 bytes 상위 N 개 각각 (small multiples)

Real ops 만 대상 (boundary 제외).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aggregate_per_core import (
    GRID_H,
    GRID_W,
    _empty_core_stats,
    aggregate_per_core,
    load_stage,
)
from stage_paths import STAGE_PATHS, validate


def draw_grid(ax, arr: np.ndarray, observed: np.ndarray, title: str, vmax_shared: float | None = None):
    vmax = vmax_shared if vmax_shared is not None else (arr.max() if arr.max() > 0 else 1.0)
    im = ax.imshow(arr, cmap="YlOrRd", origin="lower", aspect="equal", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.tick_params(labelsize=7)
    for i in range(GRID_H):
        for j in range(GRID_W):
            if not observed[i, j]:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
                continue
            v = arr[i, j]
            color = "black" if v < vmax * 0.55 else "white"
            ax.text(j, i, f"{v:.1f}" if v >= 0.1 else "0", ha="center", va="center",
                    fontsize=6, color=color)
    return im


def per_core_to_arrays(per_core: dict):
    inj = np.zeros((GRID_H, GRID_W))
    con = np.zeros((GRID_H, GRID_W))
    obs = np.zeros((GRID_H, GRID_W), dtype=bool)
    for (sx, sy), d in per_core.items():
        if 0 <= sx < GRID_W and 0 <= sy < GRID_H:
            inj[sy, sx] = d["injected"] / 1e6
            con[sy, sx] = d["consumed"] / 1e6
            obs[sy, sx] = True
    return inj, con, obs


def per_stage(all_ops_by_stage: dict, out_path: Path):
    """Stage 별 heatmap (5행 × 2열 grid)."""
    fig, axes = plt.subplots(5, 2, figsize=(12, 18))
    # 각 stage 최댓값 계산해서 시각적 대조 위한 shared vmax
    stage_arrays = {}
    inj_max = con_max = 0.0
    for st in sorted(all_ops_by_stage.keys()):
        per_core = aggregate_per_core(all_ops_by_stage[st], include_boundary=False)
        inj, con, obs = per_core_to_arrays(per_core)
        stage_arrays[st] = (inj, con, obs)
        inj_max = max(inj_max, inj.max())
        con_max = max(con_max, con.max())

    stage_titles = {
        0: "stem (7 ops → 4 real)",
        1: "layer1 (18 → 16)",
        2: "layer2 (29 → 22)",
        3: "layer3 (54 → 38)",
        4: "layer4+head (32 → 19)",
    }
    for row_idx, st in enumerate(sorted(all_ops_by_stage.keys())):
        inj, con, obs = stage_arrays[st]
        ax1, ax2 = axes[row_idx]
        im1 = draw_grid(ax1, inj, obs, f"stage_{st} Injected — {stage_titles[st]}", inj_max)
        im2 = draw_grid(ax2, con, obs, f"stage_{st} Consumed", con_max)
        ax1.set_ylabel(f"stage_{st}\nsy →", fontsize=9)
        if row_idx == 4:
            ax1.set_xlabel("sx (translated col)")
            ax2.set_xlabel("sx (translated col)")

    fig.colorbar(im1, ax=axes[:, 0], fraction=0.02, pad=0.02, label="Injected MB")
    fig.colorbar(im2, ax=axes[:, 1], fraction=0.02, pad=0.02, label="Consumed MB")
    fig.suptitle("Per-stage NoC traffic (real ops only), Eagle-N A0",
                 fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.96, bottom=0.03, hspace=0.35, wspace=0.15)
    plt.savefig(out_path, dpi=110)
    plt.close()


def cumulative(all_ops_by_stage: dict, out_path: Path):
    """Stage 누적 heatmap (stem → +layer1 → ...)."""
    fig, axes = plt.subplots(5, 2, figsize=(12, 18))
    cumulative_ops: list[dict] = []
    arrays = {}
    inj_max = con_max = 0.0
    for st in sorted(all_ops_by_stage.keys()):
        cumulative_ops.extend(all_ops_by_stage[st])
        per_core = aggregate_per_core(cumulative_ops, include_boundary=False)
        inj, con, obs = per_core_to_arrays(per_core)
        arrays[st] = (inj, con, obs)
        inj_max = max(inj_max, inj.max())
        con_max = max(con_max, con.max())

    stage_labels = {0: "stem", 1: "+layer1", 2: "+layer2", 3: "+layer3", 4: "+layer4+head (full)"}
    for row_idx, st in enumerate(sorted(all_ops_by_stage.keys())):
        inj, con, obs = arrays[st]
        ax1, ax2 = axes[row_idx]
        draw_grid(ax1, inj, obs, f"cum. {stage_labels[st]} — Injected", inj_max)
        draw_grid(ax2, con, obs, f"cum. {stage_labels[st]} — Consumed", con_max)
        ax1.set_ylabel(f"{stage_labels[st]}\nsy →", fontsize=9)
        if row_idx == 4:
            ax1.set_xlabel("sx")
            ax2.set_xlabel("sx")

    fig.suptitle("Cumulative NoC traffic through model (real ops only)",
                 fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.96, bottom=0.03, hspace=0.35, wspace=0.15)
    plt.savefig(out_path, dpi=110)
    plt.close()


def per_op_top_N(all_ops: list[dict], out_path: Path, top_n: int = 20):
    """Real ops 중 bytes 상위 N 개 각각 heatmap (small multiples)."""
    real = [op for op in all_ops if not op["is_boundary"]]
    real.sort(key=lambda o: -o["total_bytes"])
    top = real[:top_n]

    ncols = 4
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = axes.flatten() if nrows > 1 else [axes] if nrows == 1 and ncols == 1 else axes

    # 개별 op 단독으로 그리기 (op 간 부하 차이 시각화 위해 shared vmax 사용 X)
    for idx, op in enumerate(top):
        ax = axes[idx]
        # op 하나만으로 per-core 뽑기
        per_core_op = aggregate_per_core([op], include_boundary=False)
        inj, con, obs = per_core_to_arrays(per_core_op)
        # 이 op 의 injected only 표시 (consumed 도 유사 형태라 페이지 아끼려 injected 만)
        vmax = inj.max() if inj.max() > 0 else 1.0
        im = ax.imshow(inj, cmap="YlOrRd", origin="lower", aspect="equal", vmin=0, vmax=vmax)
        ax.set_title(f"stage_{op['stage']} rhid={op['run_host_id']}\n"
                     f"{op['op_name'][:22]}\n{op['total_bytes'] / 1024:.0f} KB, "
                     f"{op['cycles'] / 650:.1f} μs",
                     fontsize=8)
        ax.set_xticks(range(GRID_W))
        ax.set_yticks(range(GRID_H))
        ax.tick_params(labelsize=6)
        for i in range(GRID_H):
            for j in range(GRID_W):
                if not obs[i, j]:
                    continue
                v = inj[i, j]
                if v > 0:
                    color = "black" if v < vmax * 0.55 else "white"
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=6, color=color)

    for idx in range(len(top), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"Top {top_n} real ops by NoC bytes — Injected heatmap each "
                 f"(rank = descending by total bytes)",
                 fontsize=11, y=0.99)
    fig.subplots_adjust(top=0.94, bottom=0.03, hspace=0.5, wspace=0.15)
    plt.savefig(out_path, dpi=110)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    p.add_argument("--top-n", type=int, default=20, help="per-op small multiples 개수")
    args = p.parse_args()

    validate()
    args.out.mkdir(parents=True, exist_ok=True)

    all_ops_by_stage: dict[int, list[dict]] = {}
    all_ops: list[dict] = []
    for stage, path in sorted(STAGE_PATHS.items()):
        ops = load_stage(stage, path)
        all_ops_by_stage[stage] = ops
        all_ops.extend(ops)
        print(f"[load] stage_{stage}: {len(ops):3d} ops")
    print()

    p1 = args.out / "per_stage_heatmaps.png"
    p2 = args.out / "cumulative_progression.png"
    p3 = args.out / f"per_op_top{args.top_n}_heatmaps.png"

    print(f"[gen] {p1.name}")
    per_stage(all_ops_by_stage, p1)
    print(f"[gen] {p2.name}")
    cumulative(all_ops_by_stage, p2)
    print(f"[gen] {p3.name}")
    per_op_top_N(all_ops, p3, args.top_n)

    print(f"\n[saved] outputs → {args.out}/")


if __name__ == "__main__":
    main()
