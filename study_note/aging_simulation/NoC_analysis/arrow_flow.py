"""A0-native NoC flow arrow visualization — publication figure quality.

6x5 translated grid (empirically confirmed):
  - 20 compute cores at (row in {0,1,2,4}) x (col in {0..4})
  - Sink-only row 3 (empirical: never sources any transaction)
  - Sink-only col 5 (empirical: never sources any transaction)

Visual encoding:
  - Straight arrows (horizontal/vertical) = link (hop) demand, i.e., pass-through
  - Diagonal markers (corner arrows)      = NIU (endpoint) demand at that cell
    - NOC 0 SRC  = top-right     (out)
    - NOC 0 SINK = bottom-right  (in)
    - NOC 1 SRC  = bottom-left   (out)
    - NOC 1 SINK = top-left      (in)
  - Dashed line = torus wrap edge (short stub at grid boundary)
  - Colour + width = log(bytes), threshold p30, saturate p95
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from route_simulator import (
    GRID_COLS, GRID_ROWS,
    NOC0_EAST, NOC0_SOUTH, NOC1_NORTH, NOC1_WEST,
    ALL_LINK_TYPES,
    link_endpoints,
)


# ============================================================
# Publication rcParams
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ============================================================
# A0 identity
# ============================================================
COMPUTE_ROWS = (0, 1, 2, 4)
COMPUTE_COLS = (0, 1, 2, 3, 4)
SINK_ROW = 3
SINK_COL = 5

# Muted palette for cells
CELL_COMPUTE = "#eef4fa"   # very light blue
CELL_SINK = "#f5f0f0"      # very light warm gray
CELL_BORDER = "#7f8a99"
CELL_BORDER_LW = 0.6

# Sequential colour ramp for demand (muted red)
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "demand", ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"]
)


def cell_color(r: int, c: int) -> str:
    if c == SINK_COL or r == SINK_ROW:
        return CELL_SINK
    return CELL_COMPUTE


def cell_role(r: int, c: int) -> str:
    if c == SINK_COL or r == SINK_ROW:
        return "sink-only"
    return "compute"


# ============================================================
# Data loaders
# ============================================================
def load_link_demand(path: Path) -> dict[tuple, dict]:
    d: dict[tuple, dict] = {}
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (int(row["from_row"]), int(row["from_col"]), row["direction"])
            d[key] = {
                "bytes_real": int(row["bytes_real"]),
                "bytes_all": int(row["bytes_all"]),
                "hop_count": int(row["hop_count_all"]),
            }
    return d


def load_per_op_link_demand(path: Path) -> dict[tuple, dict]:
    d: dict[tuple, dict] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            op = (int(row["stage"]), int(row["run_host_id"]))
            key = (int(row["from_row"]), int(row["from_col"]), row["direction"])
            d[op][key] = int(row["bytes"])
    return dict(d)


def load_niu_demand(path: Path) -> dict[tuple, dict]:
    d: dict[tuple, dict] = {}
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (int(row["row"]), int(row["col"]), row["niu_type"])
            d[key] = {
                "bytes_real": int(row["bytes_real"]),
                "bytes_all": int(row["bytes_all"]),
            }
    return d


def load_per_op_niu_demand(path: Path) -> dict[tuple, dict]:
    d: dict[tuple, dict] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            op = (int(row["stage"]), int(row["run_host_id"]))
            key = (int(row["row"]), int(row["col"]), row["niu_type"])
            d[op][key] = int(row["bytes"])
    return dict(d)


# ============================================================
# Grid drawing
# ============================================================
CELL_HALF = 0.44  # cell 반지름


def draw_grid(ax) -> None:
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            ax.add_patch(plt.Rectangle(
                (c - CELL_HALF, r - CELL_HALF), CELL_HALF * 2, CELL_HALF * 2,
                facecolor=cell_color(r, c),
                edgecolor=CELL_BORDER, linewidth=CELL_BORDER_LW,
                zorder=1,
            ))
            # coord label at top-left corner of cell
            ax.text(c - CELL_HALF + 0.04, r - CELL_HALF + 0.04,
                    f"({r},{c})",
                    ha="left", va="top", fontsize=7.5,
                    color="#444444", zorder=3,
                    family="monospace")
            # role at bottom-right, muted
            ax.text(c + CELL_HALF - 0.04, r + CELL_HALF - 0.04,
                    cell_role(r, c),
                    ha="right", va="bottom", fontsize=6.5,
                    color="#888888", zorder=3, style="italic")

    ax.set_xlim(-0.7, GRID_COLS - 0.3)
    ax.set_ylim(GRID_ROWS - 0.3, -0.7)  # invert y so row 0 at top
    ax.set_aspect("equal")
    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.set_xlabel("col  (sx / dx)", fontsize=10, color="#333333")
    ax.set_ylabel("row  (sy / dy)", fontsize=10, color="#333333")
    ax.tick_params(axis="both", which="both", length=0, pad=3)


# ============================================================
# Demand → visual style
# ============================================================
def _style(bytes_val: int, vmin: int, vmax: int) -> tuple:
    if bytes_val <= 0:
        return None, 0
    b = max(vmin, min(vmax, bytes_val))
    lv = math.log10(b) - math.log10(vmin)
    rng = math.log10(vmax) - math.log10(vmin) + 1e-9
    ratio = max(0.0, min(1.0, lv / rng))
    color = CMAP(0.15 + 0.85 * ratio)
    lw = 0.6 + 4.0 * ratio
    return color, lw


def _clip_range(vals, threshold_pct: float, clip_pct: float) -> tuple:
    arr = np.array(sorted([v for v in vals if v > 0]))
    if len(arr) == 0:
        return 1, 1
    vmin = arr[int(len(arr) * threshold_pct / 100)] if len(arr) > 1 else arr[0]
    vmax = arr[min(len(arr) - 1, int(len(arr) * clip_pct / 100))]
    if vmin >= vmax:
        vmin, vmax = arr[0], arr[-1]
    return vmin, vmax


# ============================================================
# Link arrows (horizontal/vertical, hop pass-through)
# ============================================================
# 링크 방향별 perpendicular offset (겹침 방지):
#   NOC 0 EAST/SOUTH: 살짝 위쪽/왼쪽 offset
#   NOC 1 NORTH/WEST: 살짝 아래쪽/오른쪽 offset
LINK_OFFSET = 0.09


def _link_geometry(r: int, c: int, d: str) -> tuple:
    """(start, end, is_wrap) 반환. start/end 는 셀 경계 기준 정확 정렬."""
    if d == NOC0_EAST:
        wrap = c + 1 >= GRID_COLS
        start = (c + CELL_HALF * 0.05, r - LINK_OFFSET)
        end = (c + CELL_HALF * (1 if wrap else 0.90) + (0.35 if not wrap else 0.15),
               r - LINK_OFFSET)
        # simpler exact:
        start = (c, r - LINK_OFFSET)
        end = (c + (1 if not wrap else 0.55), r - LINK_OFFSET)
    elif d == NOC0_SOUTH:
        wrap = r + 1 >= GRID_ROWS
        start = (c - LINK_OFFSET, r)
        end = (c - LINK_OFFSET, r + (1 if not wrap else 0.55))
    elif d == NOC1_NORTH:
        wrap = r - 1 < 0
        start = (c + LINK_OFFSET, r)
        end = (c + LINK_OFFSET, r - (1 if not wrap else 0.55))
    elif d == NOC1_WEST:
        wrap = c - 1 < 0
        start = (c, r + LINK_OFFSET)
        end = (c - (1 if not wrap else 0.55), r + LINK_OFFSET)
    else:
        return None, None, False
    return start, end, wrap


def draw_link_arrows(ax, link_demand, bytes_key, noc_filter,
                     vmin=None, vmax=None,
                     threshold_pct=30.0, clip_pct=95.0):
    def _val(v):
        return v[bytes_key] if isinstance(v, dict) else v

    filtered = link_demand
    if noc_filter:
        filtered = {k: v for k, v in link_demand.items()
                    if k[2].startswith(noc_filter)}

    if vmin is None or vmax is None:
        vmin, vmax = _clip_range([_val(v) for v in filtered.values()],
                                  threshold_pct, clip_pct)

    for link, val in filtered.items():
        r, c, d = link
        b = _val(val)
        if b < vmin:
            continue
        color, lw = _style(b, vmin, vmax)
        if color is None:
            continue
        start, end, wrap = _link_geometry(r, c, d)
        if start is None:
            continue

        if wrap:
            # 톤다운: alpha 낮게, 색상 muted, dash 촘촘
            wrap_color = mcolors.to_rgba(color, alpha=0.35)
            ax.annotate("",
                xy=end, xytext=start,
                arrowprops=dict(
                    arrowstyle="-|>", color=wrap_color, lw=lw * 0.7,
                    mutation_scale=8, linestyle=(0, (1.5, 1.5)),
                    shrinkA=0, shrinkB=0,
                ),
                zorder=4,
            )
        else:
            ax.annotate("",
                xy=end, xytext=start,
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=lw,
                    mutation_scale=10, alpha=0.92,
                    shrinkA=0, shrinkB=0,
                ),
                zorder=5,
            )
    return vmin, vmax


# ============================================================
# NIU markers (diagonal corner arrows)
# ============================================================
# NIU markers — 우측에 몰아서 SRC/SINK 만 그림 (NOC 분리는 subplot 이 처리)
# 코너 위치 + 화살표 방향 (짧고 얇게 → link 와 시각적 구분)
NIU_CORNERS = {
    # SRC = 우상단, 화살표 밖으로 (out)
    "SRC":  (+0.28, -0.28, +0.15, -0.15, "out"),
    # SINK = 우하단, 화살표 안으로 (in)
    "SINK": (+0.28, +0.28, +0.15, +0.15, "in"),
}


def draw_niu_arrows(ax, niu_demand, bytes_key, noc_filter,
                    vmin=None, vmax=None,
                    threshold_pct=30.0, clip_pct=95.0):
    """noc_filter 는 필수 — 이 그림이 어느 NoC 를 담당하는지 명시.
    NIU 화살표는 SRC(우상단, out) / SINK(우하단, in) 두 종류만.
    """
    if not noc_filter:
        return  # combined 그림엔 NIU 안 그림

    def _val(v):
        return v[bytes_key] if isinstance(v, dict) else v

    # 이 NOC 의 NIU 만
    filtered = {k: v for k, v in niu_demand.items() if k[2].startswith(noc_filter)}

    if vmin is None or vmax is None:
        vmin, vmax = _clip_range([_val(v) for v in filtered.values()],
                                  threshold_pct, clip_pct)

    for (r, c, niu_type), val in filtered.items():
        b = _val(val)
        # SRC / SINK 추출 (NOC0_SRC → SRC)
        role = niu_type.split("_")[1]
        if b < vmin or role not in NIU_CORNERS:
            continue
        color, lw = _style(b, vmin, vmax)
        if color is None:
            continue

        dc_o, dr_o, dc_i, dr_i, direction = NIU_CORNERS[role]
        corner = (c + dc_o, r + dr_o)
        inside = (c + dc_i, r + dr_i)
        if direction == "out":
            start, end = inside, corner
        else:
            start, end = corner, inside

        # link 대비 얇고 짧게 + 화살촉 다르게 (fancy) → 한눈에 다른 종류로 인식
        ax.annotate("",
            xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="wedge,tail_width=0.3", color=color,
                lw=lw * 0.5, alpha=0.95,
                shrinkA=0, shrinkB=0,
            ),
            zorder=6,
        )


# ============================================================
# Colorbar + legend
# ============================================================
def add_colorbar(fig, ax, vmin: int, vmax: int, label: str = "bytes"):
    norm = mcolors.LogNorm(vmin=max(1, vmin), vmax=max(vmin + 1, vmax))
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=30)
    cbar.set_label(f"{label}   (log, p30-p95 clip)", fontsize=9, color="#333333")
    cbar.ax.tick_params(labelsize=8, colors="#333333", length=0)
    cbar.outline.set_edgecolor("#cccccc")
    cbar.outline.set_linewidth(0.7)


LEGEND_TEXT = (
    "Encoding\n"
    "  Straight arrow    link (hop pass-through)\n"
    "  Dashed (muted)    torus-wrap edge stub\n"
    "  Wedge (right side of cell) NIU endpoint activity:\n"
    "     top-right      SRC   (cell injecting)\n"
    "     bottom-right   SINK  (cell absorbing)\n"
    "Colour & width    log(bytes),  shared scale across NOC 0 / NOC 1"
)


def add_encoding_legend(fig, x=0.02, y=0.02):
    fig.text(x, y, LEGEND_TEXT, fontsize=7.5, family="monospace",
             va="bottom", ha="left", color="#333333",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                        edgecolor="#cccccc", linewidth=0.7))


# ============================================================
# Figures
# ============================================================
def draw_aggregate(link_demand, niu_demand, path, title,
                   bytes_key="bytes_real", noc_filter=None,
                   vmin=None, vmax=None, draw_niu=True):
    """draw_niu=False → link only (combined 뷰 등)."""
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(1, 1, left=0.06, right=0.92, top=0.92, bottom=0.20)
    ax = fig.add_subplot(gs[0, 0])
    draw_grid(ax)
    vm_in, vm_ax = draw_link_arrows(ax, link_demand, bytes_key, noc_filter,
                                     vmin=vmin, vmax=vmax)
    if draw_niu and niu_demand and noc_filter:
        draw_niu_arrows(ax, niu_demand, bytes_key, noc_filter,
                        vmin=vm_in, vmax=vm_ax)
    add_colorbar(fig, ax, vm_in, vm_ax)
    ax.set_title(title, fontsize=12, pad=10, color="#222222", loc="left")
    add_encoding_legend(fig, x=0.06, y=0.02)
    plt.savefig(path)
    plt.close(fig)


def compute_shared_range(link_demand, niu_demand, bytes_key,
                          threshold_pct=30.0, clip_pct=95.0):
    """NOC 0 / NOC 1 두 그림 공통 vmin/vmax. Link + NIU 통합."""
    def _val(v):
        return v[bytes_key] if isinstance(v, dict) else v
    all_vals = [_val(v) for v in link_demand.values() if _val(v) > 0]
    if niu_demand:
        all_vals.extend(_val(v) for v in niu_demand.values() if _val(v) > 0)
    return _clip_range(all_vals, threshold_pct, clip_pct)


def draw_single_op(per_op_links, per_op_nius, path, title):
    # 공통 스케일 계산
    all_vals = [v for v in per_op_links.values() if v > 0]
    if per_op_nius:
        all_vals.extend(v for v in per_op_nius.values() if v > 0)
    vmin, vmax = _clip_range(all_vals, threshold_pct=0, clip_pct=100)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, left=0.05, right=0.95, top=0.90, bottom=0.20,
                          wspace=0.15)
    for i, (noc, tag) in enumerate([("NOC0", "NOC 0  (+col / +row)"),
                                     ("NOC1", "NOC 1  (-row / -col)")]):
        ax = fig.add_subplot(gs[0, i])
        draw_grid(ax)
        draw_link_arrows(ax, per_op_links, None, noc, vmin=vmin, vmax=vmax)
        if per_op_nius:
            draw_niu_arrows(ax, per_op_nius, None, noc, vmin=vmin, vmax=vmax)
        if vmax > vmin:
            add_colorbar(fig, ax, vmin, vmax)
        ax.set_title(tag, fontsize=11, pad=8, color="#222222", loc="left")
    fig.suptitle(title, fontsize=13, y=0.96, x=0.05, ha="left", color="#111111")
    add_encoding_legend(fig, x=0.05, y=0.02)
    plt.savefig(path)
    plt.close(fig)


def draw_empty_op(path, title):
    """No-NoC-traffic op 용 빈 격자."""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, left=0.05, right=0.95, top=0.90, bottom=0.20,
                          wspace=0.15)
    for i, tag in enumerate(["NOC 0  (+col / +row)", "NOC 1  (-row / -col)"]):
        ax = fig.add_subplot(gs[0, i])
        draw_grid(ax)
        ax.set_title(tag, fontsize=11, pad=8, color="#222222", loc="left")
        ax.text(0.5, 0.5, "no NoC traffic",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=22, color="#aaaaaa", style="italic",
                zorder=10)
    fig.suptitle(title, fontsize=13, y=0.96, x=0.05, ha="left", color="#111111")
    add_encoding_legend(fig, x=0.05, y=0.02)
    plt.savefig(path)
    plt.close(fig)


def draw_per_op_poster(per_op, per_op_niu, out_dir, top_n=12, op_names=None):
    op_totals = {op: sum(v.values()) for op, v in per_op.items()}
    top = sorted(op_totals.items(), key=lambda kv: -kv[1])[:top_n]

    ncols = 3
    nrows = math.ceil(top_n / ncols)
    fig = plt.figure(figsize=(ncols * 6.2, nrows * 5.5))
    gs = fig.add_gridspec(nrows, ncols, left=0.04, right=0.98,
                          top=0.95, bottom=0.06, hspace=0.30, wspace=0.20)

    for i, (op, total) in enumerate(top):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        draw_grid(ax)
        vmin, vmax = draw_link_arrows(ax, per_op[op], None, None,
                                       threshold_pct=0, clip_pct=100)
        if op in per_op_niu:
            draw_niu_arrows(ax, per_op_niu[op], None, None,
                            threshold_pct=0, clip_pct=100)
        stage, rhid = op
        name = op_names.get(op, "?") if op_names else "?"
        ax.set_title(f"stage {stage}   rhid {rhid}   {name}\n"
                     f"link-bytes: {total / 1e6:.2f} MB",
                     fontsize=10, pad=6, color="#222222", loc="left")

    fig.suptitle("Per-op link + NIU demand — top 12 real ops on A0 6×5 grid",
                 fontsize=14, y=0.985, x=0.04, ha="left", color="#111111")
    plt.savefig(out_dir / "per_op_arrow_top12.png")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    args = p.parse_args()

    ld = load_link_demand(args.out / "link_demand.csv")
    per_op = load_per_op_link_demand(args.out / "link_demand_by_op.csv")
    niu = load_niu_demand(args.out / "niu_demand.csv")
    per_op_niu = load_per_op_niu_demand(args.out / "niu_demand_by_op.csv")

    op_names = {}
    with open(args.out / "per_op.csv", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            op_names[(int(row["stage"]), int(row["run_host_id"]))] = row["op_name"]

    # ---- Real ops, shared colour scale across NOC 0 / NOC 1 ----
    v_real = compute_shared_range(ld, niu, "bytes_real")
    for noc, tag in [("NOC0", "noc0"), ("NOC1", "noc1")]:
        draw_aggregate(
            ld, niu, args.out / f"arrow_flow_real_{tag}.png",
            title=f"ResNet-50 on Eagle-N A0 — {noc}, real ops   (shared scale)",
            bytes_key="bytes_real", noc_filter=noc,
            vmin=v_real[0], vmax=v_real[1],
        )
        print(f"[saved] arrow_flow_real_{tag}.png")

    # ---- All ops (real+boundary), shared scale ----
    v_all = compute_shared_range(ld, niu, "bytes_all")
    for noc, tag in [("NOC0", "noc0"), ("NOC1", "noc1")]:
        draw_aggregate(
            ld, niu, args.out / f"arrow_flow_all_{tag}.png",
            title=f"ResNet-50 on Eagle-N A0 — {noc}, real+boundary   (shared scale)",
            bytes_key="bytes_all", noc_filter=noc,
            vmin=v_all[0], vmax=v_all[1],
        )
        print(f"[saved] arrow_flow_all_{tag}.png")

    # ---- Combined reference: link only, no NIU ----
    draw_aggregate(
        ld, None, args.out / "arrow_flow_real_linkonly.png",
        title=("ResNet-50 on Eagle-N A0 — link demand only, both NOCs combined "
               "(reference)"),
        bytes_key="bytes_real", noc_filter=None,
        draw_niu=False,
    )
    print(f"[saved] arrow_flow_real_linkonly.png")

    real_per_op = {op: v for op, v in per_op.items()
                   if op_names.get(op, "") not in {
                       "InterleavedToShardedDeviceOperation",
                       "ShardedToInterleavedDeviceOperation",
                       "TilizeWithValPadding", "Tilize",
                       "UntilizeWithUnpadding", "Untilize",
                       "ReshardDeviceOperation", "MoveDeviceOperation",
                   }}
    draw_per_op_poster(real_per_op, per_op_niu, args.out, top_n=12,
                       op_names=op_names)
    print(f"[saved] per_op_arrow_top12.png")

    known = {
        "op121_stage1_rhid5120_Conv2d":  (1, 5120),
        "op172_stage3_rhid12288_Conv2d": (3, 12288),
        "op235_stage4_rhid31744_Matmul": (4, 31744),
    }
    for tag, key in known.items():
        if key in per_op:
            name = op_names.get(key, "?")
            draw_single_op(
                per_op[key], per_op_niu.get(key, {}),
                args.out / f"arrow_{tag}.png",
                title=(f"{tag}   ({name})   "
                       f"total link-bytes: {sum(per_op[key].values()) / 1e6:.2f} MB"),
            )
            print(f"[saved] arrow_{tag}.png")

    # ---- Batch export: all ops per stage (empty ops included) ----
    per_op_dir = args.out / "per_op_arrows"
    per_op_dir.mkdir(exist_ok=True)
    # per_op.csv 에서 전체 op 목록 (empty 도 포함)
    all_ops_meta = []
    with open(args.out / "per_op.csv", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            all_ops_meta.append({
                "stage": int(row["stage"]),
                "run_host_id": int(row["run_host_id"]),
                "op_name": row["op_name"],
                "txn_count": int(row["txn_count"]),
                "total_bytes": int(row["total_bytes"]),
            })

    n_total = 0
    n_empty = 0
    for meta in sorted(all_ops_meta, key=lambda m: (m["stage"], m["run_host_id"])):
        stage, rhid, name = meta["stage"], meta["run_host_id"], meta["op_name"]
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        stage_dir = per_op_dir / f"stage_{stage}"
        stage_dir.mkdir(exist_ok=True)
        fname = f"op_rhid{rhid:05d}_{safe_name}.png"

        links = per_op.get((stage, rhid), {})
        nius = per_op_niu.get((stage, rhid), {})

        if not links and not nius:
            # empty op: no NoC traffic 표시
            draw_empty_op(
                stage_dir / fname,
                title=(f"stage {stage}   rhid {rhid}   {name}   "
                       f"txn_count: {meta['txn_count']}   "
                       f"no NoC traffic"),
            )
            n_empty += 1
        else:
            total_mb = sum(links.values()) / 1e6
            draw_single_op(
                links, nius,
                stage_dir / fname,
                title=(f"stage {stage}   rhid {rhid}   {name}   "
                       f"total link-bytes: {total_mb:.3f} MB"),
            )
        n_total += 1
        if n_total % 20 == 0:
            print(f"  ...{n_total} ops exported")
    print(f"[saved] per_op_arrows/  ({n_total} PNGs, {n_empty} empty, 5 stages)")


if __name__ == "__main__":
    main()
