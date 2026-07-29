"""시간축 데이터 추출 — 지시서 (time_window/시간축_데이터_추출_지시서.md) 준수.

Op 순차 실행 확인됨 (겹침 없음, Step 1 결과).
=> Case A 로 처리: op별 bytes ÷ duration = 그 구간 rate.

좌표 신뢰도:
  real_ops:      compute↔compute 만 (신뢰 높음)
  real_boundary: 전체 (참고, boundary 라벨)

산출물:
  ops.csv, timeline.csv, link_timeseries.csv, niu_timeseries.csv,
  fig_timeline.png (최우선), fig_link_heatmap.png, fig_cdf.png, fig_op_scatter.png,
  summary.md
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from route_simulator import GRID_ROWS, GRID_COLS, ALL_LINK_TYPES, route

CHIP_FREQ_MHZ = 650  # 650 MHz — cycles → μs conversion


# =====================================================================
# Empirical A0 identity
# =====================================================================
COMPUTE_ROWS = (0, 1, 2, 4)
COMPUTE_COLS = (0, 1, 2, 3, 4)
SINK_ROW = 3
SINK_COL = 5


def is_compute(r: int, c: int) -> bool:
    return r in COMPUTE_ROWS and c in COMPUTE_COLS


def is_txn_compute_only(sx, sy, dx, dy) -> bool:
    return is_compute(sy, sx) and is_compute(dy, dx)


# =====================================================================
# Style
# =====================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# =====================================================================
# Loaders
# =====================================================================
def load_per_txn(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "stage": int(row["stage"]),
                "run_host_id": int(row["run_host_id"]),
                "op_name": row["op_name"],
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


def load_per_op(path: Path) -> list[dict]:
    """per_op.csv"""
    rows = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "stage": int(row["stage"]),
                "run_host_id": int(row["run_host_id"]),
                "op_name": row["op_name"],
                "is_boundary": int(row["is_boundary"]),
                "txn_count": int(row["txn_count"]),
                "total_bytes": int(row["total_bytes"]),
                "cycles": int(row["cycles"]),
            })
    return rows


# =====================================================================
# 2.1 ops.csv
# =====================================================================
def classify_op_type(op_name: str) -> str:
    n = op_name.lower()
    if "conv2d" in n: return "Conv2d"
    if "matmul" in n: return "Matmul"
    if "halo" in n: return "Halo"
    if "pool" in n: return "Pool"
    if "binary" in n or "add" in n: return "Eltwise"
    if "unary" in n: return "Unary"
    if "reduce" in n: return "Reduce"
    if "interleaved" in n or "sharded" in n or "reshard" in n: return "Layout"
    if "tilize" in n or "untilize" in n: return "Tilize"
    if "move" in n: return "Move"
    return "Other"


def build_ops_csv(per_op: list[dict], txns: list[dict], out_path: Path):
    # global start cycle per op = cumulative prior op cycles (op-sequential model)
    per_op_sorted = sorted(per_op, key=lambda o: (o["stage"], o["run_host_id"]))
    running = 0
    op_time: dict[tuple, tuple[int, int]] = {}
    for op in per_op_sorted:
        start = running
        end = running + op["cycles"]
        op_time[(op["stage"], op["run_host_id"])] = (start, end)
        running = end

    # per-op traffic breakdown (real = compute-only, total = all)
    by_op = defaultdict(lambda: {
        "noc0_real": 0, "noc1_real": 0,
        "noc0_total": 0, "noc1_total": 0,
        "active_cores": set(),
    })
    for tx in txns:
        key = (tx["stage"], tx["run_host_id"])
        real = is_txn_compute_only(tx["sx"], tx["sy"], tx["dx"], tx["dy"])
        nb = tx["num_bytes"]
        if tx["noc"] == "NOC_0":
            by_op[key]["noc0_total"] += nb
            if real: by_op[key]["noc0_real"] += nb
        elif tx["noc"] == "NOC_1":
            by_op[key]["noc1_total"] += nb
            if real: by_op[key]["noc1_real"] += nb
        # active cores: any compute core participating as src or dst
        if is_compute(tx["sy"], tx["sx"]):
            by_op[key]["active_cores"].add((tx["sy"], tx["sx"]))
        if is_compute(tx["dy"], tx["dx"]):
            by_op[key]["active_cores"].add((tx["dy"], tx["dx"]))

    rows = []
    for op in per_op_sorted:
        key = (op["stage"], op["run_host_id"])
        t_start_c, t_end_c = op_time[key]
        t_start_us = t_start_c / CHIP_FREQ_MHZ
        t_end_us = t_end_c / CHIP_FREQ_MHZ
        duration_us = t_end_us - t_start_us
        b = by_op[key]
        active = sorted(b["active_cores"])
        active_str = ";".join(f"({r},{c})" for r, c in active)
        n_active = len(active)
        op_id = f"op_stage{op['stage']}_rhid{op['run_host_id']:05d}_{op['op_name']}"

        def rate(bytes_val):
            return bytes_val / duration_us if duration_us > 0 else 0

        rows.append({
            "op_id": op_id,
            "op_type": classify_op_type(op["op_name"]),
            "stage": op["stage"],
            "is_boundary": op["is_boundary"],
            "t_start_us": round(t_start_us, 3),
            "t_end_us": round(t_end_us, 3),
            "duration_us": round(duration_us, 3),
            "n_active_cores": n_active,
            "active_cores": active_str,
            "noc0_bytes_real": b["noc0_real"],
            "noc1_bytes_real": b["noc1_real"],
            "noc0_bytes_total": b["noc0_total"],
            "noc1_bytes_total": b["noc1_total"],
            "noc0_rate_real": round(rate(b["noc0_real"]), 3),
            "noc1_rate_real": round(rate(b["noc1_real"]), 3),
            "noc0_rate_total": round(rate(b["noc0_total"]), 3),
            "noc1_rate_total": round(rate(b["noc1_total"]), 3),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows, op_time


# =====================================================================
# 2.2 / 2.3 / 2.4 : Time series (link, NIU, timeline)
# =====================================================================
def build_timeseries(txns, op_time, bin_us: float, total_us: float,
                     class_filter="real"):
    """class_filter: 'real' (compute↔compute only) or 'total' (all)."""
    n_bins = int(np.ceil(total_us / bin_us))
    bin_cycles = bin_us * CHIP_FREQ_MHZ

    # link 시계열: {(noc, from_r, from_c, direction, bin) : bytes}
    link_bins = defaultdict(int)
    # NIU 시계열: {(noc, r, c, 'src' or 'sink', bin) : bytes}
    niu_bins = defaultdict(int)
    # timeline 통계
    noc0_total_by_bin = np.zeros(n_bins)
    noc1_total_by_bin = np.zeros(n_bins)
    noc0_max_link_by_bin = np.zeros(n_bins)
    noc1_max_link_by_bin = np.zeros(n_bins)
    active_cores_by_bin = [set() for _ in range(n_bins)]
    active_ops_by_bin = [set() for _ in range(n_bins)]

    # 임시로 link demand per (bin, noc) 계산해 최대 링크 산출용
    per_bin_link_max: dict[tuple, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))

    for tx in txns:
        # grid bounds — skip invalid endpoints (-1)
        if not (0 <= tx["sx"] < GRID_COLS and 0 <= tx["sy"] < GRID_ROWS):
            continue
        if not (0 <= tx["dx"] < GRID_COLS and 0 <= tx["dy"] < GRID_ROWS):
            continue

        real = is_txn_compute_only(tx["sx"], tx["sy"], tx["dx"], tx["dy"])
        if class_filter == "real" and not real:
            continue

        nb = tx["num_bytes"]
        noc_str = tx["noc"]
        noc_idx = 0 if noc_str == "NOC_0" else 1 if noc_str == "NOC_1" else -1
        if noc_idx < 0:
            continue

        key = (tx["stage"], tx["run_host_id"])
        t_start_c, t_end_c = op_time[key]
        op_duration_c = t_end_c - t_start_c
        if op_duration_c <= 0:
            continue

        # spread bytes uniformly over the op's active bins
        b_start = int(t_start_c / bin_cycles)
        b_end = int(t_end_c / bin_cycles)
        n_bin = max(1, b_end - b_start + 1)
        nb_per_bin = nb / n_bin

        # 라우팅
        hops = route(tx["sy"], tx["sx"], tx["dy"], tx["dx"], noc=noc_str)

        for b in range(b_start, min(b_end + 1, n_bins)):
            # NIU (same-cell 포함 — NIU 는 여전히 사용됨)
            niu_bins[(noc_idx, tx["sy"], tx["sx"], "src", b)] += nb_per_bin
            niu_bins[(noc_idx, tx["dy"], tx["dx"], "sink", b)] += nb_per_bin

            # link (hop-by-hop). same-cell txn 은 hops 가 empty → 링크 안 씀
            for (fr, fc, d) in hops:
                link_bins[(noc_idx, fr, fc, d, b)] += nb_per_bin
                per_bin_link_max[(noc_idx, b)][(fr, fc, d)] += nb_per_bin

            # active cores
            if is_compute(tx["sy"], tx["sx"]):
                active_cores_by_bin[b].add((tx["sy"], tx["sx"]))
            if is_compute(tx["dy"], tx["dx"]):
                active_cores_by_bin[b].add((tx["dy"], tx["dx"]))
            active_ops_by_bin[b].add(key)

    # totals = 링크 사용 합 (byte-hops).
    # 이렇게 하면 total == 0 iff max_link == 0 (self-referencing txn 은 링크 안 쓰므로 제외).
    for (noc_idx, fr, fc, d, b), bytes_val in link_bins.items():
        if noc_idx == 0:
            noc0_total_by_bin[b] += bytes_val
        else:
            noc1_total_by_bin[b] += bytes_val

    # max link per bin (bytes/bin → convert to bytes/us)
    for (noc_idx, b), links in per_bin_link_max.items():
        max_b = max(links.values()) if links else 0
        if noc_idx == 0:
            noc0_max_link_by_bin[b] = max_b / bin_us
        else:
            noc1_max_link_by_bin[b] = max_b / bin_us

    # convert totals to bytes/us
    noc0_rate = noc0_total_by_bin / bin_us
    noc1_rate = noc1_total_by_bin / bin_us

    return {
        "n_bins": n_bins,
        "bin_us": bin_us,
        "noc0_rate": noc0_rate,       # B/μs
        "noc1_rate": noc1_rate,
        "noc0_max_link_rate": noc0_max_link_by_bin,
        "noc1_max_link_rate": noc1_max_link_by_bin,
        "n_active_cores": np.array([len(s) for s in active_cores_by_bin]),
        "active_ops": active_ops_by_bin,
        "link_bins": link_bins,
        "niu_bins": niu_bins,
    }


def write_timeline_csv(ts_real, ts_total, ops_rows, op_time, out_path: Path):
    """timeline.csv — 지시서 2.4"""
    n = ts_real["n_bins"]
    bin_us = ts_real["bin_us"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t_us",
            "noc0_total_rate_real", "noc1_total_rate_real",
            "noc0_total_rate_total", "noc1_total_rate_total",
            "noc0_max_link_rate_real", "noc1_max_link_rate_real",
            "n_active_cores", "n_idle_cores",
            "active_op_ids",
        ])
        for b in range(n):
            active_ids = ";".join(
                f"stage{s}_rhid{r}" for (s, r) in sorted(ts_real["active_ops"][b])
            )
            w.writerow([
                round(b * bin_us, 3),
                round(ts_real["noc0_rate"][b], 3),
                round(ts_real["noc1_rate"][b], 3),
                round(ts_total["noc0_rate"][b], 3),
                round(ts_total["noc1_rate"][b], 3),
                round(ts_real["noc0_max_link_rate"][b], 3),
                round(ts_real["noc1_max_link_rate"][b], 3),
                int(ts_real["n_active_cores"][b]),
                int(20 - ts_real["n_active_cores"][b]),
                active_ids,
            ])


def write_link_timeseries_csv(ts_real, ts_total, out_path: Path):
    """link_timeseries.csv — 지시서 2.2"""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_us", "noc", "link_id", "bytes_per_us", "traffic_class"])
        bin_us = ts_real["bin_us"]

        def write_one(ts, cls):
            # link_id: (from_r, from_c, direction) — convert to human string
            # aggregate by (noc, link, bin) and emit
            for (noc_idx, fr, fc, d, b), bytes_val in ts["link_bins"].items():
                # convert bytes-in-bin to bytes/us
                rate = bytes_val / bin_us
                # link_id 문자열: "(sy,sx)->(dy,dx)"
                # 방향으로부터 계산
                from route_simulator import link_endpoints
                (frr, frc), (tr, tc) = link_endpoints((fr, fc, d))
                link_id = f"({frr},{frc})->{tr},{tc}"
                w.writerow([
                    round(b * bin_us, 3), noc_idx, link_id,
                    round(rate, 4), cls,
                ])
        write_one(ts_real, "real")
        write_one(ts_total, "total")


def write_niu_timeseries_csv(ts_real, ts_total, out_path: Path):
    """niu_timeseries.csv — 지시서 2.3"""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_us", "noc", "core", "src_bytes_per_us",
                    "sink_bytes_per_us", "traffic_class"])
        bin_us = ts_real["bin_us"]

        def write_one(ts, cls):
            # accumulate src/sink separately per (noc, core, bin)
            agg = defaultdict(lambda: {"src": 0.0, "sink": 0.0})
            for (noc_idx, r, c, kind, b), bytes_val in ts["niu_bins"].items():
                agg[(noc_idx, r, c, b)][kind] += bytes_val
            for (noc_idx, r, c, b), sv in agg.items():
                w.writerow([
                    round(b * bin_us, 3), noc_idx, f"({r},{c})",
                    round(sv["src"] / bin_us, 4),
                    round(sv["sink"] / bin_us, 4),
                    cls,
                ])
        write_one(ts_real, "real")
        write_one(ts_total, "total")


# =====================================================================
# 3.1  fig_timeline.png — 최우선
# =====================================================================
OP_TYPE_COLORS = {
    "Conv2d":  "#1f77b4",
    "Matmul":  "#d62728",
    "Halo":    "#ff7f0e",
    "Pool":    "#2ca02c",
    "Eltwise": "#9467bd",
    "Unary":   "#8c564b",
    "Reduce":  "#e377c2",
    "Layout":  "#7f7f7f",
    "Tilize":  "#bcbd22",
    "Move":    "#17becf",
    "Other":   "#888888",
}


def draw_fig_timeline(ts_real, ts_total, ops_rows, out_path: Path, total_us):
    bin_us = ts_real["bin_us"]
    n = ts_real["n_bins"]
    t = np.arange(n) * bin_us

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(
        4, 1, height_ratios=[3, 2, 2, 1],
        hspace=0.20, left=0.06, right=0.98, top=0.96, bottom=0.06,
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # KEY: 관측 가능 창 (idle >= 4 AND NOC0 idle) 녹색 음영 — 모든 패널
    observability = (ts_real["n_active_cores"] <= 16) & (ts_real["noc0_rate"] == 0)
    if observability.any():
        d = np.diff(observability.astype(int))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if observability[0]: starts = np.insert(starts, 0, 0)
        if observability[-1]: ends = np.append(ends, len(observability))
        for ax_shade in (ax1, ax2, ax3, ax4):
            for s, e in zip(starts, ends):
                ax_shade.axvspan(s * bin_us, e * bin_us,
                                  color="#c7e9b4", alpha=0.35, zorder=0)

    # ---- Panel 1: NoC demand
    ax1.plot(t, ts_real["noc0_rate"], color="#1f77b4", lw=1.4,
             label="NOC0 (real)")
    ax1.plot(t, ts_real["noc1_rate"], color="#d62728", lw=1.4,
             label="NOC1 (real)")
    ax1.plot(t, ts_total["noc0_rate"], color="#1f77b4", lw=0.8,
             linestyle=(0, (2, 2)), alpha=0.7, label="NOC0 (total)")
    ax1.plot(t, ts_total["noc1_rate"], color="#d62728", lw=0.8,
             linestyle=(0, (2, 2)), alpha=0.7, label="NOC1 (total)")
    ax1.set_ylabel("bytes / μs  (link-byte sum)")
    ax1.set_title("NoC demand — real (solid) vs. total (dashed)   "
                  "[shaded = observability window: idle≥4 & NOC0 idle]",
                  loc="left", fontsize=11)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    # initial weight load 주석
    peak_t0 = ts_total["noc0_rate"][:200].argmax() * bin_us + \
              max(ts_total["noc0_rate"][:200].max(), ts_total["noc1_rate"][:200].max())
    max_early = max(ts_total["noc0_rate"][:200].max(), ts_total["noc1_rate"][:200].max())
    ax1.annotate("initial weight load",
                  xy=(0, max_early), xytext=(150, max_early * 0.75),
                  fontsize=9, color="#555555",
                  arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7))

    # ---- Panel 2: max link rate
    ax2.plot(t, ts_real["noc0_max_link_rate"], color="#1f77b4", lw=1.2,
             label="NOC0 max link")
    ax2.plot(t, ts_real["noc1_max_link_rate"], color="#d62728", lw=1.2,
             label="NOC1 max link")
    ax2.set_ylabel("bytes / μs  (max link)")
    ax2.set_title("Max link rate — real ops", loc="left", fontsize=11)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    # ---- Panel 3: active cores
    ax3.step(t, ts_real["n_active_cores"], color="#333333", lw=1.0,
             where="post")
    mean_ac = ts_real["n_active_cores"].mean()
    ax3.axhline(mean_ac, color="#e74c3c", lw=1.0, linestyle="--", alpha=0.8)
    ax3.text(total_us * 0.985, mean_ac + 0.4,
              f"mean = {mean_ac:.1f}", ha="right", fontsize=8, color="#e74c3c")
    ax3.axhline(16, color="#2ecc71", lw=0.7, linestyle=":", alpha=0.7)
    ax3.text(total_us * 0.985, 16.4, "16 (= 4 idle)",
              ha="right", fontsize=7, color="#2ecc71")
    ax3.set_ylabel("active compute cores (of 20)")
    ax3.set_title("Active core count — real ops   "
                  "[green line = 16 threshold for observability window]",
                  loc="left", fontsize=11)
    ax3.set_ylim(-0.5, 21)
    ax3.grid(alpha=0.3)

    # ---- Panel 4: op bands
    for i, op in enumerate(ops_rows):
        if op["is_boundary"]:
            continue
        c = OP_TYPE_COLORS.get(op["op_type"], "#888888")
        ax4.axvspan(op["t_start_us"], op["t_end_us"],
                    color=c, alpha=0.7, zorder=1)
    ax4.set_yticks([])
    ax4.set_xlabel("t  (μs)")
    ax4.set_title("Op timeline — colour = op_type", loc="left", fontsize=11)
    # legend
    legend_items = []
    for op_type, color in OP_TYPE_COLORS.items():
        if any(op["op_type"] == op_type and not op["is_boundary"] for op in ops_rows):
            legend_items.append((op_type, color))
    for op_type, color in legend_items:
        ax4.plot([], [], color=color, lw=8, label=op_type)
    ax4.legend(loc="upper right", fontsize=7, ncol=len(legend_items))

    plt.savefig(out_path)
    plt.close(fig)


# =====================================================================
# 3.2  fig_link_heatmap.png
# =====================================================================
def draw_fig_link_heatmap(ts_real, out_path: Path):
    """link × time heatmap, height_ratios=[1, 2.5], NOC0 top 15 / NOC1 top 40."""
    bin_us = ts_real["bin_us"]
    n_bins = ts_real["n_bins"]

    per_link_series = defaultdict(lambda: np.zeros(n_bins))
    for (noc_idx, fr, fc, d, b), bytes_val in ts_real["link_bins"].items():
        per_link_series[(noc_idx, fr, fc, d)][b] += bytes_val

    def prep(noc_idx, top_n):
        items = [((fr, fc, d), s) for (n, fr, fc, d), s in per_link_series.items()
                  if n == noc_idx]
        items.sort(key=lambda kv: -kv[1].sum())
        return items[:min(top_n, len(items))]

    n0 = prep(0, 15)   # NOC 0 는 상위 15 개
    n1 = prep(1, 40)   # NOC 1 는 상위 40 개

    def link_label(link):
        fr, fc, d = link
        from route_simulator import link_endpoints
        (_, _), (tr, tc) = link_endpoints((fr, fc, d))
        wrap = (
            (d == "NOC0_EAST" and tc == 0) or
            (d == "NOC0_SOUTH" and tr == 0) or
            (d == "NOC1_NORTH" and tr == GRID_ROWS - 1) or
            (d == "NOC1_WEST" and tc == GRID_COLS - 1)
        )
        arrow = "⤾" if wrap else "→"
        return f"({fr},{fc}){arrow}({tr},{tc}) {d.split('_')[1]}"

    # 방향별 색
    DIR_COLORS = {
        "EAST":  "#1f77b4",   # NOC 0 +col
        "SOUTH": "#2ca02c",   # NOC 0 +row
        "NORTH": "#d62728",   # NOC 1 -row
        "WEST":  "#ff7f0e",   # NOC 1 -col
    }

    def build_mat(items):
        mat = np.array([s for _, s in items]) if items else np.zeros((1, n_bins))
        labels = [link_label(l) for l, _ in items] if items else ["(none)"]
        colors = [DIR_COLORS.get(l[2].split("_")[1], "#333") for l, _ in items] \
                 if items else ["#333"]
        return mat / bin_us, labels, colors

    m0, lbl0, col0 = build_mat(n0)
    m1, lbl1, col1 = build_mat(n1)

    display0 = np.log10(m0 + 1)
    display1 = np.log10(m1 + 1)
    vmax = max(display0.max(), display1.max(), 1e-9)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"hspace": 0.16, "height_ratios": [1, 2.5]},
    )
    im0 = ax0.imshow(display0, aspect="auto", cmap="magma_r",
                     interpolation="nearest", origin="upper",
                     vmin=0, vmax=vmax,
                     extent=[0, n_bins * bin_us, len(lbl0) - 0.5, -0.5])
    ax0.set_yticks(range(len(lbl0)))
    ax0.set_yticklabels(lbl0, fontsize=6, family="monospace")
    for ticklabel, c in zip(ax0.get_yticklabels(), col0):
        ticklabel.set_color(c)
    ax0.tick_params(axis="y", pad=1, length=0)
    ax0.set_ylabel("NOC 0 links (top 15)", fontsize=10)
    ax0.set_title("Link rate over time — real ops, log10(bytes/μs + 1). "
                   "Colour of y-label = link direction.",
                   loc="left", fontsize=11)

    im1 = ax1.imshow(display1, aspect="auto", cmap="magma_r",
                     interpolation="nearest", origin="upper",
                     vmin=0, vmax=vmax,
                     extent=[0, n_bins * bin_us, len(lbl1) - 0.5, -0.5])
    ax1.set_yticks(range(len(lbl1)))
    ax1.set_yticklabels(lbl1, fontsize=6, family="monospace")
    for ticklabel, c in zip(ax1.get_yticklabels(), col1):
        ticklabel.set_color(c)
    ax1.tick_params(axis="y", pad=1, length=0)
    ax1.set_ylabel("NOC 1 links (top 40)", fontsize=10)
    ax1.set_xlabel("t  (μs)")

    cbar = fig.colorbar(im1, ax=[ax0, ax1], fraction=0.02, pad=0.01)
    cbar.set_label("log10(bytes/μs + 1)", fontsize=9)

    # direction color legend
    legend_items = [plt.Line2D([0], [0], color=c, lw=4, label=d)
                     for d, c in DIR_COLORS.items()]
    ax1.legend(handles=legend_items, loc="lower right", fontsize=7,
                title="direction", title_fontsize=7)

    fig.text(0.5, 0.005,
              "logical coordinates; real ops (compute↔compute) only",
              fontsize=7, color="#888888", ha="center", va="bottom", style="italic")

    plt.savefig(out_path)
    plt.close(fig)

    return {
        "noc0_top_links": [(link_label(l), s.sum() / bin_us) for l, s in n0[:10]],
        "noc1_top_links": [(link_label(l), s.sum() / bin_us) for l, s in n1[:10]],
    }


# =====================================================================
# 3.3  fig_cdf.png
# =====================================================================
def draw_fig_cdf(ts_real, out_path: Path):
    """부하 rate 의 CDF — log x, in-line labels."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

    def draw_one(ax, r0, r1, xlabel):
        for label, arr, color in [("NOC 0", r0, "#1f77b4"),
                                    ("NOC 1", r1, "#d62728")]:
            arr_sorted = np.sort(arr)
            cdf = np.arange(1, len(arr_sorted) + 1) / len(arr_sorted)
            # shift by 1 so log works (log(0) 방지)
            ax.step(arr_sorted + 1, cdf, where="post", label=label, color=color, lw=1.6)

        ax.set_xscale("log")
        # 기준선 in-line 라벨
        for q in (0.50, 0.80, 0.95):
            ax.axhline(q, color="#888888", lw=0.6, linestyle="--", alpha=0.7)
            # NOC1 곡선에서의 x 값 찾기 (더 뻗어있는 쪽)
            idx = np.searchsorted(np.sort(r1), np.quantile(r1, q))
            x_at_q = np.sort(r1)[min(idx, len(r1) - 1)] + 1
            ax.text(x_at_q, q + 0.015, f"  {int(q*100)}% at ≈{x_at_q:.0f} B/μs",
                     fontsize=7, ha="left", va="bottom", color="#666666")

        ax.set_xlabel(xlabel + "  (log, +1 shift)")
        ax.set_ylabel("cumulative time fraction")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_ylim(-0.02, 1.02)

    draw_one(ax0, ts_real["noc0_rate"], ts_real["noc1_rate"],
              "total NoC rate  (bytes/μs)")
    ax0.set_title("Total NoC rate CDF — real ops", loc="left", fontsize=11)

    draw_one(ax1, ts_real["noc0_max_link_rate"], ts_real["noc1_max_link_rate"],
              "max link rate  (bytes/μs)")
    ax1.set_title("Max link rate CDF — real ops", loc="left", fontsize=11)

    fig.text(0.5, 0.01,
              "logical coordinates; real ops (compute↔compute) only",
              fontsize=7, color="#888888", ha="center", va="bottom", style="italic")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# =====================================================================
# 3.4  fig_op_scatter.png
# =====================================================================
def draw_fig_op_scatter(ops_rows, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 8))

    real_ops = [o for o in ops_rows if not o["is_boundary"]
                and o["noc1_bytes_real"] > 0 and o["duration_us"] > 0]
    xs_all = np.array([o["duration_us"] for o in real_ops])
    ys_all = np.array([o["noc1_bytes_real"] for o in real_ops])

    for op_type, color in OP_TYPE_COLORS.items():
        pts = [o for o in real_ops if o["op_type"] == op_type]
        if not pts:
            continue
        xs = [o["duration_us"] for o in pts]
        ys = [o["noc1_bytes_real"] for o in pts]
        ax.scatter(xs, ys, s=60, color=color, alpha=0.7,
                    label=op_type, edgecolors="white", linewidth=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # 데이터 범위 기반 xlim/ylim
    ax.set_xlim(xs_all.min() * 0.7, xs_all.max() * 1.5)
    ax.set_ylim(ys_all.min() * 0.7, ys_all.max() * 1.5)

    # iso-rate 대각선 (실제 범위 내)
    x_lo, x_hi = ax.get_xlim()
    xs_iso = np.logspace(np.log10(x_lo), np.log10(x_hi), 100)
    for rate, label in [(1e3, "1 kB/μs"), (1e4, "10 kB/μs"),
                        (1e5, "100 kB/μs"), (1e6, "1 MB/μs")]:
        ys_iso = rate * xs_iso
        # 그림 안에 있는 부분만
        y_lo, y_hi = ax.get_ylim()
        mask = (ys_iso >= y_lo) & (ys_iso <= y_hi)
        if mask.any():
            ax.plot(xs_iso[mask], ys_iso[mask], "--",
                     color="#cccccc", lw=0.8, alpha=0.7)
            # 라벨 우측 상단
            idx = np.where(mask)[0][-1] - 3
            if idx > 0:
                ax.annotate(label, xy=(xs_iso[idx], ys_iso[idx]),
                             fontsize=7, color="#888888")

    ax.set_xlabel("op duration  (μs, log)")
    ax.set_ylabel("NOC 1 real bytes  (log)")
    ax.set_title("Per-op scatter — real ops. Colour = op_type.",
                  loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.text(0.99, 0.01, "logical coordinates; real ops (compute↔compute) only",
             transform=ax.transAxes, fontsize=7, color="#888888",
             ha="right", va="bottom", style="italic")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# =====================================================================
# summary.md
# =====================================================================
def compute_derived_metrics(ts_real, ts_total, ops_rows):
    m = {}

    # 1. NOC0 : NOC1 total bytes ratio
    tot0 = ts_real["noc0_rate"].sum()
    tot1 = ts_real["noc1_rate"].sum()
    m["noc0_share"] = tot0 / (tot0 + tot1) * 100 if (tot0 + tot1) > 0 else 0
    m["noc1_share"] = tot1 / (tot0 + tot1) * 100 if (tot0 + tot1) > 0 else 0
    m["noc0_total"] = tot0
    m["noc1_total"] = tot1

    # 2. NOC0 == 0 시간 비율
    m["noc0_idle_pct"] = (ts_real["noc0_rate"] == 0).sum() / len(ts_real["noc0_rate"]) * 100
    m["noc1_idle_pct"] = (ts_real["noc1_rate"] == 0).sum() / len(ts_real["noc1_rate"]) * 100

    # 3. active cores 분포
    ac = ts_real["n_active_cores"]
    m["active_mean"] = float(ac.mean())
    m["active_median"] = float(np.median(ac))
    m["active_min"] = int(ac.min())
    m["active_max"] = int(ac.max())

    # 4. idle >= 4 && noc0 == 0
    for k in (4, 8):
        idle_k = ac <= (20 - k)
        combined = idle_k & (ts_real["noc0_rate"] == 0)
        m[f"idle{k}_noc0_zero_pct"] = combined.sum() / len(ac) * 100

    # 5. max link rate percentiles + active-only + consistency check
    for noc, arr, tot in [("noc0", ts_real["noc0_max_link_rate"], ts_real["noc0_rate"]),
                            ("noc1", ts_real["noc1_max_link_rate"], ts_real["noc1_rate"])]:
        m[f"{noc}_max_link_p50"] = float(np.percentile(arr, 50))
        m[f"{noc}_max_link_p95"] = float(np.percentile(arr, 95))
        m[f"{noc}_max_link_max"] = float(arr.max())
        m[f"{noc}_max_zero_pct"] = (arr == 0).sum() / len(arr) * 100
        m[f"{noc}_total_zero_pct"] = (tot == 0).sum() / len(tot) * 100
        active = arr[arr > 0]
        m[f"{noc}_active_p50"] = float(np.percentile(active, 50)) if len(active) else 0
        m[f"{noc}_active_p95"] = float(np.percentile(active, 95)) if len(active) else 0
        m[f"{noc}_active_n"] = int(len(active))

    # warmup 제외 (t > 300 μs) 통계
    bin_us = ts_real["bin_us"]
    warmup_bin = int(300.0 / bin_us)
    ac_post = ts_real["n_active_cores"][warmup_bin:]
    n0_post = ts_real["noc0_rate"][warmup_bin:]
    n1_post = ts_real["noc1_rate"][warmup_bin:]
    if len(ac_post):
        m["post_warmup_active_mean"] = float(ac_post.mean())
        m["post_warmup_active_median"] = float(np.median(ac_post))
        m["post_warmup_noc0_idle_pct"] = (n0_post == 0).sum() / len(n0_post) * 100
        m["post_warmup_noc1_idle_pct"] = (n1_post == 0).sum() / len(n1_post) * 100
        combined4 = (ac_post <= 16) & (n0_post == 0)
        combined8 = (ac_post <= 12) & (n0_post == 0)
        m["post_warmup_idle4_noc0_zero_pct"] = combined4.sum() / len(ac_post) * 100
        m["post_warmup_idle8_noc0_zero_pct"] = combined8.sum() / len(ac_post) * 100
    else:
        m["post_warmup_active_mean"] = 0
        m["post_warmup_active_median"] = 0
        m["post_warmup_noc0_idle_pct"] = 0
        m["post_warmup_noc1_idle_pct"] = 0
        m["post_warmup_idle4_noc0_zero_pct"] = 0
        m["post_warmup_idle8_noc0_zero_pct"] = 0

    # 6. op별 바이트 top 10
    real_ops_sorted = sorted(
        [o for o in ops_rows if not o["is_boundary"]],
        key=lambda o: -(o["noc0_bytes_real"] + o["noc1_bytes_real"]),
    )[:10]
    m["top_ops"] = [
        {"op_id": o["op_id"], "op_type": o["op_type"], "duration_us": o["duration_us"],
         "noc0_bytes": o["noc0_bytes_real"], "noc1_bytes": o["noc1_bytes_real"]}
        for o in real_ops_sorted
    ]

    # 7. op overlap 확인 결과 (이미 순차 확인됨, 문서화)
    m["op_overlap"] = "sequential (no overlap, verified from profile_log_device.csv)"

    return m


def write_summary(m, ts_real, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ResNet-50 on Eagle-N A0 — 시간축 요약\n\n")
        f.write(f"- Total simulated (device-time): {ts_real['n_bins'] * ts_real['bin_us']:.1f} μs\n")
        f.write(f"- Bin size: {ts_real['bin_us']:.3f} μs  ({ts_real['n_bins']} bins)\n")
        f.write(f"- Clock: {CHIP_FREQ_MHZ} MHz\n\n")

        f.write("## 1. NOC 0 : NOC 1 총 바이트 비 (real ops)\n\n")
        f.write(f"- **NOC 0: {m['noc0_share']:.1f}%**\n")
        f.write(f"- **NOC 1: {m['noc1_share']:.1f}%**\n")
        f.write(f"- 비대칭: **NOC 1 이 {m['noc1_total']/max(m['noc0_total'],1):.1f}× 지배**\n\n")

        f.write("## 2. 채널별 완전 유휴 시간\n\n")
        f.write(f"- NOC 0 rate == 0: **{m['noc0_idle_pct']:.1f}%** of run\n")
        f.write(f"- NOC 1 rate == 0: {m['noc1_idle_pct']:.1f}% of run\n\n")

        f.write("## 3. 활성 compute 코어 분포 (of 20)\n\n")
        f.write(f"- mean {m['active_mean']:.1f}, median {m['active_median']:.0f}, "
                f"min {m['active_min']}, max {m['active_max']}\n\n")

        f.write("## 4. 관측 가능 창 (핵심 지표)\n\n")
        f.write("동시 조건: **여유 코어 ≥ N** AND **NOC 0 완전 유휴**\n\n")
        f.write(f"- 여유 코어 ≥ 4 AND NOC 0 == 0: **{m['idle4_noc0_zero_pct']:.1f}%** of time\n")
        f.write(f"- 여유 코어 ≥ 8 AND NOC 0 == 0: **{m['idle8_noc0_zero_pct']:.1f}%** of time\n\n")

        f.write("## 5. 최대 링크 rate 분위수 (real ops)\n\n")
        f.write("| NoC | p50 (B/μs) | p95 (B/μs) | max (B/μs) |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| NOC 0 | {m['noc0_max_link_p50']:.3f} | "
                f"{m['noc0_max_link_p95']:.3f} | {m['noc0_max_link_max']:.3f} |\n")
        f.write(f"| NOC 1 | {m['noc1_max_link_p50']:.3f} | "
                f"{m['noc1_max_link_p95']:.3f} | {m['noc1_max_link_max']:.3f} |\n\n")

        # 정합성 검증
        f.write(f"**정합성 검증**: NOC0 max_link==0 pct **{m['noc0_max_zero_pct']:.2f}%**, "
                f"NOC0 total==0 pct **{m['noc0_total_zero_pct']:.2f}%** → "
                f"{'일치' if abs(m['noc0_max_zero_pct']-m['noc0_total_zero_pct']) < 0.01 else '불일치'}\n\n")

        # active-only p50/p95 (0 제외)
        f.write("### 활성 bin 만 (rate > 0) 분위수\n\n")
        f.write("| NoC | active p50 | active p95 | active count |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| NOC 0 | {m['noc0_active_p50']:.3f} | "
                f"{m['noc0_active_p95']:.3f} | {m['noc0_active_n']} |\n")
        f.write(f"| NOC 1 | {m['noc1_active_p50']:.3f} | "
                f"{m['noc1_active_p95']:.3f} | {m['noc1_active_n']} |\n\n")

        f.write("## 6. op 별 바이트 top 10 (real ops)\n\n")
        f.write("| # | op_type | duration μs | NOC0 bytes | NOC1 bytes | op_id |\n")
        f.write("|---|---|---:|---:|---:|---|\n")
        for i, o in enumerate(m["top_ops"], 1):
            f.write(f"| {i} | {o['op_type']} | {o['duration_us']:.2f} | "
                    f"{o['noc0_bytes']:,} | {o['noc1_bytes']:,} | `{o['op_id']}` |\n")

        f.write("\n## 7. op 겹침 여부\n\n")
        f.write(f"- {m['op_overlap']}\n")
        f.write(f"- 각 stage 내 op 은 FastDispatch 순차 실행. Stage 간 host-gap 은 "
                f"device-compressed timeline 에선 무시됨.\n\n")

        f.write("## 8. Warmup 제외 통계 (t > 300 μs)\n\n")
        f.write("초기 300 μs (stem broadcast + weight 로딩) 을 제외한 정상 실행 구간:\n\n")
        f.write(f"- 평균 활성 코어: **{m['post_warmup_active_mean']:.1f}** "
                f"(median {m['post_warmup_active_median']:.0f})\n")
        f.write(f"- NOC 0 완전 유휴: **{m['post_warmup_noc0_idle_pct']:.1f}%**\n")
        f.write(f"- NOC 1 완전 유휴: {m['post_warmup_noc1_idle_pct']:.1f}%\n")
        f.write(f"- 관측 가능 창 (여유≥4 & NOC0 유휴): "
                f"**{m['post_warmup_idle4_noc0_zero_pct']:.1f}%**\n")
        f.write(f"- 관측 가능 창 (여유≥8 & NOC0 유휴): "
                f"**{m['post_warmup_idle8_noc0_zero_pct']:.1f}%**\n\n")

        f.write("## 좌표 신뢰도 주의\n\n")
        f.write("- 좌표는 logical (translated). 물리 배치와 1:1 대응 아님.\n")
        f.write("- 주 분석은 compute↔compute (real). y=3 sink 및 x=5 관련은 boundary "
                "라벨로 분리.\n")
        f.write("- 절대 대역폭 (GB/s) 은 미확보. bytes/μs 원단위 유지.\n")

    return


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    p.add_argument("--bin-us", type=float, default=None,
                    help="If None: min(1.0, shortest_op_us/10)")
    args = p.parse_args()

    tw_dir = args.out / "time_window"
    tw_dir.mkdir(exist_ok=True)

    per_txn = load_per_txn(args.out / "per_txn.csv")
    per_op = load_per_op(args.out / "per_op.csv")

    # ---- ops.csv (2.1)
    ops_rows, op_time = build_ops_csv(per_op, per_txn, tw_dir / "ops.csv")
    total_us = max(o["t_end_us"] for o in ops_rows)
    print(f"[saved] ops.csv  ({len(ops_rows)} ops, total {total_us:.1f} μs)")

    # ---- decide bin size
    if args.bin_us is None:
        shortest_op_us = min(o["duration_us"] for o in ops_rows
                              if o["duration_us"] > 0)
        args.bin_us = min(1.0, shortest_op_us / 10)
    print(f"[bin]  {args.bin_us:.3f} μs / bin  ({int(total_us / args.bin_us)} bins)")

    # ---- time series (real & total)
    print("[build] real time series...")
    ts_real = build_timeseries(per_txn, op_time, args.bin_us, total_us,
                                class_filter="real")
    print("[build] total time series...")
    ts_total = build_timeseries(per_txn, op_time, args.bin_us, total_us,
                                 class_filter="total")

    # ---- 2.4  timeline.csv
    write_timeline_csv(ts_real, ts_total, ops_rows, op_time,
                       tw_dir / "timeline.csv")
    print(f"[saved] timeline.csv  ({ts_real['n_bins']} rows)")

    # ---- 3.1  fig_timeline.png
    draw_fig_timeline(ts_real, ts_total, ops_rows,
                      tw_dir / "fig_timeline.png", total_us)
    print(f"[saved] fig_timeline.png")

    # 중간 보고 지점 — 여기서 STOP 여부는 caller 가 결정
    # 이후 단계 (link_timeseries, heatmap, cdf, scatter, summary) 는
    # 별도 script 또는 이 script 확장으로 진행

    # ---- 2.2  link_timeseries.csv
    write_link_timeseries_csv(ts_real, ts_total, tw_dir / "link_timeseries.csv")
    print(f"[saved] link_timeseries.csv")

    # ---- 2.3  niu_timeseries.csv
    write_niu_timeseries_csv(ts_real, ts_total, tw_dir / "niu_timeseries.csv")
    print(f"[saved] niu_timeseries.csv")

    # ---- 3.2 fig_link_heatmap.png
    heat_meta = draw_fig_link_heatmap(ts_real, tw_dir / "fig_link_heatmap.png")
    print(f"[saved] fig_link_heatmap.png")

    # ---- 3.3 fig_cdf.png
    draw_fig_cdf(ts_real, tw_dir / "fig_cdf.png")
    print(f"[saved] fig_cdf.png")

    # ---- 3.4 fig_op_scatter.png
    draw_fig_op_scatter(ops_rows, tw_dir / "fig_op_scatter.png")
    print(f"[saved] fig_op_scatter.png")

    # ---- Section 4  summary.md
    metrics = compute_derived_metrics(ts_real, ts_total, ops_rows)
    write_summary(metrics, ts_real, tw_dir / "summary.md")
    print(f"[saved] summary.md")


if __name__ == "__main__":
    main()
