"""ResNet50 stage-split NoC trace per-core aggregation.

5개 stage 의 raw noc_trace_ID*_merged.json 을 훑어서 (op × 물리 코어) 매트릭스를 만들고,
축을 접어 per-core / per-op 여러 뷰로 출력. Boundary artifact op (stage-split 경계에서만
발생하는 layout/sharding 전환) 은 이름 매치로 필터.

산출물 (기본 outputs/ 하위):
  per_core.csv          - 코어 20개 각각의 injected/consumed bytes, NoC별 breakdown, real ops만
  per_core_all.csv      - 위 + boundary 포함 (비교용)
  per_op.csv            - op별 총 traffic (real / boundary 라벨 유지)
  matrix.csv            - (op, 코어) 셀별 injected/consumed 상세 (drill-down)
  per_core_heatmap.png  - 5x4 grid 시각화 (real ops만)
  summary.txt           - 콘솔 요약 저장본
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from boundary_filter import is_boundary
from stage_paths import STAGE_PATHS, validate

# Eagle-N physical NoC grid (실측 관찰: x=0..5, y=0..4). Compute·DRAM·router 등 혼재.
GRID_W, GRID_H = 6, 5

_OP_LINE_RE = re.compile(
    r'TT_DNN_DEVICE_OP:\s*"(?P<name>[^"]+)"'
    r'\s*,\s*(?P<hash>\d+)'
    r'\s*,\s*(?P<dev>\d+)'
    r'\s*,\s*(?P<rhid>\d+)'
)


# ============================================================
# 파싱
# ============================================================
def parse_tracy_ops_data(path: Path) -> dict[int, str]:
    """run_host_id -> op_name"""
    id_to_name: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _OP_LINE_RE.search(line)
            if m:
                id_to_name.setdefault(int(m.group("rhid")), m.group("name"))
    return id_to_name


def parse_noc_trace(path: Path) -> dict:
    """한 op merged JSON -> traffic breakdown."""
    with open(path) as f:
        events = json.load(f)

    run_host_id = None
    bytes_by_src: dict[tuple[int, int], int] = defaultdict(int)
    bytes_by_dst: dict[tuple[int, int], int] = defaultdict(int)
    bytes_by_src_noc: dict[tuple[int, int, str], int] = defaultdict(int)
    bytes_by_dst_noc: dict[tuple[int, int, str], int] = defaultdict(int)
    bytes_by_type: Counter = Counter()
    zone_starts: list[int] = []
    zone_ends: list[int] = []
    txn_count = 0
    total_bytes = 0

    for ev in events:
        rhid = ev.get("run_host_id")
        if rhid is not None:
            run_host_id = rhid
        typ = ev.get("type", "")
        if "num_bytes" in ev and "zone_phase" not in ev:
            if "BARRIER" in typ or "SEMAPHORE" in typ:
                continue
            nb = ev["num_bytes"]
            if nb == 0:
                continue
            sx, sy = ev["sx"], ev["sy"]
            noc = ev.get("noc", "UNKNOWN")
            bytes_by_src[(sx, sy)] += nb
            bytes_by_src_noc[(sx, sy, noc)] += nb
            if "dx" in ev and "dy" in ev:
                dx, dy = ev["dx"], ev["dy"]
                bytes_by_dst[(dx, dy)] += nb
                bytes_by_dst_noc[(dx, dy, noc)] += nb
            bytes_by_type[typ] += nb
            total_bytes += nb
            txn_count += 1
        elif ev.get("zone_phase") == "ZONE_START":
            zone_starts.append(ev.get("timestamp", 0))
        elif ev.get("zone_phase") == "ZONE_END":
            zone_ends.append(ev.get("timestamp", 0))

    cycles = (max(zone_ends) - min(zone_starts)) if zone_starts and zone_ends else 0
    return {
        "run_host_id": run_host_id,
        "txn_count": txn_count,
        "total_bytes": total_bytes,
        "bytes_by_src": dict(bytes_by_src),
        "bytes_by_dst": dict(bytes_by_dst),
        "bytes_by_src_noc": dict(bytes_by_src_noc),
        "bytes_by_dst_noc": dict(bytes_by_dst_noc),
        "bytes_by_type": dict(bytes_by_type),
        "cycles": cycles,
    }


def load_stage(stage: int, path: Path) -> list[dict]:
    logs = path / "logs"
    id_to_name = parse_tracy_ops_data(logs / "tracy_ops_data.csv")
    ops: list[dict] = []
    files = sorted(
        logs.glob("noc_trace_ID*_merged.json"),
        key=lambda p: int(re.search(r"ID(\d+)_", p.name).group(1)),
    )
    for fp in files:
        op = parse_noc_trace(fp)
        op["stage"] = stage
        op["op_name"] = id_to_name.get(op["run_host_id"], "UnknownOP")
        op["is_boundary"] = is_boundary(op["op_name"])
        ops.append(op)
    return ops


# ============================================================
# Aggregation
# ============================================================
def _empty_core_stats() -> dict:
    return {
        "injected": 0,
        "consumed": 0,
        "injected_noc0": 0,
        "injected_noc1": 0,
        "consumed_noc0": 0,
        "consumed_noc1": 0,
        "ops_as_src": 0,
        "ops_as_dst": 0,
        "stages_active": set(),
    }


def aggregate_per_core(all_ops: list[dict], include_boundary: bool) -> dict:
    per_core: dict[tuple[int, int], dict] = defaultdict(_empty_core_stats)
    for op in all_ops:
        if op["is_boundary"] and not include_boundary:
            continue
        for coord, b in op["bytes_by_src"].items():
            per_core[coord]["injected"] += b
        for coord, b in op["bytes_by_dst"].items():
            per_core[coord]["consumed"] += b
        for (sx, sy, noc), b in op["bytes_by_src_noc"].items():
            if noc == "NOC_0":
                per_core[(sx, sy)]["injected_noc0"] += b
            elif noc == "NOC_1":
                per_core[(sx, sy)]["injected_noc1"] += b
        for (dx, dy, noc), b in op["bytes_by_dst_noc"].items():
            if noc == "NOC_0":
                per_core[(dx, dy)]["consumed_noc0"] += b
            elif noc == "NOC_1":
                per_core[(dx, dy)]["consumed_noc1"] += b
        for coord in op["bytes_by_src"]:
            per_core[coord]["ops_as_src"] += 1
            per_core[coord]["stages_active"].add(op["stage"])
        for coord in op["bytes_by_dst"]:
            per_core[coord]["ops_as_dst"] += 1
            per_core[coord]["stages_active"].add(op["stage"])
    return dict(per_core)


# ============================================================
# Writers
# ============================================================
def write_per_core_csv(per_core: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sx", "sy",
            "injected_bytes", "consumed_bytes",
            "injected_noc0", "injected_noc1",
            "consumed_noc0", "consumed_noc1",
            "ops_as_src", "ops_as_dst",
            "stages_active",
        ])
        for (sx, sy), d in sorted(per_core.items()):
            w.writerow([
                sx, sy,
                d["injected"], d["consumed"],
                d["injected_noc0"], d["injected_noc1"],
                d["consumed_noc0"], d["consumed_noc1"],
                d["ops_as_src"], d["ops_as_dst"],
                len(d["stages_active"]),
            ])


def write_per_op_csv(all_ops: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "stage", "run_host_id", "op_name", "is_boundary",
            "txn_count", "total_bytes", "cycles",
        ])
        for op in sorted(all_ops, key=lambda o: (o["stage"], o["run_host_id"])):
            w.writerow([
                op["stage"], op["run_host_id"], op["op_name"],
                int(op["is_boundary"]),
                op["txn_count"], op["total_bytes"], op["cycles"],
            ])


def write_matrix_csv(all_ops: list[dict], path: Path) -> None:
    """long format: (stage, rhid, op_name, boundary, sx, sy, injected, consumed)"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "stage", "run_host_id", "op_name", "is_boundary",
            "sx", "sy", "injected_bytes", "consumed_bytes",
        ])
        for op in sorted(all_ops, key=lambda o: (o["stage"], o["run_host_id"])):
            coords = set(op["bytes_by_src"]) | set(op["bytes_by_dst"])
            for coord in sorted(coords):
                w.writerow([
                    op["stage"], op["run_host_id"], op["op_name"],
                    int(op["is_boundary"]),
                    coord[0], coord[1],
                    op["bytes_by_src"].get(coord, 0),
                    op["bytes_by_dst"].get(coord, 0),
                ])


def draw_heatmap(per_core: dict, path: Path, title_suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    inj = np.zeros((GRID_H, GRID_W))
    con = np.zeros((GRID_H, GRID_W))
    observed = np.zeros((GRID_H, GRID_W), dtype=bool)
    for (sx, sy), d in per_core.items():
        if 0 <= sx < GRID_W and 0 <= sy < GRID_H:
            inj[sy, sx] = d["injected"] / 1e6
            con[sy, sx] = d["consumed"] / 1e6
            observed[sy, sx] = True

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    for ax, arr, name in [(ax1, inj, "Injected"), (ax2, con, "Consumed")]:
        im = ax.imshow(arr, cmap="YlOrRd", origin="lower", aspect="auto")
        ax.set_title(f"{name} NoC bytes (MB){title_suffix}\n"
                     f"Eagle-N A0 raw NoC trace coords (roles not resolved)")
        ax.set_xlabel("sx (translated col)")
        ax.set_ylabel("sy (translated row)")
        ax.set_xticks(range(GRID_W))
        ax.set_yticks(range(GRID_H))
        vmax = arr.max() if arr.max() > 0 else 1.0
        for i in range(GRID_H):
            for j in range(GRID_W):
                v = arr[i, j]
                if not observed[i, j]:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=10, color="gray")
                    continue
                color = "black" if v < vmax * 0.55 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ============================================================
# Summary
# ============================================================
def summarize(all_ops: list[dict], per_core_real: dict, per_core_all: dict) -> list[str]:
    lines: list[str] = []
    n_real = sum(1 for op in all_ops if not op["is_boundary"])
    n_bnd = len(all_ops) - n_real
    total_bytes_real = sum(op["total_bytes"] for op in all_ops if not op["is_boundary"])
    total_bytes_all = sum(op["total_bytes"] for op in all_ops)
    total_cyc_real = sum(op["cycles"] for op in all_ops if not op["is_boundary"])
    total_cyc_all = sum(op["cycles"] for op in all_ops)

    lines.append(f"=== ResNet50 stage-split aggregation ===")
    lines.append(f"total ops               : {len(all_ops)}  ({n_real} real + {n_bnd} boundary)")
    lines.append(f"total NoC bytes (real)  : {total_bytes_real / 1e6:8.2f} MB")
    lines.append(f"total NoC bytes (all)   : {total_bytes_all / 1e6:8.2f} MB")
    lines.append(f"total cycles (real)     : {total_cyc_real:>10d}  = {total_cyc_real / 650:.1f} μs (@650MHz)")
    lines.append(f"total cycles (all)      : {total_cyc_all:>10d}  = {total_cyc_all / 650:.1f} μs")
    lines.append(f"boundary overhead       : bytes {(1 - total_bytes_real / total_bytes_all) * 100:.1f}%, "
                 f"cycles {(1 - total_cyc_real / total_cyc_all) * 100:.1f}%")
    lines.append("")

    # Per-stage
    by_stage_real: Counter = Counter()
    by_stage_bytes: dict[int, int] = defaultdict(int)
    by_stage_cyc: dict[int, int] = defaultdict(int)
    for op in all_ops:
        if op["is_boundary"]:
            continue
        by_stage_real[op["stage"]] += 1
        by_stage_bytes[op["stage"]] += op["total_bytes"]
        by_stage_cyc[op["stage"]] += op["cycles"]
    lines.append("=== per stage (real ops only) ===")
    lines.append(f"{'stage':>6} {'ops':>5} {'bytes_MB':>10} {'cycles':>10} {'μs':>8}")
    for st in sorted(by_stage_real):
        lines.append(f"{st:>6} {by_stage_real[st]:>5} "
                     f"{by_stage_bytes[st] / 1e6:>10.2f} "
                     f"{by_stage_cyc[st]:>10} "
                     f"{by_stage_cyc[st] / 650:>8.1f}")
    lines.append("")

    # Per-core ranking
    lines.append("=== per-core injected ranking (real ops only) ===")
    lines.append(f"{'coord':>8} {'inj_MB':>8} {'con_MB':>8} {'noc0_in':>10} {'noc1_in':>10} {'ops_src':>8} {'stages':>7}")
    ranked = sorted(per_core_real.items(), key=lambda kv: -kv[1]["injected"])
    for (sx, sy), d in ranked[:25]:
        lines.append(f"  ({sx},{sy})  {d['injected'] / 1e6:>7.2f}  {d['consumed'] / 1e6:>7.2f}  "
                     f"{d['injected_noc0'] / 1e6:>9.2f}  {d['injected_noc1'] / 1e6:>9.2f}  "
                     f"{d['ops_as_src']:>7}  {len(d['stages_active']):>6}")
    lines.append("")

    # Top ops
    lines.append("=== top 10 real ops by NoC bytes ===")
    lines.append(f"{'stage':>5} {'rhid':>5} {'op_name':<40} {'bytes_KB':>10} {'cycles':>10} {'μs':>7}")
    ranked_ops = sorted(
        [op for op in all_ops if not op["is_boundary"]],
        key=lambda op: -op["total_bytes"],
    )
    for op in ranked_ops[:10]:
        lines.append(f"{op['stage']:>5} {op['run_host_id']:>5} {op['op_name']:<40} "
                     f"{op['total_bytes'] / 1024:>10.1f} "
                     f"{op['cycles']:>10} "
                     f"{op['cycles'] / 650:>7.1f}")
    return lines


# ============================================================
# Main
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    args = p.parse_args()

    validate()
    args.out.mkdir(parents=True, exist_ok=True)

    all_ops: list[dict] = []
    for stage, path in sorted(STAGE_PATHS.items()):
        ops = load_stage(stage, path)
        n_real = sum(1 for op in ops if not op["is_boundary"])
        print(f"[load] stage_{stage}: {len(ops):3d} ops  ({n_real:3d} real, "
              f"{len(ops) - n_real:2d} boundary)")
        all_ops.extend(ops)
    print()

    per_core_real = aggregate_per_core(all_ops, include_boundary=False)
    per_core_all = aggregate_per_core(all_ops, include_boundary=True)

    write_per_core_csv(per_core_real, args.out / "per_core.csv")
    write_per_core_csv(per_core_all, args.out / "per_core_all.csv")
    write_per_op_csv(all_ops, args.out / "per_op.csv")
    write_matrix_csv(all_ops, args.out / "matrix.csv")
    draw_heatmap(per_core_real, args.out / "per_core_heatmap.png", " — real ops")
    draw_heatmap(per_core_all, args.out / "per_core_heatmap_all.png", " — incl. boundary")

    summary = summarize(all_ops, per_core_real, per_core_all)
    text = "\n".join(summary)
    print(text)
    (args.out / "summary.txt").write_text(text, encoding="utf-8")

    print(f"\n[saved] outputs → {args.out}/")


if __name__ == "__main__":
    main()
