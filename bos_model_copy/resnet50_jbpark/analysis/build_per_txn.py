"""Per-transaction flat table builder.

5 stage 의 noc_trace_ID*_merged.json 을 훑어 개별 트랜잭션 단위로 CSV 저장.
Global cycle = sum(preceding op cycles) + local timestamp.

Boundary op 은 is_boundary=1 로 라벨만 하고 필터링은 하지 않음 (뷰가 알아서).

산출물: outputs/per_txn.csv
컬럼:
    stage, run_host_id, op_name, is_boundary,
    op_global_start, op_cycles,
    sx, sy, dx, dy, num_bytes,
    local_timestamp, global_cycle,
    type, noc, vc, proc
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from boundary_filter import is_boundary
from stage_paths import STAGE_PATHS, validate

# aggregate_per_core.py 의 op-line 파서 재사용
_OP_LINE_RE = re.compile(
    r'TT_DNN_DEVICE_OP:\s*"(?P<name>[^"]+)"'
    r'\s*,\s*(?P<hash>\d+)'
    r'\s*,\s*(?P<dev>\d+)'
    r'\s*,\s*(?P<rhid>\d+)'
)


def parse_tracy_ops_data(path: Path) -> dict[int, str]:
    id_to_name: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _OP_LINE_RE.search(line)
            if m:
                id_to_name.setdefault(int(m.group("rhid")), m.group("name"))
    return id_to_name


def parse_noc_trace_txns(path: Path) -> dict:
    """Op 하나의 raw merged JSON → txn list + zone span."""
    with open(path) as f:
        events = json.load(f)

    run_host_id = None
    txns: list[dict] = []
    zone_starts: list[int] = []
    zone_ends: list[int] = []

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
            txns.append({
                "sx": ev["sx"], "sy": ev["sy"],
                "dx": ev.get("dx", -1), "dy": ev.get("dy", -1),
                "num_bytes": nb,
                "local_timestamp": ev.get("timestamp", -1),
                "type": typ,
                "noc": ev.get("noc", ""),
                "vc": ev.get("vc", -1),
                "proc": ev.get("proc", ""),
            })
        elif ev.get("zone_phase") == "ZONE_START":
            zone_starts.append(ev.get("timestamp", 0))
        elif ev.get("zone_phase") == "ZONE_END":
            zone_ends.append(ev.get("timestamp", 0))

    cycles = (max(zone_ends) - min(zone_starts)) if zone_starts and zone_ends else 0
    return {
        "run_host_id": run_host_id,
        "txns": txns,
        "op_cycles": cycles,
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
        op = parse_noc_trace_txns(fp)
        op["stage"] = stage
        op["op_name"] = id_to_name.get(op["run_host_id"], "UnknownOP")
        op["is_boundary"] = is_boundary(op["op_name"])
        ops.append(op)
    return ops


def assign_global_cycles(all_ops: list[dict]) -> None:
    """Op 을 (stage, run_host_id) 순으로 정렬해 누적 cycle 부여."""
    all_ops.sort(key=lambda o: (o["stage"], o["run_host_id"]))
    running = 0
    for op in all_ops:
        op["op_global_start"] = running
        running += op["op_cycles"]


def write_per_txn_csv(all_ops: list[dict], path: Path) -> tuple[int, int]:
    n_rows = 0
    n_ops = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "stage", "run_host_id", "op_name", "is_boundary",
            "op_global_start", "op_cycles",
            "sx", "sy", "dx", "dy", "num_bytes",
            "local_timestamp", "global_cycle",
            "type", "noc", "vc", "proc",
        ])
        for op in all_ops:
            n_ops += 1
            gs = op["op_global_start"]
            for tx in op["txns"]:
                w.writerow([
                    op["stage"], op["run_host_id"], op["op_name"],
                    int(op["is_boundary"]),
                    gs, op["op_cycles"],
                    tx["sx"], tx["sy"], tx["dx"], tx["dy"], tx["num_bytes"],
                    tx["local_timestamp"], gs + tx["local_timestamp"],
                    tx["type"], tx["noc"], tx["vc"], tx["proc"],
                ])
                n_rows += 1
    return n_ops, n_rows


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
        tot_txns = sum(len(op["txns"]) for op in ops)
        print(f"[load] stage_{stage}: {len(ops):3d} ops  "
              f"({n_real:3d} real, {len(ops) - n_real:2d} boundary)  "
              f"{tot_txns:5d} txns")
        all_ops.extend(ops)

    assign_global_cycles(all_ops)
    total_cycles = all_ops[-1]["op_global_start"] + all_ops[-1]["op_cycles"]
    print(f"\n[timeline] total global cycles: {total_cycles:>10d}  = {total_cycles/650:.1f} μs @ 650 MHz")

    n_ops, n_rows = write_per_txn_csv(all_ops, args.out / "per_txn.csv")
    print(f"[saved] outputs/per_txn.csv  ({n_ops} ops, {n_rows} txn rows)")


if __name__ == "__main__":
    main()
