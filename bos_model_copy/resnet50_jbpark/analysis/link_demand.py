"""Per-link byte demand aggregator.

per_txn.csv 를 훑어 각 txn 마다 A0 6×5 격자에서 route 시뮬 → 홉별 bytes 누적.

산출물:
  outputs/link_demand.csv   — 링크별 총 bytes (real / all / real_noc0 / real_noc1 등)
  outputs/link_demand_by_op.csv  — (op, link) 매트릭스 (drill-down)

Semantic note:
  READ 트랜잭션에서 실제 데이터 방향은 (dx,dy)→(sx,sy) 지만, hop-level 링크 점유는
  request path + response path 양쪽 모두 발생. 우리 시뮬은 request path 만 트래킹
  (Tracy num_bytes 는 payload size 인데 sx→dx 방향에 attribute 됨).
  즉 READ 든 WRITE 든 우리 link demand 는 sx→dx 방향 홉 경로에 bytes 를 누적.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from route_simulator import (
    GRID_COLS, GRID_ROWS, ALL_LINK_TYPES,
    route, iter_all_links,
)


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


def is_in_grid(x: int, y: int) -> bool:
    return 0 <= x < GRID_COLS and 0 <= y < GRID_ROWS


def aggregate(txns: list[dict]) -> dict:
    """Returns link demand + NIU demand.

    NIU = Network Interface Unit. 셀 당 4 type:
      NOC0_SRC: 이 셀이 NOC 0 으로 injection
      NOC0_SINK: 이 셀이 NOC 0 으로부터 absorb
      NOC1_SRC / NOC1_SINK: 같은 것 NOC 1
    """
    lb_real: dict[tuple, int] = defaultdict(int)
    lb_all: dict[tuple, int] = defaultdict(int)
    lb_count: dict[tuple, int] = defaultdict(int)
    per_op: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))

    # NIU demand: (row, col, niu_type) -> bytes
    niu_real: dict[tuple, int] = defaultdict(int)
    niu_all: dict[tuple, int] = defaultdict(int)
    per_op_niu: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))

    skipped_grid = 0
    skipped_multicast = 0
    total = 0

    for tx in txns:
        total += 1
        if not (is_in_grid(tx["sx"], tx["sy"]) and is_in_grid(tx["dx"], tx["dy"])):
            skipped_grid += 1
            continue

        if tx["type"] == "WRITE_MULTICAST":
            skipped_multicast += 1

        # NIU 태그 결정
        noc = tx["noc"]  # "NOC_0" or "NOC_1"
        if noc == "NOC_0":
            src_niu = "NOC0_SRC"
            sink_niu = "NOC0_SINK"
        elif noc == "NOC_1":
            src_niu = "NOC1_SRC"
            sink_niu = "NOC1_SINK"
        else:
            src_niu = sink_niu = None

        nb = tx["num_bytes"]
        op_key = (tx["stage"], tx["run_host_id"])

        # NIU demand (initiator + target)
        if src_niu:
            k_src = (tx["sy"], tx["sx"], src_niu)
            k_sink = (tx["dy"], tx["dx"], sink_niu)
            niu_all[k_src] += nb
            niu_all[k_sink] += nb
            per_op_niu[op_key][k_src] += nb
            per_op_niu[op_key][k_sink] += nb
            if not tx["is_boundary"]:
                niu_real[k_src] += nb
                niu_real[k_sink] += nb

        # route → link demand
        r = route(tx["sy"], tx["sx"], tx["dy"], tx["dx"], noc=noc)
        if not r:
            continue
        for link in r:
            lb_all[link] += nb
            lb_count[link] += 1
            per_op[op_key][link] += nb
            if not tx["is_boundary"]:
                lb_real[link] += nb

    return {
        "link_bytes_real": dict(lb_real),
        "link_bytes_all": dict(lb_all),
        "link_hop_count": dict(lb_count),
        "per_op_link_bytes": {k: dict(v) for k, v in per_op.items()},
        "niu_bytes_real": dict(niu_real),
        "niu_bytes_all": dict(niu_all),
        "per_op_niu_bytes": {k: dict(v) for k, v in per_op_niu.items()},
        "skipped_out_of_grid": skipped_grid,
        "skipped_multicast_hint": skipped_multicast,
        "total_txns": total,
    }


def write_link_csv(agg: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "from_row", "from_col", "direction",
            "bytes_real", "bytes_all", "hop_count_all",
            "to_row", "to_col",
        ])
        from route_simulator import link_endpoints
        # iterate all possible links so absent ones show as 0
        for link in iter_all_links():
            (fr, fc), (tr, tc) = link_endpoints(link)
            w.writerow([
                fr, fc, link[2],
                agg["link_bytes_real"].get(link, 0),
                agg["link_bytes_all"].get(link, 0),
                agg["link_hop_count"].get(link, 0),
                tr, tc,
            ])


def write_per_op_csv(agg: dict, path: Path) -> None:
    """(stage, rhid, from_row, from_col, direction, bytes) — long format"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "stage", "run_host_id",
            "from_row", "from_col", "direction", "bytes",
        ])
        for (stage, rhid), links in sorted(agg["per_op_link_bytes"].items()):
            for link, b in sorted(links.items()):
                w.writerow([stage, rhid, link[0], link[1], link[2], b])


def write_niu_csv(agg: dict, path: Path) -> None:
    """NIU per-cell demand (aggregate)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "niu_type", "bytes_real", "bytes_all"])
        real = agg["niu_bytes_real"]
        all_ = agg["niu_bytes_all"]
        for key in sorted(set(list(real.keys()) + list(all_.keys()))):
            w.writerow([key[0], key[1], key[2],
                        real.get(key, 0), all_.get(key, 0)])


def write_per_op_niu_csv(agg: dict, path: Path) -> None:
    """(stage, rhid, row, col, niu_type, bytes) — long format."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "run_host_id", "row", "col", "niu_type", "bytes"])
        for (stage, rhid), nius in sorted(agg["per_op_niu_bytes"].items()):
            for key, b in sorted(nius.items()):
                w.writerow([stage, rhid, key[0], key[1], key[2], b])


def print_summary(agg: dict) -> None:
    real = agg["link_bytes_real"]
    total_link_bytes = sum(real.values())
    print(f"total txns processed  : {agg['total_txns']}")
    print(f"skipped (out of grid) : {agg['skipped_out_of_grid']}")
    print(f"multicast hint (included as unicast) : {agg['skipped_multicast_hint']}")
    print(f"active links (real ops)             : {sum(1 for v in real.values() if v > 0)} / 120")
    print(f"total link-bytes (real, MB)         : {total_link_bytes / 1e6:.2f}")
    print()
    print("=== top 15 hottest links (real ops) ===")
    print(f"{'from':>7}   {'dir':<12}  {'bytes_MB':>10}  {'n_hops':>8}")
    ranked = sorted(real.items(), key=lambda kv: -kv[1])[:15]
    for link, b in ranked:
        r, c, d = link
        n = agg["link_hop_count"].get(link, 0)
        print(f"({r},{c})   {d:<12}  {b / 1e6:>10.3f}  {n:>8}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    args = p.parse_args()

    txns = load_per_txn(args.out / "per_txn.csv")
    print(f"[loaded] {len(txns)} txns from per_txn.csv")

    agg = aggregate(txns)
    write_link_csv(agg, args.out / "link_demand.csv")
    write_per_op_csv(agg, args.out / "link_demand_by_op.csv")
    write_niu_csv(agg, args.out / "niu_demand.csv")
    write_per_op_niu_csv(agg, args.out / "niu_demand_by_op.csv")

    print_summary(agg)
    print()
    print(f"[saved] link_demand.csv ({sum(1 for _ in iter_all_links())} rows)")
    print(f"[saved] link_demand_by_op.csv ({sum(len(v) for v in agg['per_op_link_bytes'].values())} rows)")
    print(f"[saved] niu_demand.csv ({len(agg['niu_bytes_all'])} rows)")
    print(f"[saved] niu_demand_by_op.csv ({sum(len(v) for v in agg['per_op_niu_bytes'].values())} rows)")


if __name__ == "__main__":
    main()
