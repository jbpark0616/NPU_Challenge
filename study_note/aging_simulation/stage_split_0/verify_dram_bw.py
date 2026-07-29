"""
Stage-split noc_trace_ID*.json 을 파싱해서 op 별 실측 NoC / DRAM 트래픽을 계산.

목적:
- CSV `bytes_total`, `bytes_dram` (padded logical shape 기반) 이 실제 NoC 트래픽과 일치하는지 검증
- Achieved DRAM BW per op 실측 (GB/s)
- Roofline peak 재검증의 근거

DRAM core 식별 (heuristic):
- topology.json 에 좌표 없음 → 트랜잭션에서 empirical 하게 잡음
- 각 (sx, sy) 별 총 sourced bytes 랭킹 → 상위 3개 (dram_grid_size = 3×1) 를 DRAM 후보로

사용법:
    python verify_dram_bw.py <stage_folder>
    예: python verify_dram_bw.py stage_0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# 1. NoC trace 파싱
# ============================================================
def parse_noc_trace(path: Path) -> dict:
    """
    한 op 의 noc_trace_ID*_merged.json 을 파싱.
    반환: {
        'run_host_id': int,
        'txn_count': int,
        'total_bytes': int,
        'bytes_by_src': {(sx,sy): int},
        'bytes_by_dst': {(dx,dy): int},
        'bytes_by_type': {type: int},
        'bytes_by_noc': {noc: int},
        'kernel_start_cycle': int,
        'kernel_end_cycle': int,
    }
    """
    with open(path) as f:
        events = json.load(f)

    run_host_id = None
    txns = []
    zone_starts, zone_ends = [], []

    for ev in events:
        rhid = ev.get('run_host_id')
        if rhid is not None:
            run_host_id = rhid
        # 트랜잭션은 num_bytes 필드가 있고 zone 이 아닌 것
        if 'num_bytes' in ev and 'zone_phase' not in ev:
            txns.append(ev)
        elif ev.get('zone_phase') == 'ZONE_START':
            zone_starts.append(ev.get('timestamp', 0))
        elif ev.get('zone_phase') == 'ZONE_END':
            zone_ends.append(ev.get('timestamp', 0))

    bytes_by_src = defaultdict(int)
    bytes_by_dst = defaultdict(int)
    bytes_by_type = Counter()
    bytes_by_noc = Counter()
    total_bytes = 0
    for t in txns:
        nb = t.get('num_bytes', 0)
        # BARRIER 등 실제 데이터 이동 아닌 것 제외 (num_bytes 있어도)
        typ = t.get('type', '')
        if 'BARRIER' in typ or 'SEMAPHORE' in typ:
            continue
        total_bytes += nb
        bytes_by_src[(t['sx'], t['sy'])] += nb
        if 'dx' in t and 'dy' in t:
            bytes_by_dst[(t['dx'], t['dy'])] += nb
        bytes_by_type[typ] += nb
        if 'noc' in t:
            bytes_by_noc[t['noc']] += nb

    return {
        'run_host_id': run_host_id,
        'txn_count': len(txns),
        'total_bytes': total_bytes,
        'bytes_by_src': dict(bytes_by_src),
        'bytes_by_dst': dict(bytes_by_dst),
        'bytes_by_type': dict(bytes_by_type),
        'bytes_by_noc': dict(bytes_by_noc),
        'kernel_start_cycle': min(zone_starts) if zone_starts else 0,
        'kernel_end_cycle': max(zone_ends) if zone_ends else 0,
    }


# ============================================================
# 2. Tracy ops_data.csv 파싱 (run_host_id → op_name)
# ============================================================
# tracy_ops_data.csv 실제 형식:
#   `TT_DNN_DEVICE_OP: "HaloDeviceOperation", 16234281905634118301, 0, 1024 ->
# 4 필드: op_name, op_hash, device_id, run_host_id
_OP_LINE_RE = re.compile(
    r'TT_DNN_DEVICE_OP:\s*"(?P<name>[^"]+)"'
    r'\s*,\s*(?P<hash>\d+)'
    r'\s*,\s*(?P<dev>\d+)'
    r'\s*,\s*(?P<rhid>\d+)'
)


def parse_tracy_ops_data(path: Path) -> dict:
    """
    tracy_ops_data.csv 에서 run_host_id → op 이름 매핑 추출.
    """
    id_to_name = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = _OP_LINE_RE.search(line)
            if not m:
                continue
            op_name = m.group('name')
            run_host_id = int(m.group('rhid'))
            id_to_name.setdefault(run_host_id, op_name)
    return id_to_name


# ============================================================
# 3. DRAM core 식별 (heuristic)
# ============================================================
def identify_dram_cores(all_ops: list[dict], top_n: int = 3) -> set:
    """
    모든 op 의 src coord 별 총 sourced bytes 를 랭킹.
    상위 top_n 개 (dram_grid_size = 3×1 이니 3) 를 DRAM 후보로 반환.

    가정: DRAM core 는 워크로드 전반에서 큰 양의 데이터를 source 함 (weight/activation).
    """
    global_bytes_by_src = defaultdict(int)
    for op in all_ops:
        for coord, b in op['bytes_by_src'].items():
            global_bytes_by_src[coord] += b

    ranked = sorted(global_bytes_by_src.items(), key=lambda kv: -kv[1])
    dram_candidates = set(coord for coord, _ in ranked[:top_n])
    return dram_candidates, ranked


# ============================================================
# 4. 메인
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage_folder', type=Path,
                   help='stage_N/ 폴더 경로 (그 안에 logs/ 있음)')
    p.add_argument('--top-dram', type=int, default=3,
                   help='DRAM 후보 좌표 상위 몇 개 (기본 3 = dram_grid_size)')
    p.add_argument('--freq-mhz', type=float, default=650.0,
                   help='Blackhole 클럭 (기본 650 MHz, cycle → ns 변환)')
    args = p.parse_args()

    logs = args.stage_folder / 'logs'
    if not logs.is_dir():
        print(f"ERROR: {logs} 없음. stage_N/ 폴더를 지정하세요.")
        sys.exit(1)

    # noc trace 파일 로드
    noc_files = sorted(logs.glob('noc_trace_ID*_merged.json'))
    print(f"[parse] {len(noc_files)} noc_trace files")
    all_ops = []
    for fp in noc_files:
        try:
            op = parse_noc_trace(fp)
            op['file'] = fp.name
            all_ops.append(op)
        except Exception as e:
            print(f"  skip {fp.name}: {e}")

    # tracy_ops_data.csv → op 이름
    ops_data_csv = logs / 'tracy_ops_data.csv'
    id_to_name = {}
    if ops_data_csv.is_file():
        id_to_name = parse_tracy_ops_data(ops_data_csv)
        print(f"[parse] tracy_ops_data.csv → {len(id_to_name)} unique run_host_id 매핑")
    else:
        print(f"[warn] tracy_ops_data.csv 없음 — op 이름 매핑 스킵")

    # DRAM core 식별
    dram_coords, ranked_srcs = identify_dram_cores(all_ops, top_n=args.top_dram)
    print(f"\n[DRAM candidate coords] (top {args.top_dram} by total sourced bytes)")
    for coord, b in ranked_srcs[:10]:
        mark = ' ★ DRAM candidate' if coord in dram_coords else ''
        print(f"  {coord}: {b/1e6:8.2f} MB{mark}")

    # 각 op 별 분석
    print(f"\n[per-op traffic breakdown]")
    print(f"{'op#':>3} {'run_host_id':>11} {'op_name':<40} "
          f"{'txn':>5} {'total_KB':>10} {'dram_KB':>10} {'dram_%':>7} "
          f"{'cycles':>10} {'GBps_dram':>10}")
    print('-' * 130)

    total_all = 0
    total_dram = 0
    for i, op in enumerate(all_ops):
        rhid = op['run_host_id']
        name = id_to_name.get(rhid, '(unknown)')[:40]

        # DRAM 관련 트래픽: src OR dst 가 DRAM 좌표
        dram_bytes = 0
        for coord, b in op['bytes_by_src'].items():
            if coord in dram_coords:
                dram_bytes += b
        for coord, b in op['bytes_by_dst'].items():
            if coord in dram_coords:
                dram_bytes += b

        cycles = op['kernel_end_cycle'] - op['kernel_start_cycle']
        # cycles → ns → GB/s
        if cycles > 0:
            time_ns = cycles * 1000.0 / args.freq_mhz    # ns
            gbps_dram = dram_bytes / time_ns              # bytes/ns == GB/s
        else:
            time_ns = 0
            gbps_dram = 0.0

        total_all += op['total_bytes']
        total_dram += dram_bytes

        pct_dram = dram_bytes / op['total_bytes'] * 100 if op['total_bytes'] else 0
        print(f"{i:>3} {rhid:>11} {name:<40} "
              f"{op['txn_count']:>5} {op['total_bytes']/1024:>10.1f} "
              f"{dram_bytes/1024:>10.1f} {pct_dram:>6.1f}% "
              f"{cycles:>10} {gbps_dram:>10.2f}")

    print('-' * 130)
    print(f"{'TOTAL':>15}{'':>40} "
          f"{'':>5} {total_all/1024:>10.1f} {total_dram/1024:>10.1f} "
          f"{total_dram/total_all*100 if total_all else 0:>6.1f}%")

    # 트랜잭션 타입 · NoC 별 breakdown (전체 op 총합)
    # NOTE: DRAM 특화 breakdown 은 raw 재파싱 필요 (bytes_by_type/noc 은 op 총합만 담고 있음)
    print(f"\n[전체 트래픽 타입 breakdown]")
    total_by_type = Counter()
    total_by_noc = Counter()
    for op in all_ops:
        for typ, b in op['bytes_by_type'].items():
            total_by_type[typ] += b
        for noc, b in op['bytes_by_noc'].items():
            total_by_noc[noc] += b

    print("  by type:")
    for typ, b in total_by_type.most_common():
        print(f"    {typ:<25} {b/1e6:8.2f} MB")
    print("  by noc:")
    for noc, b in total_by_noc.most_common():
        print(f"    {noc:<25} {b/1e6:8.2f} MB")


if __name__ == '__main__':
    main()
