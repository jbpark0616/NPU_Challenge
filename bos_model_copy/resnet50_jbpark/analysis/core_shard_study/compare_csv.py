"""
Compare two ResNet50 profiler CSV files:
- baseline: ops_perf_results_2026_07_08_05_15_43.csv (BOS default, batch 4, ~10 iterations)
- exp3:     ops_perf_results_resnet50_exp3_2026_07_20_14_20_32.csv (teammate: 16-core forced, batch 1, 1 iteration)

NOTE (2026-07-21): 이 비교는 batch 다르기 때문에 wall time 직접 대조 불가.
차량용 NPU 는 batch=1 이 표준이므로 baseline 도 batch=1 로 다시 뽑아야 정합 비교 가능.
=> 내일 batch=1 baseline 새로 수집 후 재실행.

무엇을 확인했나 (partial):
- baseline 은 CORE COUNT 분포에 20 코어 op 이 25% 존재 → Config.py 의 CORE_SET_14/16 은 reshard 지점만 강제, 개별 op auto-shard 는 20 사용
- exp3 은 batch=1 결과로 auto-picker 가 작은 grid (4, 8, 9) 선택하는 op 도 존재
- 두 파일 모두 "16 코어 강제" 아님. auto-shard heuristic 결과.

Goals:
1. Basic shape: rows, unique op types, batch size inference
2. CORE COUNT distribution: what's the actual core count used per op
3. Total wall time (DEVICE KERNEL DURATION sum)
4. Per-op-type comparison
5. Detect where they diverge (batch? shard scheme? mem layout?)
"""

import csv
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path("C:/NPU_Challenge/bos_model_copy/resnet50/resnet_note")
BASELINE = BASE_DIR / "ops_perf_results_2026_07_08_05_15_43.csv"
EXP3     = BASE_DIR / "ops_perf_results_resnet50_exp3_2026_07_20_14_20_32.csv"


def parse_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def safe_int(s, default=0):
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return default


def analyze(rows, tag):
    print(f"\n{'='*70}")
    print(f"== {tag}   (rows={len(rows)})")
    print("="*70)

    op_types = defaultdict(int)
    for r in rows:
        op_types[r["OP CODE"]] += 1
    print("\n[op counts]")
    for op, n in sorted(op_types.items(), key=lambda x: -x[1]):
        print(f"  {op:40s} {n:>4d}")

    core_counts = defaultdict(int)
    for r in rows:
        c = safe_int(r.get("CORE COUNT", ""))
        core_counts[c] += 1
    print("\n[core count distribution]")
    for c, n in sorted(core_counts.items()):
        print(f"  cores={c:>3d}   ops={n:>4d}")

    total_kernel_ns = 0
    valid_rows = 0
    for r in rows:
        d = safe_int(r.get("DEVICE KERNEL DURATION [ns]", ""))
        if d > 0:
            total_kernel_ns += d
            valid_rows += 1
    print(f"\n[wall time]")
    print(f"  total DEVICE KERNEL DURATION = {total_kernel_ns/1e6:>10.2f} ms   ({valid_rows} valid rows)")

    op_time = defaultdict(lambda: [0, 0])
    for r in rows:
        d = safe_int(r.get("DEVICE KERNEL DURATION [ns]", ""))
        if d > 0:
            op_time[r["OP CODE"]][0] += d
            op_time[r["OP CODE"]][1] += 1
    print("\n[per-op-type wall time]")
    for op, (t, n) in sorted(op_time.items(), key=lambda x: -x[1][0]):
        print(f"  {op:40s} total={t/1e6:>8.3f} ms   count={n:>4d}   avg={t/max(n,1)/1e3:>8.2f} us")

    stem_row = None
    for r in rows:
        if r["OP CODE"] == "HaloDeviceOperation":
            stem_row = r
            break
    if stem_row:
        print(f"\n[stem HaloDeviceOperation attrs (first 500 chars)]")
        attrs = stem_row.get("ATTRIBUTES", "")[:500]
        print(f"  {attrs}")

    return {
        "rows": len(rows),
        "op_types": dict(op_types),
        "core_counts": dict(core_counts),
        "total_ns": total_kernel_ns,
        "op_time": {k: (v[0], v[1]) for k, v in op_time.items()},
    }


def main():
    baseline = parse_csv(BASELINE)
    exp3 = parse_csv(EXP3)

    b = analyze(baseline, "BASELINE (default BOS)")
    e = analyze(exp3, "EXP3 (16-core forced)")

    print("\n\n" + "="*70)
    print("== COMPARISON")
    print("="*70)
    print(f"\n[shape]")
    print(f"  baseline rows: {b['rows']}")
    print(f"  exp3     rows: {e['rows']}")
    print(f"  ratio: {b['rows']/max(e['rows'],1):.2f}x more ops in baseline")

    print(f"\n[wall time]")
    print(f"  baseline total: {b['total_ns']/1e6:>10.3f} ms")
    print(f"  exp3     total: {e['total_ns']/1e6:>10.3f} ms")
    if e['total_ns']:
        print(f"  exp3/baseline = {e['total_ns']/b['total_ns']:.3f}x")

    print(f"\n[core count comparison]")
    print(f"  baseline: {sorted(b['core_counts'].items())}")
    print(f"  exp3    : {sorted(e['core_counts'].items())}")


if __name__ == "__main__":
    main()
