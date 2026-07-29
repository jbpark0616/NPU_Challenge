"""Op 시간 겹침 여부 확인 — profile_log_device.csv 로부터 실측.

profile_log_device.csv 는 cycles_since_reset 절대값 보유.
Op 별로 min(ZONE_START)/max(ZONE_END) 추출 → 다음 op 와 겹치는지 확인.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from stage_paths import STAGE_PATHS


def parse_stage(stage_path: Path):
    """rhid → (device_start_cycle, device_end_cycle)"""
    p = stage_path / "logs" / "profile_log_device.csv"
    starts: dict[int, list[int]] = defaultdict(list)
    ends: dict[int, list[int]] = defaultdict(list)
    with open(p, encoding="utf-8") as f:
        next(f)
        r = csv.DictReader(f)
        # 필드명에 leading space 있음
        for row in r:
            row = {k.strip(): v for k, v in row.items()}
            try:
                rhid = int(row["run host ID"].strip())
                cyc = int(row["time[cycles since reset]"].strip())
                phase = row["type"].strip()
            except (ValueError, KeyError):
                continue
            if phase == "ZONE_START":
                starts[rhid].append(cyc)
            elif phase == "ZONE_END":
                ends[rhid].append(cyc)
    ops = {}
    for rhid in starts:
        if rhid in ends:
            ops[rhid] = (min(starts[rhid]), max(ends[rhid]))
    return ops


def analyze():
    for stage, path in sorted(STAGE_PATHS.items()):
        ops = parse_stage(path)
        if not ops:
            print(f"stage {stage}: no ops parsed")
            continue
        by_rhid = sorted(ops.items())
        print(f"\n=== stage {stage}: {len(by_rhid)} ops ===")

        overlaps = 0
        gaps = []
        for i in range(len(by_rhid) - 1):
            rhid_a, (s_a, e_a) = by_rhid[i]
            rhid_b, (s_b, e_b) = by_rhid[i + 1]
            gap = s_b - e_a  # positive = sequential with gap, negative = overlap
            gaps.append(gap)
            if gap < 0:
                overlaps += 1

        # 통계
        first_start = by_rhid[0][1][0]
        last_end = by_rhid[-1][1][1]
        total_span = last_end - first_start
        # sum of individual durations
        sum_durations = sum(e - s for _, (s, e) in by_rhid)

        print(f"  first op {by_rhid[0][0]}: cycles [{by_rhid[0][1][0]}, {by_rhid[0][1][1]}]  "
              f"dur={by_rhid[0][1][1] - by_rhid[0][1][0]}")
        print(f"  last op  {by_rhid[-1][0]}: cycles [{by_rhid[-1][1][0]}, {by_rhid[-1][1][1]}]  "
              f"dur={by_rhid[-1][1][1] - by_rhid[-1][1][0]}")
        print(f"  total span:            {total_span:>12,} cycles = {total_span / 650:>10.1f} μs")
        print(f"  sum of op durations:   {sum_durations:>12,} cycles = {sum_durations / 650:>10.1f} μs")
        print(f"  gaps (host dispatch):  {total_span - sum_durations:>12,} cycles = "
              f"{(total_span - sum_durations) / 650:>10.1f} μs")
        print(f"  overlapping op pairs:  {overlaps} / {len(by_rhid) - 1}")

        # gap 통계
        pos_gaps = [g for g in gaps if g >= 0]
        neg_gaps = [g for g in gaps if g < 0]
        if pos_gaps:
            print(f"  gap median: {sorted(pos_gaps)[len(pos_gaps) // 2]:>12,} cycles = "
                  f"{sorted(pos_gaps)[len(pos_gaps) // 2] / 650:>8.1f} μs")
            print(f"  gap max:    {max(pos_gaps):>12,} cycles = {max(pos_gaps) / 650:>8.1f} μs")
            print(f"  gap min:    {min(pos_gaps):>12,} cycles = {min(pos_gaps) / 650:>8.1f} μs")
        if neg_gaps:
            print(f"  ⚠ overlap median: {sorted(neg_gaps)[len(neg_gaps) // 2]:>12,} cycles")


if __name__ == "__main__":
    analyze()
