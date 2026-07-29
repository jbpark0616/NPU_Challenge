"""summary.md, timeline.csv, fig_cdf.png 3-way 정합성 검증.

방법: timeline.csv 에서 직접 계산 → summary.md 텍스트 파싱 값과 대조.
CDF 는 timeline.csv 로부터 파생되므로 timeline == CDF 는 자동 보장. summary 만 비교.
"""
from __future__ import annotations
import csv
import re
from pathlib import Path
import numpy as np

OUT = Path("outputs/time_window")


def load_timeline():
    rows = []
    with open(OUT / "timeline.csv", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return {
        "t_us": np.array([float(r["t_us"]) for r in rows]),
        "n0_total": np.array([float(r["noc0_total_rate_real"]) for r in rows]),
        "n1_total": np.array([float(r["noc1_total_rate_real"]) for r in rows]),
        "n0_max": np.array([float(r["noc0_max_link_rate_real"]) for r in rows]),
        "n1_max": np.array([float(r["noc1_max_link_rate_real"]) for r in rows]),
        "ac": np.array([int(r["n_active_cores"]) for r in rows]),
    }


def parse_summary():
    txt = (OUT / "summary.md").read_text(encoding="utf-8")
    def grab(pattern):
        m = re.search(pattern, txt)
        return float(m.group(1).replace(",", "")) if m else None

    # NOC 0/1 share 문자열: **NOC 0: X.X%**
    def grab_star(pattern):
        m = re.search(pattern, txt)
        return float(m.group(1)) if m else None

    return {
        "noc0_share": grab_star(r"\*\*NOC 0: ([\d.]+)%\*\*"),
        "noc1_share": grab_star(r"\*\*NOC 1: ([\d.]+)%\*\*"),
        "noc0_idle": grab(r"NOC 0 rate == 0: \*\*([\d.]+)%"),
        "noc1_idle": grab(r"NOC 1 rate == 0: ([\d.]+)%"),
        "active_mean": grab(r"mean ([\d.]+), median"),
        "active_max": grab(r"max (\d+)"),
        "obs_4": grab(r"NOC 0 == 0: \*\*([\d.]+)%\*\* of time"),
        "obs_8": grab(r"≥ 8 AND NOC 0 == 0: \*\*([\d.]+)%"),
        "n0_p50": grab(r"NOC 0 \| ([\d.]+) \|"),
        "n0_p95": None,  # 두 번째 그룹 별도 파싱
        "n1_p50": grab(r"NOC 1 \| ([\d.]+) \|"),
        "n0_max_link_max": None,
        "post_active_mean": grab(r"평균 활성 코어: \*\*([\d.]+)\*\*"),
        "post_noc0_idle": grab(r"NOC 0 완전 유휴: \*\*([\d.]+)%"),
        "post_obs_4": grab(r"관측 가능 창 \(여유≥4 & NOC0 유휴\): \*\*([\d.]+)%"),
        "post_obs_8": grab(r"관측 가능 창 \(여유≥8 & NOC0 유휴\): \*\*([\d.]+)%"),
        "consistency_pass": "일치" in txt,
    }


def compute_from_timeline(t):
    n0 = t["n0_total"]; n1 = t["n1_total"]
    m0 = t["n0_max"]; m1 = t["n1_max"]
    ac = t["ac"]
    n = len(n0)

    d = {
        "noc0_share": n0.sum() / (n0.sum() + n1.sum()) * 100,
        "noc1_share": n1.sum() / (n0.sum() + n1.sum()) * 100,
        "noc0_idle": (n0 == 0).sum() / n * 100,
        "noc1_idle": (n1 == 0).sum() / n * 100,
        "active_mean": ac.mean(),
        "active_median": np.median(ac),
        "active_max": int(ac.max()),
        "obs_4": ((ac <= 16) & (n0 == 0)).sum() / n * 100,
        "obs_8": ((ac <= 12) & (n0 == 0)).sum() / n * 100,
        "n0_max_link_p50": np.percentile(m0, 50),
        "n0_max_link_p95": np.percentile(m0, 95),
        "n0_max_link_max": m0.max(),
        "n1_max_link_p50": np.percentile(m1, 50),
        "n1_max_link_p95": np.percentile(m1, 95),
        "n1_max_link_max": m1.max(),
        "noc0_max_zero_pct": (m0 == 0).sum() / n * 100,
        "noc0_total_zero_pct": (n0 == 0).sum() / n * 100,
    }

    # warmup 제외 (t > 300 μs)
    bin_us = t["t_us"][1] - t["t_us"][0] if len(t["t_us"]) > 1 else 0.067
    warmup_bin = int(300.0 / bin_us)
    ac_p = ac[warmup_bin:]
    n0_p = n0[warmup_bin:]
    d["post_active_mean"] = ac_p.mean()
    d["post_noc0_idle"] = (n0_p == 0).sum() / len(n0_p) * 100
    d["post_obs_4"] = ((ac_p <= 16) & (n0_p == 0)).sum() / len(ac_p) * 100
    d["post_obs_8"] = ((ac_p <= 12) & (n0_p == 0)).sum() / len(ac_p) * 100
    return d


def cmp(name, computed, summary_val, tol=0.05):
    if summary_val is None:
        print(f"  {name:35s}  computed={computed:>10.3f}   summary=?  (parse failed)")
        return None
    diff = abs(computed - summary_val)
    ok = diff < tol
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name:33s}  computed={computed:>10.3f}   summary={summary_val:>10.3f}   diff={diff:.3f}")
    return ok


def main():
    print("=" * 80)
    print("Loading timeline.csv ...")
    t = load_timeline()
    print(f"  {len(t['t_us'])} bins, {t['t_us'][-1]:.1f} μs")

    print("\nComputing from raw timeline ...")
    d = compute_from_timeline(t)

    print("\nParsing summary.md ...")
    s = parse_summary()

    print("\n" + "=" * 80)
    print("3-WAY CROSS-CHECK  (summary.md ↔ timeline.csv computed)")
    print("=" * 80)

    print("\n[Section 1] NOC 0 / NOC 1 share")
    cmp("noc0_share (%)", d["noc0_share"], s["noc0_share"])
    cmp("noc1_share (%)", d["noc1_share"], s["noc1_share"])

    print("\n[Section 2] Channel idle %")
    cmp("noc0_idle_pct", d["noc0_idle"], s["noc0_idle"])
    cmp("noc1_idle_pct", d["noc1_idle"], s["noc1_idle"])

    print("\n[Section 3] Active cores")
    cmp("active mean", d["active_mean"], s["active_mean"])

    print("\n[Section 4] Observability windows")
    cmp("idle>=4 & NOC0=0 pct", d["obs_4"], s["obs_4"])
    cmp("idle>=8 & NOC0=0 pct", d["obs_8"], s["obs_8"])

    print("\n[Section 5] Max link consistency (Priority 0 verification)")
    print(f"  NOC0 max==0 pct   = {d['noc0_max_zero_pct']:.4f}%")
    print(f"  NOC0 total==0 pct = {d['noc0_total_zero_pct']:.4f}%")
    if abs(d['noc0_max_zero_pct'] - d['noc0_total_zero_pct']) < 0.001:
        print(f"  ✓ CONSISTENCY OK (diff < 0.001)")
    else:
        print(f"  ✗ MISMATCH")

    print("\n[Section 8] Post-warmup (t > 300 μs)")
    cmp("post active mean", d["post_active_mean"], s["post_active_mean"])
    cmp("post noc0 idle pct", d["post_noc0_idle"], s["post_noc0_idle"])
    cmp("post obs>=4 pct", d["post_obs_4"], s["post_obs_4"])
    cmp("post obs>=8 pct", d["post_obs_8"], s["post_obs_8"])

    print("\n[Consistency claim in summary.md]")
    print(f"  {'✓' if s['consistency_pass'] else '✗'} summary.md 에 '일치' 문구 있음")

    # CDF 는 timeline.csv 를 직접 소비하므로 자동 정합
    # (별도 렌더링 로직인 draw_fig_cdf 는 numpy percentile 만 사용)
    print("\n[CDF] fig_cdf.png 는 timeline.csv 를 직접 quantile 계산 → 자동 정합")
    print(f"  timeline: NOC1 p50 rate = {np.percentile(t['n1_total'], 50):.3f} B/μs")
    print(f"  timeline: NOC1 p95 rate = {np.percentile(t['n1_total'], 95):.3f} B/μs")
    print(f"  timeline: NOC1 max rate = {t['n1_total'].max():.3f} B/μs")


if __name__ == "__main__":
    main()
