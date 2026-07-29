# Aging-Aware Core Rotation — Feasibility Spike #1
# Measures the cost of cycling through K precompiled placement variants of
# ttnn.matmul on Blackhole (Eagle-N).
#
# Two phases:
#   Phase 1 (PCC gate): confirm MatmulMultiCoreReuseMultiCastProgramConfig
#     produces numerically correct output when allowed_worker_cores is offset
#     from (0,0). tt-metal docstring says this path is not validated without
#     a sub-device. If PCC fails, fall back to MatmulMultiCoreReuseProgramConfig.
#   Phase 2 (K sweep): measure per-call latency and program cache growth for
#     K in {1, 2, 3, 4} rectangular-shift variants.
#
# Run: `python bench_matmul_rotation.py`  inside bos_host docker after
#      `source env_set.sh`.

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import ttnn

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

BLOCK_COLS = 2
BLOCK_ROWS = 4
NUM_CORES = BLOCK_COLS * BLOCK_ROWS  # 8

K_DIM = 1024
M_TILES_PER_CORE = 4
N_TILES_PER_CORE = 4
M = 32 * BLOCK_ROWS * M_TILES_PER_CORE   # 512
N = 32 * BLOCK_COLS * N_TILES_PER_CORE   # 256

WARMUP_ITERS = 3
MEASURE_ITERS = 100
K_SWEEP = [1, 2, 3, 4]

PCC_THRESHOLD = 0.99


def crs(cols: int, rows: int, start_x: int = 0, start_y: int = 0) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(start_x, start_y),
                ttnn.CoreCoord(start_x + cols - 1, start_y + rows - 1),
            )
        }
    )


def mcast2d_cfg(awc: ttnn.CoreRangeSet) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(BLOCK_COLS, BLOCK_ROWS),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M_TILES_PER_CORE,
        per_core_N=N_TILES_PER_CORE,
        transpose_mcast=False,
        allowed_worker_cores=awc,
    )


def reuse_cfg(awc: ttnn.CoreRangeSet) -> ttnn.MatmulMultiCoreReuseProgramConfig:
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(BLOCK_COLS, BLOCK_ROWS),
        in0_block_w=K_DIM // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M_TILES_PER_CORE,
        per_core_N=N_TILES_PER_CORE,
        allowed_worker_cores=awc,
    )


def make_tensors(device, batched_b: bool):
    torch.manual_seed(42)
    ta = torch.randn(1, 1, M, K_DIM, dtype=torch.bfloat16)
    tb_shape = (1, 1, K_DIM, N) if batched_b else (K_DIM, N)
    tb = torch.randn(tb_shape, dtype=torch.bfloat16)
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ref = (ta @ tb).float()
    return a, b, ref


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().float()
    bf = b.flatten().float()
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = (af.norm() * bf.norm()).item()
    if denom == 0.0:
        return 0.0
    return float((af @ bf).item() / denom)


@dataclass
class PhaseResult:
    factory: str
    pcc_at_origin: float
    pcc_offset: float
    offset_supported: bool


def phase1_pcc_gate(device) -> PhaseResult:
    print("=" * 70)
    print("PHASE 1 — PCC gate (offset shift support)")
    print("=" * 70)

    a, b, ref = make_tensors(device, batched_b=False)

    print(f"[mcast2d] at-origin ({BLOCK_COLS}x{BLOCK_ROWS} @ (0,0))")
    cfg0 = mcast2d_cfg(crs(BLOCK_COLS, BLOCK_ROWS, 0, 0))
    y0 = ttnn.to_torch(ttnn.matmul(a, b, program_config=cfg0))
    p_origin = pcc(ref, y0)
    print(f"    PCC = {p_origin:.6f}")

    print(f"[mcast2d] offset  ({BLOCK_COLS}x{BLOCK_ROWS} @ (1,0))")
    cfg1 = mcast2d_cfg(crs(BLOCK_COLS, BLOCK_ROWS, 1, 0))
    y1 = ttnn.to_torch(ttnn.matmul(a, b, program_config=cfg1))
    p_offset = pcc(ref, y1)
    print(f"    PCC = {p_offset:.6f}")

    if p_offset >= PCC_THRESHOLD:
        print(">>> mcast2d offset is CORRECT. Proceeding with mcast2d for K sweep.")
        return PhaseResult("mcast2d", p_origin, p_offset, True)

    print(f">>> mcast2d offset FAILS PCC ({p_offset:.4f} < {PCC_THRESHOLD}).")
    print(">>> Falling back to Reuse factory (batched B) for K sweep.")

    a4, b4, ref4 = make_tensors(device, batched_b=True)
    print(f"[reuse]  offset  ({BLOCK_COLS}x{BLOCK_ROWS} @ (1,0))")
    cfg_r = reuse_cfg(crs(BLOCK_COLS, BLOCK_ROWS, 1, 0))
    yr = ttnn.to_torch(ttnn.matmul(a4, b4, program_config=cfg_r))
    p_reuse = pcc(ref4, yr)
    print(f"    PCC = {p_reuse:.6f}")
    assert p_reuse >= PCC_THRESHOLD, f"Reuse offset also fails: {p_reuse}"

    return PhaseResult("reuse", p_origin, p_offset, False)


def build_variants(factory: str, k: int):
    max_shift = max(1, k)
    variants = []
    for i in range(k):
        start_x = i % max_shift
        awc = crs(BLOCK_COLS, BLOCK_ROWS, start_x=start_x, start_y=0)
        if factory == "mcast2d":
            variants.append(mcast2d_cfg(awc))
        else:
            variants.append(reuse_cfg(awc))
    return variants


def measure_k(device, factory: str, k: int):
    print(f"\n--- K = {k} ({factory}) ---")
    a, b, _ = make_tensors(device, batched_b=(factory == "reuse"))
    configs = build_variants(factory, k)

    cache_before = device.num_program_cache_entries()

    # Warmup — compiles each variant into program cache.
    for _ in range(WARMUP_ITERS):
        for cfg in configs:
            _ = ttnn.matmul(a, b, program_config=cfg)
        ttnn.synchronize_device(device)

    cache_after_warmup = device.num_program_cache_entries()

    latencies_ns = []
    variants_seen = []
    for _ in range(MEASURE_ITERS):
        for vi, cfg in enumerate(configs):
            t0 = time.perf_counter_ns()
            out = ttnn.matmul(a, b, program_config=cfg)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter_ns()
            latencies_ns.append(t1 - t0)
            variants_seen.append(vi)
            del out

    cache_after_measure = device.num_program_cache_entries()

    mean_ns = statistics.mean(latencies_ns)
    p50 = statistics.median(latencies_ns)
    p95 = statistics.quantiles(latencies_ns, n=20)[18]
    minv = min(latencies_ns)
    maxv = max(latencies_ns)

    print(f"    cache entries: before={cache_before} after_warmup={cache_after_warmup} after_measure={cache_after_measure}")
    print(f"    latency (us)  : mean={mean_ns/1e3:.2f}  p50={p50/1e3:.2f}  p95={p95/1e3:.2f}  min={minv/1e3:.2f}  max={maxv/1e3:.2f}")

    return {
        "k": k,
        "factory": factory,
        "cache_before": cache_before,
        "cache_after_warmup": cache_after_warmup,
        "cache_after_measure": cache_after_measure,
        "cache_delta": cache_after_warmup - cache_before,
        "mean_ns": mean_ns,
        "p50_ns": p50,
        "p95_ns": p95,
        "min_ns": minv,
        "max_ns": maxv,
        "n_samples": len(latencies_ns),
        "raw_latencies_ns": latencies_ns,
        "raw_variants": variants_seen,
    }


def write_csvs(phase: PhaseResult, results):
    ts = time.strftime("%Y%m%d_%H%M%S")

    summary_path = OUT_DIR / f"summary_{ts}.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k", "factory", "n_samples",
            "cache_before", "cache_after_warmup", "cache_after_measure", "cache_delta",
            "mean_ns", "p50_ns", "p95_ns", "min_ns", "max_ns",
        ])
        for r in results:
            w.writerow([
                r["k"], r["factory"], r["n_samples"],
                r["cache_before"], r["cache_after_warmup"], r["cache_after_measure"], r["cache_delta"],
                f"{r['mean_ns']:.1f}", f"{r['p50_ns']:.1f}", f"{r['p95_ns']:.1f}",
                r["min_ns"], r["max_ns"],
            ])

    raw_path = OUT_DIR / f"raw_{ts}.csv"
    with raw_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "factory", "iter_flat", "variant_idx", "latency_ns"])
        for r in results:
            for i, (lat, vi) in enumerate(zip(r["raw_latencies_ns"], r["raw_variants"])):
                w.writerow([r["k"], r["factory"], i, vi, lat])

    meta_path = OUT_DIR / f"meta_{ts}.txt"
    with meta_path.open("w") as f:
        f.write(f"factory              : {phase.factory}\n")
        f.write(f"pcc_at_origin        : {phase.pcc_at_origin:.6f}\n")
        f.write(f"pcc_offset           : {phase.pcc_offset:.6f}\n")
        f.write(f"mcast_offset_ok      : {phase.offset_supported}\n")
        f.write(f"BLOCK_COLS x BLOCK_ROWS : {BLOCK_COLS} x {BLOCK_ROWS} = {NUM_CORES} cores\n")
        f.write(f"M, K, N              : {M}, {K_DIM}, {N}\n")
        f.write(f"WARMUP_ITERS         : {WARMUP_ITERS}\n")
        f.write(f"MEASURE_ITERS        : {MEASURE_ITERS}\n")
        f.write(f"K_SWEEP              : {K_SWEEP}\n")

    print(f"\nWrote:\n  {summary_path}\n  {raw_path}\n  {meta_path}")


def main():
    device = ttnn.open_device(device_id=0)
    try:
        try:
            device.enable_program_cache()
        except Exception:
            pass

        grid = device.compute_with_storage_grid_size()
        print(f"device grid: {grid.x} x {grid.y}")
        needed_x = BLOCK_COLS + max(K_SWEEP) - 1
        assert grid.x >= needed_x, f"grid.x={grid.x} but K sweep needs >= {needed_x}"
        assert grid.y >= BLOCK_ROWS, f"grid.y={grid.y} but block needs >= {BLOCK_ROWS}"

        phase = phase1_pcc_gate(device)

        print("\n" + "=" * 70)
        print(f"PHASE 2 — K sweep on factory={phase.factory}")
        print("=" * 70)
        results = [measure_k(device, phase.factory, k) for k in K_SWEEP]

        write_csvs(phase, results)

        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)
        baseline = results[0]["mean_ns"]
        for r in results:
            overhead = (r["mean_ns"] - baseline) / baseline * 100.0
            print(f"K={r['k']}: mean {r['mean_ns']/1e3:8.2f} us  "
                  f"(+{overhead:+6.2f}% vs K=1)  cache_delta={r['cache_delta']}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
