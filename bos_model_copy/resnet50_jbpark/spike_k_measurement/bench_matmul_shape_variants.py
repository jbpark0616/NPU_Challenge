# Aging-Aware Core Rotation — Feasibility Spike #1 v2
# Measures the cost of cycling through K precompiled MATMUL SHAPE variants.
#
# Context: this BOS-fork tt-metal does not have `allowed_worker_cores` on
# matmul configs — shift-based rotation is impossible in software today. But we
# CAN still ask "how many precompiled program variants can we cycle through
# without excessive overhead?" by varying the grid SHAPE (all anchored at (0,0)
# for correctness). Different shapes touch different physical core sets and
# each produces its own program cache entry.
#
# The K cost characteristic we measure here is exactly the cost we'd pay for
# real rotation once we patch the (0,0) hardcoding in C++ — the program-cache
# hit/miss mechanics are the same regardless of what distinguishes variants.
#
# Run inside bos_host docker after `source /tt-metal/env_set.sh`:
#   python bench_matmul_shape_variants.py

import csv
import statistics
import time
from pathlib import Path

import torch
import ttnn

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# Fixed matmul shape (batched B for Reuse family; per-core dims derived per variant).
M = 512
N = 256
K_DIM = 1024

# Candidate grid shapes, all anchored at (0,0). Each is a MatmulMultiCoreReuseMultiCast
# config with compute_with_storage_grid_size = CoreCoord(gx, gy).
# On a 5x4 grid these all fit and are legal. Ordered by cell count ascending.
SHAPES = [
    (2, 2),   # 4  cores
    (4, 1),   # 4
    (1, 4),   # 4
    (2, 3),   # 6
    (3, 2),   # 6
    (2, 4),   # 8
    (4, 2),   # 8
    (3, 4),   # 12
    (4, 3),   # 12
    (4, 4),   # 16
]

K_SWEEP = [1, 2, 3, 4, 6, 8, 10]

WARMUP_ITERS_PER_VARIANT = 3
MEASURE_ITERS = 60

PCC_THRESHOLD = 0.99


def make_cfg(gx: int, gy: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    per_core_M = max(1, (M // 32) // gy)
    per_core_N = max(1, (N // 32) // gx)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def make_tensors(device):
    torch.manual_seed(42)
    ta = torch.randn(1, 1, M, K_DIM, dtype=torch.bfloat16)
    tb = torch.randn(1, 1, K_DIM, N, dtype=torch.bfloat16)
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


def pcc_gate(device, shapes):
    """Verify each shape produces numerically correct output before benchmarking."""
    print("=" * 70)
    print("PHASE 1 — PCC gate (all shapes at (0,0) anchor)")
    print("=" * 70)
    a, b, ref = make_tensors(device)
    good = []
    for (gx, gy) in shapes:
        try:
            cfg = make_cfg(gx, gy)
            y = ttnn.to_torch(ttnn.matmul(a, b, program_config=cfg))
            p = pcc(ref, y)
            status = "OK " if p >= PCC_THRESHOLD else "SKIP"
            print(f"  {gx}x{gy} ({gx*gy:2d} cores) : PCC = {p:.6f}   [{status}]")
            if p >= PCC_THRESHOLD:
                good.append((gx, gy))
        except Exception as e:
            print(f"  {gx}x{gy} ({gx*gy:2d} cores) : EXCEPTION  [{type(e).__name__}]")
            print(f"      {str(e)[:500]}")
    print(f"\n>>> {len(good)}/{len(shapes)} shapes are numerically correct.")
    return good


def measure_k(device, all_variants, k: int):
    if k > len(all_variants):
        print(f"\n--- K = {k} : SKIP (only {len(all_variants)} valid variants) ---")
        return None
    print(f"\n--- K = {k} ---")
    variants = all_variants[:k]
    print(f"    shapes: {variants}")
    a, b, _ = make_tensors(device)
    cfgs = [make_cfg(gx, gy) for (gx, gy) in variants]

    cache_before = device.num_program_cache_entries()

    for _ in range(WARMUP_ITERS_PER_VARIANT):
        for cfg in cfgs:
            _ = ttnn.matmul(a, b, program_config=cfg)
        ttnn.synchronize_device(device)

    cache_after_warmup = device.num_program_cache_entries()

    latencies_ns = []
    variants_seen = []
    for _ in range(MEASURE_ITERS):
        for vi, cfg in enumerate(cfgs):
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

    print(f"    cache : before={cache_before}  after_warmup={cache_after_warmup}  "
          f"after_measure={cache_after_measure}  delta={cache_after_warmup-cache_before}")
    print(f"    lat us: mean={mean_ns/1e3:7.2f}  p50={p50/1e3:7.2f}  "
          f"p95={p95/1e3:7.2f}  min={minv/1e3:7.2f}  max={maxv/1e3:7.2f}")

    return {
        "k": k,
        "shapes": variants,
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


def write_csvs(results, valid_shapes):
    ts = time.strftime("%Y%m%d_%H%M%S")

    summary_path = OUT_DIR / f"shape_summary_{ts}.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k", "shapes", "n_samples",
            "cache_before", "cache_after_warmup", "cache_after_measure", "cache_delta",
            "mean_ns", "p50_ns", "p95_ns", "min_ns", "max_ns",
        ])
        for r in results:
            if r is None:
                continue
            w.writerow([
                r["k"], ";".join(f"{gx}x{gy}" for (gx, gy) in r["shapes"]),
                r["n_samples"],
                r["cache_before"], r["cache_after_warmup"],
                r["cache_after_measure"], r["cache_delta"],
                f"{r['mean_ns']:.1f}", f"{r['p50_ns']:.1f}", f"{r['p95_ns']:.1f}",
                r["min_ns"], r["max_ns"],
            ])

    raw_path = OUT_DIR / f"shape_raw_{ts}.csv"
    with raw_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "iter_flat", "variant_idx", "shape", "latency_ns"])
        for r in results:
            if r is None:
                continue
            for i, (lat, vi) in enumerate(zip(r["raw_latencies_ns"], r["raw_variants"])):
                gx, gy = r["shapes"][vi]
                w.writerow([r["k"], i, vi, f"{gx}x{gy}", lat])

    meta_path = OUT_DIR / f"shape_meta_{ts}.txt"
    with meta_path.open("w") as f:
        f.write(f"M x K x N            : {M} x {K_DIM} x {N}\n")
        f.write(f"candidate shapes     : {SHAPES}\n")
        f.write(f"valid shapes (PCC OK): {valid_shapes}\n")
        f.write(f"WARMUP_ITERS/variant : {WARMUP_ITERS_PER_VARIANT}\n")
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

        valid_shapes = pcc_gate(device, SHAPES)
        if len(valid_shapes) < 2:
            print("\n!!! Fewer than 2 valid shapes — cannot run K sweep.")
            return

        print("\n" + "=" * 70)
        print("PHASE 2 — K sweep on valid shape variants")
        print("=" * 70)
        results = [measure_k(device, valid_shapes, k) for k in K_SWEEP]

        write_csvs(results, valid_shapes)

        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)
        baseline = next((r for r in results if r and r["k"] == 1), None)
        if baseline is None:
            print("(no K=1 baseline)")
            return
        for r in results:
            if r is None:
                continue
            overhead = (r["mean_ns"] - baseline["mean_ns"]) / baseline["mean_ns"] * 100.0
            entry_ok = "cache-hit steady" if r["cache_after_warmup"] == r["cache_after_measure"] else "CACHE GREW!"
            print(f"K={r['k']:2d}: mean {r['mean_ns']/1e3:8.2f} us  "
                  f"({overhead:+7.2f}% vs K=1)  cache_delta={r['cache_delta']:2d}  [{entry_ok}]")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
