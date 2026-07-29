"""
ResNet50 on Eagle-N — Tracy CSV 기반 성능 시각화.

YoloV8s용 스크립트 (../../yolov8s/yolo_study/roofline_analysis.py) 를 복사해서
ResNet50 baseline 분석용으로 재활용. 하드웨어 상수와 FLOPs 로직은 그대로.
- Op-type 커버리지: Matmul(340) · Conv2d(200) · Binary(160, residual add) 등 ResNet 특성 반영
- Peak/DRAM 상수는 Eagle-N 동일 (64 TOPS spec · 53 TOPS effective · 150 GB/s)

⚠️ RUN CONDITION 표기 필수:
  아래 RUN_LABEL 상수를 CSV의 실제 실행 조건에 맞춰 매번 갱신할 것.
  Batch 크기가 다른 CSV로 재실행할 때 이 라벨을 잊고 그대로 두면 plot이 오해 유발.

기존의 hand-derive FLOPs/Byte 방식 폐기. CSV 안의 PM_* 컬럼과 실측 지표만 씀.

CSV 위치: generated/profiler/reports/YYYY_MM_DD_hh_mm_ss/ops_perf_results_*.csv

사용법:
    python roofline_analysis.py --csv ops_perf_results_XXXX.csv --out-dir plots/

    # baseline + modified 비교
    python roofline_analysis.py \\
        --csv baseline.csv \\
        --csv-compare modified_16core.csv \\
        --out-dir plots/
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# 0-a. Run condition label (CSV 재취득 시 반드시 갱신)
# ============================================================
# 현재 CSV `ops_perf_results_2026_07_08_05_15_43.csv` = ResNet50 default run:
#   batch=4, input HW = 115×115 (첫 halo 관측치), trace on, -n 10 (첫 iter warmup 후 replay)
# → batch=1 CSV로 재실행 시 이 문자열을 "batch=1 · 224×224" 등으로 갱신 필요.
RUN_LABEL = "ResNet50 · batch=4 · 115×115"

# ============================================================
# 0. 하드웨어 상수 (Eagle-N / Blackhole arch, LoFi)
# ============================================================
# 근거 (2026-07-09 세션에서 소스 검증 후 갱신, 노트 §13.14):
#   - Matrix engine LoFi: 4096 mul-adds/cycle/core
#     (tt-metal matmul_device_operation.cpp:2173, GEMM_FLOPS.md line 47-51)
#   - Eagle-N 실측 clock: 650 MHz (profile_log_device.csv: CHIP_FREQ)
#     → per-core = 4096 × 650e6 / 1e12 = 2.66 TOPS/core
#   - Physical grid: 6×4 = 24 cores (BOS spec 기준, 항상 사용 가능한 건 아님)
#   - Compute-usable: 5×4 = 20 cores (harvested 4 core 제외, baseline이 실제 쓰는 그리드)
#
# BOS 공식 스펙 "64 TOPS" 실체 확인: 24 × 2.66 = 63.84 ≈ 64.  즉 physical grid 기준.
# TOPS vs TFLOPS: LoFi matmul에서는 mul-add를 1 op로 세는 tt-metal 관행이라 동일.
BH_PEAK_TOPS_PER_CORE_LOFI    = 2.66     # 4096 × 650 MHz / 1e12
EAGLE_N_PHYSICAL_CORES        = 24
EAGLE_N_COMPUTE_CORES         = 20
EAGLE_N_PEAK_TOPS_SPEC        = 64.0     # BOS 공식 spec (24 physical cores) — ceiling
EAGLE_N_PEAK_TOPS_EFFECTIVE   = 53.0     # 20 compute-usable cores (실효 가용 peak)
EAGLE_N_PEAK_TFLOPS_LOFI      = EAGLE_N_PEAK_TOPS_SPEC   # 파생: primary ceiling

# DRAM: LPDDR5-6400 × 3 controller × 4 ch × 16-bit  ≈  150 GB/s (산술 계산치)
# 노트 §12.8: 산술은 정확 (LPDDR5-6400 × 2 bytes × 12 ch = 153.6),
# 다만 "3 controller × 4 channel" 구성이 Eagle-N 실제인지 BOS 문서로 미확인 → debt.
EAGLE_N_PEAK_DRAM_GBPS       = 150.0

# Bytes per element (Tracy CSV의 datatype 문자열 → bytes/elem)
DTYPE_BYTES = {
    "FLOAT32":  4.0, "BFLOAT16": 2.0, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625,
    "UINT32":   4.0, "INT32":    4.0, "UINT16":   2.0,     "UINT8":    1.0,
}

# ============================================================
# 1. CSV 컬럼 상수 (실측된 이름)
# ============================================================
COL_OP_CODE       = "OP CODE"                      # "Conv2d", "HaloDeviceOperation" 등
COL_OP_TYPE       = "OP TYPE"                      # "tt_dnn_device" 등
COL_CALL_COUNT    = "GLOBAL CALL COUNT"            # dispatch 번호
COL_CORE_COUNT    = "CORE COUNT"                   # 실제 사용 코어 수
COL_ATTRIBUTES    = "ATTRIBUTES"                   # 긴 config 문자열
COL_KERNEL_NS     = "DEVICE KERNEL DURATION [ns]"  # 실측 wall time
COL_FW_NS         = "DEVICE FW DURATION [ns]"      # firmware duration
COL_HOST_NS       = "HOST DURATION [ns]"

# Perf model 예측치 (tt-metal이 계산해준 것)
COL_PM_IDEAL      = "PM IDEAL [ns]"
COL_PM_COMPUTE    = "PM COMPUTE [ns]"
COL_PM_BANDWIDTH  = "PM BANDWIDTH [ns]"

# 실측 utilization
COL_PM_FPU_UTIL   = "PM FPU UTIL (%)"
COL_NOC_UTIL      = "NOC UTIL (%)"
COL_DRAM_BW_UTIL  = "DRAM BW UTIL (%)"



# ============================================================
# 2. Op record
# ============================================================
@dataclass
class OpRecord:
    idx: int                 # CSV 안 순서 (0부터)
    op_id: int               # GLOBAL CALL COUNT
    op_code: str             # 원본 (Conv2d 등)
    op_type: str             # 정규화된 소문자 (conv2d, halo, silu ...)
    core_count: int
    kernel_ns: float
    fw_ns: float
    host_ns: float
    pm_ideal_ns: float
    pm_compute_ns: float
    pm_bandwidth_ns: float
    fpu_util: float          # %
    noc_util: float
    dram_bw_util: float
    bound: str               # "compute" | "memory" | "null"  (null = PM 미구현 op의 default {1,1,1})
    efficiency: float        # actual / pm_ideal (1.0 = 이상적)
    # ── Derived (shape/attr 기반) ──
    flops: float = 0.0                 # 실측 정확 FLOPs (conv/matmul/pool). 0 = 정의 안 됨
    bytes_total: float = 0.0           # 모든 입출력 tensor 크기 합 (L1+DRAM 무관)
    bytes_dram: float = 0.0            # DRAM 표시된 tensor만 (실제 off-chip 트래픽)
    achieved_tflops: float = 0.0       # flops / kernel_ns * 1e-3 (TFLOPS)
    achieved_gbps_total: float = 0.0   # bytes_total / kernel_ns  (bytes/ns == GB/s)
    achieved_gbps_dram: float = 0.0    # bytes_dram / kernel_ns
    arith_intensity: float = 0.0       # flops / bytes_dram (FLOPs per DRAM byte)


def _f(row: dict, key: str) -> float:
    """숫자 안전 파싱. 없거나 이상하면 0."""
    v = row.get(key, "").strip()
    if not v:
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def _int(row: dict, key: str) -> int:
    return int(_f(row, key))


def _normalize_op_type(op_code: str) -> str:
    """CamelCase → snake_case 소문자. 'Conv2d' → 'conv2d'."""
    if not op_code:
        return "unknown"
    # 앞에 ttnn. 붙어있으면 떼기
    s = op_code.split(".")[-1]
    # CamelCase → snake
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    # DeviceOperation 접미사 제거
    s = s.replace("_device_operation", "").replace("_operation", "")
    # Unary는 SiLU가 대부분이라 라벨을 살짝 구체화 (attr 확인은 나중에)
    return s


# ============================================================
# 2.5. Shape / attribute 파서 — bytes와 FLOPs 유도용
# ============================================================
_PAD_RE = re.compile(r"^\s*(\d+)")  # "16384[16384]" → 16384

def _pad_int(v: str) -> int:
    """'16384[16384]' 또는 '32' 형식에서 padded logical dim 뽑음."""
    if not v:
        return 0
    m = _PAD_RE.match(v)
    return int(m.group(1)) if m else 0

def _tensor_bytes(row: dict, prefix: str) -> tuple[float, bool]:
    """
    한 tensor의 크기 (bytes)와 DRAM 여부 반환.
    prefix ∈ {"INPUT_0", "INPUT_1", "INPUT_2", "INPUT_3", "OUTPUT_0"}.
    Padded logical shape × datatype bytes 사용.
    """
    w = _pad_int(row.get(f"{prefix}_W_PAD[LOGICAL]", ""))
    z = _pad_int(row.get(f"{prefix}_Z_PAD[LOGICAL]", ""))
    y = _pad_int(row.get(f"{prefix}_Y_PAD[LOGICAL]", ""))
    x = _pad_int(row.get(f"{prefix}_X_PAD[LOGICAL]", ""))
    if not (w and z and y and x):
        return 0.0, False
    elems = float(w) * z * y * x
    dtype = row.get(f"{prefix}_DATATYPE", "").strip()
    bpe = DTYPE_BYTES.get(dtype, 2.0)   # 모르면 BF16 가정
    mem = row.get(f"{prefix}_MEMORY", "").upper()
    is_dram = "DRAM" in mem
    return elems * bpe, is_dram

def _sum_bytes(row: dict) -> tuple[float, float]:
    """(total_bytes, dram_only_bytes) 반환. IO(INPUT_0~3, OUTPUT_0) 전부 합."""
    total = 0.0
    dram = 0.0
    for prefix in ("INPUT_0", "INPUT_1", "INPUT_2", "INPUT_3", "OUTPUT_0"):
        b, is_dram = _tensor_bytes(row, prefix)
        total += b
        if is_dram:
            dram += b
    return total, dram


# --- ATTRIBUTES 파서 ---
_SLIDING_RE = re.compile(
    r"SlidingWindowConfig\("
    r".*?batch_size=(?P<b>\d+).*?"
    r"input_hw=\((?P<ih>\d+);(?P<iw>\d+)\).*?"
    r"window_hw=\((?P<kh>\d+);(?P<kw>\d+)\).*?"
    r"stride_hw=\((?P<sh>\d+);(?P<sw>\d+)\).*?"
    r"padding=\(\((?P<ph0>\d+); (?P<ph1>\d+)\); \((?P<pw0>\d+); (?P<pw1>\d+)\)\)"
)
_OUT_CH_RE = re.compile(r"'output_channels'\s*:\s*'(\d+)'")
_INPUT_SHAPE_RE = re.compile(r"'input_tensor_shape'\s*:\s*'\{(\d+);\s*(\d+);\s*(\d+);\s*(\d+)\}'")

def _parse_sliding(attr: str) -> dict | None:
    m = _SLIDING_RE.search(attr)
    if not m:
        return None
    d = m.groupdict()
    return {k: int(v) for k, v in d.items()}

def _parse_output_channels(attr: str) -> int:
    m = _OUT_CH_RE.search(attr)
    return int(m.group(1)) if m else 0

def _parse_input_shape(attr: str) -> tuple[int, int, int, int] | None:
    m = _INPUT_SHAPE_RE.search(attr)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


# --- Op별 FLOPs 추정 ---
def _estimate_flops(op_type: str, row: dict) -> float:
    """
    op-type별 FLOPs 근사값. 정의되지 않으면 0.

    - conv2d : 2 · B · H_out · W_out · C_in · C_out · K_h · K_w
    - pool2d : output_elems · K_h · K_w         (compare/add per output element)
    - matmul : 2 · M · K · N  (INPUT_0 = [.., M, K], INPUT_1 = [.., K, N])
    - unary/binary : output_elems  (1 FLOP/elem — 근사)
    - concat/upsample/halo/sharded_to_interleaved : 0 (data-movement)
    """
    attr = row.get(COL_ATTRIBUTES, "")

    if op_type == "conv2d":
        sw = _parse_sliding(attr)
        c_out = _parse_output_channels(attr)
        in_shape = _parse_input_shape(attr)
        if not (sw and c_out and in_shape):
            return 0.0
        _, ih, iw, c_in = in_shape          # (B,H,W,C_in) real
        b = sw["b"]
        kh, kw = sw["kh"], sw["kw"]
        sh, sw_ = sw["sh"], sw["sw"]
        ph = sw["ph0"] + sw["ph1"]
        pw = sw["pw0"] + sw["pw1"]
        h_out = (ih + ph - kh) // sh + 1
        w_out = (iw + pw - kw) // sw_ + 1
        return 2.0 * b * h_out * w_out * c_in * c_out * kh * kw

    if op_type == "pool2_d" or op_type == "pool2d":
        sw = _parse_sliding(attr)
        if not sw:
            return 0.0
        b = sw["b"]
        ih, iw = sw["ih"], sw["iw"]
        kh, kw = sw["kh"], sw["kw"]
        sh, sw_ = sw["sh"], sw["sw"]
        ph = sw["ph0"] + sw["ph1"]
        pw = sw["pw0"] + sw["pw1"]
        h_out = (ih + ph - kh) // sh + 1
        w_out = (iw + pw - kw) // sw_ + 1
        # 채널: INPUT_0의 X 자리 (row-major NHWC의 C)
        c = _pad_int(row.get("INPUT_0_X_PAD[LOGICAL]", ""))
        return float(b * h_out * w_out * c * kh * kw)

    if op_type == "matmul":
        # 크기가 attribute에 있으면 그거, 아니면 INPUT shape로 근사
        w0 = _pad_int(row.get("INPUT_0_W_PAD[LOGICAL]", ""))
        z0 = _pad_int(row.get("INPUT_0_Z_PAD[LOGICAL]", ""))
        y0 = _pad_int(row.get("INPUT_0_Y_PAD[LOGICAL]", ""))
        x0 = _pad_int(row.get("INPUT_0_X_PAD[LOGICAL]", ""))
        y1 = _pad_int(row.get("INPUT_1_Y_PAD[LOGICAL]", ""))
        x1 = _pad_int(row.get("INPUT_1_X_PAD[LOGICAL]", ""))
        if not (w0 and z0 and y0 and x0 and x1):
            return 0.0
        # 관습: [B, ., M, K] × [., ., K, N]. K = x0 = y1, M = y0, N = x1
        batch = w0 * z0
        m, k, n = y0, x0, x1
        if y1 and y1 != k:
            k = y1  # transpose 케이스
        return 2.0 * batch * m * k * n

    # SiLU/GELU 등 unary: 1 FLOP/elem 근사
    if op_type in ("unary", "binary", "eltwise_binary", "eltwise_unary"):
        w = _pad_int(row.get("OUTPUT_0_W_PAD[LOGICAL]", ""))
        z = _pad_int(row.get("OUTPUT_0_Z_PAD[LOGICAL]", ""))
        y = _pad_int(row.get("OUTPUT_0_Y_PAD[LOGICAL]", ""))
        x = _pad_int(row.get("OUTPUT_0_X_PAD[LOGICAL]", ""))
        return float(w * z * y * x)

    return 0.0


def parse_csv(path: Path) -> list[OpRecord]:
    records: list[OpRecord] = []
    # ATTRIBUTES 컬럼이 매우 길어서 csv 기본 field size limit을 넉넉히 확장
    csv.field_size_limit(50_000_000)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            op_code = row.get(COL_OP_CODE, "").strip()
            if not op_code:
                continue

            op_type = _normalize_op_type(op_code)
            kernel_ns = _f(row, COL_KERNEL_NS)
            pm_ideal = _f(row, COL_PM_IDEAL)
            pm_compute = _f(row, COL_PM_COMPUTE)
            pm_bandwidth = _f(row, COL_PM_BANDWIDTH)

            # bound 판정:
            #   PM_COMPUTE = PM_BANDWIDTH = 1 (or 0)  → tt-metal PM 미구현 op의 default constructor 값
            #   (operation.hpp:95 OpPerformanceModelGeneral{}). Unary/Halo/UpSample/BinaryNg 해당.
            #   → "null"로 분리 (기존엔 memory로 흡수됐음)
            if pm_compute <= 1 and pm_bandwidth <= 1:
                bound = "null"
            elif pm_compute > 0 and pm_bandwidth > 0:
                bound = "compute" if pm_compute > pm_bandwidth else "memory"
            else:
                bound = "null"

            efficiency = (pm_ideal / kernel_ns) if kernel_ns > 0 and pm_ideal > 0 else 0.0

            # ── Derived: shape → bytes, attribute → FLOPs ──
            bytes_total, bytes_dram = _sum_bytes(row)
            flops = _estimate_flops(op_type, row)
            achieved_tflops = (flops / kernel_ns * 1e-3) if (kernel_ns > 0 and flops > 0) else 0.0
            achieved_gbps_total = (bytes_total / kernel_ns) if kernel_ns > 0 else 0.0
            achieved_gbps_dram  = (bytes_dram  / kernel_ns) if kernel_ns > 0 else 0.0
            # Roofline 정공 정의: DRAM 트래픽 기준. L1 재사용은 intensity 밖으로 뺌.
            # bytes_dram == 0 인 op (L1 완전 재사용) 은 DRAM roofline 대상 아님 → 0 반환.
            arith_intensity = (flops / bytes_dram) if (bytes_dram > 0 and flops > 0) else 0.0

            records.append(OpRecord(
                idx=i,
                op_id=_int(row, COL_CALL_COUNT),
                op_code=op_code,
                op_type=op_type,
                core_count=_int(row, COL_CORE_COUNT),
                kernel_ns=kernel_ns,
                fw_ns=_f(row, COL_FW_NS),
                host_ns=_f(row, COL_HOST_NS),
                pm_ideal_ns=pm_ideal,
                pm_compute_ns=pm_compute,
                pm_bandwidth_ns=pm_bandwidth,
                fpu_util=_f(row, COL_PM_FPU_UTIL),
                noc_util=_f(row, COL_NOC_UTIL),
                dram_bw_util=_f(row, COL_DRAM_BW_UTIL),
                bound=bound,
                efficiency=efficiency,
                flops=flops,
                bytes_total=bytes_total,
                bytes_dram=bytes_dram,
                achieved_tflops=achieved_tflops,
                achieved_gbps_total=achieved_gbps_total,
                achieved_gbps_dram=achieved_gbps_dram,
                arith_intensity=arith_intensity,
            ))
    return records


# ============================================================
# 3. 요약 통계
# ============================================================
def summary(recs: list[OpRecord]) -> None:
    n = len(recs)
    total_ns = sum(r.kernel_ns for r in recs)
    total_host_ns = sum(r.host_ns for r in recs)
    total_flops = sum(r.flops for r in recs)
    total_bytes = sum(r.bytes_total for r in recs)
    total_dram = sum(r.bytes_dram for r in recs)

    by_type: dict[str, list[OpRecord]] = {}
    for r in recs:
        by_type.setdefault(r.op_type, []).append(r)

    by_bound: dict[str, list[OpRecord]] = {"compute": [], "memory": [], "null": []}
    for r in recs:
        by_bound[r.bound].append(r)

    print(f"[summary] {n} ops")
    print(f"  kernel total   : {total_ns/1e6:8.2f} ms")
    print(f"  host   total   : {total_host_ns/1e6:8.2f} ms  (kernel/host ratio "
          f"{total_ns / total_host_ns * 100 if total_host_ns else 0:.1f}%)")
    print(f"  FLOPs total    : {total_flops/1e9:8.2f} GFLOPs  (conv/pool/matmul/unary 정의된 것만)")
    print(f"  bytes I/O total: {total_bytes/1e6:8.2f} MB  (모든 INPUT + OUTPUT 파팅된 크기)")
    print(f"  bytes DRAM     : {total_dram /1e6:8.2f} MB  (DRAM 표시된 tensor만)")
    if total_ns > 0 and total_flops > 0:
        overall_tflops = total_flops / total_ns * 1e-3
        print(f"  overall achieved: {overall_tflops:.3f} TFLOPS "
              f"(vs peak {EAGLE_N_PEAK_TFLOPS_LOFI:.0f} → "
              f"{overall_tflops / EAGLE_N_PEAK_TFLOPS_LOFI * 100:.2f}%)")
    if total_ns > 0 and total_dram > 0:
        overall_gbps = total_dram / total_ns  # bytes/ns == GB/s
        print(f"  overall DRAM BW : {overall_gbps:.2f} GB/s "
              f"(vs peak {EAGLE_N_PEAK_DRAM_GBPS:.0f} → "
              f"{overall_gbps / EAGLE_N_PEAK_DRAM_GBPS * 100:.1f}%)")

    print("  by bound:")
    for b, rs in by_bound.items():
        if not rs:
            continue
        t = sum(r.kernel_ns for r in rs)
        print(f"    {b:8s}: {len(rs):4d} ops, {t/1e6:8.2f} ms ({t/total_ns*100:.1f}%)")

    print("  by op_type (top 10):")
    typed = sorted(by_type.items(),
                   key=lambda kv: -sum(r.kernel_ns for r in kv[1]))
    for op_type, rs in typed[:10]:
        t = sum(r.kernel_ns for r in rs)
        cores = set(r.core_count for r in rs)
        print(f"    {op_type:22s}: {len(rs):4d}, {t/1e6:7.2f} ms "
              f"({t/total_ns*100:5.1f}%), cores={sorted(cores)}")


def dump_optype_table(recs: list[OpRecord], out_path: Path) -> None:
    """
    op-type × (n, ms, %, cores, avg TFLOPS, avg GB/s, dominant bound) 표를
    Markdown으로 저장. 노트에 붙여넣기 용도.
    """
    total_ns = sum(r.kernel_ns for r in recs) or 1
    by_type: dict[str, list[OpRecord]] = {}
    for r in recs:
        by_type.setdefault(r.op_type, []).append(r)

    ranked = sorted(by_type.items(), key=lambda kv: -sum(r.kernel_ns for r in kv[1]))

    lines = [
        "| op_type | n | time (ms) | % | cores | avg TFLOPS | avg GB/s (I/O) | avg GB/s (DRAM) | dominant bound |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for op_type, rs in ranked:
        t_ns = sum(r.kernel_ns for r in rs)
        pct = t_ns / total_ns * 100
        cores = sorted({r.core_count for r in rs})
        rs_with_flops = [r for r in rs if r.flops > 0]
        avg_tflops = (sum(r.achieved_tflops for r in rs_with_flops) /
                      len(rs_with_flops)) if rs_with_flops else 0.0
        avg_gbps_total = sum(r.achieved_gbps_total for r in rs) / len(rs) if rs else 0.0
        avg_gbps_dram  = sum(r.achieved_gbps_dram  for r in rs) / len(rs) if rs else 0.0
        # dominant bound: kernel time가 가장 많은 것
        bnd_t = {"compute": 0.0, "memory": 0.0, "null": 0.0}
        for r in rs:
            bnd_t[r.bound] += r.kernel_ns
        dominant = max(bnd_t.items(), key=lambda kv: kv[1])[0]
        tflops_str = f"{avg_tflops:.2f}" if avg_tflops > 0 else "—"
        lines.append(
            f"| `{op_type}` | {len(rs)} | {t_ns/1e6:.2f} | {pct:.1f}% | "
            f"{cores} | {tflops_str} | {avg_gbps_total:.1f} | {avg_gbps_dram:.1f} | {dominant} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_path}")


def dump_e2e_metrics(recs: list[OpRecord], out_path: Path) -> None:
    """
    E2E 관련 지표를 Markdown 표로.
    - kernel total, host total, kernel/host ratio
    - Kernel-only theoretical FPS
    - trace 사용 시 실제 latency과 kernel time 차이는 tt-metal이 삼킴 → 참고용
    """
    kernel_ms = sum(r.kernel_ns for r in recs) / 1e6
    host_ms   = sum(r.host_ns   for r in recs) / 1e6
    fw_ms     = sum(r.fw_ns     for r in recs) / 1e6
    theo_fps  = 1000.0 / kernel_ms if kernel_ms > 0 else 0.0
    lines = [
        "| metric | value |",
        "|---|---:|",
        f"| Total device kernel time | {kernel_ms:.2f} ms |",
        f"| Total device firmware time | {fw_ms:.2f} ms |",
        f"| Total host-side op dispatch time | {host_ms:.2f} ms |",
        f"| Kernel / host ratio | {(kernel_ms/host_ms*100 if host_ms else 0):.1f}% |",
        f"| Kernel-only theoretical FPS | **{theo_fps:.1f}** |",
        f"| Total ops | {len(recs)} |",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_path}")


def dump_hw_constants(out_path: Path) -> None:
    """하드웨어 상수 표 (근거 포함)."""
    lines = [
        "| quantity | value | 근거 |",
        "|---|---:|---|",
        f"| Per-core peak (LoFi) at Eagle-N clock 650 MHz | {BH_PEAK_TOPS_PER_CORE_LOFI:.2f} TOPS | "
        "4096 mul-adds/cycle (matmul_device_operation.cpp:2173) × 650 MHz |",
        f"| Physical cores (Eagle-N, 6×4) | {EAGLE_N_PHYSICAL_CORES} | BOS spec (physical grid) |",
        f"| Compute-usable cores (Eagle-N, 5×4) | {EAGLE_N_COMPUTE_CORES} | "
        "`project_eagle_n_hardware.md` (harvested 4 core 제외) |",
        f"| **Aggregate peak (BOS spec)** | **{EAGLE_N_PEAK_TOPS_SPEC:.0f} TOPS** | "
        f"{BH_PEAK_TOPS_PER_CORE_LOFI:.2f} × {EAGLE_N_PHYSICAL_CORES} — BOS 마케팅 '64 TOPS' 실체 |",
        f"| **Aggregate peak (effective)** | **{EAGLE_N_PEAK_TOPS_EFFECTIVE:.0f} TOPS** | "
        f"{BH_PEAK_TOPS_PER_CORE_LOFI:.2f} × {EAGLE_N_COMPUTE_CORES} — 실제 baseline 가용치 |",
        f"| Peak DRAM BW (LPDDR5-6400 × 3 ch × 4) | ≈ {EAGLE_N_PEAK_DRAM_GBPS:.0f} GB/s | "
        "산술 계산치 (노트 §12.8) — 12-channel 구성이 Eagle-N 실제인지 BOS spec 확인 필요 |",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out_path}")


# ============================================================
# 4. Plots (publication style, English labels)
# ============================================================
BOUND_COLOR = {"compute": "#c1272d", "memory": "#0072b5", "null": "#8c8c8c"}

# Publication-quality defaults
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def _setup_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_STYLE)


def plot_top_ops_by_time(recs: list[OpRecord], out_path: Path, top_n: int = 25) -> None:
    """Top-N ops by kernel duration, colored by bound type."""
    import matplotlib.pyplot as plt
    _setup_style()

    top = sorted(recs, key=lambda r: -r.kernel_ns)[:top_n]
    labels = [f"{r.op_type[:14]:<14} (id {r.op_id})" for r in top]
    times_us = [r.kernel_ns / 1e3 for r in top]
    colors = [BOUND_COLOR[r.bound] for r in top]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top)), times_us, color=colors, edgecolor="none")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Kernel duration (μs)")
    ax.set_title(f"Figure 1. Top {top_n} operations by wall time\n[{RUN_LABEL}]", fontsize=12)

    from matplotlib.patches import Patch
    legend_labels = {"compute": "compute-bound", "memory": "memory-bound",
                     "null": "PM not modeled"}
    ax.legend(handles=[Patch(color=BOUND_COLOR[b], label=legend_labels[b])
                       for b in ["compute", "memory", "null"]],
              loc="lower right", title="Bound type")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def plot_pm_scatter(recs: list[OpRecord], out_path: Path) -> None:
    """PM_COMPUTE vs PM_BANDWIDTH scatter — roofline equivalent.

    Diagonal separates compute-bound (above) from memory-bound (below).
    Uses perf-model estimates from CSV (no hand-derived FLOPs).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    valid = [r for r in recs if r.pm_compute_ns > 0 and r.pm_bandwidth_ns > 0]
    xs = np.array([r.pm_bandwidth_ns for r in valid])
    ys = np.array([r.pm_compute_ns for r in valid])
    sizes = np.array([r.kernel_ns for r in valid])
    if sizes.max() > 0:
        sizes = 15 + 180 * (sizes / sizes.max())
    colors = [BOUND_COLOR[r.bound] for r in valid]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.55,
               edgecolors="black", linewidths=0.4)

    # diagonal (y = x): above = compute-bound, below = memory-bound
    lo = max(1, min(xs.min(), ys.min()))
    hi = max(xs.max(), ys.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5,
            label="PM_COMPUTE = PM_BANDWIDTH")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PM_BANDWIDTH  [ns]  — memory-limited time")
    ax.set_ylabel("PM_COMPUTE  [ns]  — compute-limited time")
    ax.set_title("Figure 2. Perf-model bottleneck classification\n"
                 "(above diagonal = compute-bound, below = memory-bound; marker size ∝ kernel time)")

    from matplotlib.patches import Patch
    handles = [Patch(color=BOUND_COLOR[b], label=b) for b in ["compute", "memory"]]
    handles.append(plt.Line2D([0], [0], color="k", ls="--", lw=1, label="diagonal"))
    ax.legend(handles=handles, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def plot_efficiency(recs: list[OpRecord], out_path: Path) -> None:
    """Achieved / ideal ratio per op, sorted by call count."""
    import matplotlib.pyplot as plt
    _setup_style()

    recs_ok = [r for r in recs if r.efficiency > 0]
    if not recs_ok:
        return

    xs = list(range(len(recs_ok)))
    ys = [r.efficiency * 100 for r in recs_ok]
    colors = [BOUND_COLOR[r.bound] for r in recs_ok]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(xs, ys, c=colors, s=18, alpha=0.65, edgecolors="none")
    ax.axhline(100, color="black", ls="--", lw=1, alpha=0.6)
    ax.text(len(xs) * 0.99, 103, "ideal (100%)", ha="right", va="bottom",
            fontsize=10, color="black")
    ax.set_xlabel("Operation index (execution order)")
    ax.set_ylabel("Achieved / ideal (%)")
    ax.set_title("Figure 3. Per-operation efficiency vs perf-model ideal")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BOUND_COLOR[b], label=b) for b in ["compute", "memory"]],
              loc="upper right", title="Bound type")

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def plot_bound_pie(recs: list[OpRecord], out_path: Path) -> None:
    """Total wall-time split by bound type."""
    import matplotlib.pyplot as plt
    _setup_style()

    totals = {b: sum(r.kernel_ns for r in recs if r.bound == b)
              for b in ["compute", "memory", "null"]}
    totals = {k: v for k, v in totals.items() if v > 0}
    total_ms = sum(totals.values()) / 1e6

    labels = [f"{k}\n{v/1e6:.2f} ms" for k, v in totals.items()]
    values = list(totals.values())
    colors = [BOUND_COLOR[b] for b in totals.keys()]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%",
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
        startangle=90,
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(13)
    ax.set_title(f"Figure 4. Wall-time by bound classification  [{RUN_LABEL}]\n"
                 f"(total kernel time: {total_ms:.2f} ms)")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def plot_fpu_hist(recs: list[OpRecord], out_path: Path) -> None:
    """FPU utilization distribution weighted by kernel time."""
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    valid = [r for r in recs if r.kernel_ns > 0]
    utils = np.array([r.fpu_util for r in valid])
    weights = np.array([r.kernel_ns for r in valid])

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 100, 21)
    ax.hist(utils, bins=bins, weights=weights / weights.sum() * 100,
            color=BOUND_COLOR["memory"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("FPU utilization (%)  [PM estimate]")
    ax.set_ylabel("Share of total wall time (%)")
    ax.set_title("Figure 5. Wall-time-weighted distribution of FPU utilization\n"
                 "(concentration near 0 = compute engines idle most of the time)")

    # add median and mean line
    median = np.median(utils)
    mean_weighted = float(np.average(utils, weights=weights))
    ax.axvline(median, color="black", ls="--", lw=1,
               label=f"median = {median:.1f}%")
    ax.axvline(mean_weighted, color=BOUND_COLOR["compute"], ls="-", lw=1.2,
               label=f"time-weighted mean = {mean_weighted:.1f}%")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def plot_optype_stack(recs: list[OpRecord], out_path: Path, top_k: int = 8) -> None:
    """Total wall-time contribution of each op-type, bound-colored bars."""
    import matplotlib.pyplot as plt
    _setup_style()

    # aggregate by op_type
    by_type: dict[str, dict[str, float]] = {}
    for r in recs:
        d = by_type.setdefault(r.op_type, {"compute": 0.0, "memory": 0.0, "null": 0.0})
        d[r.bound] += r.kernel_ns

    # rank by total time
    ranked = sorted(by_type.items(),
                    key=lambda kv: -sum(kv[1].values()))[:top_k]

    labels = [t for t, _ in ranked]
    compute_ms = [d["compute"] / 1e6 for _, d in ranked]
    memory_ms = [d["memory"] / 1e6 for _, d in ranked]
    null_ms = [d["null"] / 1e6 for _, d in ranked]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ys = range(len(labels))
    ax.barh(ys, memory_ms,
            color=BOUND_COLOR["memory"], label="memory-bound", edgecolor="none")
    ax.barh(ys, compute_ms, left=memory_ms,
            color=BOUND_COLOR["compute"], label="compute-bound", edgecolor="none")
    ax.barh(ys, null_ms,
            left=[m + c for m, c in zip(memory_ms, compute_ms)],
            color=BOUND_COLOR["null"], label="PM not modeled", edgecolor="none")

    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, family="monospace", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Wall time (ms)")
    ax.set_title(f"Figure 6. Top {top_k} op types by total wall time (stacked by bound)  [{RUN_LABEL}]")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def plot_real_roofline(recs: list[OpRecord], out_path: Path) -> None:
    """
    Figure 7. Real Roofline — arithmetic intensity vs achieved TFLOPS.

    - X: FLOPs / byte  (op-level operational intensity, I/O 기준)
    - Y: achieved TFLOPS
    - Ceiling: horizontal line at Eagle-N peak (64 TOPS = BOS spec, 24 phys core)
                dashed line at effective peak (53 TOPS = 20 usable core)
                slanted line at DRAM BW × I  (peak GB/s × intensity → TFLOPS)
    - 색은 op-type별
    - FLOPs가 정의된 op (conv2d, matmul, pool2d, unary) 만 표시
    """
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    # DRAM roofline: DRAM 트래픽이 실제 있는 op 만 (bytes_dram > 0).
    # bytes_dram == 0 인 op 은 L1 재사용이라 DRAM ceiling 대상 아님.
    valid = [r for r in recs if r.flops > 0 and r.bytes_dram > 0 and r.kernel_ns > 0]
    if not valid:
        print("no records with FLOPs/bytes_dram — skipping roofline")
        return

    # op-type별 색상 (matplotlib tab10)
    types_ordered = sorted({r.op_type for r in valid})
    cmap = plt.get_cmap("tab10")
    color_map = {t: cmap(i % 10) for i, t in enumerate(types_ordered)}

    xs = np.array([r.arith_intensity for r in valid])
    ys = np.array([r.achieved_tflops for r in valid])
    sizes = np.array([r.kernel_ns for r in valid])
    sizes = 15 + 220 * (sizes / sizes.max())
    colors = [color_map[r.op_type] for r in valid]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.7,
               edgecolors="black", linewidths=0.35)

    # ---- Roofline ceilings ----
    x_lo = max(1e-2, xs.min() * 0.3)
    x_hi = xs.max() * 3
    x_line = np.geomspace(x_lo, x_hi, 300)

    peak_spec = EAGLE_N_PEAK_TOPS_SPEC          # 64 (BOS spec, 24 physical cores)
    peak_eff  = EAGLE_N_PEAK_TOPS_EFFECTIVE      # 53 (20 usable cores)
    peak_gbps = EAGLE_N_PEAK_DRAM_GBPS
    # BW slope: y[TOPS] = intensity[FLOPs/B] × peak_gbps[GB/s] × 1e-3
    bw_line = x_line * peak_gbps * 1e-3
    roof_spec = np.minimum(bw_line, peak_spec)

    ax.plot(x_line, roof_spec, color="black", lw=1.6, alpha=0.85)
    ax.axhline(peak_spec, color="black", lw=0.8, ls=":", alpha=0.55)
    # Effective peak (53 TOPS) — dashed reference line
    ax.axhline(peak_eff, color="black", lw=0.9, ls="--", alpha=0.5)

    # Labels for both ceilings
    peak_corner_x = peak_spec / (peak_gbps * 1e-3)   # intensity where BW roof meets spec peak
    ax.text(peak_corner_x * 1.5, peak_spec * 1.08,
            f"BOS spec  ≈ {peak_spec:.0f} TOPS  (24 phys core)",
            ha="left", va="bottom", fontsize=9)
    ax.text(peak_corner_x * 1.5, peak_eff * 0.82,
            f"Effective  ≈ {peak_eff:.0f} TOPS  (20 usable core)",
            ha="left", va="top", fontsize=8.5, style="italic", alpha=0.7)
    # BW label along slope (lower on the line, well away from peak corner)
    lbl_x = peak_corner_x * 0.15
    lbl_y = lbl_x * peak_gbps * 1e-3
    ax.text(lbl_x, lbl_y * 1.35,
            f"DRAM BW  ≈ {peak_gbps:.0f} GB/s",
            rotation=32, ha="left", va="bottom", fontsize=9)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity  (FLOPs / byte, DRAM I/O)")
    ax.set_ylabel("Achieved throughput  (TFLOPS)")
    ax.set_title(f"Figure 7. ResNet50 baseline — Real Roofline (op-level)  [{RUN_LABEL}]\n"
                 "circles = individual ops, size ∝ kernel time")

    from matplotlib.patches import Patch
    handles = [Patch(color=color_map[t], label=t) for t in types_ordered]
    ax.legend(handles=handles, loc="lower right", title="op type", fontsize=9)

    ax.set_ylim(bottom=max(1e-3, ys.min() * 0.3))
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


def plot_achieved_pct(recs: list[OpRecord], out_path: Path) -> None:
    """
    Figure 8. Op별 achieved TFLOPS의 peak 대비 %.
    실제로 얼마나 peak에 근접했는지 histogram.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    peak = EAGLE_N_PEAK_TOPS_SPEC   # BOS spec 64 TOPS 기준 (effective 53 대신 spec 사용: 더 보수적)
    valid = [r for r in recs if r.flops > 0 and r.kernel_ns > 0]
    if not valid:
        return
    pcts = np.array([r.achieved_tflops / peak * 100 for r in valid])
    weights = np.array([r.kernel_ns for r in valid])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = np.linspace(0, max(pcts.max(), 20), 21)
    ax.hist(pcts, bins=bins, weights=weights / weights.sum() * 100,
            color=BOUND_COLOR["compute"], edgecolor="white", linewidth=0.5)

    mean_w = float(np.average(pcts, weights=weights))
    med    = float(np.median(pcts))
    ax.axvline(med, color="black", ls="--", lw=1,   label=f"median = {med:.2f}%")
    ax.axvline(mean_w, color=BOUND_COLOR["memory"], ls="-", lw=1.2,
               label=f"time-weighted mean = {mean_w:.2f}%")
    ax.set_xlabel(f"Achieved TFLOPS  /  BOS spec peak ({peak:.0f} TOPS)  (%)")
    ax.set_ylabel("Share of total wall time (%)")
    ax.set_title(f"Figure 8. Wall-time-weighted distribution of achieved-vs-peak TFLOPS\n[{RUN_LABEL}]", fontsize=12)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


# ============================================================
# 5. Baseline vs Modified 비교 (선택)
# ============================================================
def diff_table(base: list[OpRecord], mod: list[OpRecord]) -> None:
    b_by_id = {r.op_id: r for r in base}
    print(f"\n{'op_id':>10} {'type':<20} "
          f"{'base(μs)':>10} {'mod(μs)':>10} {'Δ(μs)':>10} {'Δ%':>7} "
          f"{'base_core':>10} {'mod_core':>9} {'bound':<8}")
    print("-" * 105)

    b_total = m_total = 0.0
    for m in mod:
        b = b_by_id.get(m.op_id)
        if b is None:
            continue
        db  = (m.kernel_ns - b.kernel_ns) / 1e3
        pct = db / (b.kernel_ns / 1e3) * 100 if b.kernel_ns > 0 else 0
        print(f"{m.op_id:>10} {m.op_type[:20]:<20} "
              f"{b.kernel_ns/1e3:>10.2f} {m.kernel_ns/1e3:>10.2f} "
              f"{db:>10.2f} {pct:>6.1f}% "
              f"{b.core_count:>10} {m.core_count:>9} {m.bound:<8}")
        b_total += b.kernel_ns / 1e3
        m_total += m.kernel_ns / 1e3

    print("-" * 105)
    print(f"{'TOTAL':>10} {'':<20} "
          f"{b_total:>10.2f} {m_total:>10.2f} "
          f"{m_total - b_total:>10.2f} "
          f"{(m_total - b_total) / b_total * 100:>6.1f}%")


# ============================================================
# 6. main
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True,
                   help="Tracy ops_perf_results CSV (baseline)")
    p.add_argument("--csv-compare", type=Path, default=None,
                   help="비교용 두 번째 CSV (예: 16-core 실험)")
    p.add_argument("--out-dir", type=Path, default=Path("."),
                   help="plot 저장 폴더")
    p.add_argument("--top-n", type=int, default=30,
                   help="wall-time 상위 몇 개 op을 볼지")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    recs = parse_csv(args.csv)
    print(f"[baseline] parsed {len(recs)} ops from {args.csv.name}")
    summary(recs)

    plot_top_ops_by_time(recs, args.out_dir / "01_top_ops.png",   top_n=args.top_n)
    plot_pm_scatter    (recs, args.out_dir / "02_pm_scatter.png")
    plot_efficiency    (recs, args.out_dir / "03_efficiency.png")
    plot_bound_pie     (recs, args.out_dir / "04_bound_pie.png")
    plot_fpu_hist      (recs, args.out_dir / "05_fpu_hist.png")
    plot_optype_stack  (recs, args.out_dir / "06_optype_stack.png")
    plot_real_roofline (recs, args.out_dir / "07_real_roofline.png")
    plot_achieved_pct  (recs, args.out_dir / "08_achieved_pct.png")

    # Markdown 표 dump (노트에 붙여넣기)
    dump_hw_constants   (args.out_dir / "table_hw_constants.md")
    dump_optype_table   (recs, args.out_dir / "table_optype.md")
    dump_e2e_metrics    (recs, args.out_dir / "table_e2e.md")

    if args.csv_compare:
        recs_mod = parse_csv(args.csv_compare)
        print(f"\n[modified] parsed {len(recs_mod)} ops from {args.csv_compare.name}")
        summary(recs_mod)
        diff_table(recs, recs_mod)


if __name__ == "__main__":
    main()
