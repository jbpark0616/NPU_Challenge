# YoloV8s on Eagle-N 학습 노트

날짜: 2026-07-08 (초판) · 2026-07-09 (§12 정량화, §13 롱폼, §13.14 peak/null 재검증)

---

## 1. 파일 구조 (bos_model/yolov8s/)

```
yolov8s/
├── run_yolov8s.py                # 실행 엔트리
├── demo_yolov8s.py               # 비디오 데모
├── README.md                     # 실행법, visualizer 사용법
├── weights_downloader.py         # 가중치 다운로드
├── yolov8s.py                    # ★ 모델 본체 (선형 모듈 리스트)
├── configs/
│   ├── yolov8s_a0.py             # ★ Eagle-N A0 board 전용 config
│   ├── yolov8s_320x320.py        # 320×320 입력 config
│   ├── yolov8s_256x256.py        # 256×256 입력 config
│   └── utils.py                  # ConvConfig, UpsampleConfig, ConcatConfig 헬퍼
├── modules/
│   ├── YoloV8Nets.py             # BOS_TTNN_Conv, BOS_TTNN_CSP, BOS_TTNN_SPPF
│   ├── YoloV8FPN.py              # BOS_TTNN_Upsample, BOS_TTNN_Concat
│   └── YoloV8Head.py             # BOS_TTNN_Detect
├── reference/                    # PyTorch 참조 구현
├── tests/                        # 테스트
├── utilities/                    # 유틸 함수
└── yolo_study/                   # ← 이 노트가 있는 곳
    ├── yolo_study_note.md
    └── yolo_visualizer/          # L1 visualizer 스크린샷
```

---

## 2. YoloV8 아키텍처 개요

### 3-stage 구조

```
[Input 이미지]
       │
       ▼
┌──────────────────┐
│  Backbone        │ ← m[0]~m[9]: 특징 추출
│  (CSP based)     │   Conv → CSP × N → Conv → CSP × N → ...
└──────────────────┘
       │  p3, p4, p5 (3-scale feature)
       ▼
┌──────────────────┐
│  FPN + PAN       │ ← m[10]~m[21]: 다중 scale fusion
│  (feature pyramid│   Upsample + Concat + CSP (top-down + bottom-up)
│   network)       │
└──────────────────┘
       │  p3_td, p4_out, p5_out
       ▼
┌──────────────────┐
│  Detect head     │ ← m[22]: 3-scale bbox + class
│  (multi-scale)   │   각 scale마다 cv2(box) + cv3(class)
└──────────────────┘
       │
       ▼
[3-scale output]
```

### Module 유형

| Module 클래스 | 뭐 하는지 |
|---|---|
| `BOS_TTNN_Conv` | Conv2d + SiLU (하나의 conv block) |
| `BOS_TTNN_CSP` | Cross Stage Partial (residual 포함 conv 시리즈) |
| `BOS_TTNN_SPPF` | Spatial Pyramid Pooling — Fast (multi-scale pooling + concat) |
| `BOS_TTNN_Upsample` | 2× 업샘플 (FPN용) |
| `BOS_TTNN_Concat` | 다중 scale 결합 |
| `BOS_TTNN_Detect` | 최종 3-scale 검출 head |

### `yolov8s.py` 구조

```
line 30-111    class_names (COCO 80 classes 딕셔너리)
line 114-372   class Detect              # PyTorch 참조 (post-processing 포함)
line 375-717   class YoloV8              # ★ TTNN 모델 본체
line 720-868   class YoloV8sRunner       # 실행 wrapper (trace/2CQ 포함)
```

### Forward pass 전체 흐름 (line 646~717)

```
입력 (1, 3, 320, 320)
   │
   ├── Backbone ──────────────────────────
   │   m[0] Conv (stride 2)  → 32 채널
   │   m[1] Conv (stride 2)  → 64 채널
   │   m[2] CSP              → 64 채널
   │   m[3] Conv (stride 2)  → 128 채널
   │   m[4] CSP              → 128 채널 → p3 저장
   │   m[5] Conv (stride 2)  → 256 채널
   │   m[6] CSP              → 256 채널 → p4 저장
   │   m[7] Conv (stride 2)  → 512 채널
   │   m[8] CSP              → 512 채널
   │   m[9] SPPF             → 512 채널 → p5 저장
   │
   ├── FPN + PAN ─────────────────────────
   │   p3, p4, p5 → sharded_to_interleaved (line 669~671)
   │   m[10] Upsample (p5)
   │   m[11] Concat (upsample, p4)
   │   m[12] CSP → p4_td
   │   m[13] Upsample (p4_td)
   │   m[14] Concat (upsample, p3)
   │   m[15] CSP → p3_td
   │   m[16] Conv (stride 2, p3_td → p3_down)
   │   m[17] Concat (p3_down, p4_td)
   │   m[18] CSP → p4_out
   │   m[19] Conv (stride 2, p4_out → p4_down)
   │   m[20] Concat (p4_down, p5)
   │   m[21] CSP → p5_out
   │
   └── Detect ────────────────────────────
       m[22] Detect([p3_td, p4_out, p5_out])
       → output0, output1, output2 (3 scale bbox+class)
```

---

## 3. Config 시스템

### 계층 구조

```
run_yolov8s.py → YoloV8sRunner
   │
   └─ 입력 크기(H×W)에 맞는 config 파일 자동 로드:
      → configs/yolov8s_{H}x{W}.py
      → 없으면 configs/yolov8s_256x256.py fallback
   │
   ├─ configs/yolov8s_320x320.py    # 320×320용
   ├─ configs/yolov8s_256x256.py    # 256×256용  
   └─ configs/yolov8s_a0.py         # ★ Eagle-N A0 board 전용
```

**`yolov8s_a0.py`가 우리 하드웨어(Eagle-N A0)에 튜닝된 것.**

### `layer_configs` 딕셔너리 형식

각 layer의 정확한 이름을 key로 config 객체 값:

```python
layer_configs = {
    "model.0.conv": ConvConfig(core_end=(3, 3), shard_layout="Height"),
    "model.1.conv": ConvConfig(core_end=(3, 3), shard_layout="Height", 
                               reshard_if_not_optimal=False),
    "model.2.cv1_a.conv": ConvConfig(core_end=(3, 3), shard_layout="Height",
                                     reshard_if_not_optimal=True),   # ★
    ...
    "model.22.cv3.0.2.conv": ConvConfig(core_end=(3, 3), shard_layout="Height",
                                        reshard_if_not_optimal=True),  # ★
}
```

### ConvConfig 기본값 분석 (`configs/utils.py:7-24`)

```python
def ConvConfig(
    core_grid=None,
    core_start=(0, 0),
    core_end=(3, 3),                # ← 기본 4×4 = 16 코어
    shard_layout="Height",
    dtype=ttnn.bfloat16,
    weights_dtype=ttnn.bfloat16,
    math_fidelity=ttnn.MathFidelity.LoFi,
    reshard_if_not_optimal=False,   # ← 기본 False (안전)
    ...
):
```

**두 개의 중요한 기본값:**
- `core_end=(3, 3)` — 지정 안 하면 4×4 grid
- `reshard_if_not_optimal=False` — conv 자체 grid 재선택 차단

### `reshard_if_not_optimal=True` 인 layer 목록 (`yolov8s_a0.py`)

```
model.2.cv1_a.conv    ← CSP block 진입
model.2.cv1_b.conv
model.4.cv1_a.conv
model.4.cv1_b.conv
model.5.conv          ← Backbone 중간 downsample
model.16.conv         ← PAN downsample
model.19.conv         ← PAN downsample
model.22.cv3.0.2.conv ← Detect head 클래스 branch
```

이 layer들은 conv가 자체 grid 확장 허용 → **spill 원인 후보.**

### `max_grid_size=(4, 4)` 하드코딩

UpsampleConfig, ConcatConfig 안에:
```python
def UpsampleConfig(..., max_grid_size=(4, 4)):    # 하드코딩
def ConcatConfig(..., max_grid_size=(4, 4)):      # 하드코딩
```

Utility 함수 안에 assertion:
```python
assert ncores_h <= max_grid_size[0]   # 4 초과 못 함
assert ncores_w <= max_grid_size[1]
```

**설계 의도**: 4×4=16 코어 안에서 실행 유도. 근데 실측은 다름 (아래 7번).

---

## 4. Resnet50과 op 비교

### 공통 op (양쪽 다 사용)

| Op | 역할 |
|---|---|
| `ttnn.conv2d` | Convolution 본체 |
| `ttnn.add` | Residual 덧셈 |
| `ttnn.to_memory_config` | Sharding 재설정 |
| `ttnn.reshape` | 차원 재정렬 |
| `ttnn.max_pool2d` | Max pooling |
| `ttnn.deallocate` | L1 해제 |

### Resnet50에만 있음

| Op | 왜 |
|---|---|
| `ttnn.global_avg_pool2d` | Classification 헤드 |
| `ttnn.linear` | FC 분류기 |
| `ttnn.tilize_with_val_padding` | FC 앞 layout |
| `ttnn.untilize_with_unpadding` | 최종 unpad |
| Fused ReLU (conv 내부) | Activation 흡수 |

### YoloV8s에만 있음

| Op | 왜 |
|---|---|
| `ttnn.silu` | 별도 activation (unfused) |
| `ttnn.concat` | FPN/PAN 다중 scale 결합 + SPPF |
| `ttnn.upsample` | FPN 업샘플 |
| `ttnn.sharded_to_interleaved` | Layout 전환을 명시적으로 |

### 구조적 차이 4가지

**1. Activation fusion**
- Resnet50: `activation=ttnn.UnaryOpType.RELU` 를 Conv2dConfig에 명시 → 1 op = conv+ReLU
- YoloV8s: conv 후 별도 `ttnn.silu` 호출 → 2 op

**2. Feature 결합**
- Resnet50: `add` 만 (residual)
- YoloV8s: `add` + `concat` (다중 scale)

**3. 헤드 구조**
- Resnet50: avgpool → FC → 1000 class (single output)
- YoloV8s: 3-scale Detect head (P3, P4, P5) → 3 outputs

**4. Layout 전환**
- Resnet50: `reshard_if_not_optimal=True` 로 conv가 몰래 reshard
- YoloV8s: forward 안에서 `ttnn.sharded_to_interleaved` 를 직접 호출

---

## 5. 최적화 상태 (YoloV8s baseline)

### ConvConfig 기본에서 꺼져있는 것

```python
# configs/utils.py 안 ConvConfig
enable_act_double_buffer=False,       # 꺼짐
# enable_weights_double_buffer=False, # 주석 처리 (기본값)
# force_split_reader=None,            # 주석 처리
```

Resnet50 baseline은 `enable_act_double_buffer=True`, `force_split_reader=True` 등 대부분 켜놓음.

**→ YoloV8s는 성능 최적화 여지 많이 남음.** 여기가 우리 연구 확장 여지.

---

## 6. YoloV8sRunner 실행 옵션

### Trace + 2CQ (line 727)

```python
class YoloV8sRunner:
    def __init__(
        self,
        device_id=0,
        image_shape=[320, 320],
        input_channels=3,
        output_classes=80,
        trace=True,                    # ← 기본 Trace on
        ...
    ):
```

### Warmup + Trace capture (line 776~794)

```python
if self.trace:
    # Warmup
    outputs = self.bos_model()
    
    # Capture
    self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
    outputs = self.bos_model()
    ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
```

Resnet50의 `ModelRunnerWrapper._warmup()` + `trace_capture()` 와 동일한 패턴.

### 실행 명령

```bash
# 일반 실행
python ./models/bos_model/yolov8s/run_yolov8s.py

# Trace 모드
python ./models/bos_model/yolov8s/run_yolov8s.py --trace -i 320 -n 10

# Visualizer용
source models/bos_model/export_l1_vis.sh yolov8s_MYNAME
python ./models/bos_model/yolov8s/run_yolov8s.py
ttnn-visualizer --profiler-path generated/ttnn/reports/yolov8s_MYNAME_MMDD_hhmm/
```

---

## 7. Visualizer 실측 결과 (op144~160)

캡처 폴더: `yolo_visualizer/`, 17장.

### Op 명명 (renamed)

| Op | 파일명 | 의미 |
|---|---|---|
| 144 | `op144_conv2d.png` | Conv (YoloV8Nets.py:140) |
| 145 | `op145_silu.png` | Activation (line 189) |
| 146 | `op146_conv2d.png` | Conv |
| 147 | `op147_silu.png` | Act |
| 148-155 | `op148~155_conv2d/silu.png` | Conv+SiLU 반복 (CSP 내부) |
| 156 | `op156_add-residual-line361.png` | Residual add (CSP bottleneck) |
| 157 | `op157_concat-sppf-line511.png` | Concat (SPPF pyramid) |
| 158 | `op158_sharded-to-interleaved-line132.png` | Layout 전환 |
| 159 | `op159_conv2d.png` | Conv |
| 160 | `op160_silu.png` | Act |

### 캡처 구간 해석

**op144~155**: Conv+SiLU 6쌍 반복 → **CSP block 내부 conv 시리즈**  
**op156**: Residual add → CSP의 skip connection  
**op157**: SPPF concat → 여러 pooling 결과 합침  
**op158**: FPN 진입 전 sharded→interleaved 전환  
**op159~160**: FPN 첫 conv+silu

즉 사이클 하나 안에 **CSP → Residual → SPPF → FPN 진입** 담김.

### ★ 예상과 다른 발견: 모든 op이 20 코어 사용

**Config 예측**: `core_end=(3,3)` = 4×4 = 16 코어  
**실측**: 모든 op이 5×4 = **20 코어**에 L1 걸침 (컬럼 4 침범)

| Op 종류 | 활성 코어 |
|---|---|
| conv2d (144, 146, 148, 150, 152, 154, 159) | **20** |
| silu (145, 147, 149, 151, 153, 155, 160) | **20** |
| add (156) | **20** |
| concat (157) | **20** |
| sharded_to_interleaved (158) | **20** |

**즉 baseline이 이미 col 4를 쓰고 있음.** Resnet50과 마찬가지 상황.

---

## 8. Col 4 사용 원인 가설 (미검증)

### 가설 1: Weight tensor broadcast (낙관)
- Weight를 여러 코어에 mcast로 caching
- Activation shard는 여전히 16 안
- Visualizer의 "All allocations" 모드가 weight까지 표시하니 20으로 보임
- **가설이 맞으면 우리 계획 유지 가능**

### 가설 2: `reshard_if_not_optimal=True` cascade (비관)
- Config에 True 있는 layer가 자체 grid 확장 → 이후 layer 상속
- 소수 True layer가 전체를 오염
- **가설이 맞으면 config 수정 필요**

### 가설 3: `core_grid` 가 hint일 뿐 강제 아님 (비관)
- `determine_parallel_config` (conv2d_utils.cpp:169) 이 최종 결정
- Config는 참고만
- **가설이 맞으면 conv2d 내부 로직 손봐야 함**

---

## 9. 다음 검증 스텝

### 즉시 (10분)

1. **Visualizer에서 `Input/Output allocations per core` 버튼 활성화** 후 op144 재캡처
   - Col 4 비어있으면 → 가설 1 확정 → 우리 SubDevice 계획 유지
   - Col 4 여전히 차있으면 → 가설 2 또는 3

### 후속 (가설 따라)

**가설 1 확정 시:**
- SubDevice sanity check 실험 즉시 시작
- Col 4의 4코어에 hello-world kernel 올려 병렬 실행 확인

**가설 2 확정 시:**
- `yolov8s_a0.py`에서 `reshard_if_not_optimal=True` 8곳 다 False로 변경
- 재실행 후 visualizer 재확인

**가설 3 확정 시:**
- `conv2d_utils.cpp:169 determine_parallel_config` 소스 읽기
- Grid 강제 방법 (override) 찾기

---

## 10. Resnet50 vs YoloV8s — 연구 재료 관점

| 항목 | Resnet50 | YoloV8s |
|---|---|---|
| Task | Classification | **Object Detection** (automotive fit) |
| Code 명료성 | Bottleneck 중첩 | **선형 module 리스트** |
| Config 시스템 | 하드코딩 dict | **Layer별 dict + a0 전용** |
| Sharding 전환 | 암묵적 (flag) | **명시적 (sharded_to_interleaved)** |
| Op 개수 | 상대적 적음 (fused) | **많음 (unfused SiLU + concat)** |
| Baseline 최적화 | 대부분 켜있음 | **덜 켜있음 (연구 여지)** |
| 실측 col 4 사용 | 있음 (op168 등) | **있음 (전 op)** ← 예상 밖 |
| Eagle-N A0 전용 config | 없음 | **있음** |

**연구 주 target으로 YoloV8s 유력.** 다만 col 4 사용 원인 검증 후 확정.

---

## 11. 참고 문서/코드 경로

### 학습 자료
- `yolov8s.py` line 375-717 — YoloV8 클래스 (아키텍처)
- `yolov8s.py` line 720-868 — YoloV8sRunner (실행 wrapper)
- `configs/yolov8s_a0.py` — Eagle-N A0 board config
- `configs/utils.py` — ConvConfig 기본값
- `modules/YoloV8Nets.py` — Conv, CSP, SPPF 구현
- `modules/YoloV8FPN.py` — Upsample, Concat
- `modules/YoloV8Head.py` — Detect 헤드

### 관련 tt-metal 코드
- `tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:169` — `determine_parallel_config` (grid 결정)
- `tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_device_operation.cpp:33` — dispatcher
- `tt-metal/tech_reports/SubDevices/SubDevices.md` — 코어 격리 API
- `tt-metal/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md` — Trace + 2CQ

### 관련 노트
- `bos_model_copy/resnet50/resnet_note/resnet50_study_note.md` — Resnet50 학습 노트 (비교용)

---

## 12. 2026-07-08 세션 요약 — Tracy CSV 기반 정량 분석

### 12.1 세션 목적
지난 노트의 "가설 1/2/3" (visualizer의 20-core 관찰 해석) 을 넘어서, **Tracy profiler CSV로 실측 정량 지표**를 뽑고 발표/논문 근거로 사용할 수 있는 수준으로 정리.

### 12.2 워크플로우 확립

**Tracy CSV 뽑는 명령** (BOS docker 서버):
```bash
cd /tt-metal
source env_set.sh                            # 필수 (feedback_env_setup.md 참고)
ENABLE_TRACY=1 TT_METAL_DEVICE_PROFILER=1 \
python -m tracy -r -p -v \
    models/bos_model/yolov8s/run_yolov8s.py --trace -i 320 -n 10
```
- 결과: `generated/profiler/reports/<YYYY_MM_DD_hh_mm_ss>/ops_perf_results_*.csv`
- CSV 컬럼 스펙: `tt-metal/docs/source/ttnn/ttnn/profiling_ttnn_operations.rst`

**분석 스크립트**: `yolo_study/roofline_analysis.py`
```bash
python roofline_analysis.py --csv <path>.csv --out-dir plots/
```
→ 6개 plot 생성 (`plots/01_...` ~ `plots/06_...`)

### 12.3 사용한 baseline CSV
- 파일: `yolo_study/ops_perf_results_2026_07_04_16_02_37.csv`
- 실행 조건: batch=1, 입력 256×256, config 수정 없음 (baseline 그대로)
- 총 op 행: **837개**
- 확인 방법: op 분포로 YoloV8s 확정 (UnaryDeviceOperation 195, ConcatDeviceOperation 48, UpSample 6 존재)

### 12.4 실측 확정 사실 (baseline 20-core)

| 지표 | 값 | 근거 |
|---|---|---|
| Total kernel time | **13.55 ms** | CSV 합산 |
| Op 개수 | 837 | CSV |
| Memory-bound 비중 | **75.1%** (10.17 ms) | PM_COMPUTE vs PM_BANDWIDTH |
| Compute-bound 비중 | 24.9% (3.38 ms) | 동 |
| FPU util 중앙값 | **1.5%** | PM_FPU_UTIL 컬럼 |
| FPU util 시간가중 평균 | **3.0%** | 동 |
| Top op type (시간) | conv2d 39.1%, unary 22.7%, matmul 12.0%, pool2d 5.9% | 실측 |

### 12.5 **지난 노트 수정 필요한 부분**

**(a) "20 코어 강제 배정" 프레이밍 부정확**  
지난 노트는 baseline이 모든 op에서 20 코어 쓴다고 프레이밍했으나, CSV 실측:
```
conv2d       : cores=[2, 4, 8, 10, 20]
unary (SiLU) : cores=[2, 4, 8, 16, 19, 20]
matmul       : cores=[2, 4, 8, 16, 19]
pool2d       : cores=[2]              ← 겨우 2 코어!
concat       : cores=[2, 8, 16, 19, 20]
halo         : cores=[2, 4, 8, 16, 19, 20]
```
→ **op별로 코어 수 다름**. Conv2d에만 grid=5×4 강제되고 pool2d 등은 오히려 저활용. 이건 **새로운 최적화 각도**가 열림 (pool2d 코어 확대 등).

**(b) "conv2d는 대체로 compute-bound일 것" 예측 틀림**  
실측: Conv2d 시간의 대부분이 memory-bound. Weight/activation DRAM 왕복이 병목. 유일한 100% compute-bound op type은 pool2d.

**(c) "16 코어 강제 시 ~20% slowdown" 예측 과장**  
Memory-bound이 75% → 코어 감축의 영향은 훨씬 작음. 손계산 추정: **~6% total slowdown**. 실측 검증은 아직 (다음 세션).

**(d) 가설 1 (weight mcast broadcast) 기각 유지**  
어제 visualizer 실험(op144)에서 컬럼 4에도 input/output shard 있음 확인. Config's core_end=(3,3)은 hint일 뿐 실효 없음 (가설 2 또는 3 확정).

### 12.6 생성된 시각화 6개 (`yolo_study/plots/`)

| # | 파일 | 메시지 |
|---|---|---|
| 1 | `01_top_ops.png` | Top 30 op by wall time. 상위 3 = memory-bound conv2d 각 ~215μs. Pool2d만 compute-bound |
| 2 | `02_pm_scatter.png` | PM_COMPUTE vs PM_BANDWIDTH scatter (roofline 등가물). 큰 원 대부분 memory-bound 쪽 |
| 3 | `03_efficiency.png` | op별 PM_IDEAL 대비 achieved 효율. Memory-bound 대부분 0~20%. Compute-bound은 20~60% |
| 4 | `04_bound_pie.png` | **75.1% memory / 24.9% compute** 파이. 발표 핵심 슬라이드 |
| 5 | `05_fpu_hist.png` | FPU util 시간가중 히스토그램. Median 1.5%, mean 3.0%. "97% 시간 FPU idle" |
| 6 | `06_optype_stack.png` | Op-type별 wall time. Unary(SiLU) 100% memory-bound = 3.08ms 낭비 |

### 12.7 발표용 슬라이드 추천 순서
```
Figure 4 (파이)        ← 열기: "75% memory-bound"
Figure 5 (히스토그램)   ← 임팩트: "FPU 3%만 활용"
Figure 6 (op-type 스택) ← 세부: "어디서 시간 새나가나"
Figure 1 (top 30)      ← 부록: 구체 op 예시
Figure 2 (roofline)    ← 방법론: 어떻게 분류했나
Figure 3 (efficiency)  ← 부록: ideal 대비 gap
```

### 12.8 하드웨어 상수 (2026-07-09 밤 갱신 — 소스 검증 완료, §13.14 참조)

**Peak compute — 정확한 재계산 (초기 값 80 TFLOPS는 오류였음)**:
- Matrix engine LoFi: **4096 mul-adds/cycle/core** (`matmul_device_operation.cpp:2173`, `GEMM_FLOPS.md` line 47-51)
- Eagle-N 실측 clock: **650 MHz** (`profile_log_device.csv:CHIP_FREQ`)
- Per-core = 4096 × 650×10⁶ = **2.66 TOPS/core**
- **BOS 공식 spec "64 TOPS" = 24 physical cores × 2.66** (63.84 확정)
- **Effective peak = 20 compute-usable cores × 2.66 = 53 TOPS** (실제 baseline 가용치)
- ~~초기 노트의 80 TFLOPS는 Wormhole 1 GHz 값(4 TFLOPS/core)을 Blackhole architecture에 잘못 적용한 값~~

**TOPS vs TFLOPS**: LoFi mode에서 mul-add를 1 op으로 세는 tt-metal 관행이라 우리 workload에서 동일. GEMM_FLOPS.md의 "5.4 TFLOPS (BH @ 1.35 GHz)"도 같은 관행.

**Peak DRAM BW**:
- LPDDR5-6400 × 3 controller × 4 channel × 16-bit = **~150 GB/s** (산술 계산치)
- 산술 검증: 6400 MT/s × 2 bytes × 12 ch = 153.6 GB/s ✓
- **"3 controller × 4 channel" 구성이 Eagle-N 실제인지 BOS spec 문서로 미확인 (debt)**
- 참고: tt-metal 내부는 **Blackhole 참조 GDDR6 = 512 GB/s** 하드코딩 (`operation.cpp:39`) — Eagle-N 아님

### 12.9 PM (Perf Model) 개념 정리
- **PM = tt-metal 내장 정적 성능 모델**. 실행 안 하고 shape+스펙으로 예측.
- **PM_IDEAL = max(PM_COMPUTE, PM_BANDWIDTH)** — 뒤늦은 놈이 병목.
- 우리 bound 분류(75% memory)는 **PM 예측** 기반. **실측** 아님.
- HW 카운터 실측(DRAM_BW_UTIL 등)은 CSV에 **empty**. `--profiler-capture-perf-counters=fpu,pack,unpack,l1_0,instrn` 플래그 붙이고 재실행 필요.

### 12.10 아직 안 한 것 (다음 세션 후보)

**우선순위 순**:
1. **HW 카운터 실측 CSV 재취득** — Tracy에 `--profiler-capture-perf-counters` 플래그 붙여 재실행. `DRAM BW UTIL`, `NOC UTIL`, packer/unpacker efficiency 등 실측치 확보.
2. **진짜 Roofline plot** — 지금은 PM_COMPUTE vs PM_BANDWIDTH scatter만 있음. Op별 FLOPs + bytes 계산해서 (X: intensity, Y: achieved TOPS) 산점도 만들기. Peak TFLOPS/DRAM BW ceiling 그려서 완성.
3. **16-core 강제 실험** — `yolov8s_a0.py` 의 `reshard_if_not_optimal=True` 8곳 다 False. 재실행 → CSV 비교 → diff table. 예상: ~6% slowdown, 실제로 얼마인지 실측.
4. **End-to-end FPS/latency 별도 측정** — `run_yolov8s.py` 콘솔 로그의 FPS 값 수집.
5. **BOS 공식 하드웨어 spec 확인** — 64 TOPS의 정확한 조건, DRAM BW 공식값.
6. **Conv+SiLU fusion 가능성 조사** — Figure 6에서 unary가 22.7% (memory-bound). Fusion 하면 얼마 절약 가능?
7. **Pool2d가 왜 2 코어만 쓰는지 파헤치기** — Baseline이 pool2d 최적화 안 함. 코어 확대 가능성.

### 12.11 파일 지도

```
yolo_study/
├── yolo_study_note.md                              # 이 파일
├── roofline_analysis.py                            # Tracy CSV → 6 plot 스크립트
├── ops_perf_results_2026_07_04_16_02_37.csv        # baseline (YoloV8s) — 사용 중
├── ops_perf_results_2026_07_08_05_15_43.csv        # ★ ResNet50 (혼동 주의)
├── plots/                                          # 오늘 생성한 시각화 6개
│   ├── 01_top_ops.png
│   ├── 02_pm_scatter.png
│   ├── 03_efficiency.png
│   ├── 04_bound_pie.png
│   ├── 05_fpu_hist.png
│   └── 06_optype_stack.png
├── yolo_visualizer/                                # 어제 캡처
│   ├── all_allocations_per_core/
│   └── in_out_allocation_per_core/                 # op144~op158 (op149 재명명 완료)
└── yoloV8s_architecture.png
```

### 12.12 남긴 debt (버그/개선점)
- **Pylance 경고 여러 개** (`roofline_analysis.py`): dispatch table의 `lambda s: 0` unused-param 경고. 무시 가능
- **한글 폰트 없음**: matplotlib이 한글 폰트 못 찾아서 title에 □□ 나옴 → 지금은 영어 라벨로 우회. 나중에 fontproperties 명시 필요하면 손 볼 것
- **`DRAM BW UTIL = 0`은 파싱 버그 아니라 컬럼이 빈 것**: 위 12.10.1 로 해결
- **`GLOBAL CALL COUNT`는 op id 아니고 dispatch 번호**: 값이 1024, 82944 등 큰 값. 시각화에서 이 값으로 sorting 하지 않음 주의

---

## 13. 2026-07-09 세션 — Baseline spec 정량화 (실측 근거)

### 13.1 세션 목적
어제 세션의 PM 기반 bound 분류(75/25)를 **op별 FLOPs·bytes 실측**과 **하드웨어 이론값**으로 확장. 발표에서 "왜 baseline이 개선 여지가 크냐"를 **한 자리 % 숫자**로 뒷받침.

### 13.2 확장한 스크립트
`roofline_analysis.py` 안에 추가:
- Shape column (`INPUT_[0-3]_*_PAD[LOGICAL]`, `OUTPUT_0_*`) + datatype으로 op별 **bytes** 정확 계산
- ATTRIBUTES에서 `SlidingWindowConfig` (input_hw, window_hw, stride_hw, padding) + `output_channels` + `input_tensor_shape` regex 파싱 → conv2d/pool2d/matmul **FLOPs** 정확 계산 (unary/binary는 1 FLOP/elem 근사)
- 상수: `EAGLE_N_PEAK_TFLOPS_LOFI = 80`, `EAGLE_N_PEAK_DRAM_GBPS = 150` (근거 §12.8)
- 두 개 신규 plot + 세 개 신규 markdown 표

### 13.3 하드웨어 이론값 (2026-07-09 밤 갱신 — 소스 재검증 완료, §13.14 참조)

| quantity | value | 근거 |
|---|---:|---|
| Per-core peak (LoFi) at 650 MHz | **2.66 TOPS** | 4096 mul-adds/cycle (`matmul_device_operation.cpp:2173`) × Eagle-N 실측 clock |
| Physical cores (Eagle-N, 6×4) | 24 | BOS spec (physical grid) |
| Compute-usable cores (Eagle-N, 5×4) | 20 | `project_eagle_n_hardware` (harvested 4 core 제외) |
| **Aggregate peak — BOS spec** | **64 TOPS** | 2.66 × 24 = 63.84 → BOS 마케팅 "64 TOPS" 실체 확정 |
| **Aggregate peak — effective** | **53 TOPS** | 2.66 × 20 → 실제 baseline 가용치 |
| Peak DRAM BW (LPDDR5-6400 × 3 ch × 4) | ≈ 150 GB/s | 산술 계산치 (§12.8) — 12-channel 구성 BOS 문서 확인 debt |

**변경 요지**: 이전 표의 "4 TFLOPS/core × 20 = 80 TFLOPS"는 Wormhole 1 GHz 값 인용 오류. Eagle-N 650 MHz 반영하고 physical/effective 두 층으로 분리.

### 13.4 ★ Baseline 지표 요약 — **source tag 포함**

`[실측]` Tracy가 device/host 카운터로 직접 측정 · `[유도]` 실측값에서 결정론적 산술 · `[PM]` tt-metal 정적 성능 모델이 shape로 예측 (실행 안 함) · `[스펙]` tech report / spec sheet 인용, Eagle-N에서 재검증 안 함 · `[계산치]` 스펙 조합의 산술

| 지표 | 값 | source | 신뢰도 | 발표용 |
|---|---:|---|---|---|
| Total kernel time | **13.55 ms** | `[실측]` Tracy device 카운터 합 | 높음 | ✅ |
| Total host-side dispatch | **26.23 ms** | `[실측]` Tracy host 카운터 합 | 높음 | ✅ |
| Kernel / host ratio | **51.6%** | `[실측/실측]` 위 두 값 비 | 높음 | ✅ |
| Op 개수 | 837 | `[실측]` CSV row count | 높음 | ✅ |
| Op-type별 시간 %, 코어 분포 | §13.5 표 | `[실측]` Tracy `DEVICE KERNEL DURATION` + `CORE COUNT` 집계 | 높음 | ✅ |
| Kernel-only 이론 FPS | **73.8** | `[유도]` 1000 / 13.55 — **상한** 명시 필요 (host/trace overhead 미포함) | 높음 (상한 의미로) | ✅ |
| Total FLOPs (정의된 op만) | 14.59 GFLOPs | `[유도]` shape + `SlidingWindowConfig` 파싱. conv2d/matmul 정확, pool2d/unary 근사 | 중간 | 방향성만 |
| Total I/O bytes | 451.79 MB | `[유도]` `INPUT_[0-3]/OUTPUT_0`의 padded logical shape × dtype | 중간 (padded 오차, 코어 간 broadcast 미반영) | 방향성만 |
| Total DRAM-tagged bytes | 79.02 MB | `[유도]` 위와 동일, MEMORY 컬럼에 `DRAM` 포함된 tensor만 | 중간 | 방향성만 |
| Overall achieved TFLOPS | 1.077 | `[유도]` FLOPs / kernel_ns | 중간 (FLOPs 근사) | 방향성만 |
| Overall achieved DRAM BW | 5.83 GB/s | `[유도]` DRAM_bytes / kernel_ns | 중간 | 방향성만 |
| Peak TOPS (BOS spec) | **64** | `[스펙/유도]` 4096 mul-adds/cycle × 650 MHz × 24 phys core. **소스 검증됨 (§13.14)** | 높음 | ✅ |
| Peak TOPS (effective) | **53** | `[유도]` 위 × 20 usable core. Baseline이 실제 쓰는 그리드 | 높음 | ✅ |
| Peak DRAM BW | ≈ 150 GB/s | `[계산치]` LPDDR5-6400 산술. **12-channel 구성 BOS 문서 미확인 (debt)** | 중간 | 상한 근사 |
| Achieved / peak TOPS (spec 64) | **1.68%** | `[유도/스펙]` 1.077 / 64. 벤더 spec 대비 | 중간 | ✅ (bracket) |
| Achieved / peak TOPS (effective 53) | **2.03%** | `[유도]` 1.077 / 53. 실효 대비 | 중간 | ✅ (bracket) |
| Achieved / peak DRAM BW | **3.9%** | `[유도/계산치]` DRAM 구성 미검증 | **낮음 — 방향성만** | 신중히 |
| Time-weighted mean of per-op achieved/peak (spec 64) | **2.11%** | `[유도]` 위 achieved의 시간가중 평균 | 중간 | 정성적 |
| Median per-op achieved/peak (spec 64) | **0.30%** | `[유도]` | 중간 | 정성적 |

### 13.4a 정직한 발표 헤드라인 (신뢰도 반영)

**절대 안전한 문장 (실측만 씀)**:
> "Baseline은 커널 13.55 ms, host dispatch 26.23 ms — kernel/host = 51.6%. Trace 없이 봤을 때 E2E는 host도 병목. Kernel 시간만 기준 이론 FPS 상한은 73.8."

**방향성 문장 (유도값 씀, 근사 명시)**:
> "BOS 공식 spec **64 TOPS** (24 physical cores × 2.66 TOPS/core, LoFi) 대비 실측 커널에서 뽑고 있는 값은 **1.68%**. Baseline이 실제 쓰는 20 usable core 기준 **53 TOPS**로 봐도 **2.03%**. Op별 시간가중으로 봐도 median 0.30% / mean 2.11%."

**말하면 안 되는 문장 (실측이라고 오해 살 것)**:
> ~~"75%가 memory-bound다 (실측)"~~ — 이건 `[PM]` 예측이지 실측 카운터 아님. §13.7과 §12.10-1 참고.

### 13.5 Op-type별 실측 상세

| op_type | n | ms | % | cores | avg TFLOPS | avg GB/s (I/O) | avg GB/s (DRAM) | dominant bound |
|---|---:|---:|---:|---|---:|---:|---:|---|
| `conv2d` | 117 | 5.30 | 39.1% | [2, 4, 8, 10, 20] | **3.05** | 24.8 | 7.9 | memory |
| `unary` (SiLU) | 195 | 3.08 | 22.7% | [2, 4, 8, 16, 19, 20] | 0.008 | 15.6 | 0.0 | memory |
| `matmul` | 96 | 1.62 | 12.0% | [2, 4, 8, 16, 19] | **2.63** | 55.7 | 8.3 | memory |
| `pool2d` | 9 | 0.79 | 5.9% | **[2]** | 0.005 | 1.5 | 0.0 | compute |
| `interleaved_to_sharded` | 39 | 0.68 | 5.0% | [2, 8, 16, 19] | — | 221.6 | 2.2 | memory |
| `halo` | 126 | 0.61 | 4.5% | [2, 4, 8, 16, 19, 20] | — | 106.6 | 0.0 | memory |
| `concat` | 48 | 0.35 | 2.6% | [2, 8, 16, 19, 20] | — | 268.4 | 7.8 | memory |
| `untilize` | 27 | 0.29 | 2.2% | [2, 8, 16] | — | 17.1 | 0.0 | compute |
| `tilize` | 33 | 0.19 | 1.4% | [1, 16] | — | 147.4 | 0.0 | compute |
| `reshard` | 30 | 0.05 | 0.4% | [2, 8, 16, 19] | — | **702.1** | 0.0 | compute |

**정량적 발견 3가지:**
1. **Conv/matmul은 각자 실행 시 3~6 TFLOPS 뽑음 (~5% of peak). 그런데 전체 시간의 39+12 = 51%만 여기에 쓴다.** 나머지 49%가 unary·data-movement·dispatch로 흘러가서 overall이 1.35%로 폭락. **Fusion과 layout 정리가 유일한 배수 개선 지점.**
2. **Pool2d는 kernel 자체가 2 코어 강제 (compute-bound + 0.005 TFLOPS)** — 20 코어로 확장하면 이론상 10배 빨라짐. 근데 pool은 5.9%밖에 안 됨 → 절대 이득은 작음.
3. **Reshard / concat / interleaved_to_sharded 같은 L1 내부 이동이 200~700 GB/s로 아주 잘 뽑고 있음** → 여기가 병목 아님. 병목은 DRAM 왕복(conv weight 등)과 unary의 짧은 kernel/dispatch overhead.

### 13.6 신규 plot 2개

| # | 파일 | 메시지 |
|---|---|---|
| 7 | `plots/07_real_roofline.png` | X: FLOPs/byte, Y: achieved TFLOPS. **BOS spec 64 TOPS 실선 + effective 53 TOPS 점선 + DRAM 150 GB/s 대각선**. Conv/matmul은 BW 라인 근처, unary·pool2d는 roof에서 크게 이탈 (kernel launch 오버헤드 지배) |
| 8 | `plots/08_achieved_pct.png` | 시간가중 achieved/peak (%) histogram. Peak = **BOS spec 64 TOPS**. **68% wall-time이 <1% 자릿수, 시간가중 mean 2.11% / median 0.30%** |

### 13.7 어제 노트 §12.5 값 갱신 — **다른 metric임을 주의**

두 세션의 값은 서로 다른 metric이라 직접 비교 못 함. 나란히 놓고 방향성만.

| 값 | 어제 (7/8) | 오늘 (7/9) | 비고 |
|---|---:|---:|---|
| §12.4 "FPU util 시간가중 mean" | 3.0% | — | 어제는 `[PM]` PM_FPU_UTIL 컬럼 (perf-model 예측치) |
| Time-weighted mean of achieved/peak | — | 1.69% | 오늘은 `[유도]` (achieved TFLOPS / spec peak). `[스펙]` peak가 미검증이라 이 %도 근사 |
| §12.4 "FPU util 중앙값" | 1.5% | — | 어제 `[PM]` |
| Median achieved/peak | — | 0.24% | 오늘 `[유도]` |

**결론적 방향성 (metric 무관 공통)**: baseline은 시간의 60% 이상이 <1% 계산 유틸 영역에서 흐름. 두 다른 metric이 같은 정성적 방향을 가리키니 관찰은 견고.

**FLOPs 정확도 주의**:
- Conv2d/matmul FLOPs: `[유도]` `2·B·H_out·W_out·C_in·C_out·K²`. attribute의 `SlidingWindowConfig` + `input_tensor_shape` 파싱이라 **정확** (padding으로 인한 C_in 확장이 반영되지만 실효 계산량은 근사)
- Pool2d FLOPs: `[유도]` `output_elems × K²`. 최대 pool의 compare 카운트. **근사**
- Unary FLOPs: `[유도]` `1 FLOP/elem` — SiLU 실제는 sigmoid + mul ≈ 4~5 FLOPs/elem이라 **1/4~1/5 저평가**. 그래도 arithmetic intensity 0.25 수준이라 memory-bound 결론 불변.
- Bytes: `[유도]` padded logical shape 사용이라 실제 이동 bytes를 **넉넉하게 상한 근사**. Weight 코어-간 broadcast/L1 재사용 미반영이라 DRAM 값은 **하한**에 가까움.

### 13.8 E2E 지표 (kernel-only)

| metric | value |
|---|---:|
| Total device kernel time | 13.55 ms |
| Total device firmware time | 14.41 ms |
| Total host-side op dispatch | 26.23 ms |
| Kernel / host ratio | **51.6%** |
| **Kernel-only theoretical FPS** | **73.8** |

**주의**: 실제 E2E FPS는 host+dispatch+trace overhead 포함. Trace on 상태면 host time이 첫 iter만 크고 나머지는 replay되므로 실측 필요. `run_yolov8s.py --trace -n 100` 콘솔 로그로 확정할 것 (§12.10-4에 남긴 debt).

### 13.9 발표 슬라이드 재구성 제안

```
Slide  Fig            메시지 (한 줄)
─────  ─────────────  ─────────────────────────────────────────────
1      Fig 4 pie      "75% memory-bound — kernel 안에서만 봐도"
2      §13.4 표       "peak 80 TFLOPS 중 1.35%만 씀"    ← ★ 킬링
3      Fig 7 roofline "Conv/matmul은 BW ceiling에 매달림"
4      Fig 8 hist     "65% 시간이 <1% util로 흘러감"
5      §13.5 표       "Fusion·pool2d 확장·host dispatch가 3대 축"
6      Fig 6 stack    "op-type별 어디서 시간 새나가나 (부록)"
```

### 13.10 다음 세션 debt (기존 §12.10 재확인 + 신규)

**로컬로 가능:**
- Unary FLOPs 정확도 개선 (SiLU = 5 FLOPs/elem 등 op별 계수)
- Roofline plot에 op-id annotation (top 5 conv 이름 표시)

**서버 재실행 필요 (기존 debt):**
- HW 카운터 CSV 재취득 (`--profiler-capture-perf-counters=fpu,pack,unpack,l1_0,instrn`) → NOC/DRAM_BW_UTIL 실측
- 16-core 강제 실험 (`reshard_if_not_optimal=True` 8곳 False)
- E2E FPS 콘솔 로그 수집
- Conv+SiLU fusion 실험 (achievable 상한 재측정)

### 13.11 파일 추가

```
yolo_study/
├── roofline_analysis.py               # ★ 확장 (§13.2)
├── plots/
│   ├── 07_real_roofline.png           # ★ 신규
│   ├── 08_achieved_pct.png            # ★ 신규
│   ├── table_hw_constants.md          # ★ 신규
│   ├── table_optype.md                # ★ 신규
│   └── table_e2e.md                   # ★ 신규
```

---

### 13.12 ★ 출처 & 신뢰도 정리 — "이 자료가 유의미한가?"

**§13은 그 자체로 정량화된 spec sheet가 아님.** 여러 종류의 값이 섞여있고, 종류마다 신뢰도가 다르다. 발표·논문에서 인용할 때는 반드시 category를 함께 명시.

#### A. 완전 실측 (Tracy device/host 카운터 직접 측정) — **논쟁 여지 없음**

| 값 | 어디서 |
|---|---|
| 각 op의 `DEVICE KERNEL DURATION [ns]` (837개) | Tracy device profiler |
| 각 op의 `HOST DURATION [ns]` | Tracy host timer |
| 각 op의 `CORE COUNT` | 실제 배정된 tensix core 수 |
| **Total kernel 13.55 ms** | 위 합 |
| **Total host 26.23 ms** | 위 합 |
| **Kernel/host 51.6%** | 두 실측의 비 |
| Op-type별 시간 분포 (§13.5의 ms / % / cores 열) | 실측 그룹핑 |

이 값들은 **재취득 없이도 바로 발표에 쓸 수 있음**. 하드웨어에서 나온 원시 카운터.

#### B. 실측에서 결정론적으로 유도 — **거의 확실**

| 값 | 유도 방법 | 근사 요소 |
|---|---|---|
| Bytes (per op, total) | padded logical shape × dtype_bytes | Padded 오버헤드로 실제 이동보다 크게 나옴. Broadcast/L1 재사용 미반영 |
| Conv2d FLOPs | attribute의 `SlidingWindowConfig` + `input_tensor_shape` + `output_channels`로 정확 공식 | Padded C_in을 그대로 씀 (첫 conv에서 3 vs 32 차이) |
| Matmul FLOPs | INPUT shape에서 M/K/N 추출, `2·M·K·N` | tt-metal의 matmul은 padded 이므로 실효 계산량은 이 값 이하 |
| Kernel-only 이론 FPS 73.8 | 1000 ms / 13.55 ms | Host+trace overhead 미포함 → **상한**이라고 명시해야 정직 |

이 값들도 **발표에서 쓸 수 있음** — 단 "이론 상한", "padded 기준" 같은 한정어를 반드시 붙일 것.

#### C. 근사가 큰 유도 — **방향성만**

| 값 | 왜 근사인지 |
|---|---|
| Unary (SiLU) FLOPs | 1 FLOP/elem으로 잡음. 실제 SiLU는 sigmoid + mul ≈ 4~5 FLOPs/elem → **4~5배 저평가** |
| Pool2d FLOPs | `elems × K²`의 compare 카운트 근사. max-pool 실제는 비교뿐이라 이 자체가 근사. Pool2d 시간 비중 5.9%라 전체 영향은 적음 |
| Overall achieved TFLOPS (1.077) | 위의 근사가 total FLOPs에 반영 |

#### D. tt-metal 정적 성능 모델 (PM) — **실행 안 함, 예측치**

| 값 | source column |
|---|---|
| **Memory-bound 75.1% / compute-bound 24.9%** (§12.4, Fig 4) | `PM COMPUTE [ns]` vs `PM BANDWIDTH [ns]` 비교로 분류 |
| §12.4의 FPU util median 1.5% / mean 3.0% (Fig 5) | `PM FPU UTIL (%)` 컬럼 |
| Efficiency (achieved/ideal, §12.6 Fig 3) | `PM IDEAL [ns]` 사용 |

**중요**: 이 값들은 **실측 아님**. tt-metal의 정적 성능 모델이 op의 shape/dtype만 보고 예측한 값. 발표에서 "실측 memory-bound"라고 하면 안 됨. **"tt-metal perf-model이 memory-bound로 분류"**가 정확한 표현. 실측 카운터(`DRAM_BW_UTIL`, `NOC_UTIL`)는 이번 CSV에서 empty (§12.10-1).

#### E. 스펙 인용 / 계산치 — **Eagle-N에서 미검증**

| 값 | 출처 | 왜 조심해야 하나 |
|---|---|---|
| Per-core 2.66 TOPS (LoFi @ 650 MHz) | `matmul_device_operation.cpp:2173` (4096 mul-adds/cycle) × `profile_log_device.csv` clock 실측 | **소스 검증됨 (§13.14)**. 초기 노트의 "4 TFLOPS" 는 Wormhole 1 GHz 값이라 폐기 |
| Aggregate 64 TOPS (BOS spec) | 2.66 × 24 physical cores | **BOS 마케팅 값 재구성 성공** — physical grid 기준 (6×4) |
| Aggregate 53 TOPS (effective) | 2.66 × 20 compute-usable | Baseline이 실제 쓰는 그리드 (5×4). §12.10-2 GEMM microbenchmark로 sustained 검증 후속 debt |
| DRAM 150 GB/s | LPDDR5-6400 × 3 controller × 4 ch × 16-bit 산술 | 산술 자체는 정확. **12-channel 구성이 Eagle-N 실제인지 BOS 문서 미확인** |

**Achieved-vs-peak %(1.35%, 3.9%)는 D의 유도값 ÷ E의 스펙**이라 절대 % 는 신뢰도 낮음. **방향성**은 유지 (여러 배수 여유 있음).

#### 발표에서 어떻게 골라 쓸지 — 결론

**꼭 쓸 것 (A + B의 상한형)**:
- Kernel 13.55 ms / Host 26.23 ms / **Kernel/host 51.6%**
- Op-type별 시간·코어 분포 표 (§13.5의 실측 열들만)
- **Pool2d 2 코어 vs Conv2d 20 코어 극단** (실측)
- Kernel-only 이론 FPS 73.8 (상한 명시)

**"perf-model 기준"이라고 명시하고 쓸 것 (D)**:
- Memory-bound 75% (Fig 4) — 반드시 "PM 분류"로 서두

**"이 정도 자릿수"라고만 쓸 것 (C+E)**:
- Achieved 1% 대 자릿수, DRAM 몇 % 대 자릿수 — 정확 값 X, 방향성 O

#### 다음 세션에서 **꼭 해야** 신뢰도가 올라가는 것

1. **HW 카운터 재취득** — `DRAM BW UTIL`, `NOC UTIL`, packer/unpacker efficiency를 실측으로. §12.10-1의 debt. 이것만 해도 D의 memory-bound %가 실측 근거가 됨
2. **Eagle-N GEMM 벤치마크** — `tt-metal/tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/`류 하나 돌려서 우리 board에서 sustained TFLOPS 실측. 이것만 해도 E의 peak 80이 검증됨
3. **BOS 공식 DRAM BW 확인** — 사내 문서/spec sheet. 이것만 해도 150 GB/s가 검증됨
4. **E2E FPS 콘솔 로그** — trace 켜고 100 iter. 이것만 해도 host+trace overhead 실측됨

이 넷 중 셋만 해도 §13.4 표 대부분 값이 "낮음"에서 "높음"으로 올라감.

(python_env) root@7575fef6ac31:/tt-metal# ls -la /tt-metal/generated/profiler/reports/2026_07_09_06_40_40/
total 297996
drwxr-xr-x  2 root root      4096 Jul  9 06:40 .
drwxr-xr-x 12 root root      4096 Jul  9 06:40 ..
-rw-r--r--  1 root root    980940 Jul  9 06:40 ops_perf_results_2026_07_09_06_40_40.csv
-rw-r--r--  1 root root  16511454 Jul  9 06:40 profile_log_device.csv
-rw-r--r--  1 root root 287634365 Jul  9 06:40 tracy_profile_log_host.tracy
(python_env) root@7575fef6ac31:/tt-metal# find /tt-metal -type d -name "*profiler*" 2>/dev/null | head
/tt-metal/tests/tt_metal/tools/profiler
/tt-metal/generated/profiler
/tt-metal/ttnn/api/tools/profiler
/tt-metal/.build/default/tt_metal/impl/CMakeFiles/impl.dir/ASan/profiler
/tt-metal/.build/default/tt_metal/impl/CMakeFiles/impl.dir/Release/profiler
/tt-metal/.build/default/tt_metal/impl/CMakeFiles/impl.dir/Debug/profiler
/tt-metal/.build/default/tt_metal/impl/CMakeFiles/impl.dir/RelWithDebInfo/profiler
/tt-metal/.build/default/tt_metal/impl/CMakeFiles/impl.dir/TSan/profiler
/tt-metal/.build/default/tt_metal/programming_examples/profiler
/tt-metal/.build/default/tt_metal/programming_examples/profiler/CMakeFiles/test_noc_event_profiler.dir
(python_env) root@7575fef6ac31:/tt-metal# grep -rn "ops_perf_results" /tt-metal --include="*.py" 2>/dev/null | head
/tt-metal/tests/ttnn/profiling/profile_host_overhead_with_tracy.py:115:        if not basename.startswith("ops_perf_results_"):
/tt-metal/tools/tracy/process_ops_logs.py:40:OUT_NAME = "ops_perf_results"
/tt-metal/tools/tracy/process_model_log.py:22:    filename = output_report_dir / runDate / f"ops_perf_results_{runDate}.csv"
/tt-metal/python_env/lib/python3.10/site-packages/ttnn_visualizer/tests/test_file_uploads.py:483:                (BytesIO(b"op,perf\n"), "perf_run/ops_perf_results_2026.csv"),
/tt-metal/python_env/lib/python3.10/site-packages/ttnn_visualizer/tests/test_file_uploads.py:494:    assert any(p.name.startswith("ops_perf_results") for p in report_dir.iterdir())
/tt-metal/python_env/lib/python3.10/site-packages/ttnn_visualizer/views.py:1072:        if not any(file.name.startswith("ops_perf_results") for file in files):
/tt-metal/python_env/lib/python3.10/site-packages/ttnn_visualizer/views.py:1393:        pattern="ops_perf_results",
/tt-metal/python_env/lib/python3.10/site-packages/ttnn_visualizer/csv_queries.py:271:    PERF_RESULTS_PREFIX = "ops_perf_results"
/tt-metal/python_env/lib/python3.10/site-packages/ttnn_visualizer/csv_queries.py:468:        ops_perf_results = []
/tt-metal/python_env/lib/python3.10/site-packages/ttnn_visualizer/csv_queries.py:469:        ops_perf_results_reader = csv.DictReader(csv_file)
(python_env) root@7575fef6ac31:/tt-metal# find /tt-metal -name "process_ops*" 2>/dev/null
/tt-metal/tools/tracy/process_ops_logs.py
/tt-metal/tools/tracy/__pycache__/process_ops_logs.cpython-310.pyc

(python_env) root@7575fef6ac31:/tt-metal# grep -n "NOC UTIL\|DRAM BW UTIL\|NPE CONG\|noc_util\|dram_bw_util" /tt-metal/tools/tracy/process_ops_logs.py
100:    "NOC UTIL (%)",
101:    "DRAM BW UTIL (%)",
102:    "NPE CONG IMPACT (%)",
439:                    ops[op_id]["NOC UTIL (%)"] = round(op_npe_stats.result.overall_avg_link_util, 1)
440:                    ops[op_id]["DRAM BW UTIL (%)"] = round(op_npe_stats.result.dram_bw_util, 1)
441:                    ops[op_id]["NPE CONG IMPACT (%)"] = round(op_npe_stats.result.getCongestionImpact(), 2)
795:                if "NOC UTIL (%)" in opData:
796:                    rowDict["NOC UTIL (%)"] = opData.get("NOC UTIL (%)")
797:                if "DRAM BW UTIL (%)" in opData:
798:                    rowDict["DRAM BW UTIL (%)"] = opData.get("DRAM BW UTIL (%)")
799:                if "NPE CONG IMPACT (%)" in opData:
800:                    rowDict["NPE CONG IMPACT (%)"] = opData.get("NPE CONG IMPACT (%)")
(python_env) root@7575fef6ac31:/tt-metal# grep -n "profile_log_device\|device_log" /tt-metal/tools/tracy/process_ops_logs.py | head -20
22:from tracy.process_device_log import import_log_run_stats
448:    logFolder, outputFolder, date, nameAppend, export_csv=True, cleanup_device_log=False, device_analysis_types=[]
638:        if cleanup_device_log:
(python_env) root@7575fef6ac31:/tt-metal# head -60 /tt-metal/generated/profiler/reports/2026_07_09_06_40_40/profile_log_device.csv
ARCH: blackhole, CHIP_FREQ[MHz]: 650
PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID,  zone name, type, source line, source file, meta data
0,0,0,BRISC,34774,11599508671958,0,1024,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11599508672618,0,1024,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11599508674208,0,1024,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11599508674354,0,1024,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11599938596690,0,2048,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11599938599278,0,2048,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11600666607763,0,3072,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11600666608459,0,3072,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11600666850741,0,3072,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11600666854243,0,3072,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11601332988250,0,4096,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11601332988928,0,4096,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11601333034706,0,4096,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11601333034857,0,4096,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11601999637757,0,5120,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11601999638354,0,5120,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11601999761709,0,5120,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11601999762689,0,5120,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11602454179678,0,6144,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11602454180164,0,6144,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11602454182400,0,6144,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11602454182546,0,6144,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11603056767959,0,7168,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11603056768608,0,7168,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11603056773101,0,7168,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11603056773242,0,7168,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11603921451174,0,8192,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11603921452065,0,8192,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11603921458609,0,8192,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11603921458794,0,8192,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11604423107614,0,9216,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11604423108311,0,9216,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11604423122872,0,9216,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11604423123644,0,9216,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11605227417709,0,10240,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11605227418885,0,10240,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11605227453035,0,10240,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11605227485879,0,10240,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11606047765922,0,11264,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11606047766829,0,11264,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11606047788972,0,11264,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11606047789138,0,11264,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11606049000262,0,12288,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11606049001168,0,12288,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11606049023305,0,12288,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11606049023487,0,12288,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11606625340129,0,13312,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11606625340683,0,13312,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11606625350899,0,13312,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11606625351050,0,13312,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11607314520983,0,14336,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11607314521841,0,14336,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11607314534700,0,14336,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11607314534881,0,14336,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,34774,11607316072111,0,15360,BRISC-FW,ZONE_START,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,
0,0,0,BRISC,64422,11607316072822,0,15360,BRISC-KERNEL,ZONE_START,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,64422,11607316087420,0,15360,BRISC-KERNEL,ZONE_END,49,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisck.cc,
0,0,0,BRISC,34774,11607316088111,0,15360,BRISC-FW,ZONE_END,402,/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc,

---

## 13.13 ★ 안전 안 발표 구조 확정 (PM 완전 배제)

### 방침
- **§13.4의 A 카테고리 (완전 실측)만 발표에 사용**.
- PM 기반 값 (memory-bound 75%, FPU util 1.69%, achieved-vs-peak 1.35%) → **발표에서 뺌**. 노트에 debt로만 남김.
- 이유: 컴퓨터 아키텍처 랩 청중에게 "실측이냐 예측이냐"의 방어 부담 최소화. 실측만으로도 baseline의 개선 여지 서사 충분.

### 발표 슬라이드 5장 outline (실측만)

```
Slide 1  Kernel 13.55 ms / Host 26.23 ms / Kernel/host 51.6%   [실측]
         메시지: "Device compute는 짧고, host dispatch가 더 오래 걸린다.
                 E2E 병목이 device 안에 있는 게 아니라 host-device 경계에 있음."
         근거 컬럼: DEVICE KERNEL DURATION [ns], HOST DURATION [ns]

Slide 2  Op-type별 kernel 시간 분포                             [실측]
         conv2d 39.1% · unary 22.7% · matmul 12.0% · pool2d 5.9%
         메시지: "Conv2d와 fusion 안 된 SiLU 두 개가 시간의 62% 차지.
                 SiLU가 22.7%라는 건 개선 지점 하나."
         근거 컬럼: 위 그룹핑

Slide 3  Op-type × 코어 배정 실측 극단                          [실측]
         pool2d [2] · conv2d [2,4,8,10,20] · matmul [2,4,8,16,19]
         메시지: "같은 board에서 op-type마다 활용 코어 편차 극심.
                 Pool2d는 20 코어 중 2개만 쓴다."
         근거 컬럼: OP CODE + CORE COUNT 그룹핑

Slide 4  Kernel-only 이론 FPS 상한 73.8                        [실측 유도]
         근거: 1000 ms / 13.55 ms
         한정어: "kernel-only 상한. host+trace overhead 미포함이라
                 실제 E2E FPS는 이 값 이하."
         메시지: "BOS 공식 64 TOPS급 하드웨어에서 여기까지가 kernel time만의 상한.
                 나머지 개선은 host/trace overhead 감소가 병렬 축."

Slide 5  Interpretation & Next steps                          [해석]
         관찰 → 방향 3축:
           1. Unary가 22.7% → conv+SiLU fusion 여지
           2. Pool2d 2 코어 → 커널 병렬화 확대 여지
           3. Host dispatch가 kernel의 2배 → trace/2CQ 활용 검토
         Future work:
           • Dynamic bound classification (HW counter 실측) — pipeline 확정, 회수 예정
           • Peak throughput 검증 (GEMM microbenchmark on Eagle-N)
           • E2E FPS 실측 (--trace -n 100)
```

### 발표에서 **안 쓰는 것** (실수 방지)
- 기존 Fig 1~6 (모두 PM 색상/축 의존)
- ~~Fig 7/8 (achieved TFLOPS 축이 peak 미검증에 의존)~~ → **§13.14 이후 사용 가능 (peak 값 소스 검증됨)**
- "Memory-bound 75%" 문구 (PM 분류. 실제로는 §13.14 후속 재계산: 47% memory + 28% PM 미구현)
- ~~"1.35% of peak / 3.9% of DRAM" 킬링 넘버~~ → **§13.14 이후 "1.68% of BOS spec 64 TOPS"로 사용 가능**

### 필요하면 급히 만들 plot 1개 (선택)

Slide 2·3용 시각화 하나만 있으면 slide가 채워짐. 실측만으로 가능:

- **Op-type별 wall-time 막대 (색은 uniform, PM bound 색상 X)**: 기존 `plot_optype_stack`을 색 없이 재실행하면 됨. `roofline_analysis.py`에서 bound 스택 대신 단일 색 bar로 재생성.
- **Op-type별 core-count 분포 boxplot**: 새 함수. Pool2d 2 코어 극단이 시각적으로 튐.

시간 여유 있으면 추가, 없으면 표만.

### Q&A 준비 3문장

**Q: memory-bound인지 compute-bound인지는 안 봤나?**
> 정적 roofline classification은 tt-metal profiler 결과에 있고 방향성은 그쪽으로 갑니다. 다만 발표에서는 wall-time 실측과 op-type 코어 분포만 사용해서 정확도를 담보했습니다. Dynamic HW counter 실측은 다음 iteration debt입니다.

**Q: 왜 kernel/host ratio가 발표 축인가?**
> Trace off인 이번 measurement에서는 host dispatch가 kernel 시간의 두 배. 이건 trace on/2CQ로 잡을 여지가 큰 축이라 baseline의 개선 우선순위 판정 근거입니다.

**Q: Pool2d 2 코어는 baseline의 config 이슈인가 kernel 이슈인가?**
> Yolov8s_a0 config의 UpsampleConfig/ConcatConfig에 max_grid_size=(4,4) 하드코딩이 있는데 pool2d는 별개 kernel이라 이 제약 밖. Kernel 내부의 parallelization strategy 문제입니다. 노트 §3에 정리돼있습니다.

### 노트 상태 정리
- §13.9 (기존 발표 슬라이드 제안, Fig 4 pie 시작) → **이 §13.13이 대체**. §13.9은 alternative로 남김.
- §13.10의 로컬 debt (unary FLOPs 정확도, roofline annotation) → 안전 안엔 불필요, 후순위.
- §13.10의 서버 debt (HW counter, GEMM peak, DRAM spec, E2E FPS) → 발표 후 신뢰도 격상용으로 계속 유효.

---

## 13.14 Peak 값 소스 검증 & Fig 4 null 분리 (2026-07-09 밤)

### 13.14.1 세션 목적
§13.4의 "peak TFLOPS 80" 과 §12.6 "Fig 4 파이 75/25"의 근거가 약하다는 판단으로, tt-metal 소스와 CSV 실측을 직접 대조해서 값을 재확정.

### 13.14.2 발견 1 — TFLOPS peak 80은 오류 (Wormhole 값 인용)

**추적 경로**:
1. `roofline_analysis.py:32` 에 `BH_PEAK_TFLOPS_PER_CORE_LOFI = 4.0` 하드코딩
2. 근거로 인용한 `GEMM_FLOPS.md` line 47-51 실측 확인:
   - "At **1.35 GHz (BH)**, this is **5.4 TFLOPS** per matrix engine"
   - "At **1 GHz (WH)**, this is **4 TFLOPS** per matrix engine"
3. 노트는 Wormhole 값 (4)을 Blackhole architecture인 Eagle-N에 적용 → **아키텍처 혼동**
4. Eagle-N 실측 clock: `profile_log_device.csv` header에 `ARCH: blackhole, CHIP_FREQ[MHz]: 650` — Blackhole 참조(1.35 GHz)의 절반 미만

**정확한 재계산 (matmul PM 공식 직접 적용)**:
- `matmul_device_operation.cpp:2173`: `constexpr int tensix_mul_adds_per_cycle_lofi = 4096;`
- Per-core = 4096 × 650×10⁶ / 1e12 = **2.66 TOPS/core**

### 13.14.3 발견 2 — BOS "64 TOPS" 실체 확인

**결정적 산술 매칭**:
```
24 physical cores × 2.66 TOPS/core = 63.84 TOPS ≈ 64 TOPS  ✓
```

즉 **BOS 마케팅 "64 TOPS" = physical 6×4 grid 기준 spec**. Harvested 4 core 포함. Chip 마케팅 관행 (공정 수율에 따라 disable된 코어까지 다 카운트).

- **BOS spec peak = 64 TOPS** (24 physical cores)
- **Effective peak = 53 TOPS** (20 compute-usable cores, baseline이 실제 쓰는 그리드)

**TOPS ↔ TFLOPS 관행**: tt-metal은 1 mul-add를 1 op으로 셈 (GEMM_FLOPS.md도 동일: WH 4096 × 1 GHz = 4 TFLOPS). LoFi matmul에서 datatype 불문 tile당 op count 같음.

### 13.14.4 발견 3 — DRAM 150 GB/s 산술은 정확, 구성은 미확인

**검증**:
- 노트 산술: LPDDR5-6400 × 3 controller × 4 channel × 16-bit
- 재계산: 6400 MT/s × 2 bytes × 12 channels = **153.6 GB/s** ✓

**미확인 부분**: "3 controller × 4 channel = 12 channels" 구성이 Eagle-N 실제 하드웨어 사양인지 BOS 문서로 확인 안 됨 (사용자 추정치). tt-metal 소스에는 Eagle-N 전용 SoC descriptor 없음. `operation.cpp:39`의 Blackhole 참조는 GDDR6 512 GB/s (P150 보드), Eagle-N 아님.

**→ 여전히 debt**. 산술 자체는 신뢰, 구성 가정은 미확인.

### 13.14.5 코드 & Plot 갱신 완료

**`roofline_analysis.py:26-45` 상수 블록**:
```python
BH_PEAK_TOPS_PER_CORE_LOFI    = 2.66     # 4096 × 650 MHz
EAGLE_N_PHYSICAL_CORES        = 24
EAGLE_N_COMPUTE_CORES         = 20
EAGLE_N_PEAK_TOPS_SPEC        = 64.0     # BOS spec, ceiling
EAGLE_N_PEAK_TOPS_EFFECTIVE   = 53.0     # 20 usable core
EAGLE_N_PEAK_DRAM_GBPS        = 150.0    # 산술 계산치 (구성 debt)
```

**Fig 7 재생성**: BOS spec 64 TOPS 실선 + Effective 53 TOPS 점선 두 층. 서사 시각화 완결.

**Fig 8 재생성**: X축 = "Achieved TFLOPS / BOS spec peak (64 TOPS) (%)". 재계산 결과:
- **median = 0.30%** (기존 0.24%)
- **time-weighted mean = 2.11%** (기존 1.69%)
- Peak 낮아진 만큼 %가 자연스레 올라감. 방향성 (여전히 <5%) 동일.

### 13.14.6 발견 4 — Fig 4 파이의 null 분리 (부수 성과)

같은 세션에서 발견: `roofline_analysis.py:293-297`의 분류 로직이 PM null cluster (unary·halo·upsample·binary_ng, 27.6% wall time)를 memory로 흡수시킴 (pc=pb=1일 때 `pc > pb`가 False라 memory로 판정).

**수정**: `pm_compute <= 1 and pm_bandwidth <= 1`을 "null"로 별도 분류. Fig 4 파이가 **75/25 → 47.4/24.9/27.6 3분할**로 재생성.

**의미**: PM null = `create_op_performance_model()` 미구현 op의 default constructor `{compute=1, bw=1}` 값 (operation.hpp:95). tt-metal의 op coverage 문제이지 YoloV8s 특유 이슈 아님.

### 13.14.7 발표용 갱신 결과 정리

| 항목 | 이전 값 | 갱신 값 | 근거 상태 |
|---|---:|---:|---|
| Peak compute | 80 TFLOPS | **64 TOPS (spec) / 53 TOPS (effective)** | 소스 검증 완료 ✅ |
| Peak DRAM BW | 150 GB/s | 150 GB/s (유지) | 산술 확인, 구성 debt 유지 |
| Achieved / peak | 1.35% | **1.68% (spec) / 2.03% (effective)** | 검증된 peak 반영 |
| Fig 8 mean/median | 1.69% / 0.24% | **2.11% / 0.30%** | 자동 갱신 |
| Fig 4 memory 비중 | 75.1% | **47.4% memory + 27.6% PM 미구현** | 분류 로직 수정 |

### 13.14.8 남은 debt (§12.10 계승)

**변화 없음 — 여전히 유효**:
1. HW counter 재취득 (`--profiler-capture-perf-counters`) — NOC/DRAM_BW_UTIL 실측
2. Eagle-N GEMM microbenchmark — per-core 2.66 TOPS의 sustained 검증
3. BOS 공식 DRAM spec sheet — 12-channel 구성 확인
4. E2E FPS 콘솔 로그 (`--trace -n 100`)

이 넷 완료 시 §13.14의 "높음" 신뢰도가 "실측 검증" 등급으로 격상됨.

### 13.14.9 파일 상태

**변경된 파일**:
- `roofline_analysis.py` — 상수 블록 재작성, 분류 로직 수정 (293-301), Fig 7 이중 ceiling, Fig 8 라벨 갱신
- `plots/07_real_roofline.png` — 두 ceiling 층 표시
- `plots/08_achieved_pct.png` — peak 64 TOPS 기준 재계산
- `plots/04_bound_pie.png` — 3분할 (compute/memory/null)
- `plots/table_hw_constants.md` — 새 상수 반영
- `yolo_study_note.md` — §12.8 / §13.3 / §13.4 / §13.4a / §13.6 / §13.12-E / §13.13 갱신, §13.14 신규

**폐기 값 (인용 금지)**:
- ~~"80 TFLOPS peak"~~
- ~~"1.35% of peak"~~
- ~~"75.1% memory-bound (실측)"~~ — PM 예측이고 null 흡수됨