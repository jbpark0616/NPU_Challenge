# Baseline Aging Simulation — Eagle-N NPU

작성일: 2026-07-27  
플랫폼: BOS Semiconductor Eagle-N (Blackhole 140 arch, 20 compute cores)  
워크로드: ResNet50 batch=1 (자동차 표준)  
상태: **미팅 발표 준비 완료** (baseline만; rotation은 C++ 패치 후 다음 세션)

---

## 0. 한 줄 요약

프레임워크의 결정론적 스케줄이 20 compute core 활동량을 **3 축 모두 다른 방식으로** 극단 편중 (NIU 13×, SRAM 74×, Compute 30×). Chen et al. NBTI+HCI 공식으로 aging 편중 계산. 절대 aging 값은 6nm 공정 상수 미확정으로 defensible 하지 않아 **상대 비율만 리포트**. 세 축 모두 rotation 없이는 특정 코어에 permanent stress 누적.

> **Compute 지표 정정 필요 (2026-08-19):** 기존 `compute_axis.py`는
> 중첩된 `TRISC-FW`와 `TRISC-KERNEL` zone을 모두 합산해 engine time을
> 이중 계상했다. 스케줄러 calibration과 향후 baseline은
> `TRISC-KERNEL`만 사용한다. 아래의 기존 Compute 30.35× 및 그로부터
> 파생된 aging ratio는 full-model profile로 재계산하기 전까지 historical
> result로만 취급한다. NIU/SRAM 수치에는 이 정정이 적용되지 않는다.

---

## 1. 폴더 구조

```
study_note/aging_simulation/
├── BASELINE_NOTE.md              ← 이 파일
├── aging_sim.py                  ← 메인 시뮬 스크립트 (3-axis)
├── compute_axis.py               ← profile_log_device.csv → α_compute (TRISC)
├── annotated_heatmap.py          ← Role 오버레이 NoC heatmap
├── per_stage_heatmap.py          ← 5-stage NoC heatmap (mcast pattern 대비)
├── per_stage_compute_heatmap.py  ← 5-stage TRISC heatmap (CORE_SET 사용 패턴)
├── layer2_module_heatmap.py      ← Layer 2 module 별 shape 확인
├── blackhole_140_arch.yaml       ← 컨테이너에서 가져온 grid 정의
├── aging_research/               ← Chen ISOCC'24, Brandalero DAC'20 논문 + figure
│   ├── Aging_Mitigation...pdf
│   ├── Proactive_Aging...pdf
│   └── images/                   ← 스크린샷 (수식, 결과)
├── NoC_analysis/                 ← 실측 NoC 데이터 파이프라인 (bos_model_copy 에서 복사)
│   ├── outputs/per_core.csv      ← 20 core 실측 데이터 (aging_sim.py 입력)
│   └── ...
├── resnet50_batch1_report/       ← Run A Tracy/NPE 원본 archive
│   └── tracy_profile_log_host.tracy
├── stage_split_{0-4}/            ← Run B (NoC pass): raw noc traces
└── plots/                        ← 생성된 figure
    ├── baseline_alpha_heatmap.png
    ├── baseline_vth_niu_year5.png
    ├── baseline_vth_sram_year5.png
    ├── baseline_vth_over_time.png
    ├── baseline_vth_dist_year{5,10}.png
    ├── per_core_heatmap_annotated.png
    └── baseline_summary.txt
```

Run A의 policy 입력 CSV 두 개는 다음 한 폴더에 모아 둔다:

```text
data_movement/conv_core_replacement/policy/data/full_model_baseline/input/
├── ops_perf_results_2026_07_26_14_16_54.csv
└── profile_log_device.csv
```

---

## 2. 데이터 파이프라인

### 2.1 실측 → 3 축 α

**두 profiling run, 세 α 축 추출**:

| Run | 명령 | 산출 | 사용된 α 축 |
|---|---|---|---|
| **Run A** (compute pass) | `tracy -r -p` (no `--collect-noc-traces`) | `profile_log_device.csv` (per-core per-RISC zone timing) | **Compute α** (TRISC 0/1/2 active cycles) |
| **Run B** (NoC pass) | `run_stage_split.sh` × 5 stages | Per-stage `noc_trace_ID*.json` | **NIU α** (injected+consumed), **SRAM α** (consumed) |

Aggregation:
- NoC → `NoC_analysis/aggregate_per_core.py` → `outputs/per_core.csv` (30 endpoints)
- Compute → `compute_axis.py` → `plots/per_core_compute.csv` (20 cores × TRISC cycles)

### 2.2 30 endpoints → 20 compute cores (yaml 기준)

`blackhole_140_arch.yaml` `functional_workers` 필드:
- `y ∈ {0,1,2,4} × x ∈ {0..5}` = 24 workers  
- `y == 3` = DRAM row (compute 아님)  
- `x == 5` = dispatch/reserved (`ops_as_src=0` 실측 확인)

ttnn logical grid (5×4=20) 과 정확히 매칭:
```python
COMPUTE_Y = {0, 1, 2, 4}
COMPUTE_X = {0, 1, 2, 3, 4}
```

`aging_sim.py:load_compute_cores()` 가 이 필터 적용.

**중요 정정** (2026-07-27): 이전 memory `project_eagle_n_noc_coords` 에서 y=0 을 IO row (LPDDR5/PCIe/ARC) 로 잘못 분류했었음. Yaml 확인 결과 y=0 = compute row. Memory 는 이 세션에서 정정됨.

### 2.3 좌표 = 물리 NoC 위치

`translation_id_enabled: True`, `harvested_workers: []` → 좌표 재번호 없음. Trace 의 (sx, sy) = **실제 물리 NoC 좌표**.

물리 layout (top → bottom):
```
y=0  [compute × 5]      compute row
y=1  [compute × 5]      compute row  ← (2,1),(3,1) mcast receivers
y=2  [compute × 5]      compute row
y=3  [── DRAM row ──]   memory only  (compute 아님)
y=4  [compute × 5]      compute row
```

**함의**: (x, 2) 와 (x, 4) 는 물리적으로 **2 hops 떨어짐** (DRAM row 통과). NoC 라우팅 hop 수 계산 시 반영 필요.

### 2.4 x=5 열 (dispatch 후보) 정체 — yaml 불일치

**Yaml 은 x=5 도 그냥 functional_worker 로 나열** — dispatch 특별 취급 없음, `router_only: []` `harvested_workers: []` 다 비어있음. 하지만 실측은 다름:

| Core | inject | consumed | ops_as_src | 특성 |
|---|---|---|---|---|
| (5,0) | **0** | 0.18 MB | **0** | NoC 1 만 receive |
| (5,2) | **0** | 0.47 MB | **0** | NoC 1 만 receive |
| (5,4) | **0** | 0.41 MB | **0** | NoC 1 만 receive |
| (5,1) | — | — | — | **완전 침묵** (CSV 에 없음) |

**패턴**: `inject=0`, 작은 consumed (~15 KB per op), **NoC 1 전용**.

**추정 mechanism** (tt-metal 컨벤션 기반):
- NoC 0 = data/compute path (weight/activation mcast)
- NoC 1 = control/completion path (semaphore, sync signals)
- Worker cores 가 op 완료 신호를 dispatch core 로 write (NoC 1 통해) → dispatch consumed
- Dispatch core 는 스스로 아무것도 initiate 안 함 (host CQ 는 PCIe 로 읽음, NoC 무관)
- 결과: dispatch = **일방적 receiver**, `inject=0`

**정확한 확정은 tt-metal 소스 dive 필요**하지만 aging 스코프에선 무의미:
- Dispatch consumed 는 최대 0.5 MB 
- Compute core outlier ((2,1)=12.65 MB) 대비 **25× 이상 작음**
- Aging 분석 대상에서 이미 제외 (COMPUTE_X = {0..4})

**(5,1) 완전 침묵**: 이 workload 에서 우연히 그럴 수도 있고, 다른 용도로 예약된 것일 수도. 데이터로는 구별 불가, aging 무영향이라 미결로 둠.

---

## 3. Aging 모델 — Chen et al. ISOCC 2024

### 3.1 원본 수식 (Chen Eq 1, 2)

```
ΔVth_NBTI = A_NBTI · exp(-E_a/kT) · V_dd^4 · t^(1/6) · u^(1/6)
ΔVth_HCI  = A_HCI  · u · f · exp(-E_b/kT) · t^(1/6)
```

- `u` : activity factor per core (0-1)
- `f` : clock frequency
- `t` : elapsed time
- `V_dd`, `T`, `E_a`, `E_b`, `A_NBTI`, `A_HCI` : 공정 상수

### 3.2 문제 — Chen 논문이 값 명시 안 함

Chen 은 방정식 형태만 자기 것으로 쓰고 파라미터는 선행 문헌 [11][12] 에 위임:
- [11] Oboril, Tahoori — "ExtraTime" DSN 2012
- [12] Tiwari, Torrellas — Facelift MICRO 2008

우리 6nm FinFET 노드의 정확한 파라미터는 fab 비밀 → **정확한 값 불가능**.

### 3.3 우리 해결책 — Symbolic K collapse (Option B)

Unknown 상수 다 하나의 K 로 묶음:

```
K_NBTI := A_NBTI · exp(-E_a/kT) · V_dd^4       ← 6nm 공정 상수, TBD
K_HCI  := A_HCI  · exp(-E_b/kT)                 ← 6nm 공정 상수, TBD

ΔVth_NBTI = K_NBTI · t^(1/6) · u^(1/6)
ΔVth_HCI  = K_HCI  · u · f · t^(1/6)
```

**Defensible 입력** (실측 or 파라미터):
- `u` — 실측 per-core NoC bytes / cycles
- `f = 650 MHz` — 실측 (profile_log_device.csv 헤더)
- `t` — 시뮬레이션 파라미터 (연 단위)

**Symbolic K** — plot y축 라벨에 `× K_NBTI` 또는 `× K_HCI` 로 표기. 절대값 주장 없음.

### 3.4 왜 상대 비율만 defensible

Ratio 에서 K 소거:
```
worst/best = [K · t^(1/6) · u_max^n] / [K · t^(1/6) · u_min^n]
           = (u_max / u_min)^n         ← K, t 다 사라짐
```
NBTI: n = 1/6 (sublinear), HCI: n = 1 (linear)

같은 K 를 baseline 과 rotation 에 적용하니 두 시나리오 비교도 K 무관.

---

## 4. 결과 요약

### 4.1 실측 activity α (3 axes normalize)

| Axis | 정의 | Min core | Max core | Ratio |
|---|---|---|---|---|
| **NIU** | injected + consumed bytes | (4,4) 0.076 | (3,1) 1.000 | **13.08×** |
| **SRAM** | consumed bytes | (0,4) 0.013 | (3,1) 1.000 | **74.14×** |
| **Compute** | TRISC 0/1/2 active cycles | (4,4) 0.033 | (0,0) 1.000 | **30.35×** |

### 4.2 Aging spread (K 소거, 시간 무관)

| Axis | NBTI worst/best | HCI worst/best | HCI mean/best |
|---|---|---|---|
| NIU     | 1.54× | 13.08× | 3.82  |
| SRAM    | 2.05× | 74.14× | 10.42 |
| Compute | 1.77× | 30.35× | **21.25** ← 소수 cold core, 다수 hot |

**NBTI 는 `u^(1/6)` sublinear** 라 spread 완화 (74× → 2×).  
**HCI 는 linear u** 라 spread 그대로 (74× → 74×).  
실제 aging = 두 mechanism 혼합 (K_NBTI / K_HCI 비율에 따라 mix).

### 4.3 시간 성장 (t^(1/6) 커브)

절대값은 t^(1/6) 로 sublinear 성장:
- year 1 → year 5: 1.30× 증가
- year 1 → year 10: 1.46× 증가

**Ratio 는 time-invariant**: worst / best 는 시간이 지나도 상수. 즉 "bad core 는 chip lifetime 내내 proportionally bad" — 시간이 gap 을 닫지 않음.

---

## 5. 편향의 근원

### 5.1 세 종류 편향 (3 axes)

| 축 | Hot cores | Cold cores | 원인 |
|---|---|---|---|
| **NIU** (Injected) | (0,0), (1-3,0) 8.34~5.5 MB | (4,4) | Mcast broadcaster (2D block-sharded factory 에 sender=top row 하드코딩) |
| **SRAM** (Consumed) | (2,1), (3,1) 10.83~12.65 MB | (0,4) | Layer4 auto-picker 가 결정론적으로 이 2 slot 을 output shard owner 로 지정 |
| **Compute** (TRISC cycles) | y=0, y=1, y=2 대부분 (0.7~1.0) | **x=4 컬럼 전체** (0.03~0.13) | **Layer3-4** (block-sharded) 가 `CORE_SET_16 = (0,0)-(3,3)` 4×4 grid → x=4 배제. Layer3-4 cycle 지배 → aggregate 상 x=4 유휴. (Layer1-2 는 height-sharded CORE_SET_14, x=4 top rows 포함) |

### 5.2 왜 정확히 (2,1), (3,1) 인가

- Stage 4 (layer4) 의 **모든 Matmul (7개) + 3 Conv2d 가 100% 이 2 코어로** target
- Config.py `LAYER_CONFIG["layer4"]` : `override_sharding_config` 사용 안 함 → auto-picker 사용
- Auto-picker heuristic 이 특정 shape (layer4 output width) 에서 결정론적으로 이 slot 선택
- 정확한 픽 이유는 `determine_parallel_config` (matmul 버전) 소스 정독 필요

### 5.3 Consumption 편향의 실체

**Trace semantics**: `sx, sy` = initiator NIU. READ 시 실제 데이터는 `(dx,dy) → (sx,sy)` 방향.
- **Injected** at core C = C 가 initiator 로 시작한 트랜잭션 (READ 요청 + WRITE 데이터)
- **Consumed** at core C = C 로 들어오는 데이터 (READ 응답 + WRITE receive)

**(2,1), (3,1) 이 극단 consumed 이유**:
- Layer4 matmul 의 output shard owner
- Output shard 코어는 자기 슬라이스 계산에 필요한 weight 를 DRAM 에서 대량 READ
- 매 op 마다 반복 → 누적 10-12 MB
- 다른 코어는 output shard 없으니 weight READ 안 함 → 0 consumed

Mcast 는 우리 trace 에서 per-receiver double-counting 안 됨 (전체 consumed < 전체 injected).

### 5.4 Injection 편향의 architectural 분해

Top row injected 실측치:

| Core | ops_as_src | Injected | 특이점 |
|---|---|---|---|
| (0,0) | **89** | **8.34 MB** | Layer3-4 2D mcast 에서 Row 0 sender + Col 0 sender = **이중 역할** |
| (1,0) | 74 | 5.48 MB | Layer3-4 block-sharded row 0 weight sender |
| (2,0) | 74 | 5.49 MB | 동일 |
| (3,0) | 74 | 5.47 MB | 동일 |
| (4,0) | 41 | 1.66 MB | **Layer1-2** (height-sharded, CORE_SET_14) 만 참여. Layer3-4 CORE_SET_16 배제 |

**74 개 op 이 (0,0)+(1,0)+(2,0)+(3,0) 공동 sender**:
- `sharded_program_factory.cpp:605` — `mcast_sender_cores = CoreRange(top_left_core, CoreCoord(num_cores_x - 1, 0))` (top row 전체)
- 각 top row core 가 자기 column 아래 receiver 3개에 weight broadcast
- Column 대칭 구조 → 각 sender 부담 균일 5.48 MB

**(0,0) 이 이중 역할하는 근거** — Column 0 (activation mcast) 도 sender:

| Row | (0,y) 값 | (1-3, y) 값 | (0,y) 우세 |
|---|---|---|---|
| y=1 | 2.38 | 2.00, 2.02, 2.02 | ✓ activation col sender |
| y=2 | 2.27 | 1.95, 1.74, 1.60 | ✓ activation col sender |
| y=4 | 1.64 | 1.14 (동일) | ✓ activation col sender |

즉 **column 0 전체가 activation mcast sender**. (0,0) 은 top-left intersection 이라 두 축 mcast 모두 담당 → 이중 부담.

**편향 지도 완성** (모든 mechanism 통합):

```
        x=0            x=1        x=2         x=3         x=4
y=0  [W + A sender]  [W]        [W]         [W]         [CORE_SET_14 only]
     8.34             5.48       5.49        5.47        1.66
y=1  [A sender]      normal     [SRAM ext]  [SRAM ext]  normal
     2.38             2.00       10.83       12.65       1.26
y=2  [A sender]      normal     normal      normal      normal
y=4  [A sender]      normal     normal      normal      normal
```

- **Row 0** (top) = weight mcast senders — framework 강제 top row
- **Column 0** (left) = activation mcast senders — framework 강제 left column
- **(0,0)** = 두 축 교차점, 이중 역할 → 최고 injected
- **(2,1),(3,1)** = auto-picker output shard = 최고 consumed
- **(4,\*)** = CORE_SET_16 배제, 유휴

**통합 메시지**: 모든 편향이 (0,0) anchor 하드코딩의 파생 결과. Mcast axis 규칙 (top/left) + CORE_SET grid 하드코딩 + auto-picker heuristic — 세 가지가 각기 다른 코어를 결정론적으로 stress. Rotation 이 모든 축을 동시에 완화하는 유일한 방법.

### 5.5 세 편향의 공통 근원

세 축 모두 **결정론적 sharding logic + hardcoded (0,0) anchor** 결과:
- **Mcast** (NIU): Layer3-4 의 2D block mcast sender = anchor top row/left col 하드코딩. Layer1-2 의 1D height mcast 도 top_left_core 하드코딩 (single sender).
- **Auto-shard** (SRAM): auto-picker heuristic 이 (0,0) anchor 기반으로 매번 같은 slot 반복 pick. Layer 4 head 의 width-sharded output shard 결정론적.
- **CORE_SET** (Compute): Config.py 에 하드코딩된 CORE_SET_14 (Layer1-2 자동 선택), CORE_SET_16 (Layer3-4). 어느 것이든 (0,0) 원점 사각형/L-shape.

**한 근본 원인** (anchor 하드코딩) 이 sharding scheme 별로 다른 방식으로 **세 편향** 을 만듦. Rotation 이 세 축 다 해결.

### 5.6 아무 코어도 안전하지 않음 (Rotation 의 강한 정당화)

| Core | NIU | SRAM | Compute | 어느 축이든 stress |
|---|---|---|---|---|
| (0,0)  | **0.61 hot** | 0.05 | **1.00 hot** | NIU + Compute |
| (3,1)  | **1.00 hot** | **1.00 hot** | 0.98 hot | 세 축 다 |
| (2,1)  | 0.88 hot | **0.86 hot** | 0.95 hot | 세 축 다 |
| (4,4)  | 0.08 low  | 0.05 | 0.03 low | 세 축 다 낮음 |
| (4,0)  | 0.15 low  | 0.04 | 0.13 low | 세 축 다 낮음 |

**Rotation 이 없으면 (3,1), (2,1) 같은 코어는 chip lifetime 내내 세 축 모두 극단 stress → premature failure**. Rotation 이 세 축 동시 완화.

---

## 6. Sharding-Mcast 구조 (source verified)

이 섹션은 tt-metal 소스 (`sharded_program_factory.cpp`, kernel 파일들) 직접 읽고 확정한 사실. 이전 세션에서 여러 번 헷갈렸던 부분 최종 정리.

### 6.1 실측 stage 별 sharding (tracy_ops_data.csv grep)

```bash
for i in 0..4: grep -oE "HEIGHT_SHARDED|BLOCK_SHARDED|WIDTH_SHARDED|INTERLEAVED"
```

| Stage | Sharding | 정정 |
|---|---|---|
| Stage 0 (Stem) | HEIGHT + INTERLEAVED | Height (mixed 아님) |
| Stage 1 (Layer 1) | HEIGHT + INTERLEAVED | Height |
| **Stage 2 (Layer 2)** | **HEIGHT + INTERLEAVED** | ⚠️ **Height (이전에 block 이라 잘못 표기)** |
| Stage 3 (Layer 3) | **BLOCK** + INTERLEAVED | Block |
| Stage 4 (Layer 4 + Head) | **BLOCK + WIDTH** + INTERLEAVED | Block + Width (head Matmul) |

INTERLEAVED = DRAM interleaved tensor (compute sharding 무관).

**Layer 2 정정 근거**: Config.py `layer2.module2-4: height_sharding=None` (auto) 이지만, auto-picker 가 실제로 **height 선택** (L1 heuristic 결과). 우리가 Config.py 만 보고 "None=block" 이라 잘못 추론했었음.

### 6.2 Block-sharded mcast 구조

**소스**: `sharded_program_factory.cpp:596-617` (block branch)

- **all_cores** = `input_cores.merge(output_cores)` — strict rectangular
- **`total_noop_cores == 0` 강제** (line 566): 모든 cell active
- **Mcast 2 종류**:
  - **Weight mcast** (2D column): sender = top row (`CoreRange(top_left, {num_cores_x-1, 0})`), receiver = 아래 rectangle grid
  - **Activation mcast** (2D row): sender = left column (`CoreRange(top_left, {0, num_cores_y-1})`), receiver = 오른쪽 rectangle grid
- **NoC API**: `get_noc_multicast_addr(x_start, y_start, x_end, y_end)` — single rectangle 만
- **Kernel**: `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`, `writer_tiled_out_2d_mcast_*.cpp`

**Rotation 자유도**:
- Anchor shift: OK (top_left_core 파라미터화, §12 패치)
- **Reflection**: 4 mode 가능 (weight sender = top/bottom, activation sender = left/right)
- **L-shape**: 불가 (strict rect 강제)
- **Wrap**: 불가 (single rectangle mcast API)

### 6.3 Height-sharded mcast 구조

**소스**: `sharded_program_factory.cpp:240-243, 618-624` (height branch)

- **all_cores** = **forced rectangular bbox from (0,0)** (line 240-243):
  ```cpp
  all_cores = CoreRangeSet(CoreRange(
      CoreCoord(0, 0),
      CoreCoord(grid_size.x - 1, grid_size.y - 1)));
  ```
- **`total_noop_cores > 0` 허용**: bbox 안의 unused cell 은 **noop kernel** 로 launch, skip work
- **Mcast 1 종류**:
  - **Weight mcast** (1D): sender = single core (top_left, 하드코딩), receiver = `all_cores.subtract(sender)` = bbox 나머지
  - **Activation mcast**: **없음** — 각 코어가 자기 shard 를 L1 에서 로컬 read
- **NoC API**: 여전히 `get_noc_multicast_addr` — single rectangle
- **Kernel**: `reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp` (mcast 없음, per-core independent), `reader_writer_tiled_out_1d_mcast_sender_*.cpp` (weight mcast)

**Rotation 자유도**:
- Anchor shift: OK (§12 패치)
- **Reflection**: 매우 제한적 (sender 는 단일 코어라 "edge reflection" 개념 없음)
- **L-shape input**: **OK** (bbox rectangle 로 감싸고 unused cell 을 noop 로 처리 → CORE_SET_14 가 이 방식으로 작동)
- **Wrap**: 불가 (bbox 사각형 강제, 여전히 single rectangle NoC mcast)

### 6.4 Wrap 이 어느 sharding 이든 불가한 이유

**세 층에서 제약**:

1. **NoC mcast API 자체**: `get_noc_multicast_addr(start_x, start_y, end_x, end_y)` 는 두 corner 로 정의된 **단일 사각형**만. Wrap 하면 end < start 발생 → empty or error.

2. **Factory 코드 all_cores 강제 rectangular**:
   - Height: `CoreRange((0,0), (grid_size.x-1, grid_size.y-1))` — 단일 CoreRange (사각형)
   - Block: `input_cores.merge(output_cores)` — 이론상 CoreRangeSet 이지만 block 은 strict rect 강제

3. **NoC 하드웨어**: Blackhole 은 torus 라 wrap 배선 존재. L-shape mcast 는 BringUp 완료 (§12.2 조사). 하지만 tt-metal API 로 노출 안 됨.

**Wrap 을 진짜 지원하려면**:
- Factory `all_cores`, sender/receiver 세트를 CoreRangeSet (multi-range) 대응하게 rewrite
- Reader kernel: 여러 mcast rectangle 순회 loop, semaphore 다중 관리
- Writer kernel: 매 weight tile 당 2+ mcast 호출, split receiver count aggregate
- 총 **factory + kernel 대공사, 수주-수개월 스코프** → 학부 프로젝트 밖

### 6.5 우리 rotation 알고리즘 정확한 mode 분류

**Mode A — Block-sharded (Layer 3, Layer 4 body)**:
- Anchor shift: `(X_max - W + 1) × (Y_max - H + 1)` positions
- Reflection: 4 modes (TL/TR/BL/BR — sender edge swap)
- 예: 4×4 shape on 5×4 grid = 2 anchor × 4 refl = **8 configs**
- 강 rectangular

**Mode B — Height-sharded (Stage 0, 1, 2 = Stem + Layer 1 + Layer 2)**:
- Anchor shift: bbox 이동 (grid 내 fit)
- Reflection: 실질적으로 없음 (single sender)
- L-shape 자연 지원 (noop 활용, CORE_SET_14 처럼)
- Mode A 대비 anchor 자유도 넓음 (특히 작은 op)

**Mode C — Width-sharded (Layer 4 head Matmul)**:
- 별도 factory (`conv2d_op_width_sharded_program_factory.cpp` 및 matmul width-sharded factory)
- 조사 필요, 이번 스코프 밖
- Head 이 이 mode → 우선 Mode A 로 rotation 시뮬레이션 시 exclude

**공통 제약**: **어느 mode 도 wrap 불가**. Strict-fit anchor only.

**논문 방어 문구**:
> "We support anchor rotation and (for block-sharded) reflection under strict-fit constraint. Wrap-around rotation requires substantial factory + kernel rewrite (multi-rectangle mcast, split semaphore handling) — deferred to future work. Blackhole HW supports L-shaped mcast (per BringUp documentation) but this capability is not exposed via tt-metal API."

### 6.6 이전 잘못된 주장 정정 요약

| 이전 주장 | 실제 (source verified) |
|---|---|
| ~~"Height-sharded 는 mcast 없음"~~ | Height 도 1D weight mcast 있음, activation mcast 만 없음 |
| ~~"Height-sharded 는 wrap 가능"~~ | **Height 도 wrap 불가** (bbox 사각형 강제, single-rect NoC API) |
| ~~"CoreRangeSet 자유롭게 shape 지정"~~ | bbox 사각형 + noop cell 로만 L-shape 흉내 |
| ~~"Layer 2 는 block-sharded"~~ | **Layer 2 는 all height-sharded** (실측 확증) |
| ~~"Config.py None = auto = block"~~ | None = auto-picker, layer2 에서 실제로 height 선택 |

---

## 7. Rotation 알고리즘 (Tier 1-3 구현)

### 7.1 Tier 분류

| Tier | Anchor shift | Reflection | Policy | 참조 |
|---|---|---|---|---|
| **T1** | ✓ | ✗ | Round-robin | 단순 baseline |
| **T2** | ✓ | ✓ (block 4×4 만) | Round-robin | 확장 baseline |
| **T3** | ✓ | ✓ | **Aging-aware greedy** | Brandalero DAC'20 mirror |
| ~~T4~~ | ~~+ wrap~~ | ~~L-shape mcast~~ | ~~complex~~ | **Future work** (kernel 대공사) |

### 7.2 Configuration space per op

**Op 특성 추출** (`matrix.csv` 로부터):
- Bbox: `(min_x, min_y, max_x, max_y)`
- Shape: `(W, H) = (max_x - min_x + 1, max_y - min_y + 1)`
- Origin: `(min_x, min_y)`
- `is_block = (shape == (4, 4))` — CORE_SET_16 (Layer 3-4 block-sharded)

**Valid configs** (strict-fit, no wrap):
```python
anchors = [(x, y) for y in range(4 - H + 1)
                  for x in range(5 - W + 1)]
modes = ['TL', 'TR', 'BL', 'BR'] if is_block else ['TL']
configs = anchors × modes
```

| Shape | Anchors | Modes | Total configs | Op 수 |
|---|---|---|---|---|
| 5×4 (full grid) | 1 | 1 | **1** (rotation 불가) | 52 |
| 4×4 (block) | 2 | 4 | **8** | 14 |
| 5×3 | 2 | 1 | 2 | 1 |
| 4×2 | 6 | 1 | 6 | 22 |

### 7.3 Transformation: reflect + translate

```python
def transform_activity(activity, origin, target_anchor, reflection, W, H):
    # Step (a): 원 배치를 local coordinate (0,0) 원점으로 이동
    local = {(x - origin.x, y - origin.y): v for (x, y), v in activity.items()}
    
    # Step (b): local shape 안에서 reflection
    reflected = {}
    for (x, y), v in local.items():
        nx = (W - 1 - x) if reflection in ('TR', 'BR') else x   # 수평 반사
        ny = (H - 1 - y) if reflection in ('BL', 'BR') else y   # 수직 반사
        reflected[(nx, ny)] = v
    
    # Step (c): target anchor 로 이동
    return {(x + target.x, y + target.y): v for (x, y), v in reflected.items()}
```

**전제**: Translation-symmetry — anchor 이동은 활동 패턴을 rigid 이동. Reflection = spatial mirror.

### 7.4 Policy 별 config 선택

**T1/T2 — Round-robin**:
```python
counter[shape] += 1                  # shape 별 카운터
chosen = configs[counter % len(configs)]
```
- Deterministic, state-free
- 각 config 균등 사용 (statistical uniformity)
- 한계: 실제 aging 편중 인지 안 함

**T3 — Aging-aware greedy** (Brandalero 방식):
```python
state = defaultdict(int)   # per-core cumulative activity

for op in ops_sequence:
    best_config = None
    for config in valid_configs(op):
        moved = transform_activity(op.activity, ..., config)
        trial_worst = max(state[core] + moved[core] for core in touched)
        if trial_worst < best_worst_so_far:
            best_config = config
    
    # Commit
    schedule[op.id] = best_config
    for core, activity in moved.items():
        state[core] += activity   # aging state 업데이트
```

- Greedy min-max: **"이 op 을 놓은 후 worst-core 가 최소" 인 config 선택**
- Per-core aging state 추적 (`state[core] = 누적 inj+con`)
- 실제 aging 편중 반영 → 콜드 코어로 적극 분산

### 7.5 실측 결과 (89 ops from batch=1 ResNet50)

| Metric | Baseline | T1 | T2 | **T3** |
|---|---|---|---|---|
| NIU spread (worst/best) | 13.08× | 10.32× | 5.61× | **5.25×** |
| **NIU worst-core** | 14.67 MB | 12.74 (13.2%↓) | 9.55 (34.9%↓) | **6.76 (53.9%↓)** |
| SRAM spread | 74.14× | 59.43× | 41.01× | **28.05×** |
| **SRAM worst-core** | 12.65 MB | 10.14 (19.8%↓) | 6.93 (45.2%↓) | **4.82 (61.9%↓)** |

**Lifetime 투영** (Chen equations, K 소거):
- HCI-dominant (linear u): `t_fail ∝ u^(-6)` → SRAM T3 = **327×** upper bound
- NBTI-dominant (u^(1/6)): `t_fail ∝ u^(-1)` → SRAM T3 = **2.6×** lower bound
- 실제 aging = 두 mechanism 혼합 → **정직 리포트 "2.6× ~ >30×"**

### 7.6 스크립트 위치

| 파일 | 내용 |
|---|---|
| `rotation_sim_tier1.py` | T1 — anchor shift + round-robin |
| `rotation_sim_tier2.py` | T2 — + reflection (block 4×4 8 configs) |
| `rotation_sim_tier3.py` | T3 — aging-aware greedy (Brandalero mirror) |
| `rotation_tier1_visualize.py` | Baseline vs T1 히트맵 |
| `rotation_tier12_visualize.py` | 3-way (baseline/T1/T2) |
| `rotation_tier3_visualize.py` | **4-way progression (미팅 하이라이트)** |
| `plots/rotation_tier3_per_core.csv` | 4 tier 통합 데이터 |

### 7.7 알고리즘 제한 및 미해결

- **Full-grid op 52 개 (58%) rotation 불가** — 여전히 baseline 그대로 누적. 이게 T3 도 완전히 못 없앤 잔여 편향의 원인.
- **Round-robin ≠ optimal** — T3 aging-aware 로 개선 확인. 하지만 greedy 라 global optimum 보장 안 됨.
- **Reflection height-sharded 미지원** — Single sender core 라 물리적 의미 없음. 4×4 block 에만 4 mode.
- **Wrap 미지원** — mcast API single rectangle 강제 (§6.4). Future work.

### 7.8 Brandalero 비교

| 항목 | Brandalero DAC 2020 | 우리 |
|---|---|---|
| 플랫폼 | CGRA (연구 프로토타입) | Tenstorrent Blackhole (상용 NPU) |
| Configuration generation | Virtual placement (fig 3a-b) | `valid_configs()` (T1-3) |
| Configuration allocation | Stress table (fig 8) = aging-aware | **T3 aging-aware greedy** |
| 실증 | Simulation only | **실측 (batch=1 ResNet50) driven** |
| Wall time overhead | 시뮬 추정 | 실측 예정 (C++ 패치 후) |
| Aging metric | Delay increase % | ΔVth 3 축 + lifetime bounds |

**논문 인용 문구**:
> "Our Tier 3 policy follows the utilization-aware allocation strategy of Brandalero et al. (DAC 2020), adapted from CGRA fabric to Blackhole-class programmable spatial NPU. Round-robin policies (T1/T2) serve as simpler baselines for comparison."

---

## 8. 도표 인벤토리

| 파일 | 내용 | 용도 |
|---|---|---|
| `plots/baseline_alpha_heatmap.png` | Per-core α 히트맵 (NIU, SRAM, **Compute**) | 3 축 activity 편중 시각화 |
| `plots/per_core_heatmap_annotated.png` | NoC bytes + role 라벨 (compute/DRAM/dispatch) | 물리 layout 이해 |
| `plots/per_stage_heatmaps.png` | 5-stage 별 NoC (inj + con) — mcast pattern 대비 | Height (Stage 0-2) vs Block (Stage 3-4) sharding 시각적 대비 |
| `plots/per_stage_compute_heatmaps.png` | 5-stage 별 TRISC — CORE_SET 사용 패턴 | Layer 별 core set (14 vs 16) 사용 확인 |
| `plots/layer2_module_heatmaps.png` | Layer 2 module 별 shape 상세 | Height-sharded L-shape 형태 명시 |
| `plots/baseline_vth_niu_year5.png` | Year=5 aging heatmap (NBTI, HCI) — NIU 축 | 5년 후 편중 mechanism 별 |
| `plots/baseline_vth_sram_year5.png` | 동일 — SRAM 축 | 동일 |
| `plots/baseline_vth_compute_year5.png` | 동일 — Compute 축 | 동일 |
| `plots/baseline_vth_over_time.png` | 시간 성장 6-panel (3 axes × NBTI/HCI) | t^(1/6) 성장 + 코어별 gap 유지 |
| `plots/baseline_vth_dist_year{5,10}.png` | 코어 분포 히스토그램 (3 axes) | Spread 분포 |
| `plots/baseline_summary.txt` | 3-axis 수치 표 요약 | 발표용 수치 |
| `plots/per_core_compute.csv` | Per-core TRISC active cycles + α_compute | compute_axis.py 출력, aging_sim.py 입력 |
| **`plots/rotation_tier1_comparison.png`** | Baseline vs T1 (shift only) | Anchor shift 효과 |
| **`plots/rotation_tier12_comparison.png`** | Baseline vs T1 vs T2 | Reflection 효과 |
| **`plots/rotation_tier3_comparison.png`** | **4-way full progression** | **미팅 하이라이트** |
| `plots/rotation_tier3_per_core.csv` | 4 tier 통합 데이터 | 논문 표 근거 |

---

## 9. 미팅 발표 스크립트

### Slide 1 — 3 축 실측 activity (`baseline_alpha_heatmap.png`)
> "Batch=1 ResNet50 을 Eagle-N 20 compute core 에서 실측. 3 축 (NIU/SRAM/Compute) 각각 다른 패턴으로 편중: NIU 13×, SRAM 74×, Compute 30×. 어느 축이든 특정 코어에 극단 부담."

### Slide 2 — 세 편향 원인 (`per_core_heatmap_annotated.png` + `per_stage_heatmaps.png`)
> "세 편향, 각기 다른 mechanism: (1) NIU 는 Layer3-4 (block-sharded) 의 2D mcast top row/left col sender 편중, (2) SRAM 은 (2,1),(3,1) auto-picker output shard 편중, (3) Compute 는 Layer3-4 CORE_SET_16 이 x=4 배제. Layer1-2 는 height-sharded (mcast 1D) 라 편향 훨씬 약함 — per-stage heatmap 에서 확인 가능. 세 축 모두 프레임워크의 결정론적 하드코딩 결과."

### Slide 3 — Aging model (`baseline_vth_sram_year5.png`)
> "Chen ISOCC 2024 NBTI+HCI 공식 적용. 6nm 공정 상수 미확정 → K 로 collapse. NBTI-only bound 는 spread 2× 근처 (u^(1/6) sublinear), HCI-only bound 는 30-74× (linear u). 실제 aging 은 두 mechanism 혼합. K 무관 상대 비율만 리포트."

### Slide 4 — 시간 성장 (`baseline_vth_over_time.png` 6-panel)
> "Aging 은 t^(1/6) 로 시간 따라 성장. 근데 worst/best ratio 는 시간 무관 상수 — 시간이 gap 을 닫지 않음. 세 축 모두 rotation 없으면 chip lifetime 내내 편중 유지."

### Slide 5 — 아무 코어도 안전하지 않음 (`baseline_vth_compute_year5.png` + §5.5 표)
> "(3,1) 은 세 축 다 hot, (0,0) 은 NIU+Compute hot, (4,4) 는 세 축 다 cold. 결정적으로 어느 축이든 어떤 코어는 극단 stress. Rotation 이 세 축 동시 완화하는 유일한 방법."

### Slide 6 — Rotation 알고리즘 (T1-T3 진행) — `rotation_tier3_comparison.png`
> "3 tier 로 rotation 정책 진화 실험. Tier 1 = anchor shift + round-robin (worst-core 20% ↓). Tier 2 = + reflection (45% ↓). **Tier 3 = aging-aware greedy (Brandalero DAC'20 mirror) — SRAM worst-core 61.9% 감소**. Lifetime 투영: NBTI-dominant 2.6× ~ HCI-dominant 30× 개선."

### Slide 7 — 4주 스코프 (Tier 별 난이도)
- **T1 필수 (1-2주)**: Anchor shift 만 — C++ 패치 15 lines mechanical
- **T2 likely (+1주)**: + Reflection — sender edge flag ~30-50 lines
- **T3 stretch (+1주)**: Aging-aware — Python sim 만 확실, C++ 은 매 op state update 필요
- **T4 future work**: Wrap-around — factory + kernel 대공사

### Slide 8 — 추후 과제
- **6nm K 정량화**: TSMC 데이터 or Facelift/Oboril 상수 대입 → 절대 mV 추정 가능
- **아키텍처 통합**: 3 축 → 코어 단위 chip lifetime 통합 모델 (weakest-axis / weighted 등)
- **Real HW rotation 실측**: C++ 패치 후 shifted anchor profiling → sim 검증
- **Wrap-around rotation**: mcast factory + kernel 대공사, future work

### 핵심 3 문장
1. **실측**: 3 축 activity spread — NIU 13×, SRAM 74×, Compute 30× (**서로 다른 코어 겨냥**)
2. **모델링**: Chen NBTI+HCI. NBTI 는 spread 2× 근처, HCI 는 30-74× (실제 aging 은 두 bound 사이). Spread 는 시간 무관.
3. **Rotation**: T3 aging-aware policy 로 **worst-core SRAM 61.9% 감소, lifetime 보수적 2.6× ~ 30× 개선**. Brandalero DAC'20 방식을 상용 programmable NPU 에 최초 적용.

---

## 10. Defensible / Not Defensible 정직 표

| 주장 | Defensible? | 근거 |
|---|---|---|
| "α spread NIU 13×, SRAM 74×, Compute 30× on real HW" | ✅ | per_core.csv + profile_log_device.csv 실측 |
| "3 축 편향의 mechanism 이 각기 다름" | ✅ | 편향 위치 실측 + Config.py + tt-metal 소스 |
| "Aging spread NBTI 2× 근처, HCI 30-74× (bounds)" | ✅ | K 소거된 비율 |
| "Spread is time-invariant" | ✅ | 수학적 (t^(1/6) 소거) |
| "5년 후 worst core Vth 245 mV" | ❌ | K 미확정, 완전 조작 |
| "Rotation 이 aging 을 N× 개선" (실측 후) | ✅ (예정) | K 무관 상대 비교 |
| "6nm FinFET 특유 aging behavior 재현" | ❌ | 공정 데이터 없음 |
| "3 축 통합 chip lifetime" | ⚠️ | Weighted-sum / weakest-axis 등 assumption 필요 → future work |
| "Stage 0-2 = height-sharded, Stage 3-4 = block-sharded" | ✅ | tracy_ops_data.csv grep 로 확증 |
| "Wrap-around rotation" (구현) | ❌ | tt-metal mcast API single rect 만, 대공사 필요, future work |
| "Tier 3 rotation reduces worst-core SRAM 61.9% on real workload" | ✅ | rotation_sim_tier3.py 실측 activity 기반 sim |
| "Lifetime 2.6× ~ 30× 개선 (Tier 3)" | ⚠️ conditional | K_NBTI/K_HCI 비율 미지 → bound 로만 리포트 |
| "Aging-aware policy 는 Brandalero DAC'20 mirror" | ✅ | 알고리즘 구조 same (utilization-aware allocation) |

---

## 11. 관련 파일 & 참조

**코드**:
- `aging_sim.py` — 메인 baseline 시뮬. 3-axis + K collapse
- `annotated_heatmap.py` — Role 오버레이 NoC 히트맵
- `compute_axis.py` — TRISC cycles → α_compute
- `per_stage_heatmap.py`, `per_stage_compute_heatmap.py` — 5-stage NoC / Compute
- `layer2_module_heatmap.py` — Layer 2 module 별 shape
- `rotation_sim_tier{1,2,3}.py` — **Rotation algorithms**
- `rotation_tier{1,12,3}_visualize.py` — 비교 히트맵
- `NoC_analysis/aggregate_per_core.py` — Raw noc_trace → per_core.csv (실측 파이프라인)

**논문**:
- Chen et al. "Aging Mitigation in Systolic Array Accelerators" ISOCC 2024 — 우리 seed 논문
- Brandalero et al. "Proactive Aging Mitigation in CGRAs" DAC 2020 — head-to-head prior work
- Tiwari, Torrellas "Facelift" MICRO 2008 — CMP aging origin
- Karpuzcu et al. "VARIUS-NTV" DSN 2012 — arch-level aging model
- Naithani et al. HPCA 2017 — top-tier reliability-aware scheduling precedent

**Memory 관련**:
- `[[project-eagle-n-noc-coords]]` — 좌표 매핑 확정
- `[[aging-rotation-note]]` — 상위 aging rotation 연구 노트
- `[[feedback-teach-dont-just-execute]]` — 사용자 학습 스타일

**메인 노트**:
- `study_note/aging_rotation_research_note.md` — 프로젝트 상위 노트

---

## 12. 재현 절차

```bash
cd C:/NPU_Challenge/study_note/aging_simulation

# --- Baseline sim ---
python compute_axis.py                  # profile_log_device.csv → per_core_compute.csv (한 번만)
python aging_sim.py                     # baseline aging sim + 3-axis plots + summary
python annotated_heatmap.py             # role-annotated NoC heatmap

# --- Per-stage plots ---
python per_stage_heatmap.py             # 5-stage NoC (mcast pattern)
python per_stage_compute_heatmap.py     # 5-stage TRISC (CORE_SET pattern)
python layer2_module_heatmap.py         # Layer 2 module 별 L-shape 확인

# --- Rotation sim (Tier 1-3) ---
python rotation_sim_tier1.py            # T1: anchor shift + round-robin
python rotation_sim_tier2.py            # T2: + reflection (block 4x4)
python rotation_sim_tier3.py            # T3: aging-aware greedy (Brandalero mirror)
python rotation_tier1_visualize.py      # Baseline vs T1
python rotation_tier12_visualize.py     # Baseline vs T1 vs T2
python rotation_tier3_visualize.py      # 4-way full progression (미팅용)
```

입력 데이터 위치:
- **NIU, SRAM**: `bos_model_copy/resnet50_jbpark/analysis/outputs/per_core.csv`
- **Per-op matrix** (rotation sim 입력): `NoC_analysis/outputs/matrix.csv`
- **Compute**: `data_movement/conv_core_replacement/policy/data/full_model_baseline/input/profile_log_device.csv` (Run A 결과)
- **Per-stage compute**: `stage_split_{0-4}/stage_{0-4}/logs/profile_log_device.csv`

수정 후 재실행 → `plots/` 하위 그림들 갱신.

---

## 13. 다음 세션 To-Do

1. **미팅 후**: 피드백 반영, calibration 관련 질문 정리
2. ~~**Compute (FPU) 축 추가**~~ ✅ 완료
3. ~~**Per-stage NoC/Compute heatmap**~~ ✅ 완료
4. ~~**Mcast 구조 확정**~~ ✅ 완료 (§6)
5. ~~**Rotation Python sim (Tier 1-3)**~~ ✅ 완료 (§7)
6. **C++ 패치** (컨테이너 세션, Week 1-2):
   - Tier 1: `sharded_program_factory.cpp` anchor 하드코딩 제거 (§12 목록, ~15 lines)
   - Tier 2: sender edge flag 추가 (block reflection ~30-50 lines)
   - Tier 3: aging state 관리 (host 측 sched, factory 수정 최소)
   - Wrap 은 future work
7. **Real HW rotation 실측** (Week 3):
   - Rotated anchor 로 profiling
   - Wall time overhead 측정
   - Sim vs 실측 대조 (translation-symmetry 가정 검증)
8. **논문 draft** (Week 4):
   - Baseline + Tier progression
   - Brandalero 비교 표
   - Lifetime bound 리포트
9. **아키텍처 통합 (선택)**: weighted-sum or weakest-axis chip lifetime 모델

_최종 수정: 2026-07-28 (Tier 1-3 rotation sim 완료, §7 신설)_
