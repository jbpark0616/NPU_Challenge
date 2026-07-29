# ResNet50 on Eagle-N 학습 노트

날짜: 2026-07-06

---

## 1. 파일 구조 (bos_model/resnet50/)

```
resnet50/
├── run_resnet50.py              # 실행 엔트리 (인자 파싱, 이미지 로드)
├── README.md                    # 실행법, Tracy/Visualizer 사용법
├── tt/
│   ├── ttnn_functional_resnet50.py  # ★ 모델 본체 (아키텍처)
│   ├── Config.py                    # 튜닝 스위치 테이블
│   ├── Conv.py                      # ttnn.conv2d 얇은 래퍼
│   ├── Linear.py                    # ttnn.linear 클로저 래퍼 (FC)
│   ├── Utils.py                     # pcc_check, sharding 유틸 (부수적)
│   └── custom_preprocessing.py      # 입력 전처리
└── tests/
    ├── test_ttnn_functional_resnet50.py
    └── resnet50_test_infra.py       # 실행 세팅, warmup, batch loop
```

---

## 2. Tenstorrent 프로그래밍 스택

```
사용자 편의 ▲
  │ tt-forge / MLIR       ← PyTorch/ONNX 자동 컴파일 (실험적)
  │ ttnn (Python)         ← ★ 이 baseline이 사용하는 층
  │ tt-metal (C++)        ← Metalium, 커널 직접 짜기
낮은 레벨  ▼
```

- **ttnn** = "PyTorch처럼 생긴 Tenstorrent 라이브러리"
- 모델 개발자는 대부분 Python + ttnn API로 작업
- 커널 최적화하려면 tt-metal(C++)까지 내려감

---

## 3. API vs Op

| 개념 | 사는 곳 | 예시 |
|---|---|---|
| **API** | 소스 코드 (Python) | `ttnn.conv2d(...)`, `ttnn.Conv2dConfig(...)` |
| **Op** | 실행 시점 (하드웨어) | 실제로 도는 conv 연산 하나 |

- **API = 부탁하는 문법**, **op = 실제 실행되는 연산**
- **모든 API 호출이 op를 만드는 건 아님** (설정 API는 op 안 만듦)
- Profiler에는 op만 찍힘

---

## 4. Config.py — 튜닝 스위치 테이블

### CORE_SET 정의

```python
CORE_SET_14 = 5×2 + 4×1 = 14 코어 (L자, layer1~3용, HEIGHT_SHARDED)
CORE_SET_16 = 4×4 = 16 코어 (정사각형, layer4/FC용, BLOCK_SHARDED)
CORE_GRID_16 = ttnn.CoreGrid(x=4, y=4)  # FC 전용
```

### LAYER_CONFIG 필드 순서

```
[reshard_if_not_optimal, height_sharding, packer_l1_acc,
 enable_act_double_buffer, force_split_reader, ops_parallel_config]
```

값이 리스트로 있는 이유: `Bottleneck.__call__` 안에서 **순서대로 언패킹**해서 위치 인자로 씀. 그래서 이름 없이 True/False만 나열 가능.

### 스위치별 하드웨어 동작

| 필드 | True 시 |
|---|---|
| `reshard_if_not_optimal` | sharding 안 맞으면 자동 재배치 |
| `height_sharding` | 세로 방향으로 텐서 쪼갬 (아니면 격자) |
| `packer_l1_acc` | Packer가 결과를 L1에 직접 누적 |
| `enable_act_double_buffer` | Activation CB를 2슬롯 → reader/compute 오버랩 |
| `force_split_reader` | Reader kernel을 BRISC+NCRISC 두 코어로 분할 |

---

## 5. ttnn_functional_resnet50.py — 모델 본체

### 파일 구조 5개 덩어리

```
line 17-96    conv1x1, conv3x3            # 헬퍼
line 102-375  class Bottleneck            # 잔차 블록 (16번 반복)
line 378-717  class resnet50              # ★ 모델 전체
line 720-858  ModelRunnerWrapper          # trace/warmup 감쌈
line 860-888  BenchmarkModelRunnerWrapper # 벤치마킹 계측
```

### 전체 forward 흐름 (`__call__` line 478~)

```
입력 (batch=4, 3, 224, 224)
   ↓ fold on host → reshape (1,1, n*c*h, w)
conv1 (4×4, stride 1) + ReLU
   ↓
maxpool (3×3, stride 2)
   ↓
[to_memory_config → HEIGHT_SHARDED on CORE_SET_14]  ← ★ 첫 하드웨어 자원 결정
   ↓
layer1 (Bottleneck × 3)  ─┐
layer2 (Bottleneck × 4)   │  → 대부분의 계산이 여기 (90%+)
layer3 (Bottleneck × 6)   │
   ↓
[to_memory_config → BLOCK_SHARDED on CORE_SET_16]   ← ★ layer4 진입 시 강제 전환
layer4 (Bottleneck × 3)  ─┘
   ↓
avgpool (앞뒤로 tilize/untilize 여러 번)
   ↓
FC (Linear, CORE_GRID_16, 1024 → 1000 잘라내기)
   ↓
출력 (batch, 1, 1000)
```

### 이미 들어간 최적화 (baseline이 baseline이 아닌 이유)

| 기법 | 효과 |
|---|---|
| ReLU fuse (conv/add 안에) | dispatch 오버헤드 감소 |
| `bfloat8_b` (BFP8) + `MathFidelity.LoFi` | 대역폭/L1 절반, 연산 빠름 |
| `packer_l1_acc` | packer가 L1 직접 누적 |
| `enable_act_double_buffer` | reader/compute 오버랩 |
| `force_split_reader` | reader를 BRISC+NCRISC로 분할 |
| `deallocate_activation=True` | L1 즉시 해제 |
| Trace mode + 2CQ | host↔device dispatch 오버랩 |
| Layer4 sharding config 캐싱 | 2번째 forward부터 reshard 스킵 |

---

## 6. ResNet 아키텍처 개념 (오늘 정리)

### Bottleneck 블록 — ResNet의 기본 단위

```
입력 x  ─── 1×1 conv (256 → 64)   + ReLU     ← 채널 압축
              ↓
          3×3 conv (64 → 64)      + ReLU     ← ★ 진짜 특징 추출
              ↓
          1×1 conv (64 → 256)                ← 채널 복원
              ↓
   x ─── [downsample 필요 시]                 ← skip path 크기 맞춤
              ↓
          add + ReLU (fused via ttnn.add_)   ← 잔차 연결
              ↓
          출력 out
```

### 왜 1×1 → 3×3 → 1×1 인가

- **3×3 conv는 무거움**: 채널 많으면 계산 폭증
- **1×1로 채널 줄여놓고 (256→64)** → 좁은 상태에서 3×3 → **1×1로 다시 늘림 (64→256)**
- 병 모양이라 "bottleneck"

### 1×1 conv의 정체

- **공간 정보 안 봄**, **채널만 재조합**
- 각 픽셀에서: `256차원 벡터 → (filter 64개로 조합) → 64차원 벡터`
- Filter 개수 = 출력 채널 수
- 각 filter 원소 개수 = 입력 채널 수
- **본질적으로 각 픽셀마다 적용되는 mini FC layer**

### 3×3 conv의 정체

- **주변 3×3 이웃 픽셀 봄** → 공간 패턴 학습
- Bottleneck에서 **진짜 특징 추출하는 곳**
- 앞뒤 1×1은 이걸 효율적으로 만들어주는 조수

### Skip connection & Residual F(x)

```
out = x + F(x)

- x    = 원본 (skip으로 그대로 넘어옴)
- F(x) = Bottleneck 안 conv들이 계산한 "변화량"
```

- **구조상 skip은 항상 켜져 있음** (if 문 없음)
- **F(x)의 크기는 학습으로 결정**
  - F(x) ≈ 0 → 사실상 no-op (skip만 통과한 것과 같음)
  - F(x) ≠ 0 → x를 실제로 수정
- 즉 **conv들은 "덧붙일 수정"만 계산**, 원본은 skip이 보존
- 마지막 1×1이 다시 256채널로 늘리는 이유: **x와 형식 맞춰야 더할 수 있으니까**

### ResNet50 구성

- `layers = [3, 4, 6, 3]` → 총 16 Bottleneck
- 4개 layer로 그룹핑, layer 넘어갈 때 해상도 반, 채널 2배
- Layer 넘어갈 때 첫 Bottleneck의 skip path에도 1×1 conv (`downsample=True`)
- **"ResNet50"의 50 = conv1 (1) + Bottleneck 16 × 3 conv (48) + FC (1)**

### 저기여 블록 존재 가능성

- Veit et al. 2016 "Residual Networks Behave Like Ensembles..."
- 학습된 ResNet에서 랜덤 블록 삭제해도 정확도 거의 유지
- 특히 **중간 layer의 중간 블록들**이 저기여
- → 연구 주제 후보: 저기여 블록 식별 & 제거

---

## 7. Eagle-N 하드웨어 실측 결과

### 실측 커맨드

```python
import ttnn
d = ttnn.open_device(device_id=0)
print('compute_with_storage_grid_size:', d.compute_with_storage_grid_size())
print('dram_grid_size:', d.dram_grid_size())
print('arch:', d.arch())
ttnn.close_device(d)
```

### 결과

```
compute_with_storage_grid_size: (x=5, y=4)   ← ★ 20 compute 코어
dram_grid_size: (x=3, y=1)                    ← DRAM 컨트롤러 3개
arch: Arch.BLACKHOLE                          ← Blackhole 아키텍처 기반
Harvesting mask for chip 0 is 0x0             ← 불량 코어 없음
```

### 요약

- **물리 6×4 = 24 Tensix**
- **compute 가능 = 5×4 = 20** (컬럼 5 = 하드웨어 예약, compute 불가)
- 컬럼 5 용도는 문서에 명시 없음. Full Blackhole 패턴 참고 시 **Ethernet 코어 유력** (Router-only도 가능)
- Eagle-N 전용 SoC descriptor는 tt-metal 리포에 없음 (blackhole_140_arch.yaml만 존재)

### 격자 시각화

Compute 가능 영역 (컬럼 5 = 예약):

```
        x=0  x=1  x=2  x=3  x=4  |x=5|
     ┌─────────────────────────────────┐
y=0  │[C]  [C]  [C]  [C]  [C]  |RSV| │
y=1  │[C]  [C]  [C]  [C]  [C]  |RSV| │
y=2  │[C]  [C]  [C]  [C]  [C]  |RSV| │
y=3  │[C]  [C]  [C]  [C]  [C]  |RSV| │
     └─────────────────────────────────┘
      compute 20개               예약 4개
```

Baseline이 실제 쓰는 영역:

**CORE_SET_14** (layer1~3):
```
y=0 [■][■][■][■][■][ ]
y=1 [■][■][■][■][■][ ]
y=2 [■][■][■][■][ ][ ]   ← (4,2) 하나 빠짐
y=3 [ ][ ][ ][ ][ ][ ]   ← 행 3 통째로 미사용 (5코어 낭비)
```

**CORE_SET_16** (layer4/FC):
```
y=0 [■][■][■][■][ ][ ]
y=1 [■][■][■][■][ ][ ]
y=2 [■][■][■][■][ ][ ]
y=3 [■][■][■][■][ ][ ]   ← 컬럼 4 통째로 미사용 (4코어 낭비)
```

### 활용률 정리

| Sharding | 사용 코어 | 활용률 (compute 대비) |
|---|---|---|
| CORE_SET_14 | 14 / 20 | 70% |
| CORE_SET_16 | 16 / 20 | 80% |

---

## 8. 연구 방향 후보

### 후보 1: 5×4=20 코어 완전 활용 ★ 유력

- Baseline이 안 쓰는 4~6 코어를 활용
- 실험: `CORE_SET_20 = 5×4` 만들어 layer4/FC에 적용
- 이론상 20~40% 이득 여지, 실제로는 shape 나눔/reshard 오버헤드 감안 필요
- **관건**: shape 나눔 재튜닝 (2048/20=102.4, 1024/20=51.2 등 안 나눠떨어짐 → padding 조정)

### 후보 2: 저기여 Bottleneck 제거

- 16개 중 실질 기여도 낮은 블록 식별
- 학문적 근거 있음 (Veit 2016)
- 방법: 블록별 F(x) magnitude 측정 → 작은 놈부터 잘라 정확도/속도 트레이드오프
- `layers = [3, 4, 6, 3]` → `[2, 3, 4, 2]` 같은 축소

### 후보 3: Sharding 전환/reshard 감축

- Layer3.1의 명시적 reshard, Layer4 진입 reshard, avgpool 앞뒤 layout 변환
- Profiler에서 `to_memory_config`, `to_layout` op 시간 확인 필요
- 없앨 수 있는 것 있는지 조사

### 후보 4: 컬럼 5 예약 리소스 활용 (어려움)

- 컬럼 5가 ETH/router면 그 리소스를 계산에 재활용
- Metalium 레벨 심층 연구
- 위험/보상 큼

---

## 9. 다음 스텝

1. **Profiler report 생성** (Tracy 또는 TT-NN Visualizer)
   ```bash
   # Tracy
   python -m tracy -r -p -v -m pytest models/bos_model/resnet50/tests/test_ttnn_functional_resnet50.py

   # Visualizer
   source models/bos_model/export_l1_vis.sh resnet
   pytest models/bos_model/resnet50/tests/test_ttnn_functional_resnet50.py -k batch_4
   ttnn-visualizer --profiler-path generated/ttnn/reports/resnet_MMDD_hhmm/
   ```

2. **관찰 포인트** (실제 profiler 볼 때)
   - Op 총 개수
   - 상위 3개 hot op (거의 확실히 conv2 3×3)
   - `to_memory_config` / `to_layout` 시간 비중
   - Layer4 앞의 reshard op 시간
   - Avgpool 앞뒤 layout 변환 비용
   - 각 op가 실제 몇 코어에서 도는지

3. **`compute_with_storage_grid_size` 로 확인 완료**한 것 외 추가 확인
   - `tt-smi -a`, `tt-smi -sm` 로 코어별 타입
   - Evaluation_Board_Overview.md의 Figure 9 (NPU block diagram) 확인
   - 컬럼 5 정확한 용도 (ETH? Router?)
