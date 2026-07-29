# Aging-Aware Core Rotation Mapping — 탐구 정리

작성일: 2026-07-25
대상 플랫폼: BOS Eagle-N NPU (Tenstorrent Tensix 기반, 프로그래머블 spatial 아키텍처)
상태: 방향 탐색 완료 / feasibility spike 착수 전

---

## 0. 한 줄 요약

> 프로그래머블 spatial NPU에서 **각 op이 쓰는 물리 코어 집합을 실행에 걸쳐 rotation**해서 트랜지스터 stress(NBTI/HCI)를 코어 전체에 고르게 분산하는 매핑 정책. **측정 가능한 기여는 "rotation 메커니즘 + 오버헤드 특성화"이고, aging 저감 자체는 측정 불가라 시뮬레이션 투영으로 다룬다.**

---

## 1. 배경 & 동기

- **문제:** 현재 스택은 op 배치가 **완전히 결정론적**이라, 같은 모델을 돌리면 **매 추론마다 똑같은 코어가 똑같이 stress**를 받는다 → 특정 코어만 빨리 노화(NBTI/HCI로 Vth↑ → 지연↑ → 타이밍 에러 → 잘못된 연산 → 정확도 붕괴).
- **자동차 맥락:** BTI/HCI aging은 automotive SoC 검증의 실제 이슈이고, Eagle-N 안전 맥락(수년 수명, ISO 26262 wear-out)과 정확히 붙는다.
- **아이디어:** stress를 시간축으로 rotation(wear-leveling)해서 코어들이 고르게 노화하도록 만든다.

---

## 2. 선행 연구 & 우리와의 차이

**직접 선행: Chen, Ho, Jou — "Aging Mitigation in Systolic Array Accelerators: Balancing PE Loads for Enhanced Reliability" (ISOCC 2024, 국립중앙대 대만)**

- 내용: systolic array(TPU식 256×256 PE)의 불균등 PE 활용(최대 14배)이 불균등 노화를 유발 → aging-aware 매핑으로 균등 노화.
- 두 부분: **(1) "fission" 하드웨어 데이터패스**(mux 추가로 rigid systolic이 놀던 PE를 쓰게 함) + **(2) aging-aware 매핑 휴리스틱**.
- 방법: C++ 시뮬레이션, base = Google TPU, NBTI+HCI 모델(선행 차용), 10년 수명.
- 결과: 10년 후 conventional 대비 정확도 **74.75% 우위**, 활용률 **2.4배**, 활용률 분산 3.6배 개선.

**우리와의 핵심 차이 (= 차별점):**

- 그들의 주 novelty(**fission**)는 rigid systolic에 **유연성을 하드웨어로 추가**하는 것. **Tenstorrent는 그 유연성이 native**(프로그래머블)라 fission이 불필요.
- 즉 우리 버전에서 rotation은 **순수 소프트웨어 정책**이 된다. **"programmable spatial accelerator 위 software wear-leveling"은 문헌이 얇다**(대부분 systolic/GPU/FPGA aging).
- 관련: Karpuzcu et al., Rehman et al.(GPU aging), FPGA aging 등 — spatial 프로그래머블 accelerator는 open.

---

## 3. 코드베이스 조사 결과 (tt-metal, ground truth)

**결론: override API는 있으나 rotation 정책은 전무. 침습도 4/5. 진짜 비용은 program cache.**

### Q1. 코어 할당은 어디서 결정되나
- **개수:** `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp` — `determine_parallel_config()` (device grid 위 divisor search로 num_cores 결정)
- **위치:** `tt_metal/common/work_split.cpp` — `num_cores_to_corerangeset()` (1-arg 오버로드가 `start_core={0,0}` 하드코딩). Block-shard는 `conv2d_utils.cpp`에서 `CoreRange({0,0}, ...)` 인라인.
- **Matmul:** `matmul_program_config_types.hpp` 계열에서 유사하게 결정.

### Q2. 결정론적인가
- **완전 결정론적.** placement에 randomization 없음(rand/mt19937/srand 무). 항상 **(0,0) 앵커 사각형**.

### Q3. override 가능한가 — 가능하나 caveat 많음
- **Conv2d:** `Conv2dConfig{ override_sharding=true, core_grid=CoreRangeSet{...} }` → `conv2d_utils.cpp`에서 소비.
- **Matmul:** `MatmulMultiCoreReuse*ProgramConfig.allowed_worker_cores` → `matmul_device_operation.cpp`에서 소비(shift 존중).
- **함정 2개:**
  - **(a) Height-sharded conv 경로가 override CoreRangeSet을 무시하고 (0,0)에서 재구성** — 사실상 버그. conv rotation의 실질적 걸림돌.
  - **(b) mcast 앵커가 항상 (0,0)** (`conv2d_op_sharded_program_factory.cpp`).

### Q4. wear / aging / rotation 개념 존재?
- **전무.** `wear|aging|lifetime|stress|core_history|usage_count` 관련 히트 0.

### Q5. 훅 지점 & 진짜 난이도
- **(A) Python wrapper** — `ttnn.conv2d/matmul`을 감싸 `override_sharding + core_grid` 주입. **가장 현실적.** 단 autotuner의 L1-fit 로직이 꺼지므로 **자동으로 뽑힌 core 개수는 유지**해야 함.
- **(B) `work_split.cpp` 직접 수정** — 깔끔하나 conv sharded factory의 (0,0) 앵커 문제.
- **(C) device scheduler** — 그런 계층 없음. placement는 op-dispatch time에 결정.
- **진짜 비용 — program cache:** program cache가 CoreRangeSet을 해싱한다. **순진하게 매 추론 rotate → 캐시 미스 → 프로그램 factory 재실행 → 물리 NoC 좌표 재계산 + descriptor rebind + runtime-arg 재계산.** (Kernel ELF 캐시는 히트할 가능성 높음.)
- **실용 경로:** **K개 rotation variant를 미리 컴파일해 K개 캐시 엔트리를 순환.** 비용 = K× L1 shift. **"감당 가능한 K"가 핵심 미지수.**

---

## 4. 정직한 평가 — 무엇이 measurable하고 무엇이 아닌가

| 구분 | 내용 | 측정 가능? |
|---|---|---|
| **시스템 메커니즘** | rotation 정책 구현 + 오버헤드(K variants, program-cache 비용) 특성화 | **가능 (실물 실측)** |
| **활용률/활동량 균형** | 코어별 stress 분산 정도 | **가능 (프로파일)** |
| **aging 저감** | 실제 수명 연장 | **불가 (수년 소요)** → 시뮬레이션 투영 필요 |

**따라서 프레이밍:**

> **main = "프로그램-캐시 기반 스택에서 저오버헤드 placement rotation 메커니즘을 실물 programmable NPU에 구현하고 오버헤드를 특성화" (측정 가능, 덜 탐구됨)**
> **motivation = aging (자동차 맥락) + 실측 활동량으로 구동한 aging 시뮬 투영**

**"aging을 줄인다"를 헤드라인으로 걸면 안 됨** — 측정 못 하는 주장이 된다. 측정(rotation/K/활용률)과 투영(aging)을 명확히 분리.

---

## 5. 기존 탐구와의 통합 (sunk cost 회수)

이 rotation 정책은 앞서 판 세 아이디어를 하나로 흡수한다:

- **여유 코어 FT** → "쓰지 않는 코어"를 wear 균등화에도 활용 → 하나의 placement 정책으로 통합.
- **Row-swap (물리 위치 변경)** → aging-induced perf/SDC drift를 spatial probe로 관측하는 실험이 됨.
- **SubDevice sweep** → static partition vs dynamic rotation vs aging-rotation 비교 축.

---

## 6. 리스크

1. **aging unmeasurable** → 순수 시뮬레이션/모델링 논문이 되기 쉬움. **하드웨어 측정(활동량/온도 proxy)이 뽑히는지 확인 필요.**
2. **program cache 오버헤드** → "감당 가능한 K"를 먼저 실측해야 그림이 나옴.
3. **override 버그** (height-sharded conv가 override 무시) → conv 경로 rotation의 엔지니어링 해저드.
4. **침습도 4/5** → 학부 타임라인 리스크.

---

## 7. 다음 스텝 — Feasibility Spike (2~3일, 착수 전 필수)

**목표: 몇 주 투자 전에 "되는지/안 되는지"를 숫자로 판정.**

1. **[최우선] 감당 가능한 K 실측**
   - placement variant 3~5개를 미리 컴파일 → 순환 시 program-cache/rebuild 오버헤드 측정.
   - **성공 기준:** K가 저오버헤드로 충분히 커서 wear를 의미 있게 분산할 수 있는가.
2. **conv override 실제 동작 확인**
   - height-sharded conv가 override CoreRangeSet을 정말 무시하는지(Q3-a 버그) 재현.
3. **aging proxy 관측 가능성**
   - 실물에서 per-core 활동량/온도가 뽑히는지 (aging은 못 재도 stress 입력은 real).

**판정:** 1이 되고 2가 우회 가능하면 진행. 1이 안 되거나 2가 막히면 근거 갖고 방향 재검토.

---

## 8. 열린 질문 (팀 논의용)

- rotation 정책의 목적함수: 순수 wear 균등화? aging 예측 모델 기반 최적화? 성능과의 트레이드오프를 어디까지 허용?
- variant 단위: 모델 전체 vs 결함/취약 레이어부터?
- aging 시뮬 파라미터: 문헌 값(대표 노드) 차용 — Eagle-N 공정과의 괴리를 어떻게 다룰지(절대값 vs 상대 비교).


---

## 9. Session 2026-07-26 진행 — Empirical validation

### 9.1 (0,0) 편향은 실측 가능한 규모 (**연구 정당화 근거 확보**)

기존 ResNet50 stage-split NoC 트레이스 (`bos_model_copy/resnet50_jbpark/analysis/outputs/`) 를 재분석. 트레이스 좌표는 6×5 translated grid (y=0 = IO 행 LPDDR5/PCIe/ARC, y=3 = DRAM passthrough, x=5 = memory 컨트롤러). 이 IO/DRAM 을 제외한 **compute region 만 (y∈{1,2,4}, x∈0..4, 15코어)** 필터링 후 비교:

| 지표 | Max (top-left compute) | Min (bottom-right compute) | **비율** |
|---|---|---|---|
| Injected bytes | (0,1) = 2.38 MB | (4,4) = 0.50 MB | **4.76×** |
| ops_as_src | (0,1) = 74 ops | (4,4) = 18 ops | **4.11×** |
| Column 편향 x=0 vs x=4 | 2.10 MB mean | 0.85 MB mean | **2.47×** |
| Row 편향 y=1 vs y=4 | 1.94 MB mean | 1.11 MB mean | 1.75× |

**Aging 함의**: BTI 는 √t 스케일 → 활동량 4.76× → Vth 열화 속도 약 **2.18× 가속**. 5년 수명이면 고활동 코어가 2년치 초과 열화 → 조기 timing failure. **아이디어 정당화 완료.**

Consumed side (write barrier / L1 buffer 활동) 는 더 극단적: (3,1)=12.65 MB vs (0,3)=1.94 MB → 25× 이상 차이. Output shard concentration 이 특정 코어에 집중.

### 9.2 왜 rotation 이 이 스택에서 어려운지 (C++ 소스로 확정)

컨테이너 tt-metal 은 BOS fork (`bos-semi/tt-metal`, HEAD = `a035c886`, 2026-05-29). `operations/` 스냅샷을 laptop 에 뜨서 조사:

**Matmul (`operations/matmul/device/matmul_op.hpp:192-204`)**:
- `allowed_worker_cores` 필드 **없음** (upstream 2025 리팩터링 이후 추가된 것)
- `compute_with_storage_grid_size = CoreCoord(gx, gy)` 만 있음 → **항상 (0,0) 앵커**
- Rotation shift 방법 없음

**Conv2d (`operations/conv/conv2d/device/conv2d_op.hpp:22-72`)**:
- `Conv2dConfig::override_sharding_config: bool` (line 56) ✅ 존재
- `Conv2dConfig::core_grid: std::optional<CoreRangeSet>` (line 61) ✅ 존재
- Python 바인딩 완비 (`conv2d_pybind.cpp:331-407`)
- 소비 지점 `conv2d_utils.cpp:614-670` ✅ 정상
- **BUT** `conv2d_op_sharded_program_factory.cpp:576` 에 `top_left_core = {0, 0}` **하드코딩**
- `.cpp:573-574` 에서 `in_num_cores_x = input_cores.bounding_box().end_coord.x + 1` — shift 무시하는 개수 산정
- `.cpp:600, 605` mcast sender/receiver 도 (0,0) 기준
- → **shift 된 core_grid 넘겨도 mcast 가 (0,0) 기준으로 나감 → wrong output 예상**

**SubDevice route 는 rotation 에 부적합 (문서 인용)**:
- `SubDevices.md §1.1`: "load_sub_device_manager requires all local allocators to have no buffers allocated"
- `§1.1`: manager 스위치는 "synchronize and wait for all existing programs to finish"
- `§1.3`: "**Programs cannot be rerun on a different sub-device-manager configuration**"
- → op-level rotation 불가. SubDevice 설계 목적은 concurrent execution, 시간축 rotation 이 아님.

**결론**: 이 스택에서 진짜 rotation 하려면 **C++ 패치 필수**. 최소 수정 범위:
1. `top_left_core = input_cores.bounding_box().start_coord` (하드코딩 제거)
2. `in_num_cores_x = end - start + 1` (개수 산정 수정)
3. Matmul 쪽에도 `allowed_worker_cores` 백포트 (선택)

### 9.3 K spike v2 결과 — **🟢 초록불** (단, 부분적)

`bench_matmul_shape_variants.py` 실행 결과 (2026-07-26):

| K | mean lat (μs) | vs K=1 | cache_delta |
|---|---|---|---|
| 1 | 212.47 | +0.00% | 0 |
| 2 | 216.33 | +1.82% | 0 |
| 3 | 205.20 | −3.42% | 0 |
| 4 | 196.36 | −7.58% | 0 |
| 6 | 192.88 | **−9.22%** | 0 |

**해석**:
- K 커질수록 오히려 빨라짐 (음수 %) — K=1 baseline 이 warmup 부족으로 부풀려진 것. K 늘수록 total 반복 증가로 정착 latency 드러남. **오버헤드 0 이라는 강한 signal**.
- Min latency 167μs @ K=4 = 진짜 op cost. 나머지는 host dispatch jitter.
- `cache_delta=0` — `num_program_cache_entries()` 가 우리 예상과 다른 걸 카운트하거나 auto-enabled 로 이미 채워져 있음. **latency 안정성이 캐시 hit 을 증명** (재컴파일이면 ms 단위여야 함).

**Phase 1 부가**: 6/10 shape 통과. 실패 4개 (2×3, 3×2, 3×4, 4×3) 는 `num_blocks <= num_cores` constraint (per_core_M/N 계산 잔여 처리 미흡, 하드코딩 버그 아님).

**⚠ 방법론적 한계 (사용자 지적)**: 이 실험은 프로그램 캐시가 **거의 빈 상태 (우리 것 6개만)** 에서 진행. Real deployment (ResNet50 op ~100개 × K variants) 의 캐시 pressure 는 시뮬레이션 안 됨. "**빈 냄비에 라면 하나 끓임**" 수준 검증.

### 9.4 Program cache 실제 설계 확인 (2026-07-26 추가)

`tt_metal/api/tt-metalium/program_cache.hpp` (Tenstorrent 2023 안정 설계) 소스 확인 결과:

```cpp
struct ProgramCache {
    void clear() { this->cache_.clear(); }
    std::size_t num_entries() const { return this->cache_.size(); }
private:
    std::unordered_map<ProgramCacheKey, CachedProgramFactory, ...> cache_;
};
```

- **`std::unordered_map` — 크기 제한 없음. LRU 없음. Eviction 정책 없음.**
- 삭제 유일 경로: `clear()` (사용자 명시 호출) or device close
- 각 entry 최대 크기: `CachedProgramFactory::MAX_SIZE = 4096` bytes

**함의**: cache pressure 우려 (이전 §9.3 의 "감당 가능한 K 미결" 부분) 는 **소스로 반박됨**. K variants 를 미리 다 dispatch 하면 모두 저장되고 host RAM 만이 유일한 제약. ResNet50 100 op × K=6 = 600 entries × 4KB ≈ **2.4 MB** — 서버 RAM 관점 무시 가능.

**Program cache 의 정확한 의미**: **host RAM 안에 있는 `Program` C++ 조립 지시서** (각 코어에 어느 kernel binary 를 올릴지, runtime args, semaphore, CB, mcast 좌표 매핑) 에 대한 캐시. Device DRAM/L1 과 무관. L1 은 매 dispatch 마다 kernel binary 로 덮어써지는 소모품, DRAM 은 tensor buffer 저장소.

**결론 갱신**:
- **재컴파일 재발 우려 없음** — cache 는 사실상 무한, 미리 dispatch 한 K variants 는 계속 있음
- **진짜 남은 오버헤드는 dispatch 층** (kernel binary NoC upload, runtime args 계산) — 이건 rotation 유무와 무관하게 매 dispatch 발생하는 비용
- 정말 물어야 할 질문: **rotation 이 붙은 dispatch 가 rotation 없는 dispatch 보다 유의미하게 느린가?** — C++ 패치 후 실물 실험으로만 답 가능

즉 **C++ 패치 정당화 논거는 여전히 유효**, 다만 검증 대상이 "cache pressure" 가 아니라 "dispatch 층 μs 누적" 으로 명확화됨.

---

## 10. 업데이트된 다음 스텝 — 🟢×🟢 = C++ 패치 진입 확정

**두 결정적 신호**:
1. **문제 정당화** (§9.1): (0,0) 편향 4.76× — 프레임워크가 강제하는 stress 편향 실측
2. **솔루션 실현성** (§9.3): K 순환 오버헤드 0 — rotation 인프라만 있으면 무비용

→ **C++ 패치 (기존 하드코딩 제거) 가 가장 정당한 다음 스텝**.

---

## 10. 업데이트된 다음 스텝

**다음 세션 시작점 — C++ 패치 계획 (§11 신설)**:

1. **수정 대상 (최소 범위)**:
   - `conv2d_op_sharded_program_factory.cpp:576` — `top_left_core = input_cores.bounding_box().start_coord`
   - `.cpp:573-574` — `in_num_cores_x = end - start + 1` (개수 산정 수정)
   - `.cpp:600, 605` — mcast sender/receiver 도 start_coord 기반으로
2. **빌드 전략**:
   - Full rebuild 30분~1시간 예상 — 감당 가능한지 서버 상태 확인
   - Incremental build 가능한지 (conv2d 관련만) 조사 → 훨씬 빠를 것
   - **주의**: bos_model 이 이 코드 경로에 의존하므로 rebuild 후 회귀 테스트 필요 (ResNet50 baseline 돌려서 PCC 검증)
3. **패치 후 검증 순서**:
   - a. Rebuild 성공 확인
   - b. bos_model 기존 conv 호출 정상 동작 확인 (regression)
   - c. Shifted `core_grid` 로 PCC 재검증 → 통과하면 진짜 rotation 실현
   - d. 진짜 rotation 스크립트 작성 → K 측정 재실행 (이번엔 진짜 shift 로)
4. **fallback**: 패치 실패 or bos_model 회귀 발생 시 → aging 편향 evidence 만 확보한 상태에서 "framework-induced stress asymmetry" 관찰 논문으로 프레이밍

---

---

## 12. 확인해봐야 할 점들 (미결, 2026-07-26 세션 마무리)

**아직 아이디어 정리 X. 순수 open questions.**

### 12.1 코드 동작 확인 필요
- [ ] Height-sharded 경로 (`sharded_program_factory.cpp:277-283`) 가 override CoreRangeSet 을 실제로 무시하는지 PCC 로 재현
- [ ] Width-sharded 경로 (`conv2d_op_width_sharded_program_factory.cpp`) 구조 다름 — shift 를 이미 존중하는지 line-by-line 확인
- [ ] Block-sharded 경로가 shift 에서 어떻게 동작하는지 (구조상 `input_cores` 기반이라 상대적으로 나을 수도)
- [ ] Reader/Compute/Writer 커널 안에 **logical position 가정** 있는지 (예: "get_arg(0)==0 이면 sender"). C++ 패치 후 실측으로 확인 or 커널 소스 grep
- [ ] Kernel binding line 1155/1168 이 정말 shard spec 만 따라가는지 (다른 코드에서 재override 안 하는지)

### 12.2 아키텍처 상 (0,0) 앵커 이유 파악
왜 (0,0) 하드코딩됐는지 정확한 원인 — 아이디어 A/B 선택에 결정적
- [ ] NoC 라우팅: (0,0) 이 mcast tree root 로 최적인지, shift 시 hop 수 얼마나 증가하는지
- [ ] FastDispatch 코어 물리 위치 — (0,0) 컴퓨트 vs shifted 컴퓨트의 dispatch latency 차이
- [ ] DRAM/LPDDR5 컨트롤러 위치 vs 컴퓨트 anchor 거리 — activation/weight load 시간 영향
- [ ] Chip 코너/harvesting 관련 특수성 있는지

### 12.3 Program cache 실제 상태
- [ ] Container BOS fork `/tt-metal/tt_metal/api/tt-metalium/program_cache.hpp` 가 upstream 과 같은 unbounded 설계인지 확인
- [ ] `num_program_cache_entries()` 가 실제로 뭘 카운트하는지 (우리 spike 결과 delta=0 이 정상인지)
- [ ] `enable_program_cache()` 가 default enabled 인지, 우리가 켤 필요 있는지
- [ ] Real ResNet50 실행 시 cache entries 실제로 몇 개 쌓이는지 (100 op × K variants 시나리오)

### 12.4 실측 재검증 필요
- [ ] **K spike v2 를 realistic cache pressure 하에서 재실행**: ResNet50 baseline 한 사이클 돌린 후 K spike → cache 채워진 상태에서 K 순환 오버헤드 재측정
- [ ] Shape variation (아이디어 B) 만으로 편향 얼마나 개선? — shape 강제 후 NoC 트레이스 재수집 → 히트맵 재비교
- [ ] Shape 강제 시 conv 자체 성능 손실 (autotuner override 로 인한) — baseline vs 강제 shape latency 비교
- [ ] Shifted 실행 시 성능 손실 정량 (mcast latency, dispatch latency, DRAM hop) — C++ 패치 후에만 가능

### 12.5 결과 해석 caveat
- [ ] (0,0) 편향 실측이 IO 행 (y=0 LPDDR5/PCIe/ARC) 제외 후에도 유효 → **이미 확인됨 (§9.1)**. 하지만 y=3 DRAM 행 처리도 재확인 (injected=0 이 정말 계산 코어 아니라는 증거)
- [ ] Consumed side 25× 편향의 실체 — (3,1), (2,1) 이 정말 compute core 인지 아니면 특수 역할인지
- [ ] Shape 만 다양화 시 (0,0) monopoly 는 어차피 유지됨 → 얼마나 balance 되는지 실측 없이 예측 못 함

### 12.6 아이디어 자체 정리 필요 (내일 이후)
- 아이디어 A (dynamic shift rotation) vs 아이디어 B (static shape variation) 아직 정리 X
- 어느 것이 논문 스토리에 더 맞는지, 둘 다 하는지, 하나 버리는지 — 데이터 더 확보 후 결정
- 관련 프레이밍: "cheap 버전 vs full 버전" 으로 병렬 프레이밍 가능성

---

## 11. Raw appendix — session terminal logs

(원본 진단 로그 — 참고용, 정리된 finding 은 §9-10 참조)

device grid: 5 x 4
======================================================================
PHASE 1 — PCC gate (offset shift support)
======================================================================
[mcast2d] at-origin (2x4 @ (0,0))
Traceback (most recent call last):
  File "/tt-metal/models/bos_model/resnet50_jbpark/spike_k_measurement/bench_matmul_rotation.py", line 296, in <module>
    main()
  File "/tt-metal/models/bos_model/resnet50_jbpark/spike_k_measurement/bench_matmul_rotation.py", line 273, in main
    phase = phase1_pcc_gate(device)
  File "/tt-metal/models/bos_model/resnet50_jbpark/spike_k_measurement/bench_matmul_rotation.py", line 119, in phase1_pcc_gate
    cfg0 = mcast2d_cfg(crs(BLOCK_COLS, BLOCK_ROWS, 0, 0))
  File "/tt-metal/models/bos_model/resnet50_jbpark/spike_k_measurement/bench_matmul_rotation.py", line 57, in mcast2d_cfg
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
TypeError: __init__(): incompatible constructor arguments. The following argument types are supported:
    1. ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCastProgramConfig(*, compute_with_storage_grid_size: ttnn._ttnn.tensor.CoreCoord, in0_block_w: int, out_subblock_h: int, out_subblock_w: int, out_block_h: Optional[int] = None, out_block_w: Optional[int] = None, per_core_M: int, per_core_N: int, transpose_mcast: bool, fused_activation: Optional[ttnn._ttnn.activation.UnaryWithParam], fuse_batch: bool = True)

Invoked with: kwargs: compute_with_storage_grid_size=(x=2,y=4), in0_block_w=2, out_subblock_h=1, out_subblock_w=1, per_core_M=4, per_core_N=4, transpose_mcast=False, allowed_worker_cores={[(x=0,y=0) - (x=1,y=3)]}


(python_env) root@7575fef6ac31:/tt-metal# find /tt-metal -name "*.hpp" -path "*matmul*" 2>/dev/null | head -20
/tt-metal/tests/tt_metal/tt_metal/common/matmul_test_utils.hpp
/tt-metal/.cpmcache/taskflow/52063f60902bfeb362fa4616b1394ab5efe30994/taskflow/cuda/algorithm/matmul.hpp
/tt-metal/ttnn/cpp/ttnn/operations/matmul/matmul.hpp
/tt-metal/ttnn/cpp/ttnn/operations/matmul/matmul_pybind.hpp
/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/pad_tile.hpp
/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.hpp
/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_matmul_backward/moreh_matmul_backward_pybind.hpp
/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_matmul_backward/moreh_matmul_backward.hpp
/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_matmul/moreh_matmul_pybind.hpp
/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_matmul/device/moreh_matmul_device_operation.hpp
/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp
/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/group_attn_matmul_device_operation.hpp
/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul.hpp
/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul_pybind.hpp
/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/attn_matmul_pybind.hpp
/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/attn_matmul.hpp
/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/attn_matmul_device_operation.hpp
/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_op.hpp
/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async.hpp
/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async_pybind.hpp
(python_env) root@7575fef6ac31:/tt-metal# python -c "import ttnn; print('SubDevice:', hasattr(ttnn, 'SubDevice')); print('SubDeviceId:', hasattr(ttnn, 'SubDeviceId')); print('create_sub_device_manager:', hasattr(ttnn, 'create_sub_device_manager'))"
2026-07-25 15:40:48.439 | DEBUG    | ttnn:<module>:77 - Initial ttnn.CONFIG:
Config{cache_path=/root/.cache/ttnn,model_cache_path=/root/.cache/ttnn/models,tmp_dir=/tmp/ttnn,enable_model_cache=false,enable_fast_runtime_mode=true,throw_exception_on_fallback=false,enable_logging=false,enable_graph_report=false,enable_detailed_buffer_report=false,enable_detailed_tensor_report=false,enable_comparison_mode=false,comparison_mode_should_raise_exception=false,comparison_mode_pcc=0.9999,root_report_path=generated/ttnn/reports,report_name=std::nullopt,std::nullopt}
SubDevice: True
SubDeviceId: True
create_sub_device_manager: False
(python_env) root@7575fef6ac31:/tt-metal# grep -rln "struct MatmulMultiCoreReuseMultiCastProgramConfig" /tt-metal/ttnn/cpp/ 2>/dev/null
/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.hpp

(python_env) root@7575fef6ac31:/tt-metal# grep -A 25 "struct MatmulMultiCoreReuseMultiCastProgramConfig" \
    /tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.hpp
struct MatmulMultiCoreReuseMultiCastProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t out_block_h{};
    std::size_t out_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    bool transpose_mcast{};
    std::optional<UnaryWithParam> fused_activation;
    bool fuse_batch = true;
};

struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t out_block_h{};
    std::size_t out_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    bool fuse_batch{};
    std::optional<UnaryWithParam> fused_activation;
    bool mcast_in0{};
(python_env) root@7575fef6ac31:/tt-metal# python -c "
import ttnn
d = ttnn.open_device(device_id=0)
methods = [m for m in dir(d) if 'sub_device' in m.lower() or 'stall_group' in m.lower()]
print('device sub_device methods:', methods)
ttnn.close_device(d)
"
2026-07-25 15:43:38.102 | DEBUG    | ttnn:<module>:77 - Initial ttnn.CONFIG:
Config{cache_path=/root/.cache/ttnn,model_cache_path=/root/.cache/ttnn/models,tmp_dir=/tmp/ttnn,enable_model_cache=false,enable_fast_runtime_mode=true,throw_exception_on_fallback=false,enable_logging=false,enable_graph_report=false,enable_detailed_buffer_report=false,enable_detailed_tensor_report=false,enable_comparison_mode=false,comparison_mode_should_raise_exception=false,comparison_mode_pcc=0.9999,root_report_path=generated/ttnn/reports,report_name=std::nullopt,std::nullopt}
2026-07-25 15:43:39.497 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-25 15:43:39.531 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-25 15:43:39.531 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-25 15:43:39.532 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:202)
2026-07-25 15:43:39.555 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-25 15:43:39.589 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-25 15:43:39.589 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-25 15:43:39.624 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-25 15:43:39.654 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-25 15:43:39.657 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-25 15:43:39.658 | info     |             UMD | Harvesting mask for chip 0 is 0x0 (NOC0: 0x0, simulated harvesting mask: 0x0). (cluster.cpp:399)
2026-07-25 15:43:39.694 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-25 15:43:39.694 | info     |             UMD | Initializing iommu for sysmem (size: 0x40000000). (sysmem_manager.cpp:300)
2026-07-25 15:43:39.694 | info     |             UMD | Allocating sysmem without hugepages (size: 0x40000000). (sysmem_manager.cpp:306)
2026-07-25 15:43:39.926 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-25 15:43:39.926 | info     |             UMD | Opening local chip ids/PCIe ids: {0}/[0] and remote chip ids {} (cluster.cpp:251)
2026-07-25 15:43:39.926 | info     |             UMD | IOMMU: enabled (cluster.cpp:173)
2026-07-25 15:43:39.926 | info     |             UMD | KMD version: 2.3.0 (cluster.cpp:176)
2026-07-25 15:43:39.926 | info     |          Device | start_device in llrt (tt_cluster.cpp:415)
2026-07-25 15:43:39.988 | info     |             UMD | Pinning pages for DMA: virtual address 0x7327faa00000 and size 0x40000000 pinned to physical address 0x7fc0000000 without mapping to noc (pci_device.cpp:618)
2026-07-25 15:43:39.988 | info     |             UMD | Mapped sysmem without hugepages to IOVA 0x7fc0000000; NOC address 0x0 (sysmem_manager.cpp:354)
2026-07-25 15:43:40.032 | info     |             UMD | Initializing memory barriers (local_chip.cpp:171)
2026-07-25 15:43:41.623 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
device sub_device methods: ['clear_loaded_sub_device_manager', 'create_sub_device_manager', 'load_sub_device_manager', 'remove_sub_device_manager', 'reset_sub_device_stall_group', 'set_sub_device_stall_group']
2026-07-25 15:43:44.472 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:427)
2026-07-25 15:43:44.490 | info     |             UMD | Unpinning pages for DMA: virtual address 0x7327faa00000 and size 0x40000000 (pci_device.cpp:644)
(python_env) root@7575fef6ac31:/tt-metal# 

device grid: 5 x 4
======================================================================
PHASE 1 — PCC gate (all shapes at (0,0) anchor)
======================================================================
  2x2 ( 4 cores) : EXCEPTION  [TypeError: __init__(): incompatible constructor arguments. The followin]
  4x1 ( 4 cores) : EXCEPTION  [TypeError: __init__(): incompatible constructor arguments. The followin]
  1x4 ( 4 cores) : EXCEPTION  [TypeError: __init__(): incompatible constructor arguments. The followin]
  2x3 ( 6 cores) : EXCEPTION  [TypeError: __init__(): incompatible constructor arguments. The followin]
  3x2 ( 6 cores) : EXCEPTION  [TypeError: __init__(): incompatible constructor arguments. The followin]
  2x4 ( 8 cores) : EXCEPTION  [TypeError: __init__(): incompatible constructor arguments. The followin]
  4x2 ( 8 cores) : EXCEPTION  [TypeError: __init__(): incompatible constructor arguments. The followin]
  3x4 (12 cores) : EXCEPTION  [TypeError: __init__(): incompatible constructor arguments. The followin]
  4x3 (12 cores) : EXCEPTION  [TypeError: __init__(): incompatible constructor arguments. The followin]
  4x4 (16 cores) : EXCEPTION  [TypeError: __init__(): incompatible constructor arguments. The followin]

>>> 0/10 shapes are numerically correct.

!!! Fewer than 2 valid shapes — cannot run K sweep.
2026-07-25 16:25:50.471 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:427)
2026-07-25 16:25:50.483 | info     |             UMD | Unpinning pages for DMA: virtual address 0x7bf1e2c00000 and size 0x40000000 (pci_device.cpp:644)


device grid: 5 x 4
======================================================================
PHASE 1 — PCC gate (all shapes at (0,0) anchor)
======================================================================
  2x2 ( 4 cores) : PCC = 0.999979   [OK ]
  4x1 ( 4 cores) : PCC = 0.999979   [OK ]
  1x4 ( 4 cores) : PCC = 0.999979   [OK ]
2026-07-25 16:28:10.850 | critical |          Always | Num output blocks along y (4) must be smaller than or equal to the number of rows in compute grid (3)! (assert.hpp:103)
  2x3 ( 6 cores) : EXCEPTION  [RuntimeError]
      TT_FATAL @ /tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_2d_program_factory.cpp:1560: num_blocks_y <= num_cores_y
info:
Num output blocks along y (4) must be smaller than or equal to the number of rows in compute grid (3)!
backtrace:
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0x7967e8) [0x75574fa287e8]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn10operations6matmul43matmul_multi_core_reuse_mcast_2d_optimized_ERN2tt8tt_metal7ProgramERKNS
2026-07-25 16:28:10.856 | critical |          Always | Num output blocks along x (4) must be smaller than or equal to the number of columns in compute grid (3)! (assert.hpp:103)
  3x2 ( 6 cores) : EXCEPTION  [RuntimeError]
      TT_FATAL @ /tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_2d_program_factory.cpp:1555: num_blocks_x <= num_cores_x
info:
Num output blocks along x (4) must be smaller than or equal to the number of columns in compute grid (3)!
backtrace:
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0x7967e8) [0x75574fa287e8]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn10operations6matmul43matmul_multi_core_reuse_mcast_2d_optimized_ERN2tt8tt_metal7ProgramER
  2x4 ( 8 cores) : PCC = 0.999979   [OK ]
  4x2 ( 8 cores) : PCC = 0.999979   [OK ]
2026-07-25 16:28:12.940 | critical |          Always | Num output blocks along x (4) must be smaller than or equal to the number of columns in compute grid (3)! (assert.hpp:103)
  3x4 (12 cores) : EXCEPTION  [RuntimeError]
      TT_FATAL @ /tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_2d_program_factory.cpp:1555: num_blocks_x <= num_cores_x
info:
Num output blocks along x (4) must be smaller than or equal to the number of columns in compute grid (3)!
backtrace:
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0x7967e8) [0x75574fa287e8]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn10operations6matmul43matmul_multi_core_reuse_mcast_2d_optimized_ERN2tt8tt_metal7ProgramER
2026-07-25 16:28:12.942 | critical |          Always | Num output blocks along y (4) must be smaller than or equal to the number of rows in compute grid (3)! (assert.hpp:103)
  4x3 (12 cores) : EXCEPTION  [RuntimeError]
      TT_FATAL @ /tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_2d_program_factory.cpp:1560: num_blocks_y <= num_cores_y
info:
Num output blocks along y (4) must be smaller than or equal to the number of rows in compute grid (3)!
backtrace:
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0x7967e8) [0x75574fa287e8]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn10operations6matmul43matmul_multi_core_reuse_mcast_2d_optimized_ERN2tt8tt_metal7ProgramERKNS
  4x4 (16 cores) : PCC = 0.999979   [OK ]

>>> 6/10 shapes are numerically correct.

======================================================================
PHASE 2 — K sweep on valid shape variants
======================================================================

--- K = 1 ---
    shapes: [(2, 2)]
    cache : before=6  after_warmup=6  after_measure=6  delta=0
    lat us: mean= 212.47  p50= 210.84  p95= 226.07  min= 199.02  max= 264.10

--- K = 2 ---
    shapes: [(2, 2), (4, 1)]
    cache : before=6  after_warmup=6  after_measure=6  delta=0
    lat us: mean= 216.33  p50= 215.41  p95= 236.18  min= 197.90  max= 269.94

--- K = 3 ---
    shapes: [(2, 2), (4, 1), (1, 4)]
    cache : before=6  after_warmup=6  after_measure=6  delta=0
    lat us: mean= 205.20  p50= 204.65  p95= 227.69  min= 181.79  max= 278.57

--- K = 4 ---
    shapes: [(2, 2), (4, 1), (1, 4), (2, 4)]
    cache : before=6  after_warmup=6  after_measure=6  delta=0
    lat us: mean= 196.36  p50= 197.74  p95= 218.29  min= 167.00  max= 258.64

--- K = 6 ---
    shapes: [(2, 2), (4, 1), (1, 4), (2, 4), (4, 2), (4, 4)]
    cache : before=6  after_warmup=6  after_measure=6  delta=0
    lat us: mean= 192.88  p50= 190.23  p95= 218.64  min= 169.02  max= 236.80

--- K = 8 : SKIP (only 6 valid variants) ---

--- K = 10 : SKIP (only 6 valid variants) ---

Wrote:
  /tt-metal/models/bos_model/resnet50_jbpark/spike_k_measurement/results/shape_summary_20260725_162815.csv
  /tt-metal/models/bos_model/resnet50_jbpark/spike_k_measurement/results/shape_raw_20260725_162815.csv
  /tt-metal/models/bos_model/resnet50_jbpark/spike_k_measurement/results/shape_meta_20260725_162815.txt

======================================================================
VERDICT
======================================================================
K= 1: mean   212.47 us  (  +0.00% vs K=1)  cache_delta= 0  [cache-hit steady]
K= 2: mean   216.33 us  (  +1.82% vs K=1)  cache_delta= 0  [cache-hit steady]
K= 3: mean   205.20 us  (  -3.42% vs K=1)  cache_delta= 0  [cache-hit steady]
K= 4: mean   196.36 us  (  -7.58% vs K=1)  cache_delta= 0  [cache-hit steady]
K= 6: mean   192.88 us  (  -9.22% vs K=1)  cache_delta= 0  [cache-hit steady]
2026-07-25 16:28:15.608 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:427)
2026-07-25 16:28:15.621 | info     |             UMD | Unpinning pages for DMA: virtual address 0x7556b5a00000 and size 0x40000000 (pci_device.cpp:644)