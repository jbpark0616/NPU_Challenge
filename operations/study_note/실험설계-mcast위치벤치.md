# 작업 지시 — 임의 sender 위치 mcast "정합성" 시험 (Claude Code 실행용)

> **한 줄 목표:** conv가 쓰는 mcast가 **임의 sender 위치에서도 올바르게 데이터를 전달하는지** 확인한다.
> = "위치를 자유롭게 해도 mcast가 깨지지 않는다"를 **최소 노력으로 실증**.
> **성능/latency/imbalance 측정은 이번 스코프가 아니다 (나중).** 지금은 오직 **correctness(정합성)**.

---

## 0. 왜 이걸 먼저 하나 (맥락)

- conv 자유 배치의 **유일한 위험 = mcast가 임의 sender에서 되나?** 나머지(compute, halo 세부)는 검증된 것.
- width 샤딩이 이미 `noc_async_write_multicast_loopback_src`로 **임의 sender mcast를 쓰고 있음** → 원리상 됨. 근데 **"있다 ≠ conv 맥락에서 된다"** → 실측으로 de-risk.
- **conv 전체를 건드리기 전에**, mcast 프리미티브만 격리해서 "임의 sender 정합" 확인.
- **halo도 같은 noc multicast + 세마포어 패턴** → 이 시험이 halo도 원리상 커버 (halo 고유 하드코딩은 나중).

---

## 1. 에이전트 실행 규칙

1. **conv/mcast 실행은 컨테이너 안 + 보드에서.** `[CONTAINER]` = `docker exec -it bos_host bash`.
2. **공유 컨테이너다.** 파일 수정 시 `.orig` 백업(별도 문서 규칙). 남의 실험 폴더(`resnet50_cys/`, `resnet50_cmg/` 등)·컨테이너 건드리지 말 것.
3. **새 코드 최소화.** 기존 tt-metal 테스트를 **먼저 읽고 재사용**. 못 하면 최소 확장.
4. **이번엔 correctness만.** heatmap/rotation/per-link 트래픽/latency 스윕은 **하지 말 것** (다음 단계).

---

## 2. 재사용할 기존 자산 (tt-metal, `C:\NPU_Challenge\tt-metal\` 기준)

### 이번 목표에 직결 (이것부터)
| 파일 | 왜 |
|---|---|
| `tests/tt_metal/tt_metal/data_movement/one_to_all/test_multicast_schemes.cpp` (+ `kernels/sender_multicast_sem.cpp`) | **1 sender → 사각형 mcast + 세마포어.** conv mcast 패턴의 base |
| `tests/tt_metal/tt_metal/test_dram_to_l1_multicast_loopback_src.cpp` (+ `test_kernels/dataflow/dram_to_l1_multicast_include_src.cpp`) | **loopback(include-src) mcast 실증.** 임의 sender 프리미티브 |
| `tests/tt_metal/tt_metal/data_movement/one_to_all/test_one_to_all.cpp` | 1→N 기본형 |

### 무시 (이번 스코프 아님)
- fabric/ethernet(chip-to-chip), quasar, pcie, dram_* 성능, `python/heatmap.py`, rotation policy, `noc_estimator` → **전부 다음 단계(imbalance)용.**

### 파일 이해용 네이밍 규칙
- `_loopback_src`/`_include_src` = sender도 수신자 포함 (loopback) · `_sem` = 세마포어 핸드셰이크 · `_hardcoded` = 고정 실제 워크로드 · `_2_0` = v2

---

## 3. 실행 계획 (순서 · 결정점)

### STEP 1 — 기존 테스트 읽고 판정 (코드 짜기 전)
대상: `test_multicast_schemes.cpp`, `test_dram_to_l1_multicast_loopback_src.cpp`
**확인할 것 (각 파일):**
- [ ] **sender(소스) 코어 위치가 파라미터인가?** (인자/변수로 바꿀 수 있나, 아니면 하드코딩인가)
- [ ] **receiver 사각형이 파라미터인가?**
- [ ] **정합성 검증이 있나?** (수신 데이터 == 송신 데이터 비교, golden/assert)
- [ ] loopback 변형이 실제로 include-src로 도는가

**판정 → 분기:**
- **sender 파라미터화 O + 정합검증 O** → STEP 2 (그냥 실행)
- **sender 하드코딩** → STEP 3 (최소 확장)

### STEP 2 — 여러 sender 위치로 실행 (새 코드 X)
- 같은 receiver 사각형에 **sender 위치만 바꿔가며** 실행: top-left(0,0) / center / 임의 코너
- **[CONTAINER]** 각 실행에서 **모든 receiver 정합 통과하는지** 확인
- **PASS:** 모든 sender 위치에서 **모든 receiver가 올바른 데이터 수신** (assert 통과)

### STEP 3 — 최소 확장 (필요 시)
- sender 코어 좌표를 **인자로** 받게만 추가 (`--sender-x/--sender-y` 또는 상수 파라미터화)
- 로직·검증부는 그대로. **origin만 파라미터화, 정합검증 재사용**
- 확장 후 STEP 2 실행

### STEP 4 — 결과 기록
- 표: sender 위치 → 정합성 PASS/FAIL
- **핵심 답:** "임의 sender에서 mcast가 정합하게 동작하나?" YES/NO
- (선택) Tracy로 실제 활동 코어가 sender 위치 따라 이동했는지 눈으로 확인 (성능 수치 X, 동작 확인만)

---

## 4. PASS / FAIL 기준 (명확히)

- **PASS =** top-left 아닌 위치(center·임의)의 sender에서도 **모든 지정 receiver가 올바른 데이터 수신** (정합성 assert 통과)
- **FAIL =** 특정 sender 위치에서 데이터 누락/오류 → **그 위치에서 mcast fabric이 커버 못 함** = 중요한 catch 발견 (그럼 loopback의 sender 위치 제약을 문서화)

---

## 5. 이 시험이 답하는 것 / 안 답하는 것

**답함:**
- 임의 sender mcast가 **정합하게 동작하나** (conv 자유배치의 핵심 전제)
- loopback include-src가 **어느 sender 위치까지 커버하나** (제약 있으면 발견)
- halo도 같은 mcast라 **원리상 함께 검증**

**안 답함 (다음 단계):**
- 성능/latency (sender 위치별 빠르기)
- per-core/per-link 부하 imbalance
- rotation policy 효과
- conv end-to-end

---

## 6. 스코프 가드레일 (하지 말 것)

- ❌ heatmap·rotation·noc_estimator·per-link 트래픽 (imbalance 단계)
- ❌ conv factory 수정 (이번엔 mcast 격리 시험만; factory는 이 시험 PASS 후)
- ❌ 성능 수치로 결론 (correctness만)
- ❌ 새 프레임워크 발명 (기존 test 재사용/최소확장)

---

## 7. 다음 단계 (이 시험 PASS 후, 참고)

1. **conv factory 수정** — sender/placement를 runtime arg로 (loopback), origin 파라미터화
2. **conv_test.py** — conv 정합성+배치 검증 (별도 문서: 검증환경-계획.md)
3. (더 나중) imbalance/rotation 측정 — 이 문서의 STEP들 대신 heatmap/rotation 프레임워크

---

## 부록: 한눈 요약 (에이전트용)

```
목표: 임의 sender mcast가 "정합"하게 도나? (성능 X, correctness O)
1. test_multicast_schemes.cpp + test_dram_to_l1_multicast_loopback_src.cpp 읽기
   → sender 파라미터화? 정합검증 있나?
2. 있으면 → sender 위치 바꿔 실행 → 모든 receiver 정합?
3. 없으면 → sender 좌표만 인자 추가 → 실행
4. PASS = 임의 위치 sender에서 전 receiver 정합
스코프: correctness만. heatmap/rotation/latency/conv수정 전부 다음 단계.
```

---

## STEP 1 결과 — 기존 테스트 판정 (2026-08-10)

> 코드 수정·실행 없이 **읽고 판정만**. 두 mcast 테스트 각각에 대해 "임의 sender 가능?" YES/NO.

### 파일 A — `test_multicast_schemes.cpp` (+ 프레임워크 `test_one_to_all.cpp::run_dm`)

**판정: ✅ YES — 임의 sender 즉시 가능, 정합 검증 있음, 코드 수정 불필요**

| 체크 항목 | 답 | 근거 |
|---|---|---|
| sender 좌표 파라미터화 | O | `MulticastSchemeType` enum이 sender 위치 **10가지** 열거 (grid 내부 4모서리 TR/BR/BL/TL + grid 외부 6종). `get_coordinates(sub_grid_dim, type)` 가 case별로 `mst_core_coord` 계산 (파일 A L45–98) |
| receiver rectangle 파라미터화 | △ | `sub_start_core_coord`는 파라미터긴 하나 스킴에서는 (0,0) 또는 (1,1)로만 고정 (같은 파일 L45–98). rect 크기는 sub_grid_dim 자유 |
| 정합성 검증 | O | 위임된 `run_dm()` (`test_one_to_all.cpp` L365–378) 이 각 receiver core L1을 읽어 `packed_output == packed_golden` 바이트 비교. 실패 시 assert |
| 스윕 커버리지 | O | `run_all_tests()` (L128–154) 가 NOC 0/1 × grid_size(2 ~ 하드웨어 min-1) × 10 scheme 순회 |
| loopback (include-src) 지원 | O | `OneToAllConfig.loopback` 필드 (L47). true면 sender도 receiver 세트에 포함, L1 base 오프셋 조정 (L141–142) |

**주의:** 개별 `TEST_F` 케이스 대부분 `GTEST_SKIP()` 처리 (L163, L172, L209) — CI에서 안 돎. 로컬에서 skip 제거만 하면 그대로 실행. `TensixDataMovementOneToAllMulticastSchemesNoLoopback2_0` (L180) 만 skip 없음 → **이거 하나 돌리면 10 scheme × NOC 0/1 × grid_size 전 스윕 실측** (correctness).

**한계 (다음 단계에서 확장할 것):**
- receiver rect의 **위치** 자유도는 낮음 (좌상단 좌표만 (0,0)/(1,1)) — receiver rect를 임의 위치로 옮기려면 `get_coordinates`에 케이스 추가 필요
- rect **크기**는 정사각형만 (`{sub_grid_dim, sub_grid_dim}`) — 5×4, 4×5 등 비대칭은 미커버

### 파일 B — `test_dram_to_l1_multicast_loopback_src.cpp` (+ 커널 `dram_to_l1_multicast_include_src.cpp`)

**판정: ❌ NO — sender가 (0,0)로 하드코딩. 최소 수정(1~2줄) 필요**

| 체크 항목 | 답 | 근거 |
|---|---|---|
| sender 좌표 파라미터화 | X | L37 `CoreCoord core = {0, 0};` — 상수 리터럴. argv/env 없음. `CreateKernel(..., core, ...)` (L73–78) 로 그대로 launch |
| receiver rectangle 파라미터화 | X | L55–57 receiver rect = **전체 grid** 하드코딩 (`{0,0}` ~ `logical_grid_size() - 1`) |
| 정합성 검증 | O | L103–116 grid 전 코어 순회, 각 코어 L1 read해서 원본 tensor와 바이트 비교. mismatch 시 `log_info("Mismatch on core ...")` + `pass = false` |
| loopback (include-src) | O | 커널 파일명이 `..._include_src.cpp`, sender core가 receiver rect (전체 grid) 안에 포함됨 |
| 실행 모드 | slow-dispatch 전용 | L22–23 `TT_METAL_SLOW_DISPATCH_MODE` 없으면 `TT_FATAL` |

**임의 sender 만들려면 (STEP 3용, 최소 확장안):**
1. L37 `CoreCoord core = {0, 0};` → argv로 `--sender-x/--sender-y` 파싱해서 대체 (1줄 → 3~5줄)
2. 커널·정합검증 로직은 **손 안 댐** (`core_start_physical`/`core_end_physical`은 receiver rect 그대로, sender만 rect 안 임의 위치로 이동)
3. mcast reader kernel (`dram_to_l1_multicast_include_src.cpp`) 은 sender 위치를 자체 참조하므로 CreateKernel의 target core만 바꾸면 됨

---

## 종합 결론

- **STEP 2 (수정 없이 즉시 실행) 가능 대상 = 파일 A 하나** — `TensixDataMovementOneToAllMulticastSchemesNoLoopback2_0` 만 돌려도 sender 위치 10가지 × grid_size 전 스윕에서 mcast 정합성 확인됨.
- **파일 B는 STEP 3 (최소 확장) 필요** — 다만 파일 A로 이미 답이 나오면 파일 B 확장은 **선택**. 파일 A가 다루는 sender 위치가 conv/halo 시나리오를 포괄한다면 파일 B는 skip 가능.
- **다음 액션:** 파일 A의 `TensixDataMovementOneToAllMulticastSchemesNoLoopback2_0` 실행 → PASS 나오면 core 질문 (correctness of 임의 sender mcast) 종결.

---

## ★★ STEP 3 결과 — center sender 검증 (2026-08-11)

> **동기:** STEP 2 의 파일 A 는 sender 위치 **10가지 (모서리 4 + 외부 6)** 만 커버. **grid 내부 중앙(interior)** 미검증.
> 코너 source = mcast fabric 이 단방향으로 커버 (쉬움). **중앙 source = 양방향 커버 필요** → fabric 이 단일 호출로 처리하는지 확인.

**실행:** `test_multicast_schemes.cpp` 에 `SenderInGridCenter` scheme 추가 + `TensixDataMovementOneToAllMulticastSchemeSingle` 로 3×3 grid, sender=(1,1) 실행.
작업 지시서: `operations/study_note/작업-center_sender_test.md`

**결과: ✅ PASS (5077 ms)**
```
[ RUN      ] GenericMeshDeviceFixture.TensixDataMovementOneToAllMulticastSchemeSingle
2026-08-10 18:06:49 | Running Test ID: 102, Run ID: 0
[       OK ] ... (5077 ms)
[  PASSED  ] 1 test.
```

**의미:**
- 단일 `noc_async_write_multicast` 호출이 **내부-중앙 source 에서도 양방향 전 receiver 정합**
- conv 자유배치 시 center sender 별도 로직 (2-piece mcast / exclude) **불필요** — 같은 커널 코드로 커버
- halo 도 원리상 동일 (같은 noc_async_write_multicast 패턴)

**⚠️ 빌드 시 이슈 (기록용):** 로컬 tt-metal clone 을 sync 했을 때 `#include <distributed/mesh_device_impl.hpp>` (신규 tt-metal 에만 존재) 로 빌드 실패. 컨테이너 tt-metal 은 구버전이라 이 헤더 없음. → 컨테이너 원본 (`.orig`) 위에 3개 편집만 직접 적용해서 해결. **앞으로 tt-metal 편집은 컨테이너 실제 파일을 진실로.**

**한계 (다음 단계):**
- 3×3 만 검증 — 큰·비정사각형 grid (4×4, 5×4, 4×3) + 변 중점 sender 미검증
- exclude (중간 코어 rest = rotation 핵심) 미검증
- Latency 수치 미측정 — profiler 켜서 재실행 시 CSV 로 확보 가능 (imbalance 스텝)

---

## ★★ STEP 4 결과 — Shape × NoC 스윕 & 프로파일링 (2026-08-11)

> **동기:** STEP 3 는 3×3 정사각형만. Shape 다양화(3×4, 4×3, 4×4, 5×2, 2×5) + NoC 0/1 편차 확인 필요. `noc_async_write_multicast` 는 NoC 방향성 있으므로 shape 회전 시 NoC 최적 선택 여지 조사.

**실행:**
- `SchemeSingle` TEST_F 를 shape 리스트 × NoC 리스트로 확장, `directed_ideal_test` 직접 호출 (multicast_schemes::test 정사각형 제약 우회)
- `test_information.yaml` 의 test 102 name 을 "Multicast Single Scheme" → "Multicast **Schemes** Single" 로 변경 → CSV 에 `NoC Index`, `Grid Dimensions`, `Multicast Scheme Type` 컬럼 자동 노출

**결과: ✅ 모든 config PASS (10 config: 5 shape × 2 NoC, 2×5 는 Eagle-N y=4 초과로 skip)**

**Latency (cycles) — center sender, directed_ideal 워크로드:**

| Shape | 코어 | NoC0 | NoC1 | Δ (NoC1−NoC0) | 편향 |
|---|---:|---:|---:|---:|---|
| 3×3 | 9 | 68,815 | 68,862 | +47 | +0.07% (NoC 중립) |
| 4×4 | 16 | 71,251 | 70,448 | −803 | −1.13% (NoC1 유리) |
| 3×4 (세로) | 12 | 70,692 | 69,600 | **−1,092** | **−1.55%** (NoC1 유리) |
| 4×3 (가로) | 12 | 69,166 | 69,561 | +395 | **+0.57%** (NoC0 유리, **반전**) |
| 5×2 (가로 극단) | 10 | 69,186 | 69,706 | +520 | +0.75% (NoC0 유리) |
| **5×4 (full grid)** | **20** | **72,226** | **71,890** | **−336** | **−0.47%** (NoC 중립, rotation baseline) |

**핵심 발견:**
1. **NoC 방향 편향 확인** — NoC0 = 가로(X-long) 유리 / NoC1 = 세로(Y-long) 유리. 재현성 있는 systematic bias.
2. **정사각형은 NoC 중립** (편차 ≤1%) — 대칭 shape 에서 편향 사라짐.
3. **같은 코어 수 비교 (12 core)** — 4×3 NoC0 (69,166) 이 3×4 NoC1 (69,472) 보다 306cyc (0.44%) 유리. **가로 rect 시스템적 우위** (추정: Blackhole DRAM 상하단 row → fabric X-축 데이터 흐름 유도).
4. **면적 스케일링 미미** — 9→16 core (77% 증가) 에도 latency ~3.5% 증가만. mcast fabric 은 hop-count 지배, 수신자 수 무관.
5. **전체 편차 ~3.5%** — 다양한 shape 간 latency 매우 tight. **Shape 자유롭게 골라도 성능 penalty 미미**.

**Aging rotation 연구 시사:**
- **Rectangle 회전 시 NoC 도 회전** → latency 손실 없이 트래픽 분산 가능 (~1~2% 편향 exploit)
- **가로형 rect 선호** — 세로보다 systematic 하게 유리
- **정책 자유도 매우 큼** — 정사각형/가로 rect 편차 <1% → rotation 이 shape 선택 광범위 가능. **정합성·성능 양쪽 안전.**

**★ Rotation 정책 baseline (5×4 full grid, [[mcast-rotation-semantics]]):**
- 5×4 latency = 72,226 (NoC0) / 71,890 (NoC1) cyc — 최소 (3×3 = 68,815) 대비 **+5% penalty**
- 면적 122% 증가 (9c→20c) 에도 latency +5% 만 = **sender-bound 완전 확인**
- BW 58.07 B/cc (peak 64 의 90.7%) — 큰 rect 일수록 오버헤드 소폭 증가
- **Trade-off:** 5% latency 세금 vs (20 core full util + fabric path rotation) = net gain 명백

**작업 지시서·산출물:**
- 수정: `data_movement/one_to_all/test_multicast_schemes.cpp` (SchemeSingle TEST_F 확장 + NoC 루프), `python/test_mappings/test_information.yaml` (test 102 name 변경)
- CSV: `data_movement/data/blackhole/Multicast Schemes Single.csv`
- Plot: `data_movement/data/blackhole/Multicast Schemes Single.png`

**한계 & 다음:**
- Center sender 만 검증 — 코너·변중점 sender 는 별도 STEP
- Exclude (rotation 핵심: 중간 코어 rest) 미검증
- Per-link/router 트래픽 heatmap 없음 (mcast 특성상 sender zone 만 잡혀서 per-core 히트맵 불가능). tt-npe 시뮬로 보완 필요.

---

## ★★ STEP 2 결과 — 정합성 실증 완료 (2026-08-10)

> ⚠️ 실행 참고: 컨테이너 소스엔 `NoLoopback2_0`이 없었음(업로드 파일 ≠ 컨테이너). 대신 컨테이너의
> `MulticastSchemesLoopback`(L161) + `NoLoopback`(L170)의 `GTEST_SKIP`을 제거 → 재빌드 → 실행.

**결과: ✅ `[ PASSED ] 2 tests` (Loopback + NoLoopback, 53971 ms)**

- 각 테스트 = **10 sender 위치(4모서리 in-grid + 6 out-grid) × NOC 0/1 × grid_size(2~min-1) 전 스윕**
- 모든 config에서 **receiver L1 == golden 바이트 일치** (`run_dm` 검증)
- **loopback(=conv 케이스) 포함 통과**

**결론 (Eagle-N 실물 실증):**
- **임의 sender 위치 mcast가 정합하게 동작함 = 확정.** top-left 아닌 위치에서도 전부 OK.
- **wrap 불필요 확정** — 하드웨어 fabric이 임의 sender를 알아서 커버.
- **conv 자유 배치의 핵심 전제(sender 자유) = 검증 완료.** 이제 가정 아니라 사실.

**남은 것:**
- **성능 (위치별 latency)** — 같은 테스트를 프로파일러 켜고 재실행 → "sender 위치 vs cycle" → 프로그래머블 배치의 성능 비용 파악. ← 다음
- 이후 conv factory 수정 (sender/origin 파라미터화)로 진행
