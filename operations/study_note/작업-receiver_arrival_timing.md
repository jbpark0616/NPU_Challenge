# 작업 지시 — Per-receiver arrival timing heatmap (Claude Code 실행용)

> **목표:** mcast 시 각 receiver 코어에 데이터가 언제 도착하는지 개별 stamp → grid 상 per-core arrival heatmap.
> **동기:** 지금은 sender-side latency 만 측정. Per-receiver 편차·공간 분포 = aging rotation 연구의 hero figure.
> 배경: `실험설계-mcast위치벤치.md` STEP 4 결과 (Shape/NoC 스윕 완료, sender-side 만).

---

## 0. 왜 세마포어 인가 (짧게)

- **NoC 하드웨어 ACK 는 sender 측 aggregated** (`noc_async_write_barrier` 가 그걸 씀) — receiver 별 stamp 못 함.
- **Per-receiver arrival timing 은 receiver-side 관측 훅 필요.**
- 세마포어 방식:
  - Sender: 데이터 mcast → `sem_addr[local]=1` → sem mcast
  - Receiver: `DeviceZoneScoped { noc_semaphore_wait(sem, 1); }` → wait 종료 = arrival stamp
- 세마포어 mcast 는 같은 NoC 라우팅, 데이터 직후 (~수 cycle) 도착 → arrival proxy 정확도 충분.
- **오버헤드 미미** (기존 인프라 재사용).

---

## 1. 실행 규칙

1. **컨테이너 안 + 보드**. `docker exec -it bos_host bash`
2. **공유 컨테이너.** 소스 수정 전 `.orig` 백업 필수. `git add` 금지. 남의 폴더·컨테이너 건드리지 말 것.
3. **`run_dm` 자체는 안 건드림** — 기존 sweep 테스트 영향 X. 새 TEST_F 에서 인라인으로 sender+receiver 페어 orchestrate.
4. **로컬 tt-metal clone 은 지도용** ([[fork-container-sync]]) — 편집·명령은 컨테이너 파일이 진실.

---

## 2. 재사용할 참고 코드

| 코드 | 용도 |
|---|---|
| `data_movement/one_to_all/kernels/sender_multicast.cpp` (컨테이너) | 데이터 mcast 로직 — 여기에 sem mcast 3~5줄 추가 |
| `data_movement/one_to_all/kernels/receiver.cpp` (컨테이너) | **그대로 재사용** — 이미 `DeviceZoneScopedN("RISCV1")` 로 sem wait 감쌈 |
| `test_kernels/misc/noc_debugging/async_mcast_semaphore.cpp` (로컬 tt-metal) | 최소 sender sem 패턴 참고 (data mcast + local sem set + sem mcast) |
| `data_movement/one_to_all/test_multicast_schemes.cpp` (컨테이너) | 새 TEST_F 넣을 자리 |

---

## 3. 편집 단계

### Step 0 — 백업 (컨테이너)
```bash
cd /tt-metal/tests/tt_metal/tt_metal/data_movement/one_to_all
cp kernels/sender_multicast.cpp{,.orig}
# test_multicast_schemes.cpp 는 이미 .orig 있음
```

### Step 1 — `sender_multicast.cpp` 확장 (sem mcast 추가)

기존 코드 끝부분 (`DeviceZoneScopedN("RISCV0")` 블록 안, `noc_async_write_barrier()` 다음) 에 추가:

```cpp
// === 추가 시작 ===
// Semaphore mcast: 데이터 다 도착 후 receiver 들에 arrival 신호
constexpr uint32_t arrival_sem_id = 0;  // 세마포어 id (test 에서 CreateSemaphore 로 할당)
uint32_t arrival_sem_addr = get_semaphore(arrival_sem_id);
volatile tt_l1_ptr uint32_t* arrival_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_sem_addr);

uint64_t sem_multicast_addr = noc_index == 0
    ? get_noc_multicast_addr(start_x, start_y, end_x, end_y, arrival_sem_addr)
    : get_noc_multicast_addr(end_x, end_y, start_x, start_y, arrival_sem_addr);

*arrival_sem_ptr = 1;  // sender 자기 자신 sem set (loopback 대비)
noc_semaphore_set_multicast(arrival_sem_addr, sem_multicast_addr, num_subordinates, false);
// === 추가 끝 ===
```

**주의:**
- `DeviceZoneScopedN("RISCV0")` 블록 **안**에 넣어야 sender 프로파일에 세마포어 시간까지 포함됨. 밖에 넣어도 동작은 함.
- `arrival_sem_id = 0` 는 test 에서 첫 번째로 CreateSemaphore 할 때 자동 할당되는 id. 다른 semaphore 도 만들면 순서 주의.

### Step 2 — 새 TEST_F 추가 (`test_multicast_schemes.cpp` 맨 밑)

기존 `TensixDataMovementOneToAllMulticastSchemeSingle` **아래**에 새로 추가:

```cpp
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMulticastArrivalHeatmap) {
    uint32_t test_case_id = 103;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);
    auto grid = device->compute_with_storage_grid_size();

    bool loopback = false;
    bool is_linked = true;

    // 스윕할 (shape, sender_pos, NoC) 조합
    struct Config { uint32_t gx, gy, sx, sy; NOC noc; const char* label; };
    std::vector<Config> configs = {
        // ─── 5×4 full-grid rotation cycle (aging 논문 hero) ───
        // Rotation 정책: 항상 20-core mcast, sender 만 회전
        {5, 4, 0, 0, NOC::NOC_0, "5x4 sender(0,0) TL"},
        {5, 4, 4, 0, NOC::NOC_0, "5x4 sender(4,0) TR"},
        {5, 4, 0, 3, NOC::NOC_0, "5x4 sender(0,3) BL"},
        {5, 4, 4, 3, NOC::NOC_0, "5x4 sender(4,3) BR"},
        {5, 4, 2, 2, NOC::NOC_0, "5x4 sender(2,2) center"},
        // NoC 대칭 확인 (5x4 center 만)
        {5, 4, 2, 2, NOC::NOC_1, "5x4 sender(2,2) center NoC1"},
        // ─── 작은 shape 참조 (baseline 비교) ───
        {3, 3, 1, 1, NOC::NOC_0, "3x3 center"},
        {3, 3, 0, 0, NOC::NOC_0, "3x3 BL corner"},
        {3, 3, 2, 2, NOC::NOC_0, "3x3 TR corner"},
        {4, 4, 2, 2, NOC::NOC_0, "4x4 center"},
        {4, 4, 0, 0, NOC::NOC_0, "4x4 BL corner"},
    };

    for (const auto& cfg : configs) {
        if (cfg.gx > grid.x || cfg.gy > grid.y) {
            log_info(tt::LogTest, "SKIP {}: exceeds device grid {}x{}", cfg.label, grid.x, grid.y);
            continue;
        }

        log_info(tt::LogTest, "ARRIVAL: {} grid={}x{} sender=({},{}) NoC{}",
                 cfg.label, cfg.gx, cfg.gy, cfg.sx, cfg.sy, static_cast<int>(cfg.noc));

        // === Program setup ===
        Program program = CreateProgram();

        CoreCoord sender_logical = {cfg.sx, cfg.sy};
        CoreRange receiver_range({0, 0}, {cfg.gx - 1, cfg.gy - 1});
        CoreRangeSet receiver_set({receiver_range});
        // Loopback=false 인 경우 sender 제외
        if (!loopback) {
            CoreRangeSet sender_only({CoreRange(sender_logical)});
            receiver_set = receiver_set.subtract(sender_only);
        }

        uint32_t num_receivers = receiver_set.num_cores();

        // Physical coord range for mcast rect
        CoreCoord start_phys = device->worker_core_from_logical_core({0, 0});
        CoreCoord end_phys = device->worker_core_from_logical_core({cfg.gx - 1, cfg.gy - 1});

        // === Semaphore (receiver 들이 wait 할 것) ===
        // sender + receiver 모두 같은 id 로 접근하도록 all_cores 에 만듦
        CoreRangeSet all_cores(std::vector<CoreRange>{
            CoreRange(sender_logical),
            CoreRange({0, 0}, {cfg.gx - 1, cfg.gy - 1})});
        uint32_t arrival_sem_id = CreateSemaphore(program, all_cores, 0);

        // === Buffer 세팅 (sender L1) ===
        // 기존 dm_common 헬퍼 재사용 시도 — 없으면 하드코딩
        uint32_t bytes_per_page = 64;
        uint32_t pages_per_transaction = 256;
        uint32_t num_of_transactions = 256;
        uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;
        uint32_t total_bytes = bytes_per_transaction * num_of_transactions;

        uint32_t l1_base = 200 * 1024;  // 안전한 오프셋
        uint32_t sub_l1_base = l1_base;  // 같은 주소로 mcast

        // === Sender kernel ===
        std::vector<uint32_t> sender_cta = {
            l1_base,                    // mst_base_addr
            sub_l1_base,                // sub_base_addr
            num_of_transactions,        // num_of_transactions
            pages_per_transaction,      // pages_per_transaction
            bytes_per_page,             // bytes_per_page
            test_case_id,               // test_id
            num_receivers,              // num_subordinates
            (uint32_t)is_linked,        // is_linked
            (uint32_t)loopback,         // loopback
            (uint32_t)start_phys.x,     // start_x
            (uint32_t)start_phys.y,     // start_y
            (uint32_t)end_phys.x,       // end_x
            (uint32_t)end_phys.y,       // end_y
            0,                          // multicast_scheme_type (unused, 0=skip stamp)
            cfg.gx,                     // sub_grid_size_x
            cfg.gy,                     // sub_grid_size_y
        };

        auto sender_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/sender_multicast.cpp",
            sender_logical,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = cfg.noc,
                .compile_args = sender_cta});

        // === Receiver kernel ===
        std::vector<uint32_t> receiver_cta = {
            l1_base,                    // src_addr
            sub_l1_base,                // dst_addr
            num_of_transactions,
            pages_per_transaction,
            bytes_per_page,
            test_case_id,
        };

        auto receiver_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/receiver.cpp",
            receiver_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = cfg.noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0,  // 반대 NoC 로 wait
                .compile_args = receiver_cta});

        SetRuntimeArgs(program, receiver_kernel, receiver_set, {arrival_sem_id});

        // === Runtime ID + launch ===
        program.set_runtime_id(unit_tests::dm::runtime_host_id++);

        distributed::MeshWorkload workload;
        auto zero = distributed::MeshCoordinate(0, 0);
        auto range = distributed::MeshCoordinateRange(zero, zero);
        workload.add_program(range, std::move(program));

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, true);
    }
}
```

**포인트:**
- 각 config = 새 `Program`. `run_dm` 은 안 건드림.
- Semaphore 는 sender + receiver 모두 접근하도록 `all_cores` 에 create.
- Receiver 는 반대 NoC 로 wait (sender 와 NoC 충돌 방지 관례).
- `runtime_host_id` 는 기존과 통합 (프로파일 CSV 에서 다른 test 랑 구분).

### Step 3 — 빌드 + 프로파일 실행 (컨테이너)

```bash
cd /tt-metal

# 편집한 두 파일 sync (호스트에서)
# docker cp .../sender_multicast.cpp bos_host:/tt-metal/.../sender_multicast.cpp
# docker cp .../test_multicast_schemes.cpp bos_host:/tt-metal/.../test_multicast_schemes.cpp

cmake --build build --target unit_tests_data_movement

# Correctness 만 (프로파일 없이) 먼저 sanity 체크
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*ArrivalHeatmap*"

# 통과하면 프로파일 + 파싱
pytest tests/tt_metal/tt_metal/data_movement/python/test_data_movement.py \
  --gtest-filter='ArrivalHeatmap' \
  --plot --report --verbose-log
```

### Step 4 — `test_information.yaml` 에 test 103 등록 (CSV 이쁘게 나오게)

```yaml
  103:
    name: "Multicast Schemes Arrival Heatmap"
```
("Multicast Schemes" 포함해야 stats_reporter 가 attribute 컬럼 자동 붙임)

---

## 4. 예상 CSV 결과

파일: `data_movement/data/blackhole/Multicast Schemes Arrival Heatmap.csv`

- **Kernel: Sender** row: 8 configs × 1 = 8 rows (기존과 동일 sender-side)
- **Kernel: Receiver** row: 각 config 의 receiver 코어 수만큼 (예: 3×3 config = 8 receiver row, 5×4 = 19)
- 총 ~100 rows 예상

**핵심 컬럼 (receiver row):**
- `Latency (cycles)` = `noc_semaphore_wait` duration ≈ **arrival wait time**
- `Run Host ID` = config 구분
- `Grid Dimensions` = shape 태그

## 5. Heatmap 시각화 (post-processing)

기존 `heatmap.py` 는 sender BW 용이라 그대로 못 씀. 새 스크립트 하나 짜야 함 (100줄 이내):

```python
# arrival_heatmap.py (핵심 로직)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Multicast Schemes Arrival Heatmap.csv")
receiver_df = df[df["Kernel"] == "Receiver"]

# per-config heatmap
for run_id, group in receiver_df.groupby("Run Host ID"):
    grid = np.zeros((grid_y, grid_x))
    for _, row in group.iterrows():
        # row 에 core 좌표 있어야 함 — CSV 에 코어 좌표 컬럼 필요
        grid[core_y, core_x] = row["Latency (cycles)"]
    plt.imshow(grid, cmap="hot")
    plt.title(f"Arrival heatmap: {group.iloc[0]['Grid Dimensions']} sender=...")
    plt.savefig(f"heatmap_{run_id}.png")
```

**⚠️ 주의:** CSV 에 **코어 좌표 컬럼 없음** (`stats_reporter.py` 기본 export 는 코어 별 좌표 안 넣음). 두 옵션:
- **(A)** `stats_reporter.py` 수정해서 core coord 추가 export
- **(B)** raw tracy CSV (`profile_log_device.csv`) 를 직접 파싱 (거기엔 core 좌표 있음)

Post-processing 은 별도 스텝. 실측 데이터 나온 다음 결정.

---

## 6. PASS / FAIL 기준

- **Sanity PASS** = `[ OK ] ...ArrivalHeatmap` (모든 config 정합 통과)
- **Data PASS** = CSV 에 `Kernel: Receiver` row 가 예상 개수만큼 나오고, `Latency (cycles)` 값이 receiver 별로 **다르게** 찍힘 (같으면 stamp 실패 or fabric 이 완벽 sync)
- **FAIL 케이스:**
  - Sender-only row 만 있음 → receiver kernel launch 실패 (kernel 경로 확인)
  - Receiver latency 값 다 0 or 동일 → 세마포어 mcast 도달 실패 (sender kernel sem 추가 로직 확인)

---

## 7. 하지 말 것

- ❌ `run_dm` 수정 (다른 test 영향)
- ❌ `receiver.cpp` 수정 (그대로 재사용)
- ❌ 로컬 tt-metal clone 편집 (컨테이너 파일이 진실)
- ❌ `git add` / 커밋
- ❌ Semaphore id 를 0 이외로 (첫 번째 CreateSemaphore = id 0)

---

## 8. 이후 (성공 후)

1. **Post-processing 스크립트** (`arrival_heatmap.py`) 로 config 별 히트맵 PNG 생성
2. **5×4 rotation cycle 5개 히트맵 diff** = aging 논문 hero figure (특정 코어 부하 감소량 정량화)
3. **Exclude/skip 메커니즘 도입** (다음 스텝, 아래 참고)
4. **tt-npe per-link 시뮬** 병행 → 실측 (per-core arrival) + 시뮬 (per-link traffic) 이중 evidence

---

## 부록: Multicast Rest/Skip 메커니즘 조사 (2026-08-11)

**목표:** rotation 정책 = "항상 5×4 mcast + 특정 코어 rest". 실현 가능한 옵션들.

### Native exclude API 는 kernel 레벨 미노출

| 레벨 | 지원 |
|---|---|
| Blackhole HW firmware (`noc.h::noc_multicast_copy_exclude`) | ✅ |
| Kernel dataflow_api.h | ❌ wrapper 미제공 |

### 실현 옵션 (재정리)

**★ 채택: (A) Full-rect mcast + kernel 미배치 (Height sharding 패턴)**
- mcast rect = 항상 5×4 bounding rect
- Compute kernel 은 active core (예: 14개) 에만 deploy
- Rest core (예: 6개) 는 데이터 L1 에 도착하지만 아무 kernel 이 안 씀 → 자연스레 drop
- **Height sharding 이 이미 이 방식으로 프로덕션 검증됨**
- **장점:** 코드 단순, 프로덕션 realistic, aging 연구 결과 실제 워크로드와 정합
- **회전이 미치는 범위:**
  - ✅ Sender-outgoing NoC 채널 (sender 위치별 회전)
  - ✅ Fabric routing path (XY routing, source 위치별 변화)
  - ❌ Receiver-inbound 마지막 hop 은 매번 동일 (모든 receiver 매번 도달)
- **평가:** production ML 워크로드와 동일 특성. **realistic aging 연구엔 오히려 맞음.**

**보조: (B) 사각형 분할 mcast**
- Rest core 있는 위치 따라 rect 를 2개 이상으로 분해 (mcast rect 진짜 축소)
- 마지막 hop 까지 완전 회전 원할 때만
- **오버헤드** (mcast 호출 여러 번) + 코드 복잡
- 특정 실험 (완전 회전 vs 부분 회전 비교) 에만 유용

**최후: (C) firmware-level exclude 직접 사용**
- `noc.c::noc_multicast_copy_exclude` 를 kernel 에서 inline invoke
- 비표준, 유지보수 부담, 논문 재현성 낮음
- 실용성 없음

### 아키텍처 관점 인사이트

Full-rect + skip 방식이 부분 회전인 이유 = **Blackhole NoC 라우팅 = XY routing** (source → column 이동 → row 이동). Sender 위치가 변하면 fabric 내부 경로 (라우터 crossbar) 는 회전. **하지만** 마지막 hop (라우터 → receiver L1) 링크는 receiver 마다 고정 → 매번 도달하니 회전 X.

이건 **버그가 아니라 mcast 특성**. Rotation 이 "완전" 하려면 receiver rect 자체도 회전해야 함 (=height shard 회전).

### 결정: (A) 로 감

- 지금 arrival timing 실측 → 5×4 full mcast + sender 5 위치 회전 (`configs` 리스트 위쪽)
- Rest core 시뮬 필요 시 → 그때 (A) 방식으로 kernel deploy 만 조작 (mcast 로직 변경 없음)
- (B) 는 "완전 회전 vs 부분 회전" 대조군 실험 시에만 사용

---

## 부록: 편집 총량 예상

| 파일 | 라인 수 | 위험도 |
|---|---|---|
| `sender_multicast.cpp` | +10 | 낮음 (기존 로직 후단 append) |
| `test_multicast_schemes.cpp` | +80 (새 TEST_F) | 낮음 (독립 TEST_F) |
| `test_information.yaml` | +2 (test 103 등록) | 없음 |
| **합계** | **~90 라인** | 대부분 boilerplate |

Post-processing 스크립트 (`arrival_heatmap.py`) 는 별도, 실측 데이터 나온 후.
