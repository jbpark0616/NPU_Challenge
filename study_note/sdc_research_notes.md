# SDC Detection on Eagle-N — 연구 노트

> 작성일: 2026-07-18
> 관련: `safety_island_study.md` §7, `automotive_specs.md`, 메모리 [[project-detection-only-scope]], [[project-eagle-n-safety-context]]
> 진행 상태: 방향 확정, 실험 설계 초안

---

## TL;DR

- Eagle-N NPU 는 **연산 결과의 정확성 (SDC — Silent Data Corruption)** 을 감지하는 하드웨어가 없다. Safety Island 는 liveness (hang/crash) 만 감시하고, HW ECC 는 NoC 헤더와 일부 내부 메모리에만 걸린다.
- 오늘 grep 결과 확인된 **보호 근거를 못 찾은 SDC 표면 3곳** (⚠ negative evidence — tt-metal 오픈 소스에서 SW 인터페이스 부재이며 실측·벤더 문서 확인 전엔 "구현 안 됨" 을 단정하지 않음):
  1. **Compute-time SDC** — FPU MAC / SFPU vector unit 결과 (ECC 관련 config/counter/interrupt 3세트 모두 없음)
  2. **Storage-time SDC** — 코어 L1 SRAM (1.5MB scratchpad) — ECC/parity 관련 매크로 없음
  3. **Transport-time SDC** — NoC 페이로드 데이터 — 헤더는 SEC-DED 확인, payload 는 보호 근거 없음
- ISO 26262 ASIL-B "진단 커버리지 90%+" 요구 관점에서 이 gap 이 연구의 정당성 근거
- 연구 방향: **SDC detection primitive 특성화 + 새 primitive 제안** (컴아키 톤)
- 실험 도구: SubDevice + Global Semaphore/CB + 커널 수준 fault injection

---

## 1. 문제 정의

### 1.1 Eagle-N 안전 계층의 세 축

| 축 | 질문 | Eagle-N 현황 | 담당 |
|---|---|---|---|
| Liveness FT | NPU 가 살아있나? Hang/crash 없나? | ✅ 있음 | Safety Island |
| Storage/Transport FT | 메모리·NoC 헤더에서 bit-flip 감지되나? | 부분 있음 | HW ECC (헤더·NIU/Router 내부 메모리) |
| **Correctness FT (SDC)** | **연산 결과가 조용히 틀리지 않나?** | **❌ 없음** | **없음** |

Safety Island 는 세 번째를 커버하지 않는다 — 자기 시야에는 "NPU 가 결과 냈다" 만 보이고 그 값의 정확성은 알 수 없다.

### 1.2 자동차 컨텍스트 (ASIL-B 진단 커버리지)

- ISO 26262 ASIL-B: **진단 커버리지 90%+ 요구** (`safety_island_study.md` §4)
- Dense compute (FPU, SFPU, L1) 이 blind spot 이면 이 계산에 큰 구멍
- Safety Island 는 "NPU 감시자" 자격이지 "NPU 내부 검증자" 는 아님

### 1.3 왜 아무도 안 했나

- SASSIFI, NVBitFI (GPU FI 프레임워크), PyTorchFI, TensorFI (ML level FI) — **모두 GPU 또는 SW 수준**
- Spatial dataflow NPU (Tenstorrent, Groq, SambaNova 등) 위 SDC 연구는 **거의 없음** (문헌 리서치 미완, 확인 필요)
- Tenstorrent 스택 (tt-metal) 자체에 SDC 관련 인프라 부재 (`ABFT / checksum / redundancy / SDC` grep = 0)

---

## 2. Fact Check — Eagle-N 이 실제로 무엇을 보호하나

2026-07-18 tt-metal 소스 grep 근거.

### 2.1 보호되는 계층

**NoC 라우터 헤더** (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h` line 152~157):
```
ROUTER_CFG_0:
  bit 16: ECC_ROUTER_MEM_PARITY_EN  — 라우터 내부 메모리 파리티
  bit 17: ECC_HEADER_CHKBITS_EN     — 패킷 헤더 체크 비트
  bit 18: ECC_HEADER_SECDED_EN      — 패킷 헤더 SEC-DED (1비트 정정 + 2비트 감지)
```

**NIU (NoC Interface Unit) 내부 메모리** (`noc_parameters.h` line 139~148):
```
NIU_CFG_0:
  bit 8:  ECC_NIU_MEM_PARITY_EN      — NIU 내부 메모리 파리티
  bit 9:  ECC_MEM_PARITY_INT_EN      — 파리티 에러 인터럽트
  bit 10: ECC_HEADER_1B_INT_EN       — 헤더 1비트 에러 인터럽트
  bit 11: ECC_HEADER_2B_INT_EN       — 헤더 2비트 에러 인터럽트
```

**감지 카운터** (`noc_parameters.h` line 50~53):
```
NUM_MEM_PARITY_ERR    (NIU 내부 메모리 파리티 카운트)
NUM_HEADER_1B_ERR     (헤더 1비트 감지·정정 카운트)
NUM_HEADER_2B_ERR     (헤더 2비트 감지·비정정 카운트)
ECC_CTRL              (bit 3/4/5: MEM_PARITY/HEADER_1B/HEADER_2B FORCE 비트 → fault injection 도구)
```

**RISC-V 코어 내부 SRAM** (`tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix.h`):
- `RISCV_DEBUG_REG_ECC` 관련 매크로 — baby RISC-V (5개 per Tensix) 자체의 내부 SRAM (I-cache 등) ECC. FPU/SFPU 결과와 무관.

**Ethernet 링크 (칩간)**:
- CRC 존재 (`tt_metal/hw/inc/internal/tt-1xx/wormhole/eth_l1_address_map.h` — 유일한 CRC hit)
- 즉 **CRC 는 칩간 통신 전용**, 칩 내부 NoC 는 CRC 안 씀

### 2.2 보호 근거를 찾지 못한 계층 (SDC 표면 후보)

> **⚠ 인식적 주의**: 아래는 "구현이 없다" 의 확정이 아니라 **"tt-metal 오픈 소스와 공개 문서에서 SW 인터페이스가 발견되지 않음"** 이라는 negative evidence. 
> 
> 강한 근거로 보는 이유: HW ECC 가 있을 때 관례적으로 함께 나오는 3 세트의 SW 인터페이스 — (1) enable 비트, (2) 감지 카운터, (3) interrupt/status — 가 **세 계층 모두 하나도 발견되지 않음**. 벤더가 통계·디버깅 목적으로 최소 카운터라도 노출하는 게 업계 관례라서 이 부재는 유의미한 신호.
>
> 약한 지점: (a) 순수 HW 자동 정정으로 SW 완전 투명 가능성, (b) tt-metal 밖 내부 벤더 인터페이스 존재 가능성, (c) BOS 가 Eagle-N 실리콘에 얹은 자동차용 추가 보호가 아직 문서화 안 됐을 가능성.
>
> **확정하려면**: BOS Eagle-N 백서·Tenstorrent Blackhole architecture spec 확인 + Phase 1 실측 (bit-flip 주입 후 어떤 카운터도 안 오르는지).

**NoC 페이로드 데이터** (`noc_parameters.h` line 286~288):
```
NOC_PAYLOAD_WIDTH = 512  bit  (순수 데이터)
NOC_DATA_WIDTH    = 512+3 bit (3 bit 오버헤드는 flit type/routing 용, SEC-DED 하려면 512에 최소 10비트 필요)
```
- Payload ECC / Data ECC 관련 매크로 **grep 결과 없음**
- 페이로드 데이터의 bit-flip 감지 인터페이스 근거 없음

**코어 L1 SRAM (1.5 MB per core, scratchpad)**:
- `tensix.h` 에서 L1 관련 매크로는 다 있지만 (`PERF_CNT_L1_*`, `DBG_L1_MEM_*`, `INSTRN_SEL_L1`), **ECC/parity 관련은 grep 결과 없음**
- Tech reports (`allocator.md` 포함) 에서도 L1 ECC/parity 언급 없음
- `L1.*ECC` / `sram.*ecc` grep 이 tt-metal 실제 코드 파일에는 0 hit

**FPU / SFPU 연산 결과**:
- `FPU/SFPU/MAC/Math + ECC/parity` 조합 grep = 0 hit (2026-07-18)
- Dense compute 결과에 대한 감지 지원 근거 없음

### 2.3 종합 표

**Legend**:
- ✅ 있음 — SW 인터페이스 (enable/counter/interrupt) grep 결과 명확히 확인됨
- ⚠ 근거 없음 — tt-metal 오픈 소스 grep 결과 SW 인터페이스 발견되지 않음. 실측·벤더 문서 확인 전엔 "구현 없음" 확정 X

| 계층 | 보호 | 정정 | 감지 | 근거 강도 | SDC 표면? |
|-----|-----|-----|-----|---------|--------|
| NoC 라우터 헤더 | SEC-DED | O (1비트) | O (2비트) | ✅ | ✗ |
| NoC 라우터 내부 메모리 | Parity | X | O | ✅ | ✗ |
| NIU 내부 메모리 | Parity | X | O | ✅ | ✗ |
| Ethernet 링크 (칩간) | CRC | (link retry) | O | ✅ | ✗ |
| RISC-V 코어 내부 SRAM (I-cache) | ECC (RISCV_DEBUG_REG_ECC) | O | O | ✅ | ✗ |
| **NoC 페이로드 데이터** | **근거 없음** | ? | ? | ⚠ | **SDC 후보** |
| **L1 SRAM (코어당 1.5MB scratchpad)** | **근거 없음** | ? | ? | ⚠ | **SDC 후보** |
| **FPU / SFPU 연산 결과** | **근거 없음** | ? | ? | ⚠ | **SDC 후보** |
| DRAM (LPDDR5, 32GB) | (tt-metal 밖 — 별도 확인) | ? | ? | 미조사 | 별도 확인 필요 |

---

## 3. Three SDC Surfaces

우리 논문의 fault model 을 세 종류로 나눈다:

### 3.1 Compute-time SDC
- **원인**: FPU MAC array, SFPU vector unit 의 연산 오류
- **injection 흉내**: 커널 코드에서 조건부로 결과 원소 flip (`tile[i] ^= 0x01`)
- **감지 후보**: DMR (redundant compute), ABFT (Freivalds' algorithm for matmul), checksum
- **논문 각도**: BFP shared exponent 특성상 mantissa 1비트 flip 의 영향이 IEEE 754 와 다름

### 3.2 Storage-time SDC
- **원인**: L1 SRAM 저장 중 bit-flip
- **injection 흉내**: 호스트 또는 다른 코어에서 L1 특정 주소에 XOR write
- **감지 후보**: Checksum (tile sum 슬롯), 주기적 scrub, DMR
- **논문 각도**: L1 이 캐시가 아니라 scratchpad 이므로 flip 이 자동으로 fresh copy 로 복원되지 않음 → 전파 특성 다름

### 3.3 Transport-time SDC
- **원인**: NoC 페이로드 전송 중 bit-flip
- **injection 흉내**: 다른 코어에서 NoC 로 write 하면서 payload 조작 (SW 수준)
- **감지 후보**: End-to-end checksum (producer 가 생성, consumer 가 검증), 페이로드 재전송 후 비교
- **논문 각도**: Header 는 SEC-DED, payload 는 무보호 — **비대칭 보호 설계** 의 결과 정량화

### 3.4 세 축 조합의 novelty
GPU/CPU SDC 문헌은 대부분 compute-time (SASSIFI 등) 이나 memory (DRAM ECC) 에 집중. **Spatial NPU 에서 세 축을 동시에 특성화** 하는 건 아직 안 된 것으로 보임 (문헌 확인 필요).

---

## 4. 연구 질문

### 4.1 Primary
> **Eagle-N (spatial NPU) 상에서 세 종류의 SDC (compute/storage/transport) 를 SW 수준 감지 메커니즘으로 얼마나 잡을 수 있는가? 각 감지 메커니즘의 latency·coverage·overhead 는 어떻게 되는가?**

### 4.2 Secondary
- (Q2) FT 코어 (verifier) 의 물리적 위치가 감지 성능에 영향을 주는가? (조교님 sweep)
- (Q3) BFP dataformat 이 IEEE 754 와 SDC 전파 특성이 어떻게 다른가?
- (Q4) 현재 없는 어떤 HW primitive 가 있으면 이 감지가 획기적으로 개선되는가? (Quasar NOC ATT 처럼 다음 세대 요구사항 제안)

### 4.3 스코프 밖 (명시)
- Recovery / hot-swap — 교수님 피드백 "detection 만 해도 유의미" 반영
- SoC 수준 FT — Safety Island 책임
- DRAM ECC — LPDDR5 외부 문제

---

## 5. 실험 로드맵

### Phase 1 — Minimum viable SDC 감지 (첫 스텝)

**목표**: 가장 간단한 조합으로 "감지된다" 를 확인

- SDC 종류: 커널 논리 오류 (Compute-time SDC 흉내)
- 감지: Checksum (tile sum)
- 주입 위치·시점: 고정 (1 core, 첫 tile 만)
- 워크로드: dummy tile (hardcoded 값)

**커널 구조**:
```
Producer (SubDevice 0, Core A):
  1. tile 생성 or 계산
  2. checksum = sum(tile)
  3. IF fault_enabled: tile[0] ^= 0x01  ← SDC 주입
  4. Write tile + checksum to L1
  5. noc_semaphore_inc → Verifier

Verifier (SubDevice 1, Core B):
  1. noc_semaphore_wait
  2. Read tile + checksum from Core A
  3. Recompute sum
  4. IF sum != checksum → SDC detected, write flag
  5. noc_semaphore_inc → Host
```

**측정**: flag 세워짐 여부, latency

### Phase 2 — Sweep (조교님 방법론)
- 주입 위치 20 코어 sweep
- Verifier 위치 3~5 후보 (인접/중간/원거리/코너)
- 조합별 detection latency 3-D 지도

### Phase 3 — 세 SDC 종류 비교
- Compute / Storage / Transport 각 종류마다 fault injection 방식 확정
- 각 감지 메커니즘 (checksum, DMR, ABFT) 이 각 종류를 잡는가

### Phase 4 — 워크로드 상향
- Dummy → 실제 matmul → ResNet50 conv layer
- Overhead 정량화 (baseline vs detection-enabled)

---

## 6. 실험 설계 열린 질문

**Phase 1 확정 전 결정해야 할 것**:

1. **Tile 데이터 타입**: int32 (checksum 정확 매칭 쉬움) vs bfloat16 (실제 워크로드에 가까움, rounding 이슈)
2. **주입 확률**: 결정론적 100% (디버깅 쉬움) vs 확률적 Poisson (현실적)
3. **Baseline 비교**: SDC 감지 없는 코드도 돌려서 오버헤드 비교 필요
4. **SubDevice 배치**: 최소 1+1 (Core A + Core B) 로 시작 vs 처음부터 12+8 asymmetric

**추천**: 
1. int32 (Phase 1)
2. 결정론적 100% (Phase 1) → 확률적 (Phase 2)
3. Baseline 필수
4. 1+1 최소 (Phase 1) → 12+8 (Phase 4)

---

## 7. 관련 문헌 (아직 정독 안 함)

### GPU / CPU Fault Injection
- **SASSIFI** (Hari et al., SC 2017) — GPU SASS 명령어 레벨 FI
- **NVBitFI** — SASSIFI 계승, PTX 레벨
- **PinFI** — Intel Pin 기반

### ML-specific FI
- **PyTorchFI** (Mahmoud et al., DSN-W 2020) — model level 시뮬레이션
- **TensorFI** — TensorFlow model level

### ABFT / Algorithm-Based Fault Tolerance
- **Huang & Abraham 1984** — ABFT 원 논문 (matmul checksum)
- **Freivalds 1977** — 랜덤 검증 알고리즘 (matmul)

### 자동차 표준
- **ISO 26262 Part 5** — HW fault injection 요구
- **FIT rate** — Failures In Time / 10^9 hours

### Spatial NPU FT
- 거의 없음 (2026-07-18 시점, 확인 미완) — 이 자체가 novelty 근거

---

## 8. 오늘 확정 안 된 것 (다음 세션)

**보호 스코프 확정 (인식적 정직성 강화)**:
- [ ] **BOS Eagle-N 백서·기술문서 확인** — L1/FPU 자동차용 추가 보호가 문서화됐는지
- [ ] **Tenstorrent Blackhole architecture spec 확인** — tt-metal 오픈 소스 밖의 벤더 문서 접근 가능 여부
- [ ] **Phase 1 실측으로 negative evidence 강화** — L1 bit-flip 주입 후 어떤 카운터도 안 오르는지 확인 (아무 것도 안 오르면 강한 empirical 근거)
- [ ] DRAM (LPDDR5) 보호 여부 확인 (tt-metal 밖, BOS/삼성 문서)

**실험**:
- [ ] 문헌 정독 — 특히 SASSIFI methodology
- [ ] Phase 1 코드 뼈대 실제 작성 (컨테이너 살아나면 실행)
- [ ] Companion workload 방향과의 결합 형태 확정 ("여유 코어 = SDC 검증자" 프레이밍)

**관리**:
- [ ] `PROGRESS.md` 에 이 파일 참조 추가

---

## 9. 세션 로그

**2026-07-18**:
- Safety Island 재검토 → liveness vs correctness 축 분리 → SDC gap 확인 (`safety_island_study.md` §7 추가)
- tt-metal grep 결과:
  - **양성 근거 (있음)**: NoC 헤더 SEC-DED, 라우터 내부 메모리 파리티, NIU 내부 메모리 파리티, Ethernet CRC, RISC-V 코어 ECC
  - **음성 근거 (근거 없음)**: FPU/SFPU 결과 ECC, L1 SRAM ECC, NoC payload ECC — SW 인터페이스 (enable/counter/interrupt 3세트) 모두 grep 실패
- ROUTER_CFG_0 / NIU_CFG_0 필드 분석 → 헤더는 명시 SEC-DED
- Three SDC surfaces (compute/storage/transport) 도출
- 실험 로드맵 4 phase 초안
- 이 파일 신설
- **인식적 정직성 갱신**: "없음" → "근거 없음 (SW 인터페이스 발견 안 됨, 실측·벤더 문서 확인 전엔 확정 X)" 로 표현 완화. 확정하려면 (a) BOS/Tenstorrent 벤더 문서 확인, (b) Phase 1 실측 두 축 필요
