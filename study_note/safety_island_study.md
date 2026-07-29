# Safety Island 스터디: Eagle-N vs NVIDIA DRIVE Orin

> 목적: Eagle-N SoC 다이어그램에 있는 Safety Island 블록의 역할을 이해하고, 벤치마크로 NVIDIA DRIVE Orin의 Functional Safety Island(FSI)와 비교. 우리 연구 방향의 스코프를 명확히 하는 배경 자료.
> 관련: `automotive_specs.md`, `PROGRESS.md`.

날짜: 2026-07-12

---

## TL;DR

- **Safety Island는 SoC 안에 별도로 존재하는 "안전 담당 하드웨어 아일랜드"** — CPU/NPU와 분리된 락스텝 MCU + 에러 감시기. [(NVIDIA FSI Docs)](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/common/topics/fsi_integration/Functional_Safety_Island.html), [(Arm Safety Island 참조 아키)](https://learn.arm.com/learning-paths/automotive/openadkit2_safetyisolation/1d_safety_island_arch/)
- SoC 급 functional safety는 이 블록이 담당 → NPU 파티션에서 FT를 굳이 다시 구현할 필요 없음.
- **Eagle-N은 ASIL-B (ADAS급)** [(BOS 제품 페이지)](https://www.bos-semi.com/product), DRIVE Orin은 ASIL-D (L4급) [(NVIDIA FSI Docs)](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/common/topics/fsi_integration/Functional_Safety_Island.html). Eagle-N Safety Island는 Orin FSI보다 가벼운 스펙일 가능성 높음 (공식 상세 미공개).
- **결정적**: Eagle-N NPU는 **"4-way physical partition + Freedom from Interference"**를 하드웨어로 지원. [(BOS 제품 페이지)](https://www.bos-semi.com/product) 다만 이 문구는 chiplet 기반 풀 제품(250+ TOPS) 관점일 가능성이 크고, **우리 손의 A0 샘플(5×4=20 compute-usable 코어)에서는 4-way(=파티션당 5 코어)가 실용성이 낮음** → 실제 연구 대상은 **2-way 비대칭 파티션** (예: 12 메인 + 8 companion). "4-way 명시"는 primitive 존재의 증거로만 인용하고, A0 규모 스코프는 별도로 정의할 것.

---

## 1. Safety Island가 뭐하는 놈인가

자동차 SoC의 "안전 격리 하드웨어 아일랜드". 다음을 담당:
- **락스텝 MCU** — 같은 명령을 두 코어가 병렬 실행, 결과 비교. 발산하면 에러. [(Aivon: Dual-Core Lockstep)](https://www.aivon.com/blog/security-devices-systems/dual-core-lockstep-safety-chip-technologies/)
- **시스템 헬스 모니터** — 메인 CPU/NPU가 살아있는지, 에러 리포트 수집. [(Arm 참조)](https://learn.arm.com/learning-paths/automotive/openadkit2_safetyisolation/1d_safety_island_arch/)
- **폴백 (safe stop)** — 메인이 죽으면 최소한의 차량 제어(제동 등)로 안전한 정지. [(Arm 참조)](https://learn.arm.com/learning-paths/automotive/openadkit2_safetyisolation/1d_safety_island_arch/)
- **ECC 보호 메모리** — 비트 플립 검출/정정. [(Aivon)](https://www.aivon.com/blog/security-devices-systems/dual-core-lockstep-safety-chip-technologies/)
- **독립 전원/클럭** — 메인이 죽어도 안전 회로는 살아있음. [(NVIDIA FSI Docs)](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/common/topics/fsi_integration/Functional_Safety_Island.html)
- **Secure boot / crypto** — 인증된 SW만 실행. [(BOS+Rambus)](https://www.rambus.com/blogs/bos-semiconductors-and-rambus-securing-the-future-of-automotive-ai/)

### 격리 (Freedom from Interference, FFI)의 원리
ISO 26262가 요구하는 개념. 상위 크리티컬리티(ASIL-D)와 하위(ASIL-B)가 같은 SoC에 살 때, **하위가 상위에 간섭할 수 없도록 하드웨어 수준에서 격리**되어야 함. [(Arm 참조 아키)](https://learn.arm.com/learning-paths/automotive/openadkit2_safetyisolation/1d_safety_island_arch/) 방법:
- 물리적으로 다른 전원/클럭 도메인
- 액세스 블로킹 로직(파이어월)
- 별도 인터커넥트/버스 마스터
- 우선순위 큐

**"NPU 4-way physical partition + FFI"** 라는 Eagle-N 스펙 문구는 NPU 내부에서도 이 FFI 개념을 파티션 간에 적용한다는 뜻. [(BOS 제품 페이지)](https://www.bos-semi.com/product)

<br>

## 2. NVIDIA DRIVE Orin FSI 상세

Orin 문서 기준. 벤치마크 대상.

### CPU 복합체
- **4 × DCLS Cortex-R52 (ARMv8-R)** — 4 쌍 × 2 코어 = 8 물리 코어, 락스텝 [(NVIDIA FSI Docs)](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/common/topics/fsi_integration/Functional_Safety_Island.html)
- 코어당 메모리:
  - ATCM 256 KB, BTCM 128 KB, CTCM 128 KB
  - I-cache 32 KB, D-cache 32 KB
- 캐시 코히런시 없음 (ARMv8-R은 다중 코어 간 코히런시 미지원) [(NVIDIA FSI Docs)](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/common/topics/fsi_integration/Functional_Safety_Island.html)

### 메모리
- **공유 메모리 3 MB** (인터커넥트 통해 CPU 연결) [(NVIDIA FSI Docs)](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/common/topics/fsi_integration/Functional_Safety_Island.html)
- 영구 저장소는 없음

### 격리 메커니즘
- **독립 전원 레일 + 독립 XTAL 클럭** [(NVIDIA FSI Docs)](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/common/topics/fsi_integration/Functional_Safety_Island.html)
- **Access blocking logic / firewall** — 나머지 SoC로부터 FFI 보장

### 안전 관제
- **HSM (Hardware Safety Manager)** — SoC 전체의 안전 에러 알림을 중앙 수집, R52에 알림, **SOC_ERROR GPIO**를 외부 MCU에 assert [(NVIDIA FSI Docs)](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/common/topics/fsi_integration/Functional_Safety_Island.html)
- **CHSM (Crypto HSM)** — R52가 쓰는 하드웨어 crypto accelerator

### 외부 통신
- CAN 컨트롤러 × 2
- SPI 컨트롤러 (외부 MCU 전용)
- UART (디버그)

### 성능·인증
- **~10K ASIL-D MIPS** [(NVIDIA FSI Docs)](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/common/topics/fsi_integration/Functional_Safety_Island.html)
- **Systematics ASIL-D + Random ASIL-D** (Orin 전체 대비 더 높은 안전 인증)
- 담당 워크로드 예: 센서 퓨전, 차량 제어 (braking, steering) [(Arm 참조)](https://learn.arm.com/learning-paths/automotive/openadkit2_safetyisolation/1d_safety_island_arch/)

<br>

## 3. Eagle-N Safety Island (알 수 있는 만큼)

BOS 제품 페이지·발표 자료 기준. **상세 아키텍처는 공개된 게 매우 적음.**

### 확인된 것
- **ASIL-B** [(BOS 제품 페이지)](https://www.bos-semi.com/product) (Orin의 ASIL-D 대비 한 단계 낮음)
- **AEC-Q100 Grade 2** [(BOS 제품 페이지)](https://www.bos-semi.com/product) — 자동차 반도체 신뢰성
- **Rambus 파트너십**으로 **Secure Boot + Key Management + Anti-Tamper** [(BOS+Rambus)](https://www.rambus.com/blogs/bos-semiconductors-and-rambus-securing-the-future-of-automotive-ai/)
- ISO 26262 및 ISO/SAE 21434 (cybersecurity) 준수 지향 [(BOS Chiplet 발표)](https://www.design-reuse.com/news/202529279-bos-semiconductors-introduces-the-industry-s-first-chiplet-based-npu-samples-expanding-supply-to-global-automotive-oems/)

### 슬라이드에서 관찰
- Safety Island가 별도 블록으로 존재 (NPU/CPU와 병렬) — `study_note/eagleN_Architecture.png`
- Main Bus Interconnect에 연결
- 상세 구조는 슬라이드에도 미노출

### 추정 (공개 자료 부재로 조심스러움 — **발표 인용 전 재확인 필요**)
- Orin FSI만큼 무겁지 않을 것 (ASIL-B가 목표라 락스텝 규모/HSM 스케일이 축소일 가능성)
- Cortex-R 계열 락스텝 MCU + HSM 유사 블록 조합이 자연스러움
- 담당은 시스템 헬스 모니터·safe stop·secure boot 관제

<br>

## 4. ASIL-B vs ASIL-D — 이 차이가 우리에게 주는 함의

| 등급 | 대상 | 요구 사항 | 대표 시스템 |
|---|---|---|---|
| **ASIL-B** | 고장이 있어도 안전 유지 가능한 시스템 | 진단 커버리지 90%+ | 파킹, 크루즈, ADAS L2 |
| **ASIL-D** | 고장 시 인명 위험 큰 시스템 | 진단 커버리지 99%+, 락스텝 필수 | 자율주행 L4, 조향, 제동 |

**Eagle-N이 ASIL-B라는 것은:** [(BOS 제품 페이지)](https://www.bos-semi.com/product)
- 타겟이 **L2+/L3 ADAS** — 자율주행 결정 최종 책임은 인간
- **L4/L5용 heavy FT 요구는 안 걸림** — 우리 연구도 이 톤에 맞춰야
- NPU 파티션의 여유 코어는 **redundant inference (FT)** 가 아니라 **companion workload**에 쓰는 게 자연스러움
- Safety Island가 최소한의 시스템 헬스만 담당하고, NPU는 인식 성능 극대화 담당이라는 역할 분담 명확

<br>

## 5. 우리 연구 방향에 미치는 영향

### 재확인된 것
- **NPU 파티션은 FT 재구현이 아니라 concurrent inference 최적화용** — Safety Island가 있으니 FT는 SoC 급에서 해결됨
- BOS가 **"4-way physical partition + FFI"** 라는 파티셔닝 primitive를 스펙에 명시했다는 것은 파티션 + 격리 지향의 하드웨어 설계 의도를 확인 [(BOS 제품 페이지)](https://www.bos-semi.com/product)

### 규모 caveat (⚠ 발표 인용 전 재확인 필수)
- BOS의 "4-way" 문구는 **chiplet 기반 풀 제품(250+ TOPS)** 관점일 가능성이 큼. 여러 다이를 UCIe로 붙인 상위 티어에서 다이 단위 4-way 분할이 자연스러움.
- **우리 손의 A0 샘플은 single-die, 5×4 = 20 compute-usable 코어**. 4-way 파티션 = 파티션당 5 코어 → 실질 인식 워크로드에 적합하지 않음.
- 따라서 실제 연구 스코프는 **2-way asymmetric partition** (예: 12 main + 8 companion, 또는 14 + 6).
- BOS 문구는 "이 방향의 primitive가 하드웨어 설계 의도로 존재한다"는 배경 근거로만 인용하고, 우리 실험 대상 스케일은 별도로 명시해야 함.

### 연구 질문의 재정의
> **"Eagle-N A0(20 compute-usable 코어)의 2-way asymmetric partition 상에서, ADAS급 real-time perception(메인) + companion 워크로드(여유)를 배치했을 때 파티션 간 FFI가 실제로 성립하며 각 파티션의 latency 예산이 지켜지는가?"**

### FT 언급의 위치
- 완전히 버리지 않고, "Safety Island가 SoC 급 FT를 담당함으로써 NPU가 인식 성능에 집중할 수 있음"으로 **배경 컨텍스트**에만 사용
- 우리 연구 스코프는 **NPU 파티션 자체의 real-time + concurrent 실행**에 국한

<br>

## 6. 다음 스텝

- [ ] **"4-way physical partition"의 정확한 의미** — chiplet 급인지 다이 내부인지, Tenstorrent subdevice API에서 어떻게 노출되는지 확인 (`tt-metal` 문서·헤더 확인). A0 20 코어 안에서 4-way가 실제로 가능한지도 실측.
- [ ] **2-way asymmetric partition의 최소·최대 granularity** — subdevice API가 코어 단위로 얼마나 자유롭게 나누는지
- [ ] **FFI의 소프트웨어 관측 가능성** — 파티션 간 간섭이 실제로 얼마나 격리되는지 tt-npe로 측정 가능한지
- [ ] Eagle-N Safety Island 상세는 BOS 공식 문서·백서가 공개될 때 재확인 (현재는 공백)

<br>

## 7. Liveness vs Correctness — SDC gap 정리 (2026-07-18 추가)

> 이전 결론 ("NPU 파티션은 FT 재구현이 아니라 concurrent inference 최적화용") 은 **liveness FT** 얘기였음. **Correctness (SDC) FT** 는 여전히 열린 문제이고, 이게 새 연구 방향의 자리.

### 7.1 두 축 구분

자동차 컨텍스트에서 "FT" 는 사실 두 개의 다른 카테고리:

| 축 | 질문 | Eagle-N 현황 | 담당 |
|---|---|---|---|
| **Liveness FT** | NPU 가 살아있나? Hang/crash 없나? | ✅ 있음 | Safety Island (시스템 헬스 감시 + safe stop) |
| **Storage/Transport FT** | 메모리·NoC 에서 bit-flip 감지되나? | ✅ 있음 | HW ECC (`NUM_HEADER_*_ERR`, `NUM_MEM_PARITY_ERR`) |
| **Correctness FT (SDC)** | 연산 결과가 조용히 틀리지 않나? | ❌ **없음** | ??? |

**Safety Island 는 세 번째를 커버하지 않음** — 자기 시야에는 "NPU 가 결과 냈다" 만 보이고 그 값의 정확성은 몰라.

### 7.2 tt-metal 소스 grep 근거 (2026-07-18)

- `FPU/SFPU/MAC/Math + ECC/parity` 조합 → **hit 0**
- `ABFT / checksum / redundancy / SDC` → hit 0
- `RISCV_DEBUG_REG_ECC` 는 baby RISC-V 코어의 내부 SRAM용 (compute engine 결과 아님)

즉 dense compute (FPU MAC array, SFPU vector unit) 에 대한 어떤 감지 지원도 스택에 부재.

### 7.3 ISO 26262 ASIL-B 진단 커버리지 관점

- ASIL-B 요구: **진단 커버리지 90%+** (§4 표)
- Dense compute 가 blind spot 이면 이 계산에서 큰 구멍
- Safety Island 는 이 커버리지 계산에서 "NPU 감시자" 자격이지 "NPU 내부 검증자" 는 아님
- 위임된 곳 (NPU 내부) 에는 아무 감지 없음 → **이 gap 이 연구의 정당성 근거**

### 7.4 이전 방향과의 관계

| 방향 | 상태 |
|-----|-----|
| Companion workload (concurrent inference) | 유효, 계속 후보 |
| **SDC detection on Eagle-N** | 2026-07-18 새 주 방향 |
| **두 방향 결합**: "여유 코어를 SDC 검증 companion 으로" | 자연 통합 시나리오 |

즉 companion workload 방향은 완전히 대체되는 게 아니라, "companion 이 뭘 하냐" 의 답이 "SDC 검증" 이 되는 형태로 재조합 가능.

### 7.5 발표·인용 시 조심

- "Safety Island 가 SoC 급 FT 를 담당" 이라는 원래 문구는 **liveness** 로만 해석. Correctness/SDC 는 노출됐다고 명시.
- BOS "4-way partition + FFI" 문구는 여전히 primitive 존재 증거로만 인용, SDC 는 별개 축.
- 발표에서 "NPU 내부 연산 SDC 는 현재 검증되지 않음" 을 명확히 하면 연구 gap 이 뚜렷해짐.

<br>

## 출처 (전체 목록)

- [NVIDIA Functional Safety Island (DriveOS 6.0.9 Docs)](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/common/topics/fsi_integration/Functional_Safety_Island.html)
- [NVIDIA FSI (DriveOS 7.0.3 Docs)](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/embedded-software-components/Functional_Safety_Island_FSI/Functional_Safety_Island.html)
- [NVIDIA Orin SoC Hot Chips 34 Presentation](https://hc34.hotchips.org/assets/program/conference/day2/ADAS%20and%20Grace/HC2022.NVIDIA.Mike_Ditty.v6.pdf)
- [BOS Semiconductor Product Page](https://www.bos-semi.com/product)
- [BOS and Tenstorrent Unveil Eagle-N (PR Newswire)](https://www.prnewswire.com/news-releases/bos-and-tenstorrent-unveil-eagle-n-industrys-first-automotive-ai-accelerator-chiplet-soc-302328727.html)
- [BOS Chiplet-Based NPU Samples (Design Reuse)](https://www.design-reuse.com/news/202529279-bos-semiconductors-introduces-the-industry-s-first-chiplet-based-npu-samples-expanding-supply-to-global-automotive-oems/)
- [Arm Safety Island Reference Architecture](https://learn.arm.com/learning-paths/automotive/openadkit2_safetyisolation/1d_safety_island_arch/)
- [Dual-Core Lockstep Safety Chip Technologies (Aivon)](https://www.aivon.com/blog/security-devices-systems/dual-core-lockstep-safety-chip-technologies/)
- [BOS Semiconductor + Rambus Partnership](https://www.rambus.com/blogs/bos-semiconductors-and-rambus-securing-the-future-of-automotive-ai/)
