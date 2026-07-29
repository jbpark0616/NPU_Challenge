# Eagle-N NPU의 Silent Data Corruption 검출 연구

작성일: 2026-07-19

---

## 1. 연구 배경

- **SDC(Silent Data Corruption)**: 하드웨어 연산 오류가 예외나 정지 없이 잘못된 결과를 그대로 전달하는 현상. 검출 수단이 없으면 오류가 인지되지 않은 채 후속 연산으로 전파됨.
- 차량용 ADAS에서는 인지(perception) 결과의 무증상 오류가 판단 오류로 직결되므로 검출 메커니즘이 요구됨.
- 본 연구의 대상은 BOS Eagle-N NPU(Tenstorrent Tensix 기반, A0 기준 24 TensixNeo 코어, 3×4 클러스터, 64 TOPS).

---

## 2. 하드웨어 보호 메커니즘 분석

Tenstorrent/BOS 문서 및 소스 분석 결과, NPU 내 보호 범위는 다음과 같음.

| 계층 | 보호 방식 | SDC 발생 가능성 |
|---|---|---|
| NoC 라우터 헤더 | SEC-DED | 없음 |
| NoC 라우터·NIU 내부 메모리 | Parity | 없음 |
| Ethernet 링크(칩간) | CRC | 없음 |
| NoC payload | 없음 | **있음** |
| L1 SRAM (코어당 1.5MB) | 없음 | **있음** |
| FPU/SFPU 연산 결과 | 없음 | **있음** |
| DRAM (LPDDR5) | 미확인 | 확인 필요 |

**근거**
- NoC payload는 512-bit이나 오버헤드가 3-bit에 불과하여 SEC-DED(최소 10-bit 소요) 적용이 구조적으로 불가능함.
- L1 관련 ECC/parity 정의가 소스 내에 존재하지 않음(`L1.*ECC`, `sram.*ecc` 검색 결과 0건).

**분석**
- 해당 아키텍처는 일반 AI 워크로드(데이터센터·엣지)를 대상으로 설계되었으며, 해당 시장은 ASIL 수준의 무결성 요구가 없음.
- Eagle-N은 이 아키텍처를 안전필수 영역에 적용한 사례이므로, 보호 공백이 발생함.

**SDC 발생 지점 분류**
1. Compute-time: FPU/SFPU 연산 오류
2. Storage-time: L1 저장 데이터 오류 (주 대상)
3. Transport-time: NoC payload 전송 오류

---

## 3. 기능안전 요구사항 및 공백

### 3.1 ISO 26262 요구 지표 (목표: ASIL-B)

| ASIL | SPFM | LFM | PMHF |
|---|---|---|---|
| **B (목표)** | ≥ 90% | ≥ 60% | < 100 FIT |
| D | ≥ 99% | ≥ 90% | < 10 FIT |

- SPFM/LFM은 영구 결함(permanent fault)을 기준으로 산정됨.
- ASIL-B는 전면 이중화(lockstep)를 필수로 요구하지 않으므로, 선택적 보호 기법의 적용 여지가 있음.

### 3.2 Eagle-N 안전 개념상의 공백

BOS 공식 문서(Functional Safety Integration)에 따르면, Eagle-N은 전용 Safety MCU와 연동하여 기능안전을 충족하며, NPU는 동작 결함을 검출하여 Safety MCU에 알림을 전송하는 구조임.

- 즉 안전 개념은 **NPU의 자체 결함 검출을 전제**함.
- 그러나 2절에서 확인한 바와 같이 연산·L1·NoC payload에는 SDC 검출 수단이 존재하지 않음.
- **따라서 안전 개념의 전제와 하드웨어 구현 사이에 공백이 존재하며, 본 연구는 이 공백을 대상으로 함.**

---

## 4. 연구 목표 및 접근 방법

### 4.1 목표

하드웨어 추가 없이, NPU 내부에서 SDC를 검출하는 저비용 계층을 구현하여 ASIL-B 요구 수준의 진단 커버리지 달성 가능성을 검증함.

### 4.2 접근 방법

- 모델 실행 시 24개 코어가 전부 사용되지 않는 공간적 유휴가 관측됨(레이어별 텐서 형상과 타일 매핑 불일치에 기인).
- 해당 유휴 코어를 검증 연산에 활용함으로써 추가 하드웨어 없이 검출 기능을 확보함.
- 코어 수가 24개로 제한적이므로 안전 전용 자원의 추가 할당은 비효율적이며, 기존 유휴 자원의 회수가 타당함.

### 4.3 표면별 검증 방식

| SDC 표면 | 검증 방식 | 유휴 코어의 역할 |
|---|---|---|
| Compute | 타 코어에서 재연산 후 비교 | 재연산 |
| L1 저장 | checksum 유지 및 주기적 스캔 | checksum 검사 |
| NoC 전송 | payload end-to-end checksum | 전송 검증 |

**설계상 제약**
- L1 오류는 재연산으로 검출 불가함. 원본과 복제본이 동일한 손상 데이터를 참조하므로 결과가 일치함. 따라서 checksum 방식이 요구됨.
- 영구 결함 검출을 위해 복제본은 반드시 다른 물리 코어(가급적 다른 클러스터)에 배치해야 함.

### 4.4 핵심 논지: 선택적 보호

- 전면 이중화는 성능 손실이 과대하므로 적용 불가.
- 취약 지점을 사전에 특정한 후 해당 지점만 보호하는 방식을 채택함.
- 절차: AVF 측정 → 취약성 집중 구간 도출 → 해당 구간 선택적 검증 → SPFM 환산.

### 4.5 주요 가설: MX 공유 스케일의 취약성

- Eagle-N은 MX 계열 포맷(MXFP8/6/4, MXINT8/4/2)을 지원하며, 해당 포맷은 블록 단위로 공유 스케일 1개를 사용함.
- 공유 스케일 손상 시 블록 전체가 오염되므로 AVF가 높을 것으로 예상됨.
- Mantissa 하위 비트 손상은 신경망의 오차 내성으로 흡수되어 AVF가 낮을 것으로 예상됨.
- **"블록당 스케일 바이트가 SDC 위험의 대부분을 차지한다"는 결과가 확인될 경우, 선택적 보호의 근거가 확보됨.** 최우선 검증 대상.

---

## 5. 결함 모델

- **유형**: 영구 결함(stuck-at) 중심, 일시 단일비트 결함은 보조. SPFM이 영구 결함 기준으로 산정되며, 실리콘상 구현도 용이함.
- **위치**: L1(주), FPU/SFPU 출력, NoC payload.
- **추상화 수준**: RTL 미보유로 gate-level 주입이 불가하므로 architectural 수준(32×32 tile 단위)에서 주입함.
- **범위 제외**: NoC 헤더·라우터(ECC/parity로 검출됨), DRAM(보호 여부 확인 후 결정).
- **다중도**: 단일 결함 기준(ISO 26262 단일점 고장 분석). 다중 비트는 ablation으로 처리.

---

## 6. 실험 계획

### 6.1 AVF 측정 절차

1. Golden 확보: 입력에 대한 정상 출력 저장
2. 시행: 대상(표면·포맷·레이어) 선택 후 임의 비트 반전, 종료까지 실행, 출력 비교
3. 기록: (a) 출력 텐서의 수치 변화 여부, (b) 판정 변화 여부(오분류). 안전 관점에서는 (b)가 유의미함
4. 조건별 수천 회 시행 후 AVF 및 신뢰구간 산출

### 6.2 주입 방법

| 표면 | 방법 | 난이도 |
|---|---|---|
| L1 | host 또는 타 코어에서 해당 주소에 write | 낮음(우선 착수) |
| Compute | 커널 출력 tile 손상(에뮬레이션) | 중간 |
| NoC | endpoint 버퍼 손상(에뮬레이션) | 높음(후순위) |

- 실리콘 기반 시행이므로 추론 1회가 밀리초 단위이며, 수천 회 시행이 시간 단위로 가능함. RTL 시뮬레이션 대비 큰 이점.
- 주요 소요는 시행이 아니라 주입 하네스 구축임.

### 6.3 사용 도구 (실행 완료)

| 도구 | 대상 |
|---|---|
| Tracy (Performance Profiler) | Compute 활용률 |
| L1 Memory Visualizer | L1 사용량 및 텐서 배치 |
| tt-npe (NOC Visualizer) | NoC 트래픽 및 대역폭 여유 |

---

## 7. 현재 진행 상황

- 프로파일링 도구 3종 실행 완료
- 하드웨어 보호 메커니즘 분석 완료
- 결함 모델 및 안전 목표 확정

### 7.1 우선 도출 항목

| 항목 | 결정 사항 |
|---|---|
| NoC 대역폭 여유 | 여유 확보 시 성능 손실 최소화 가능, 포화 시 성능-안전 트레이드오프로 전환 |
| L1 여유 공간 및 텐서 배치 | checksum 배치 가능 여부, AVF 주입 대상 확정 |
| 유휴 코어 수 및 지속성 | 상시 유휴 시 정적 할당, 변동 시 동적 스케줄링 필요 |

**최우선 실험**: L1에서 MX 스케일 바이트와 mantissa의 AVF 비교

### 7.2 확인 필요 사항

- Eagle-N 안전 개념상 NPU의 ASIL 할당 여부(ASIL-B 또는 QM) — BOS 문의 필요
- DRAM(LPDDR5) inline/link ECC 적용 여부
- git 접근 신청 및 NPU Programming Guide 확보
- 실제 ADAS 모델(BEV/occupancy/transformer) 프로파일링 추가 — 현재 ResNet50 기준

---

## 8. 참고 자료

**하드웨어 및 SDK**
- BOS SDK Overview: https://developers.bos-semi.com/docs/Overview/
- Eagle-N 사양: https://developers.bos-semi.com/docs/Supported_NPU_SoCs/Eagle_N/
- tt-npe: https://github.com/tenstorrent/tt-npe
- Tenstorrent Hot Chips 2024: https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf

**관련 연구**
- Mukherjee et al., "Architectural Vulnerability Factor," MICRO 2003
- SAFFIRA (systolic array 신뢰성 평가): https://arxiv.org/html/2403.02946v1
- FORTALESA (fault-tolerant reconfigurable systolic array): https://arxiv.org/html/2503.04426v2
- Quantized systolic array 취약성 분석: https://arxiv.org/html/2401.09509v1
- ApproxABFT: https://arxiv.org/html/2302.10469v5
- FT-CNN: https://arxiv.org/pdf/2003.12203

**기능안전**
- ISO 26262-5:2018 (SPFM: 8.4.5/Table 5, LFM: 8.4.6/Table 6)
- TI SLOA294 (지표 확인용): https://www.ti.com/lit/SLOA294A
