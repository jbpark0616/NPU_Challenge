# Tenstorrent 아키텍처 학습 노트

> METALIUM_GUIDE.md + tensor_layouts.md 학습 정리
> 작성일: 2026-06-24

## 목차

1. [큰 그림 — Tenstorrent의 설계 철학](#1-큰-그림--tenstorrent의-설계-철학)
2. [칩 레벨 — Tensix Processor의 노드 구성](#2-칩-레벨--tensix-processor의-노드-구성)
3. [Tensix Core 내부 구조](#3-tensix-core-내부-구조)
4. [데이터 평면 vs 제어 평면](#4-데이터-평면-vs-제어-평면)
5. [NoC 토폴로지](#5-noc-토폴로지)
6. [HW/SW 협력 매크로 파이프라이닝](#6-hwsw-협력-매크로-파이프라이닝)
7. [3개 커널 추상화 (Reader / Compute / Writer)](#7-3개-커널-추상화-reader--compute--writer)
8. [Circular Buffer (CB)](#8-circular-buffer-cb)
9. [Scratchpad vs Cache 철학](#9-scratchpad-vs-cache-철학)
10. [Native Tile-Based Computing](#10-native-tile-based-computing)
11. [Tensor Layout (Row-Major vs Tiled, Faces)](#11-tensor-layout-row-major-vs-tiled-faces)
12. [Memory Layout (Interleaved vs Sharded)](#12-memory-layout-interleaved-vs-sharded)
13. [Storage (L1 vs DRAM)](#13-storage-l1-vs-dram)
14. [GPU와의 종합 비교](#14-gpu와의-종합-비교)
15. [핵심 통찰 요약](#15-핵심-통찰-요약)

---

## 1. 큰 그림 — Tenstorrent의 설계 철학

### 핵심 메시지

> **"HW는 시킨 일만 정확하게 한다. 모든 설계는 프로그래머의 손에 있다."**

GPU와의 본질적 차이:

| GPU | Tenstorrent |
|-----|-------------|
| HW가 알아서 해줌 (캐시, 자동 latency 숨김) | 프로그래머가 다 설계해라 |
| Massive thread parallelism으로 latency 숨김 | SW 파이프라이닝으로 latency 숨김 |
| 데이터 이동이 자동 (캐시 hierarchy) | 데이터 이동이 명시적 (DMA, NoC 직접 지정) |

→ **편의성 ↓, 통제권 ↑, 예측 가능성 ↑**

---

## 2. 칩 레벨 — Tensix Processor의 노드 구성

### 그리드 구조

Wormhole 칩 = **이질적 노드들이 격자로 배열된 분산 시스템**

| 약자 | 의미 | 역할 |
|------|------|------|
| **T** | Tensix core | 실제 연산 수행 (대부분 차지) |
| **D** | DRAM | 외부 메모리 컨트롤러 |
| **E** | Ethernet | 칩 간 통신 (멀티칩 확장) |
| **A** | ARC | 칩 관리 (전원, 클럭, 상태) |
| **P** | PCIe | 호스트 CPU와의 인터페이스 |

### 용어

- **Tensix Processor**: 격자 전체 (칩 전체)
- **Tensix (= Tensix core)**: 격자 안의 개별 T 노드

### 통신

모든 노드는 **NoC(Network-on-Chip)** 로 연결되어 있어, 임의 노드 ↔ 임의 노드 통신 가능.

---

## 3. Tensix Core 내부 구조

### 구성 요소

| 구성 요소 | 개수 | 역할 |
|----------|------|------|
| **Baby RISC-V 코어** | 5개 | 명령 디스패치/제어 전담 |
| **NoC 인터페이스** | 2개 (NoC 0, NoC 1) | 외부 데이터 송수신 |
| **Unpacker** | 1개 | 데이터 → 연산 유닛 포맷 변환 |
| **FPU (Matrix unit)** | 1개 | 행렬 연산 |
| **SFPU (Vector unit)** | 1개 | 벡터 연산 |
| **Packer** | 1개 | 연산 결과 → 저장 포맷 변환 |
| **SRAM (L1)** | **1.5MB** | 로컬 데이터 + 구성요소 간 데이터 교환 |

### 5개 Baby RISC-V 코어 역할 분담

1. **Data Movement 0 (DM0)** → NoC 0 제어
2. **Data Movement 1 (DM1)** → NoC 1 제어
3. **Unpack** → Unpacker 제어
4. **Math** → FPU/SFPU 제어
5. **Pack** → Packer 제어

### "Baby"의 의미

- **직접 연산하지 않음** — 주변 유닛에게 명령만 던지는 **디스패처**
- 의도적으로 미니멀하게 설계 (MIPS R3000 수준)
- 현대의 슈퍼스칼라/벡터 CPU와 다름

---

## 4. 데이터 평면 vs 제어 평면

### 핵심 분리

```
[제어 평면 — RISC-V 코어들이 명령만 던짐]
   DM0          Unpack        Math         Pack         DM1
    │명령         │명령         │명령         │명령         │명령
    ▼             ▼            ▼            ▼            ▼
[데이터 평면 — HW 유닛이 SRAM과 직접 데이터 교환]
   NoC0 ──→ Unpacker ──→ FPU ──→ Packer ──→ NoC1
            ▲            ▲       ▲
            └────────────┴───────┴──── SRAM (L1) ─────────
```

### 주요 포인트

- **RISC-V 코어는 데이터를 직접 만지지 않음** (공장 관리자 역할)
- **데이터는 SRAM ↔ HW 유닛 사이만 흐름** (공장 작업자가 옮김)
- 블록 다이어그램의 **파란 화살표(명령)** vs **갈색 화살표(데이터)** 분리

---

## 5. NoC 토폴로지

### 4가지 핵심 개념

#### 1. Unidirectional (단방향)
- 각 NoC는 물리적으로 한 방향으로만 흐름
- 이유: 전력 절약, 실리콘 면적 절약

#### 2. Wraparound (2D Torus)
- 격자의 끝이 반대편과 연결되어 도넛 모양
- 이유: full connectivity, 최악 거리 단축

#### 3. Opposite Directions (반대 방향)
- NoC 0과 NoC 1이 반대 방향으로 순회
- 이유: 어느 위치든 둘 중 하나는 짧은 경로

#### 4. Quasi-Full-Duplex (준-전이중)
- 각 NoC는 단방향이지만, 두 NoC 합치면 양방향 통신 가능

### NoC 사용의 유연성

- **NoC 0/1은 입구/출구가 아닌, 양방향 동등한 채널**
- 보통 NoC0=읽기, NoC1=쓰기로 쓰지만, **둘 다 읽기/쓰기에 자유롭게 활용 가능**
- 두 NoC를 같은 방향으로 쓰면 **대역폭 2배** (단, 읽기/쓰기 겹침은 불가)

### Tensix 간 통신

- **모든 Tensix 간 통신 가능** (full connectivity)
- 비용:
  - Latency (hop 수)
  - NoC 대역폭 점유 (경합 발생 가능)
  - 전력

→ **일반 워크로드는 SPMD로**, **고성능 커널은 토폴로지를 의식하여 데이터 분할**

---

## 6. HW/SW 협력 매크로 파이프라이닝

### 전통적 CPU 파이프라이닝과의 차이

| 항목 | 전통 HW 파이프 | Tensix 파이프 |
|------|------------------|----------------|
| 스테이지 간 저장소 | **Flip-Flop** (pipeline register) | **SRAM의 Circular Buffer** |
| 스테이지 분리 | 같은 프로세서 내 논리적 분할 | **물리적으로 분리된 유닛** |
| 동기화 주체 | HW (클럭, 해저드 검출) | **SW (세마포어 핸드셰이크)** |
| 파이프라인 단위 | 명령어 1개 | **타일(32×32) 처리 작업** |
| 유연성 | 고정 | **프로그래머가 재구성 가능** |

### 계층적 파이프라이닝

Tensix는 **두 계층의 파이프라인이 공존**:

1. **거시(macro) 파이프** — SW 협력
   - NoC0 → Unpacker → Math → Packer → NoC1
   - 타일 단위
   - CB + 세마포어로 동기화

2. **미시(micro) 파이프** — 전통적 HW 파이프
   - FPU/SFPU 내부
   - MAC 연산 단위
   - Flip-Flop 기반

### 전형적 데이터 흐름

```
DRAM/다른 Tensix
      │
      ▼
[NoC 0] ──► SRAM ──► [Unpacker] ──► [FPU/SFPU] ──► [Packer] ──► SRAM ──► [NoC 1]
                                                                              │
                                                                              ▼
                                                                    DRAM/다른 Tensix
```

---

## 7. 3개 커널 추상화 (Reader / Compute / Writer)

### 5 RISC-V → 3 커널 매핑

| 커널 | 담당 RISC-V 코어 | 역할 |
|------|----------------|------|
| **Reader kernel** | DM0 (1개) | NoC 0으로 DRAM → CB |
| **Compute kernel** | Unpack + Math + Pack (3개) | 연산 |
| **Writer kernel** | DM1 (1개) | NoC 1로 CB → DRAM |

### 왜 Compute가 3코어를 1커널로 묶는가

- Unpack → Math → Pack은 항상 같은 순서, 강한 데이터 의존성
- **컴파일러가 같은 소스를 3번 컴파일**해서 각 코어에 다른 바이너리로 배포
- 프로그래머는 하나의 compute kernel만 작성 → Metalium이 자동 분배

### 왜 Reader/Writer는 분리하는가

- NoC 0/1은 독립적으로 동시 동작해야 함
- 타이밍이 다름 (Reader는 입력 대기, Writer는 출력 대기)
- 분리해야 파이프라인의 이점 유지

### 재사용성

- Reader/Writer는 데이터 이동 패턴이 표준적 → **여러 연산에 재사용**
- 새 연산 추가 시 보통 Compute만 새로 작성

---

## 8. Circular Buffer (CB)

### 정체

> **CB = SRAM 영역 + FIFO 사용 규칙 + HW 동기화 카운터**

새로운 하드웨어가 아니라 **SRAM의 특정 주소 범위를 FIFO 규율로 쓰자는 약속**.

### 비유: 회전 초밥 컨베이어 벨트

```
   요리사 (Producer)              손님 (Consumer)
       │                              │
       ▼                              ▼
   ┌───┬───┬───┬───┬───┬───┬───┬───┐
   │   │🍣│🍣 │🍣 │   │   │   │   │  ← CB (Circular FIFO)
   └───┴───┴───┴───┴───┴───┴───┴───┘
```

### 4가지 핵심 동작

**Producer 측**:
- `cb_reserve_back(N)`: N개 자리 비워질 때까지 대기
- `cb_push_back(N)`: N개 채웠다고 신고

**Consumer 측**:
- `cb_wait_front(N)`: N개 도착할 때까지 대기
- `cb_pop_front(N)`: N개 가져갔다고 신고

### Race Condition 방지 메커니즘

- 가득 찬 CB에 push 시도 → reserve_back에서 자동 대기
- 빈 CB에서 pop 시도 → wait_front에서 자동 대기
- → **데이터 없는데 읽거나, 공간 없는데 쓰는 일 원천 차단**

### CB 할당 흐름

```
[1단계] 호스트 코드: CreateCircularBuffer(idx=0, size=8KB)
                ↓
[2단계] Metalium 런타임: SRAM의 특정 주소 범위 할당, HW에 등록
                ↓
[3단계] HW: 메타데이터 (back/front 포인터, count) 관리
```

### CB의 5가지 역할

1. **동기화** — race condition 방지
2. **버퍼링** — 깊이 N으로 jitter 흡수, 진정한 비동기 파이프라인
3. **순서 보존** — FIFO (먼저 push된 게 먼저 pop)
4. **SRAM 공간 관리** — circular로 재사용
5. **표준 통신 채널** — 매번 새로 설계할 필요 없음

### HW 메타데이터 동기화

- `push_back` 호출 → HW가 카운터 atomic 증가
- `wait_front` 호출 → HW가 카운터 > 0 까지 대기 (busy-wait 없이)
- → SW는 짧은 코드만, **실제 동기화 프리미티브는 HW가 제공**

### 다중 CB

- SRAM 안에 여러 CB 공존 가능 (CB 0, CB 1, CB 16, ...)
- vector add 예시: cb_in0(A), cb_in1(B), cb_out(결과) — 3개 동시 사용

---

## 9. Scratchpad vs Cache 철학

### 본질적 차이

| 항목 | GPU L1/L2 (캐시) | Tensix L1 (스크래치패드) |
|------|------------------|-------------------------|
| **데이터 이동** | HW 자동 | **SW 명시적 명령** |
| **무엇이 들어있나** | "최근 본 DRAM 내용" | **프로그래머가 직접 넣은 것만** |
| **주소 의미** | DRAM 주소 임시 거주 | **SRAM 자체가 고유 주소 공간** |
| **누락 시(miss)** | HW가 자동 fetch | **그냥 없음, SW가 가져와야** |
| **교체 정책** | LRU 등 HW가 결정 | **없음** |
| **일관성** | HW 프로토콜 (MESI 등) | **없음, SW가 동기화** |
| **Latency** | 예측 어려움 (hit/miss) | **항상 일정 (deterministic)** |

### Operator Fusion 부담 비교

**GPU**: 캐시 evict 때문에 중간 텐서가 DRAM 왕복 → fusion 필수

**Tensix**: SRAM은 자동 evict 없음 → 중간 결과를 그냥 둔 채 다음 연산 호출 가능

→ **컴파일러 부담 ↓, 프로그래밍 복잡도 ↓**

### "deliberate absence"

> "The **deliberate absence** of cache hierarchies provides deterministic and consistent memory access patterns"

캐시를 못 만든 게 아니라 **의도적으로 빼버린** 설계 결정.

### 이유 (ML 워크로드 특성)

- 데이터 흐름이 컴파일 타임에 예측 가능
- 같은 데이터를 무작위로 재방문하지 않음
- → **캐시의 자동성이 별로 도움 안 됨, 오히려 오버헤드**

---

## 10. Native Tile-Based Computing

### 32×32 타일이 모든 레이어의 통일 단위

```
┌──────────────────────────────────────────┐
│  텐서 저장 페이지     = 32×32 타일          │
│  NoC 전송 단위        = 32×32 타일          │
│  CB push/pop 단위     = 32×32 타일          │
│  Unpacker/Packer 단위 = 32×32 타일          │
│  FPU 연산 단위        = 32×32 타일 (내부 16×16 face × 4) │
│  컴파일러 추상화       = 32×32 타일          │
└──────────────────────────────────────────┘
```

### Row-major의 문제

- 큰 행렬에서 다음 행 접근 시 **메모리상 큰 점프**
- 한 행 전체를 버퍼링 필요 → SRAM 낭비
- 또는 SW로 직접 zigzag 타일링 구현 → 복잡

### Tiled Layout의 해법

- 한 타일 = 32×32 = 1024 원소
- 다음 행이 **32 원소 뒤**에 있음 (메모리 지역성 최대화)
- **compute-to-SRAM ratio** 향상

### 패딩

- 32 배수가 아닌 텐서는 패딩 필요
- 예: 48×1024 → 64×1024 (16행 패딩)
- 손해: ~33% 메모리 낭비
- 이득: HW 단순, 시간 예측 가능

### Tilization / Untilization

- Tilization: row-major → tiled
- Untilization: tiled → row-major
- **on-device 수행 가능** → CPU 오버헤드 감소

### GPU와의 차이

- GPU: vector 기반 설계 위에 Tensor Core 얹음 → 데이터 공급 병목
- Tensix: **모든 레이어가 타일 단위로 통일** → 활용률 높음, 예측 가능

---

## 11. Tensor Layout (Row-Major vs Tiled, Faces)

### 텐서의 3가지 속성

```
① Tensor Layout   : Row-Major  vs  Tiled
② Memory Layout   : Interleaved  vs  Sharded
③ Storage         : L1  vs  DRAM
```

### Row-Major Layout

- 한 페이지 = 한 행
- 64×64 텐서 → 64 페이지
- API: `layout=ttnn.ROW_MAJOR_LAYOUT`

### Tiled Layout

- 한 페이지 = 한 타일 (기본 32×32)
- 64×64 텐서 → 4 페이지 (2×2 타일)
- API: `layout=ttnn.TILE_LAYOUT`

### 지원 Tile Shape

- HW: 32×32, 16×32, 4×32, 2×32, 1×32
- **현재 Metalium은 32×32만 정식 지원** (16×32는 matmul 일부)

### Faces — 타일 안의 16×16 단위

**중요한 새 개념**:

```
   32×32 Tile (4개의 16×16 face로 구성):
   ┌─────────┬─────────┐
   │ Face 0  │ Face 1  │
   │ 16×16   │ 16×16   │
   ├─────────┼─────────┤
   │ Face 2  │ Face 3  │
   │ 16×16   │ 16×16   │
   └─────────┴─────────┘
```

**이유**:
- FPU의 원시 연산 단위 = **16×16 행렬 곱셈**
- 32×32 타일 곱셈 = 여러 번의 16×16 face 곱셈으로 분해

### 계층 구조

```
Tensor (논리적)
  └─ Page (Tensor Layout이 결정)
       └─ Tile (Tiled layout일 때, 32×32)
            └─ Face (4개의 16×16, HW MatMul 단위)
                 └─ 개별 원소
```

---

## 12. Memory Layout (Interleaved vs Sharded)

### Bank 개념

- 디바이스 메모리는 **bank**로 물리적 분할
- DRAM 뱅크: Grayskull 8개, **Wormhole 12개**
- L1 뱅크: 대부분 Tensix당 1개 (1.5MB = 1뱅크)

### Interleaved (기본)

```
페이지를 모든 뱅크에 round-robin 분배:
   Bank 0: P0, P3, P6, ...
   Bank 1: P1, P4, P7, ...
   Bank 2: P2, P5, P8, ...
```

- ✅ 로드 밸런싱, 범용 워크로드에 좋음
- ✅ 기본값, 단순
- ❌ 위치 정보 무시, NoC 트래픽 가능
- 모든 텐서는 **Bank 0부터 할당 시작** → 단편화 가능
- API: `ttnn.DRAM_MEMORY_CONFIG` / `ttnn.L1_MEMORY_CONFIG`

### Sharded

```
텐서를 의도적으로 특정 위치에 분할 배치:
   Tensix A: 텐서의 1/4
   Tensix B: 텐서의 1/4
   ...
```

- ✅ 데이터-연산 지역성 최대화
- ✅ NoC 거리 단축
- ✅ **Attention, Convolution**에 특히 유리
- ❌ 프로그래머 직접 설계 필요
- ❌ 패턴이 안 맞으면 비효율

### 두 모드 비교

| 항목 | Interleaved | Sharded |
|------|-------------|---------|
| 분배 방식 | 타일 round-robin | 토폴로지 기반 |
| NoC 거리 | 평균적 | 짧음 (의도적) |
| 프로그래머 노력 | 적음 | 많음 |
| 최적 워크로드 | 일반 | Attention, Conv |

---

## 13. Storage (L1 vs DRAM)

### L1 (SRAM)

- Tensix당 1.5MB
- 칩 전체 ~120MB (수십~80개 Tensix)
- 빠름, 활성 데이터/중간 결과
- `ttnn.L1_MEMORY_CONFIG`

### DRAM

- GB 단위
- Wormhole 12 뱅크
- 느림, 가중치/큰 텐서
- `ttnn.DRAM_MEMORY_CONFIG`

### 하이브리드 전략

- SRAM only 아키텍처(Cerebras 등): 모델 크기 제약
- DRAM 의존(전통 GPU): 대역폭 병목
- **Tenstorrent**: 둘 다 사용, **프로그래머가 명시적 결정**

---

## 14. GPU와의 종합 비교

| 측면 | GPU | Tenstorrent |
|------|-----|-------------|
| **연산 모델** | SIMT (massive threads) | Cooperative pipelining (코어 분업) |
| **Latency 숨김** | 수많은 스레드로 | **SW 파이프라이닝** |
| **메모리 계층** | L1/L2 캐시 (자동) | **L1 스크래치패드 (명시적)** |
| **데이터 이동** | 자동 (HW 캐시) | **명시적 DMA, NoC 명령** |
| **연산 단위** | Vector → Tensor Core (4×4×4) | **32×32 타일 (모든 레이어 통일)** |
| **일관성 처리** | HW 프로토콜 | **없음 (SW가 동기화)** |
| **Latency 예측** | 캐시 미스로 어려움 | **결정론적** |
| **Operator Fusion** | 필수 (캐시 evict) | **상대적으로 부담 적음** |
| **편의성 vs 통제** | 편의성 ↑ | **통제 ↑, 예측성 ↑** |
| **칩 간 확장** | NVLink (전용) | **Ethernet (표준)** |

---

## 15. 핵심 통찰 요약

### 한 줄 정리

> **Tenstorrent는 격자형 분산 노드 + 명시적 데이터 이동 + 32×32 타일 단위 통일 + SW 협력 매크로 파이프라인을 통해, GPU의 "HW 자동화" 대신 "프로그래머 명시 제어"로 ML 워크로드의 효율성을 추구하는 아키텍처입니다.**

### 머릿속 그림 5장

1. **칩 그림** — T/D/E/A/P 노드들이 격자로 배열, NoC로 전부 연결
2. **Tensix 내부** — 5 RISC-V가 지휘, HW 유닛이 SRAM 거쳐 데이터 처리
3. **NoC 흐름** — 두 NoC가 반대 방향으로 도는 2D torus
4. **3 커널 + CB** — Reader/Compute/Writer가 Circular Buffer로 동기화하는 파이프라인
5. **타일 통일** — 모든 레이어가 32×32 타일을 다룸 (안에서 16×16 face × 4)

### 설계 철학의 일관성

모든 결정이 **하나의 원칙**으로 수렴:

> **"HW는 단순하고 결정론적으로, 모든 자원 관리는 프로그래머에게."**

- 캐시 없음 → 스크래치패드 (명시적)
- 자동 prefetch 없음 → DMA (명시적)
- 자동 thread scheduling 없음 → 3 커널 SPMD (명시적)
- 자동 메모리 분배 없음 → Interleaved/Sharded 선택 (명시적)
- 자동 데이터 포맷 없음 → Row-Major/Tiled 선택 (명시적)

→ **개발 난이도는 높지만, 예측 가능성과 효율성은 최대**

### 다음 학습 단계

- [ ] **TT-Metalium 호스트 API** (CreateDevice, CreateBuffer, EnqueueProgram 등)
- [ ] **Compute Kernel 내부 디테일** (tile_regs, pack_tile 등)
- [ ] **SPMD in Metalium** (여러 Tensix에 작업 분배)
- [ ] **Multi-chip scaling** (n300, QuietBox, Galaxy, tt-Fabric)
- [ ] **Tensor Sharding 상세** (tensor_sharding.md)
- [ ] **Flash Attention 사례** (토폴로지 활용 실전)
- [ ] **GEMM 성능 분석** (GEMM_FLOPS.md)

---

# 추가 학습 (2026-06-28)

> 학습 자료 카탈로그 + TTNN 레이어 + 프로파일링 도구 + Tile/Face 디테일 + CB 설계 철학

## 16. 학습 자료 카탈로그

### 공식 문서
- `tt-metal/docs/source/tt-metalium/` (Sphinx RST)
  - `get_started/get_started.rst` — 첫 프로그램
  - `tt_metal/advanced_topics/` — `tiles.rst`, `memory_for_kernel_developers.rst`, `compute_engines_and_dataflow_within_tensix.rst`, `fp32_accuracy.rst`
  - `tt_metal/apis/host_apis/`, `kernel_apis/` — API 레퍼런스
  - `tools/` — `tracy_profiler.rst`, `watcher.rst`, `inspector.rst`, `noc_debug_dump.rst`, `device_print.rst`

### Tech Reports (실전적, 가장 가치 높음)
- `tt-metal/tech_reports/`
  - `matrix_engine/matrix_engine.md` — FPU 동작
  - `GEMM_FLOPS/GEMM_FLOPS.md` — peak FLOPS 계산
  - `data_formats/`, `tensor_layouts/`, `tensor_sharding/`, `tensor_accessor/`
  - `prog_examples/` — NoC_tile_transfer, matmul_multi_core_optimized, multicast, sfpu_eltwise_chain, shard_data_rm
  - `CNNs/cnn_optimizations.md`, `ttcnn.md`
  - `YoloV4-TTNN/yolov4.md`
  - `Debugging/Kernel_Debugging_Tips.md`
  - `op_kernel_dev/accuracy_tips/`
  - `NamedKernelArgs/`, `Saturating_DRAM_bandwidth/`

### 압축 가이드 (Cursor/Copilot용)
- `tt-metal/.github/instructions/` — `llk-kernels.instructions.md`, `metal-api.instructions.md`, `metal-dispatch.instructions.md`, `cpp.instructions.md`

### 기여자 가이드
- `tt-metal/contributing/BestPractices.md`, `ErrorMessageBestPractices.md`

---

## 17. TT-Metal ↔ TTNN 레이어 구조

```
[ Python 모델 코드 ]                    models/demos/.../*.py
        ↓ ttnn.conv2d(...)
[ TTNN C++ op (Python binding + host) ] ttnn/cpp/ttnn/operations/.../
        ↓ Program 생성 + kernel 파일 지정
[ Metal 커널 (.cpp, Tensix 실행) ]      ttnn/cpp/ttnn/operations/.../device/kernels/*.cpp
        ↓
[ Tensix 하드웨어 ]
```

- **TT-Metal** = 저수준 SDK (`tt_metal::CreateProgram`, `CreateKernel`, `EnqueueProgram`)
- **TTNN** = metal 위의 op 라이브러리 (PyTorch ↔ CUDA 관계와 유사)
- TTNN op 1콜 = host C++가 program 만들고 kernel .cpp 선택해서 디스패치

---

## 18. 핵심 통찰: TTNN은 컴파일러가 아니라 라이브러리

`ttnn.conv2d(...)` 호출 시:
- ❌ Python → metal 자동 변환
- ✅ **이미 손으로 짜둔 metal 커널들 중 입력 조건에 맞는 걸 선택 + 파라미터 세팅 + 디스패치**

| 무엇이 | 누가 | 언제 |
|---|---|---|
| 실제 metal 커널 (.cpp) | TT 엔지니어가 손으로 작성 | 미리 |
| Python에서 op 호출 | 사용자 / 모델 코드 | 런타임 |
| 알맞은 커널 선택 + 파라미터 | TTNN op infra (C++) | 매 호출마다 |
| 칩 실행 | metal runtime | EnqueueProgram |

**새 op 필요 = metal 커널 직접 작성 + TTNN 등록**

**참고**: `tt-mlir`은 별도 프로젝트로 PyTorch/ONNX → TTNN 호출 자동 변환을 시도하지만, 결국 마지막엔 똑같이 기존 metal 커널을 디스패치.

---

## 19. 모델 데모 위치 (실제로 돌릴 수 있는 것)

### `models/demos/` (안정 데모)
- **LLM**: llama3_70b_galaxy, deepseek_v3, gpt_oss, gemma4, falcon7b_common, qwen3_vl, qwen25_vl
- **BERT 계열**: bert, metal_BERT_large_11, sentence_bert, bge_large_en, distilbert, squeezebert
- **Vision**: `vision/classification/` (resnet50, mobilenetv2, vgg, vit, mnist), `vision/segmentation/` (segformer, vanilla_unet, vgg_unet), `vision/generative/` (stable_diffusion)
- **Audio/Multimodal**: audio/whisper, multimodal/gemma3, informer
- **Wormhole 전용**: wormhole/falcon7b, wormhole/mamba, wormhole/mistral7b, wormhole/mnist, wormhole/owl_vit

### `models/experimental/` (덜 다듬어졌지만 종류 더 많음)
- yolo_common (+ 버전별), nanogpt, llama2, bloom, swin, retinanet, mobilenet, efficientnet, hrnet, ssd, lenet, t5, openvla

### 실행 패턴
거의 모두 **pytest 기반** 데모/테스트:
```bash
pytest models/demos/vision/classification/resnet50/demo/demo.py
```

### 학부생 추천 순서
1. `wormhole/mnist` — sanity check
2. `vision/classification/resnet50`
3. `wormhole/falcon7b` 또는 `bert/` — LLM 입문
4. YOLO (experimental) + `tech_reports/YoloV4-TTNN/`

---

## 20. 실제 Metal 커널 위치

`tt-metal/ttnn/cpp/ttnn/operations/` 아래 op별:
```
operations/
├── conv/conv2d/device/kernels/       ← reader/compute/writer .cpp 12개
├── eltwise/binary/device/kernels/
│   ├── compute/    (TRISC)
│   └── dataflow/   (BRISC/NCRISC)
├── matmul/device/kernels/
├── normalization/, pool/, reduction/, transformer/
```

### 한 op 따라가는 법
1. **Python API** — `ttnn/ttnn/operations/binary.py`
2. **C++ binding** — `binary_pybind.cpp` (Python ↔ C++)
3. **Host op (program factory)** — `binary_device_operation.cpp` (CreateProgram, CreateKernel 호출)
4. **실제 커널** — `device/kernels/compute/*.cpp`, `device/kernels/dataflow/*.cpp`

---

## 21. Python에서의 메모리/Sharding 제어

**C++로 안 내려가도 Python에서 메모리 전략 다 노출됨**:
```python
mem_cfg = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=ttnn.ShardSpec(
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(...)}),
        shard_shape=[32, 128],
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
x = ttnn.from_torch(t, ..., memory_config=mem_cfg)
y = ttnn.conv2d(x, w, ..., memory_config=mem_cfg_out)
```

→ **"어느 코어에 얼마씩 어떤 모양으로"는 Python에서 결정**

### C++로 내려가는 진짜 이유
1. 새 op 작성
2. 기존 op의 새 sharding variant
3. 학습 목적

### 레벨별 컨트롤
| 레벨 | 결정 사항 | 위치 |
|---|---|---|
| L4 모델 | 어떤 op을 어떤 순서로 | Python |
| L3 메모리/샤딩 | DRAM/L1, layout, core grid | Python (MemoryConfig) |
| L2 op 알고리즘 | 어떤 커널을 어느 코어에 | C++ program factory |
| L1 커널 코드 | NoC, CB, FPU/SFPU | C++ kernels |
| L0 raw metal | 처음부터 | C++ programming_examples |

---

## 22. 프로파일링 / 관찰 도구

**모두 tt-metal과 별개의 외부 도구**:

| 도구 | 용도 | 비고 |
|---|---|---|
| **Op perf CSV** | op별 시간, core grid | 내장, `TT_METAL_DEVICE_PROFILER=1` |
| **Tracy** | 코어별 timeline 시각화 | 외부 GUI (노트북 설치) |
| **TTNN Visualizer** | op 구조 + 메모리 + NoC | `pip install ttnn-visualizer` |
| **TT-NPE** | NoC 트래픽 추정 | 별도 repo |
| **Inspector** | program 구조, kernel/CB 배치 | 내장 |
| **Watcher** | 코어 PC, NoC 큐 라이브 | 내장, `TT_METAL_WATCHER=N` |

### Tracy 구조
- Tracy는 원래 game/C++ 진영의 오픈소스 프로파일러
- tt-metal이 빌드 타임에 Tracy 라이브러리 통합 (`--enable-profiler`)
- 실행 시 각 코어가 timestamp dump → 노트북의 Tracy GUI로 열기

### 워크플로
```
[정적] 모델 코드 grep "CoreRange|shard_spec" → 설계 의도
[실행] op perf CSV                          → 실측 시간 + 실제 core grid
[시각] TTNN Visualizer                       → 모델 전체 그림
[심층] Tracy Profiler                       → 코어별 timeline, idle gap
[NoC]  TT-NPE / NoC dump                    → 라우팅/bandwidth
```

---

## 23. Tile-Tile MatMul의 Face 단위 분해 (정확한 디테일)

`C(32×32) = A(32×32) @ B(32×32)`를 face 단위로 풀면 **8번 face matmul**:

```
A = [A00 A01]    B = [B00 B01]    C = [C00 C01]
    [A10 A11]        [B10 B11]        [C10 C11]

C00 = A00·B00 + A01·B10     ← 2 face matmul + 누적
C01 = A00·B01 + A01·B11     ← 2 face matmul + 누적
C10 = A10·B00 + A11·B10     ← 2 face matmul + 누적
C11 = A10·B01 + A11·B11     ← 2 face matmul + 누적
→ 총 face matmul 8번
```

### 추상화 계층
- **Kernel 코드**: `matmul_tiles()` API 1콜
- **LLK (Low Level Kernels)**: TRISC가 FPU에게 face matmul 명령 8번 시퀀스
- **FPU 하드웨어**: 16-wide MAC array가 SIMD로 face 1개당 수십 사이클
- **결과**: Dst register에 누적

**Tile matmul 1번 ≈ 100 사이클 범위** (정확한 수는 칩/데이터 포맷에 따라). 정확한 수치 → `tech_reports/matrix_engine/`

### 단위 정리
| 단위 | 크기 | 누가 다룸 |
|---|---|---|
| Element | 1 | 안 다룸 |
| **Face** | 16×16 = 256 | **FPU 하드웨어 단위** |
| **Tile** | 32×32 = 1024 = 4 faces | **CB / kernel API 단위** |
| Block | 여러 tile | sharding / loop |

### "1 코어 = 미니 행렬 가속기" (NOT 1 PE)
- 한 사이클에 1024 MAC ops가 SIMD로 처리
- 64 코어 Wormhole에서 8×8 출력 타일 = 코어당 1 출력 타일 담당 가능
- GPU의 "thread = element" 패러다임과 달리 **"코어 = tile"** 매핑

---

## 24. CB를 직접 코드로 짜는 이유 (Tensix 설계 철학 심화)

### 비교
| 시스템 | 동기화 처리 |
|---|---|
| CPU + OS | mutex, condvar (OS 라이브러리) |
| GPU | warp scheduler, `__syncthreads()` (HW 스케줄러) |
| **Tensix** | **raw semaphore를 직접 wire-up** |

### 5가지 이유
1. **Bare-metal RISC-V** — OS도 스케줄러도 mutex 라이브러리도 없음
2. **HW가 일부러 raw 노출** — 추상화 레이어 없음
3. **명시적 제어 = 최대 성능 + deterministic timing**
4. **유연성** — 같은 API로 in-core / cross-core / mcast 다 됨
5. **자동화하면 conservative해져 성능 손해** — 컴파일러 자동 동기화는 보수적

### 비용
- `cb_push_back()` = atomic increment 1~2 사이클
- `cb_wait_front()` = atomic load + 비교 (데이터 있으면 즉시 통과)
- HW semaphore = L1의 32-bit 카운터
- → **거의 zero overhead**

### 트레이드오프
| | 추상화 비용 | 자유도 | 학습 곡선 |
|---|---|---|---|
| CPU + OS | 높음 | 매우 높음 | 낮음 |
| GPU (CUDA) | 중간 | 중간 | 중간 |
| **Tensix Metal** | **거의 0** | **매우 높음** | **높음** |

**"자유도 + zero overhead = 학습 비용으로 지불"**. TTNN은 이 학습 비용을 대신 치러 라이브러리화.

---

## 25. 같은 4 API로 다양한 토폴로지

각 RISC-V마다 자기 .cpp 파일 따로:
```
operations/eltwise/binary/device/kernels/
├── dataflow/
│   ├── reader_*.cpp      ← NCRISC
│   └── writer_*.cpp      ← BRISC
└── compute/
    └── eltwise_binary.cpp ← TRISC
```

### 패턴 카탈로그
- **A. 표준**: Reader (NCRISC) → CB → Compute (TRISC) → CB → Writer (BRISC)
- **B. 2 Reader 분리**: 입력 두 개를 다른 RISC-V에 → NoC0/NoC1 동시 fetch → 대역폭 2배
- **C. Cross-core mcast**: 한 코어가 fetch + NoC mcast → N 코어 CB에 동시 push (matmul weight 공유)
- **D. Producer chain (op fusion)**: TRISC0 → TRISC1 → TRISC2 (matmul → bias → activation)
- **E. Back-pressure**: 모든 패턴에서 자동 (CB 가득/빔이 자동 stall)

→ **5개 패턴 다 똑같은 4개 API**로 짜짐. 하드웨어가 "악기"만 주고 "악보"는 개발자가 씀.

### HW 제공 vs 개발자 결정
| HW 제공 | 개발자 결정 |
|---|---|
| L1 메모리 (1.5MB) | CB 위치/크기/page 크기 |
| Semaphore 카운터 (32개) | 누가 누구에게 신호 |
| NoC | 어디서 어디로 |
| FPU/SFPU 명령어 | 언제 무엇을 |
| RISC-V 5개 | 각자의 역할 |

---

## 26. 환경 정리 (개인 셋업)

```
[ 노트북 (Windows) ]   ─ SSH ─   [ 서버 (Linux) ]   ─ PCIe ─   [ Eagle/Tensix 보드 ]
   - SSH 클라이언트                  - Docker (tt-metal 환경)
   - VSCode Remote-SSH               - tt-metal 빌드/실행
   - (선택) Tracy GUI                - NPU 드라이버
   - Docker ❌                       
```

- **노트북에는 Docker/NPU 툴 깔지 않음**
- 빌드/실행 모두 서버에서 (SSH 컨테이너 안)
- 코드 편집은 노트북, 실행은 서버

### 서버에서 확인할 것
1. SSH 접속 정보
2. tt-metal 환경이 docker인지 native인지
3. NPU 공유 정책 (lock 등)

---

## 27. 실험 아이디어 (학기 프로젝트 스코프)

1. **ResNet50 데모 실행 + op perf CSV로 가장 느린 conv 찾기**
2. 그 conv의 `MemoryConfig` (sharding 전략) 바꿔서 latency 비교
3. height_sharded vs block_sharded vs width_sharded
4. Tracy로 코어별 idle gap 시각화
5. 새 sharding pattern을 conv2d program factory에 추가 (L2)
6. 작은 custom op을 .cpp로 작성하고 TTNN에 등록 (L1+L2)

---

## 28. 오늘의 추가 핵심 통찰

1. **TTNN = 라이브러리, not 컴파일러** — 미리 짜둔 metal 커널을 선택해서 디스패치
2. **메모리 컨트롤은 Python에서도 가능** — C++로 가는 이유는 "더 많은 제어"가 아니라 "새 알고리즘"
3. **CB는 page 단위 FIFO, page = tile (32×32)** — AI dataflow의 sequential 특성과 자연스럽게 맞음
4. **FPU는 face 단위로 동작** — tile-tile matmul = 8 face matmul로 풀림
5. **CB의 4 API + NoC = 모든 통신 패턴** — 하드웨어는 최소, 소프트웨어가 조립
6. **"1 코어 = 미니 행렬 가속기"** — GPU의 "thread=element" 사고를 "코어=tile"로 전환

---

# 추가 학습 (2026-07-02)

> `allocator.md` 정독 + 대화형 정리. §12, §13 을 심화하는 내용.

## 29. Allocator 의 정체

### 29.1 무엇을 하는가

> **호스트 CPU 에서 도는 "메모리 지도 관리자" 소프트웨어. 디바이스 실행 전에 모든 자리 배정을 끝내고 물러남.**

핵심 특징:
- **호스트 CPU 에서 실행** — 디바이스 RISC-V 에서 X
- **실행 "전"에만 일함** — 커널 실행 중에는 관여 없음
- **데이터를 옮기지 않음** — 순수 주소 배정만. 실제 DMA 는 커널/NoC HW가 담당
- **"자동화 vs 명시적" 스펙트럼의 절충점** — 기본값을 제공하되 프로그래머가 override 가능

### 29.2 호텔 비유

| 역할 | TT | 호텔 |
|---|---|---|
| Allocator | Buffer 를 어느 뱅크·어느 주소에 놓을지 결정 | 프론트 데스크. 방 번호 배정 |
| 커널 (RISC-V) | 미리 받은 주소로 데이터 read/write | 손님이 방에 들어가서 생활 |
| Reserved 의 metadata | 주소값이 전달되어 저장된 곳 | 손님이 받아 든 방 카드 |

프론트 데스크는 손님 체크인 후엔 무슨 일이 벌어져도 관여 안 함 → Allocator 도 실행 시작 후엔 물러남.

---

## 30. L1 SRAM 의 두 층위 구조

L1 이 나뉜다는 게 두 축이 있어서 헷갈리기 쉬움. 축을 분리해야 함.

### 30.1 첫 번째 축 — 코어 종류 (Blackhole 실무: 잊어도 됨)

| 코어 종류 | L1 뱅크 |
|---|---|
| Compute core | 1.5 MB = 뱅크 1개 |
| Storage-only core | 1.5 MB = N개 뱅크로 쪼갬 |

Storage-only 는 Grayskull 유산. **Blackhole/Eagle-N 에서는 사실상 다 compute core → 코어당 뱅크 1개** 만 기억하면 됨.

### 30.2 두 번째 축 — 한 코어의 L1 (1.5MB) 안의 3영역

```
높은 주소  ▲
   ┌────────────────────────────┐
   │  SRAM Buffer               │  ← top-down 할당 ↓
   │  ─── 명시적 텐서 저장       │
   │  ─── 여러 Op 걸쳐 상주 가능 │
   ├────────────────────────────┤
   │  Circular Buffer (CB)      │
   │  ─── 파이프라인 FIFO        │
   │  ─── 한 Op 안에서만 존재    │
   ├────────────────────────────┤
   │  Reserved                  │
   │  ─── FW + 커널 바이너리     │
   │  ─── + 런타임 metadata      │
   │  ─── 프로그래머 접근 X      │
   └────────────────────────────┘
낮은 주소  ▼
```

DRAM 은 반대로 **bottom-up** 할당 → 두 종류가 서로 단편화 안 만듦.

### 30.3 세 영역의 역할 비교

|  | Reserved | Circular Buffer | SRAM Buffer |
|---|----------|-----------------|-------------|
| **누가 관리** | Metal 런타임 | Program (또는 Allocator) | Allocator |
| **누가 넣나** | 시스템 자동 | Kernel 이 push | 프로그래머가 명시 |
| **수명** | 디바이스 전체 | 한 Op | 여러 Op 가능 |
| **개념** | 시스템 코드 | 흐르는 데이터 (동사) | 놓아둔 데이터 (명사) |
| **비유** | 분전반 | 컨베이어 벨트 | 자재 보관함 |

**핵심 인사이트**:
- 컴파일된 커널 바이너리는 **각 Tensix 의 L1 Reserved 로 DMA** 됨
- RISC-V 는 자기 L1 에서만 명령어 fetch 가능 → Reserved 가 필수
- 5 RISC-V 각자 다른 커널 바이너리를 병렬 실행 → 3커널 파이프라인의 물리적 근거

---

## 31. 두 개의 독립 메모리 풀

Allocator 는 메모리 종류마다 별도 풀 관리. **두 풀은 완전 독립.**

| 풀 | 뱅크 수 (Blackhole 기준) | 뱅크 수 (Eagle-N) |
|---|---|---|
| DRAM 풀 | 8 뱅크 × ~4 GB | 별도 확인 필요 |
| L1 풀 | compute 코어 수 만큼 (~130) | 24 (코어당 L1 1개) |

- `ttnn.DRAM_MEMORY_CONFIG` → DRAM 풀 사용. 코어 무관.
- `ttnn.L1_MEMORY_CONFIG` → L1 풀 사용. 각 Tensix 의 L1 뱅크가 참여.

두 풀 사이 주소는 완전 별세계. DRAM buffer 만들어도 L1 offset 은 영향 없음.

---

## 32. Lockstep Allocation ⭐ 핵심 개념

### 32.1 규칙

> **한 풀 안의 모든 뱅크가 하나의 Buffer 에 대해 같은 offset 을 공유.**

### 32.2 매커니즘

- Allocator 는 **풀당 free list 1개** 만 유지 (뱅크별 X)
- Buffer 할당 = 그 하나의 free list 에서 slot 뽑기
- 뽑힌 offset 이 **모든 뱅크에 동일하게 적용**
- "리더 뱅크" 개념 없음 — 호스트의 Allocator 가 통합 관점을 유지

### 32.3 "예약" ≠ "데이터 있음"

Lockstep 은 **자리(offset) 예약** 이야기. 실제 데이터가 그 자리에 있냐는 별개.

```
         Core 0        Core 1        ...   Core 7        Core 8        ...
         ┌───────┐    ┌───────┐            ┌───────┐    ┌───────┐
0xE0000  │  P0   │    │  P1   │            │  P7   │    │ (빈)  │      ← offset 은 모든 코어에 예약
         │       │    │       │    ...     │       │    │       │
         └───────┘    └───────┘            └───────┘    └───────┘
         ↑ 데이터 있음                       ↑ 있음      ↑ 자리는 예약, 데이터 없음
```

Core 8~23 도 0xE0000 을 예약한 상태로 유지 (Buffer 가 어디까지 뻗을지 모르니까). Interleaved 든 Sharded 든 offset 예약 규칙은 동일.

### 32.4 왜 이렇게 설계했나

Kernel 이 페이지 접근할 때 주소 계산이 이렇게 끝남:

```cpp
bank_id  = page_idx % num_banks
offset   = (page_idx / num_banks) * page_size
address  = (bank_id, buffer_handle.base + offset)
          //         ↑ 모든 뱅크에서 동일한 값
```

**뱅크별 base 가 다르면** `base_table[bank_id]` 조회 오버헤드 발생 → 파이프라인 처리량 병목. Lockstep 이 이 조회를 없앰.

### 32.5 트레이드오프

| 대가 | 얻는 것 |
|---|---|
| 참여 안 하는 뱅크도 자리 예약 (약간의 공간 낭비) | 커널 주소 계산이 단순 |
| Allocator 가 뱅크별 상태 못 씀 | Allocator 도 단순 (free list 1개) |

---

## 33. Interleaved vs Sharded — 데이터 배치 규칙

**Lockstep 은 offset 공유. 어느 뱅크에 실제 데이터가 있냐는 아래 두 규칙 중 하나가 결정.**

### 33.1 Interleaved

- 페이지가 라운드로빈으로 뱅크에 뿌려짐
- 페이지 인덱스 순으로 Bank 0, 1, 2, ... 하나씩, 넘치면 다시 Bank 0
- 기본값 — API: `ttnn.DRAM_MEMORY_CONFIG` / `ttnn.L1_MEMORY_CONFIG`

**언제 좋은가**: element-wise 연산 등 위치 무관 워크로드. 뱅크 로드밸런싱.

**예시**: Eagle-N (24 코어) 에 4 타일 텐서를 L1 interleaved
```
Core 0: T0    Core 1: T1    Core 2: T2    Core 3: T3    Core 4~23: 자리 예약, 비어있음
```

### 33.2 Sharded

- 프로그래머가 명시적으로 지정 (어느 코어에, 어떤 shape 으로, 어떻게 나눠서)
- 세 종류:
  - **Height-sharded**: 행 방향으로 잘라 분산
  - **Width-sharded**: 열 방향으로 잘라 분산  
  - **Block-sharded**: 2D 블록으로 잘라 분산
- API: `ttnn.MemoryConfig(shard_spec=...)`

**언제 좋은가**: Matmul (M/N 차원 분산), Attention (head 분산), Convolution (spatial locality)

**핵심 이득**: 데이터-연산 지역성 → NoC 트래픽 최소화. 각 코어가 자기 조각을 자기 L1 에 가짐.

### 33.3 표로 정리

|  | Interleaved | Sharded |
|---|-------------|---------|
| 배치 방식 | 라운드로빈 | 프로그래머 명시 |
| 위치 인식 | 안 함 | 함 |
| NoC 트래픽 | 평균적 | 최소화 가능 |
| 프로그래머 노력 | 적음 (한 줄) | 많음 (shard spec) |
| 최적 워크로드 | element-wise | matmul, attention, conv |

### 33.4 TTNN 이 사용하는 매커니즘

`memory_config` 는 TTNN 이 kernel 선택에 사용:

```
Python: memory_config=sharded_cfg
   ↓
TTNN C++: 이 sharding 패턴에 맞는 pre-written kernel 선택
   ↓ (ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv2d_height_sharded.cpp 등)
Metal 런타임: 그 커널 컴파일 · Reserved 로 DMA · Runtime args 세팅 · Enqueue
```

**중요**: TTNN 은 컴파일러가 아니라 카탈로그. Sharding 패턴이 미리 지원되어 있어야 사용 가능. 없으면 Metal 커널 직접 작성 (§18 재확인).

---

## 34. 실행 흐름 — 호스트 시점 시간순

Allocator 가 실제 언제 일하는지 정리:

```
[호스트 CPU]

① Buffer 생성
   auto a = CreateBuffer(...)
   auto b = CreateBuffer(...)
   → Allocator: 각 풀의 free list 에서 offset 뽑아 예약
   → address 반환

② 커널 컴파일 (완전히 별개의 단계)
   auto reader  = CreateKernel("reader.cpp", ...)
   auto compute = CreateKernel("compute.cpp", ...)
   auto writer  = CreateKernel("writer.cpp", ...)
   → 컴파일러: .cpp → RISC-V 기계어
   → 이 시점엔 buffer 주소가 코드에 없음

③ Runtime args 로 주소 전달
   SetRuntimeArgs(reader, {a.address(), b.address(), n_tiles})
   → ①의 주소를 커널 파라미터로 심음

④ EnqueueProgram
   → 컴파일된 커널 바이너리 → 각 Tensix Reserved 로 DMA
   → runtime args → metadata 영역

──── 여기까지가 실행 전. Allocator 는 여기서 임무 완료 ────

⑤ [디바이스] RISC-V 실행
   → runtime args 읽음
   → NoC 로 데이터 접근
```

**핵심**: 
- Allocator 는 ①에서만 일함
- 컴파일 (②) 은 Allocator 와 독립. 컴파일된 커널은 주소 몰라도 됨
- 커널 코드의 `get_arg_val<uint32_t>(0)` 자리에 ③에서 주소가 주입

---

## 35. Alignment & Memory Reports (부록)

### 35.1 Alignment
- DRAM_ALIGNMENT 로 정렬된 주소에만 할당 (NoC 정렬 요구사항)
- Buffer 크기가 정렬 안 맞으면 Allocator 가 자동 패딩 → 내부 단편화 조금 발생

### 35.2 Memory Reports (디버깅)

`$TT_METAL_HOME/generated` 에 자동 생성:

| 파일 | 내용 |
|---|---|
| `l1_usage_summary.csv` | Program 별 최소 free block · 최대 interleaved buffer 크기 |
| `memory_usage_summary.csv` | 뱅크별 total / allocated / free / largest free block |
| `detailed_memory_usage.csv` | 각 block 의 주소·사이즈·상태 (가장 상세) |

**API**:
```cpp
DumpDeviceMemoryState(device, prefix="");
EnableMemoryReports();
```
```python
ttnn.device.dump_device_memory_state(device, prefix="")
ttnn.device.EnableMemoryReports()
```

**용도**: OOM 원인 추적, 단편화 분석, sharding 최적화.

---

## 36. 아키텍처 철학과의 연결

이 문서를 관통하는 세 가지 메시지:

### 36.1 결정론적 메모리 접근
GPU 는 캐시 히트/미스로 latency 예측 어려움. TT 는 Lockstep + 명시적 배치 → **항상 예측 가능**.

### 36.2 캐시 부재의 실무적 대안
- GPU: HW 가 자동 캐시 관리 (강제 자동화)
- TT: SW Allocator 가 배치 (합리적 기본값 + 프로그래머 override)
- **"자동화 없음" 이 아니라 "자동화가 프로그래머를 이기지 못함"**

### 36.3 CB 가 이미 SRAM 지도에 자리를 예약받고 있음
- CB 는 SW 개념 같지만, 사실 **SRAM 배치도에 물리적 자리가 지정된 영역**
- Part 2 의 3커널 파이프라인이 왜 자연스럽게 작동하는지의 물리적 근거

---

## 37. Part 1 발표 관점 — 뭘 넣고 뭘 뺄까

### 포함 가치 있음 ✅
- Blackhole DRAM 스펙 (8 × 4 GB)
- SRAM 배치 3층 다이어그램 (Reserved / CB / SRAM Buffer)
- Top-down / Bottom-up 방향의 이유
- "캐시 없음" 을 Allocator 가 대체한다는 프레이밍
- Lockstep 을 아주 짧게 (한 문장) — 왜 커널 코드가 단순한지의 답

### 스피커 노트로만 ⚪
- Lockstep 의 세부 예시 (뱅크당 offset 계산)
- Interleaved vs Sharded 비교 (Part 2 에서 오히려 다루기 좋음)
- Storage-only 구분 (혼란만 야기)

### 뺄 것 ❌
- First-fit vs Best-fit 논의
- Alignment 세부
- Memory reports CSV 이름
- ③ ~ ⑤ 실행 흐름 (Part 2 host code 시간에 자연스럽게 나옴)
