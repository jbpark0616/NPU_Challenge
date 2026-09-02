# 논문 정리 — 텐스토런트 NoC 성능 특성

> 두 논문 중심. 내 연구(코어 배치 → 부하/열화 unfairness)의 **선행연구·근거**로 활용.
> 이미지는 채팅 첨부라 파일 삽입 불가 → 각 Figure가 뭘 보여주는지 + 원논문 Figure 번호로 기록.

---

## 논문 A: KAIST NoC Characterization (Wormhole NoC 저수준 성능)

### 배경 (Ch 2)
- **2.1 Related Works:** 기존 연구는 컴파일러/시스템 레벨, **저수준 NoC 동작은 분석 안 함**
  → 이 논문이 그 gap(저수준 성능)을 채움. **내 연구는 한 겹 더: 전체 모델 unfairness + aging (아무도 안 함)**
- **2.2.1 하드웨어 (Table 2.1, N300):** Tensix 80 + DRAM 16 + Ethernet 18 = 120코어.
  코어당 SRAM 1.5MB, 칩 96MB, DRAM 288GB/s, 12nm, 300W. **DRAM은 column-wise 배치**
- **2.2.2 NoC:** 2D folded torus, NoC 2개.
  - **NoC0 = right→down, NoC1 = up→left** (방향 고정 = 결정적 라우팅)
  - VC 16개: unicast(VC0-3), **multicast(VC4-5)**, response(VC6-7), dateline/deadlock방지(VC8-15)
  - VC = 대기줄, 종류별 분리로 head-of-line blocking 방지
- **2.2.3 샤딩 정의:** HEIGHT(y만)/WIDTH(x만)/BLOCK(둘 다). "샤딩 방식마다 부하/트래픽 다름" 명시.
  TT-Metal은 각 코어에 다른 인자(SetRuntimeArgs)로 명시적 제어

### ★ Figure 3.1 — Read latency/bandwidth 히트맵 (핵심 근거)
- (0,0)이 각 코어 L1 읽을 때 latency. **위치별로 크게 다름:**
  - 로컬(자기) 69.7 / 같은 행 165.7 / 같은 열 181.7 / 대각(행·열 다름) 269.7 cycles
  - per-hop ≈ 9 cycles. 세로(12라우터)가 가로(10라우터)보다 느림
- (b) bandwidth = 다 28.x 균등 → **latency만 위치 민감, bandwidth는 둔감**
- **의미:** 코어 위치가 read latency를 1.6~4배 좌우 (내 "배치가 부하 결정" 근거)
- 주의: zero-load(혼잡 없을 때) + 로컬이 압도적 빠름

### ★ Figure 3.2 — Write latency 히트맵 (hop/wrap 증거)
- (0,0)이 각 코어에 write. latency = time(write 시작 → 목적지 세마포어 증가)
- **(a) NoC0(right-down):** (7,7)이 최장 237 cycles (오른쪽아래로 14 hop)
- **(b) NoC1(up-left):** (1,1)이 최장 250 cycles ← **wrap!** 방향 반대라 folded torus 한 바퀴 돎
- **Observation #3: write latency는 hop 수(경로 길이)가 결정**
- hop = 라우터 한 칸 건너기. latency ∝ hop 수

### ★ Figure 3.8 — 2D 트래픽 경로 겹침 (§3.2.3)
- read time vs throughput, 코어 수 늘리며. (a)transpose (b)random
- **transpose(각 코어 (x,y)→(y,x)):** 결정적 라우팅이라 경로 겹침 → 코어 조금만 켜도 일찍 backpressure(혼잡)
- **random:** 경로 안 겹쳐 degradation 없음
- **Observation #9: transpose는 패킷이 몇 겹치는 경로에 집중돼 일찍 contention**
- 의미: 트래픽 패턴(누가 누구랑 통신)에 따라 특정 경로 과부하 = 부하 불균형의 또다른 축

### ★ Figure 3.9 — mcast 혼잡 (ACK) (§3.2.3)
- (0,0)이 grid에 mcast, grid 커질수록 throughput. 4모드(linked/unlink × reserve/non)
- **grid 커질수록 throughput↓ = ACK 혼잡** (receiver 많을수록 ACK 폭증)
- unlinked+NoC1 = 15GB/s 최악 (ACK이 데이터와 같은 링크 공유)
- linked(경로 독점) = ACK 간섭↓ 유지
- **Observation #10: mcast throughput은 ACK 혼잡으로 저하. 라우팅 방향이 영향, transaction 모드로 완화 가능**

### §5.2 — naive(top_left) vs center 라우팅 (Figure 6은 논문 B? — 아래 참고)
- ※ 이 §5.2 관련 그래프(center vs naive)는 별도 확인 필요 (global reduction 연산)

### ★★★ Figure 4.1 — Data movement: grid 모양이 throughput 좌우 (§4.1, 내 연구 직결)
- 측정: interleaved→sharded 변환. 각 코어가 자기 슬라이스를 remote L1/DRAM에서 read
- (a) L1→L1, (b) DRAM→L1. x축=throughput(GB/s), y축=exec time(ns)
- 색=reader grid 모양 (2,8)/(4,4)/(8,2)/(8,8) [x,y], 선=NoC0(실선)/NoC1(점선)
- **★ 발견: 같은 16코어라도 grid 모양+NoC 방향 궁합으로 throughput 크게 다름**
  - NoC0(right-down): x축에 코어 많을수록(8,2) 유리 / y축 많으면(2,8) 불리
  - NoC1(up-left): 정반대 (y축 많을수록 유리)
  - 4x4(균형): NoC0/NoC1 차이 작음
  - **결론: "data movement 성능은 목적지 코어의 공간적 분포에 크게 영향받는다"**
- 원인: 결정적 라우팅이라 grid 긴 축이 NoC 진행 방향과 맞아야 잘 흐름
- **★ 내 연구 의미 (논문 B와 층위 다름, 모순 아님):**
  - origin 이동 → 성능 둔감 (논문 B) = 위치는 자유롭게 옮겨도 됨
  - **grid 모양(aspect ratio) 변경 → data movement에서 성능 민감** = 모양은 함부로 못 바꿈
  - **∴ 시프팅 전략: "모양 유지 + 위치만 오프셋"이 안전** (수정계획서 전략과 일치, 근거 확보)

#### §4.1 마무리 + Implication #1 (data movement)
- **목적지 세로 몰림 → NoC0 혼잡:** 64→16코어 merge 시 목적지 근처 트래픽 집중. NoC0(right-down)는 목적지 세로 배치 시 y축 혼잡
- **reader 많을수록 트래픽 밀도↑:** L1+NoC0는 오히려 throughput drop
- **DRAM read NoC 불균형:** NoC1(세로 먼저)은 source 주변 혼잡 심함 ← **DRAM 코어가 column-wise(세로 열) 배치**라서 (Fig 1.1)
- **read size 클수록 완화:** TTNN interleaved read 커널이 read 사이 **barrier 삽입** → NoC 주입압력 조절 → throughput 안정 (posted+barrier의 실제 용례)
- **★★ Implication #1: data movement 지배 연산은 코어 배치가 어긋나면(misaligned) 결정적 라우팅 하에서 심각한 NoC 혼잡 → L1 트래픽이 DRAM read보다 낮은 throughput**
- **내 연구 의미:**
  - "misaligned placement" = 시프팅을 **잘못하면** 내가 만들 수 있는 상황. **시프팅은 양날의 검** (잘=분산, 못=혼잡)
  - **DRAM 코어 column-wise 고정 = 못 옮기는 제약.** 목적지 origin 옮길 때 DRAM 열과의 상대 위치 변화로 트래픽 궁합 달라짐 → 실험 시 측정 포인트
  - barrier가 주입압력 방어 → mcast 범위 넓혀도 어느정도 보호됨 (ACK 완화 논의와 연결)

### ★★ Chapter 4 — 실제 워크로드에서 NoC 혼잡의 영향 (메인 챕터)
- 3장이 마이크로벤치(개별 latency/throughput)였다면, **4장은 진짜 TTNN 연산에서 그게 어떻게 나타나나**
- 대상 연산 3종, 각각 통신 특성이 다름:
  1. **Data movement (interleaved→sharded 변환):** 지속적 point-to-point 트래픽
  2. **Matrix multiplication:** **partitioning-dependent** 통신 패턴 (어떻게 쪼개느냐가 통신 결정 ← 내 conv 주제 직결)
  3. **Attention:** reduction 기반 데이터 집계(여러 코어에 걸친 aggregation)
- **★ 결정적 사실: 현 TTNN은 NoC0=read 트래픽, NoC1=write 트래픽으로 고정 배정**
  - 목적: DRAM read와 write의 overlap 개선
  - → 내 "ACK을 반대 NoC로 받아 혼잡 완화" 아이디어와 직결. read/write가 이미 물리적으로 분리돼 있음
- 4장 목표: 이 NoC 배정이 개별 연산 성능에 어떻게 기여하는지 + 혼잡 유발하는 아키텍처/트래픽 요인 분석
- **내 연구 연결:** 이 챕터는 "개별 연산"의 혼잡. **나는 여기서 한 겹 위 — 모델 전체를 돌렸을 때 누적 부하 불균형(unfairness) + aging.** 4장이 개별 연산 근거를 주고, 내가 거시적으로 확장

---

## 논문 B: Numerical Kernels on Spatial Accelerator (Wormhole, 2026)

### ★ Figure 6 — center vs naive 라우팅 speedup (§5.2)
- **global reduction 연산** (dot product 등, conv 아님!)
- naive = top-left로 모음 / center = 중앙으로 모음
- y축 = center가 naive보다 몇 % 빠른가 / x축 = 코어 수 / 선 = 코어당 타일(1/4/64/128)
- **결과: 타일 많을수록 라우팅 차이 무의미** (1 tile ±15%, 128 tiles ≈ 0)
- **§5.2 결론: NoC가 low latency라 큰 문제에선 naive(top_left)로 충분** → conv가 top_left 박은 이유
- ⚠️ **주의: reduction 벤치마크. conv에 정확 수치 적용 불가, "경향"만**

---

## 핵심 개념 정리 (논문에서 배운 것)

- **hop** = 라우터 한 칸 건너기. latency ∝ hop 수 (per-hop ≈ 9cycle)
- **wrap** = folded torus라 방향 안 맞으면 고리 끝↔처음으로 빙 돌아 hop 폭증
- **결정적 라우팅** = 경로 고정(NoC0 right-down, NoC1 up-left). 단순하나 혼잡 못 피함 → 위치가 부하 결정
- **ACK** = NoC **하드웨어** 로직. write 물리적 완료 신호 (posted=안받음/non-posted=받음). `noc_wr_ack_received()` 레지스터로 확인
- **세마포어** = **소프트웨어** 논리적 준비/완료 (커널이 noc_semaphore_set/wait)
- **ACK ≠ 세마포어:** ACK=저수준 물리완료, 세마포어=고수준 논리준비. mcast는 둘 다 씀
- **posted+barrier:** ACK 끄면 혼잡↓ 근데 세마포어만으론 순서보장 안됨 → barrier로 데이터완료 먼저

---

## ★★ 내 연구에 주는 종합 (제일 중요)

### 연구 논리 뼈대 (선행연구로 뒷받침됨)
```
[거시적 관점] 모델 전체 = 연산 지배 → 위치가 성능에 대체로 둔감 (논문B Fig6)
   ↓ 그래서
[내 기회] 배치 바꿔도 성능 손해 미미 = 배치를 자유롭게 조정 가능
   +
[구분] 성능(순간/병목) ≠ 열화(누적 활동량)  ← 방어 논리
   ↓
[내 관찰] 누적 활동량은 위치별 편차 큼 (내 데이터 30배 spread)
   ↓
[내 기여] 배치 균등화 → 성능 유지 + 부하/열화 분산
```

### 부하 불균형의 원인 (논문 근거)
1. **hop 거리** (Fig 3.1/3.2): 위치별 latency 1.6~4배
2. **경로 겹침** (Fig 3.8, Obs#9): 트래픽 패턴 따라 특정 경로 집중
3. **ACK 혼잡** (Fig 3.9, Obs#10): mcast receiver 많을수록

### 연구 포지셔닝
- 기존: 컴파일러/시스템 레벨(gap1) → 논문들: 저수준 성능 characterization(gap2, 개별 연산)
- **내 연구: 전체 모델 unfairness + aging (gap3, 아무도 안 함)** ← 블루오션
- 차량용(batch=1, 고온)이 성립 조건에 딱 (코어 덜 써서 편중 큼)

### 주의 (냉정)
- 논문들은 마이크로벤치(특정 패턴/연산) → 원리만 참고, **수치는 conv/모델로 직접 측정해야**
- ResNet conv 타일 추정: Stage1~2 ~12, Stage3 ~6, Stage4 ~3, Stage5 ~1.5 (batch=1) → 위치 유의미 구간일 가능성 (단 추정치, 실측 필요)

### 측정 도구 (논문에서 확인, 내가 쓸 것)
- **Algorithm 1:** timestamp(시작/끝) + read_barrier로 코어별 latency 측정
- **Tracy (DeviceZoneScopedN):** 모델 전체 실행 코어별 활동 측정 (내 30배 spread가 이걸로 나옴)
- end-to-end = 최초 시작 ~ 최후 끝

---

_다음: 개념/논문 그만 파고 → 시프팅 구현 → Tracy로 before/after 측정 (실험)_
