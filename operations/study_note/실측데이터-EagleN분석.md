# 실측 데이터 분석 — Eagle-N 칩 (ResNet50 per-stage TRISC 활동)

> 실물 보드 측정. Per-stage TRISC(compute) activity, CORE_SET usage. **연구의 핵심 증거.**

---

## 칩 레이아웃 (Eagle-N, KAIST Wormhole과 다름)

- 격자: 6열(sx 0–5) × 5행(sy 0–4)
- **sx=5 열 = Dispatch/reserved (회색)** = 사용 불가
- **sy=3 행 = DRAM row (주황, 6 endpoints)** = 가로 행 배치 ★
- **Compute = 20 workers** = sx{0,1,2,3,4} × sy{0,1,2,4} = 5×4 = 20

### ★ KAIST 논문과 결정적 차이
- KAIST(Wormhole N300): DRAM = **column-wise (세로 열)**
- **Eagle-N: DRAM = row (sy=3, 가로 행)**
- → NoC 방향 vs DRAM 배치 궁합 분석이 KAIST와 다름. **내 칩 고유 조건으로 재측정 필요**

---

## Stage별 총 compute cycles

| Stage | 샤딩 | total (M cyc) | 비중 |
|---|---|---|---|
| 0 Stem (fold+conv1+maxpool) | height | 24.38 | 1.2% |
| 1 Layer1 | height | 24.11 | 1.1% |
| 2 Layer2 | height | 29.53 | 1.4% |
| **3 Layer3** | **block** | **1754.40** | **★ 83.2%** |
| 4 Layer4+Head | block+width | 275.62 | 13.1% |
| 합계 | | 2108.04 | 100% |

- **Stage 3 하나가 전체의 83%.** aging/부하 논의 = 사실상 Stage 3.
- Stage 3 + 4 = 96% → block/width 샤딩 스테이지가 전부. height 스테이지(0-2)는 무시가능.

---

## ★★ Stage 3 불균형 (핵심 발견)

- **sx 0–3 (16코어): 각 ~101–112 M cycles (풀가동)**
- **sx=4 (4코어): 0.72 / 3.65 / 7.69 / 7.74 (사실상 유휴)**
- 검산: 16×~108 + 4×~5 = 1748 ≈ 1754 ✓

### 정량화
- 완전 균등: 1754 / 20 = **87.7 /코어**
- 현재 최대: **111.68**
- **균등 분배 시 최대 부하 코어 ~21% 감소** (111.68 → 87.7)
- **aging: sx0–3이 sx4보다 수명 내내 ~20배 활동** = unfairness 실측 확인 (추정 아님)

### 원인 (냉정)
- **origin 시프팅 문제 아님.** block-sharding이 16조각(4열)으로 쪼개서 5번째 열(sx=4) 안 씀
- 최적화기가 나누기 좋은 모양(4의 배수) 선택 → sx=4 방치

---

## ★ 두 가지 해법과 판단

1. **grid 모양 확대 (16→20코어):**
   - sx=4 포함시키려 모양 변경
   - ⚠️ Fig 4.1: 모양 바꾸면 data movement throughput 영향. 20이 4로 안 나눠떨어져 조각 불균등 위험
2. **op마다 origin/배치 회전 (모양 유지):** ← **채택**
   - 매 conv마다 16코어 위치를 rotate → 매번 다른 코어가 "노는 코어"가 됨
   - 순간 성능 동일(모양 유지) + 장기 누적부하(=aging) 평준화
   - **내 시프팅 전략과 정확히 일치. 성능 안전하면서 aging 분산**

---

## 실험 설계에 주는 것

- **측정 지표 확정:** per-core TRISC active cycles (이 데이터 포맷 그대로)
- **불균형 지표:** max/mean = 111.68/87.7 = 1.27 (baseline). 회전 후 재측정해 개선 확인
- **타겟 스테이지: Stage 3 (block-sharded Layer3)** — 83% 지배, 여기만 균등화해도 전체 aging 대부분 해결
- **성능 trade-off:** 회전 전후 end-to-end 시간 비교 (모양 유지 → 거의 0이어야 내 가설 성립)
- **DRAM row(sy=3) 위치 고정** = 회전 시 DRAM 행과의 상대 거리 변화 → 트래픽 영향 측정 포인트
