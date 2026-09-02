# Eagle-N multicast microbenchmark 정리 — 2026-08-11

> 목적: row/column rest rotation 정책을 conv에 연결하기 전에, multicast의 정합성·rest 제외 방법·NoC/segment 비용을 분리해 확정한다.
>
> 공통 장비: Eagle-N Blackhole, logical compute grid `5 x 4` (20 workers), 650 MHz.

---

## 1. 결론 먼저

1. **sender 위치는 자유롭다.** 기존 corner/out-of-grid sweep과 `3 x 3` center sender 정합성 시험이 모두 PASS했다. 단일 multicast 호출은 interior sender에서도 전체 rectangle을 정확히 전달한다.
2. **`CoreRangeSet`의 skip은 NoC destination exclude가 아니다.** receiver/compute kernel 배치만 제외한다. full `5 x 4` rectangle으로 multicast하면 rest core의 L1에도 payload가 도착한다.
3. **rest core의 L1 write까지 막으려면 segmented multicast가 필요하다.** row/column rest를 피해 1개(edge) 또는 2개(middle) 연속 rectangle으로 쪼개는 방식이 10개 구성, 각 16회 반복에서 전부 PASS했다.
4. 현 측정 조건에서는 **NoC0가 NoC1보다 모든 config에서 빠르다.** 단, 이는 top-left sender, 64 B payload, 독립 microbenchmark의 결론이다. conv 전체의 NoC 정책을 뒤집는 근거는 아니다.
5. 다음 정책 탐색의 가장 큰 미측정 변수는 **sender 위치**다. 현재 segmented 시험 sender는 모두 active bounding-box top-left이며, 이 위치 편향을 먼저 분리해야 한다.

---

## 2. 실험별 확정 결과

### A. 임의 sender multicast correctness

| 시험 | 설정 | 결과 | 의미 |
|---|---|---|---|
| 기존 scheme sweep | in-grid 4 corner + out-of-grid 6 위치, NoC0/1, loopback/non-loopback | PASS | sender가 top-left일 필요 없음 |
| center sender | `3 x 3`, sender `(1,1)`, NoC0, non-loopback | PASS | interior sender도 단일 mcast로 양방향 destination을 커버 |

따라서 sender 위치 변경 자체를 위해 mcast primitive를 새로 만들 필요는 없다. 다만 성능 최적 위치는 별도 문제다.

### B. shape × NoC baseline (center sender, directed-ideal workload)

이전 `SchemeSingle` 프로파일 결과다. 이 workload의 latency는 약 69k–72k cycles로, 아래 segmented 64 B 시험과 **절대값을 직접 비교하면 안 된다.** 방향성만 참고한다.

| Shape | NoC0 cycles | NoC1 cycles | 더 빠른 쪽 |
|---|---:|---:|---|
| `3 x 3` | 68,815 | 68,862 | 사실상 동률 |
| `4 x 4` | 71,251 | 70,448 | NoC1 |
| `3 x 4` (세로) | 70,692 | 69,600 | NoC1 |
| `4 x 3` (가로) | 69,166 | 69,561 | NoC0 |
| `5 x 2` (가로) | 69,186 | 69,706 | NoC0 |
| `5 x 4` | 72,226 | 71,890 | 사실상 동률 |

방향성은 존재한다. 세로형은 NoC1, 가로형은 NoC0 쪽으로 기우는 경향이 있었지만, full grid에서는 차이가 작았다.

### C. full-rectangle skip의 한계

`active_cores`에서 행/열을 제거하고도 physical mcast destination을 항상 full `5 x 4`로 둔 시험 결과:

- active core는 모두 golden payload와 일치했다.
- skip row/column core도 모두 `data-arrived`였다.

이는 `CoreRangeSet`이 **kernel/compute 참여 집합**을 정할 뿐, `noc_async_write_multicast()`에 전달한 physical rectangle의 L1 write를 필터링하지 않음을 실증한다.

### D. generalized segmented row/column-rest correctness

전용 sender/receiver kernel로 source와 destination L1 주소를 분리했다.

- source L1: golden payload
- 모든 core destination L1: 실행 전 sentinel
- non-loopback: sender destination은 sentinel 유지
- active receiver: golden payload
- rest line: sentinel 유지
- data segment 전체 발행 후 global data barrier는 한 번만 실행
- semaphore multicast도 동일 segment들에만 발행

| Config | Active | Segments | Destinations | 16회 correctness |
|---|---:|---:|---:|---|
| `full_5x4` | 20 | 1 | 19 | PASS |
| `skip_row_0`, `skip_row_3` | 15 | 1 | 14 | PASS |
| `skip_row_1`, `skip_row_2` | 15 | 2 | 14 | PASS |
| `skip_col_0`, `skip_col_4` | 16 | 1 | 15 | PASS |
| `skip_col_1`, `skip_col_2`, `skip_col_3` | 16 | 2 | 15 | PASS |

middle rest는 큰 bounding rectangle으로 보내지 않고 rest 양쪽을 두 rectangle으로 나눴다. 모든 config에서 `destination count 합 = active core 수 - 1`이었다. `Finish()`가 반환했고 active receiver의 semaphore wait도 hang 없이 완료했다.

따라서 segmented multicast는 **rest line에 data/semaphore L1 write를 보내지 않는 primitive**로 검증됐다.

---

## 3. segmented latency: NoC0 vs NoC1

### 측정 정의와 주의점

- payload: 64 B, non-loopback, 16 repetitions/config
- sender zone: **모든 data segment 발행 + 정확히 한 번의 data barrier + 모든 semaphore segment 발행**
- receiver zone: semaphore wait 종료까지의 시간. receiver start skew도 포함하므로 순수 network arrival latency로 해석하지 않는다.
- sender: active bounding-box의 top-left 고정. `skip_row_0=(0,1)`, `skip_col_0=(1,0)`, 그 외 대부분 `(0,0)`.
- NoC1은 reversed multicast endpoint ordering을 적용해 같은 physical rectangle을 대상으로 측정했다.

| Config | Segments | NoC0 mean cycles | NoC1 mean cycles | NoC1 − NoC0 |
|---|---:|---:|---:|---:|
| `full_5x4` | 1 | 534.0 | 561.0 | +5.1% |
| `skip_row_0` | 1 | 512.6 | 554.2 | +8.1% |
| `skip_row_1` | 2 | 682.3 | 698.9 | +2.4% |
| `skip_row_2` | 2 | 590.5 | 647.2 | +9.6% |
| `skip_row_3` | 1 | 426.6 | 482.4 | +13.1% |
| `skip_col_0` | 1 | 511.8 | 564.9 | +10.4% |
| `skip_col_1` | 2 | 743.2 | 754.2 | +1.5% |
| `skip_col_2` | 2 | 739.9 | 770.0 | +4.1% |
| `skip_col_3` | 2 | 730.8 | 756.6 | +3.5% |
| `skip_col_4` | 1 | 512.6 | 564.1 | +10.0% |

NoC0가 10/10 config에서 우세했다. 650 MHz에서 1 cycle은 약 1.538 ns다.

- baseline full-grid 차이: `27 cycles ≈ 42 ns`
- 최대 차이: `55.8 cycles ≈ 86 ns` (`skip_row_3`)
- two-segment middle-column은 NoC 차이가 1.5–4.1%로 비교적 작지만, NoC1이 역전한 사례는 없다.

기존 NoC0-only figure: [segmented_mcast_latency_paper.png](../../data_movement/data/blackhole/segmented_mcast_latency_paper.png)

---

## 4. 현재 정책 초안

### 확정해도 되는 부분

1. **연산만 rest로 만들면 되는 경우**: full rectangle mcast + rest core에 compute/receiver kernel 미배치가 가장 단순하다. 단, rest L1에 data가 써지는 것은 허용해야 한다.
2. **rest L1 write도 막아야 하는 경우**: row/column rest를 1개(edge) 또는 2개(middle) physical rectangle segment로 분할한다. 각 segment의 destination count는 sender 포함 여부를 반영하고, 합계는 `active_count - 1`로 검증한다.
3. **기본 NoC**: 현재 sender 위치와 64 B 조건에서는 NoC0를 default로 둔다.
4. **conv의 기존 dual-NoC 역할 분리**: writer/weight mcast와 reader/activation 흐름의 분리는 유지한다. 단독 mcast가 NoC0에서 빠르다고 conv 전체에서 NoC1 역할을 NoC0로 바꾸면 동시 트래픽 overlap이 깨질 수 있다.

### 아직 확정하면 안 되는 부분

- “NoC0는 항상 NoC1보다 빠르다.”
- “segmented mcast의 microbenchmark overhead가 conv end-to-end overhead와 같다.”
- “top-left sender가 최선이다.”
- “모든 rest 상황에서 segmented mcast가 full rectangle보다 낫다.”

---

## 5. 다음 측정: 경우의 수를 통제한 sender-position sweep

전수 sender sweep은 `20 + 4×15 + 5×16 = 160` sender 위치다. NoC 2개와 16회 반복까지 하면 5,120 program executions이라 가능은 하지만, 첫 단계로는 해석 대비 비용이 크다.

먼저 각 config에서 다음 네 위치만 비교한다.

1. active bound top-left (현재 baseline)
2. active bound top-right에 가장 가까운 active core
3. active bound bottom-left에 가장 가까운 active core
4. active bound bottom-right에 가장 가까운 active core

각 위치 × NoC0/1 × 16회면 현재 dual-NoC 시험의 약 4배(1,280 executions)다. 여기서 NoC1 역전 또는 sender 위치의 큰 편차가 보일 때만 160-position 전수 sweep으로 확대한다.

그 다음에야 실제 Stage 3 conv의 shard/origin rotation에서 허용되는 sender 후보만 남겨 policy를 연결한다.

---

## 6. 산출물과 코드

- segmented test orchestration: `data_movement/one_to_all/test_multicast_schemes.cpp`
- segmented sender: `data_movement/one_to_all/kernels/segmented_multicast_sender.cpp`
- segmented receiver: `data_movement/one_to_all/kernels/segmented_multicast_receiver.cpp`
- NoC0 figure generator: `data_movement/python/plot_segmented_mcast_latency.py`
- raw profiler log: `data_movement/data/blackhole/profile_log_device.csv`

이 노트는 microbenchmark의 결정 기록이다. conv/halo runtime mapping, end-to-end 성능, 실제 aging 균등화는 별도 단계에서 이 기준과 비교한다.

---

## 7. 구현 로드맵 (2026-08-12 합의)

목표는 모든 op가 segmented path를 타게 만드는 것이 아니다. primitive는 programmable하게 제공하되, 프로그래머와 기존 conv가 보는 기본 경로는 single-rectangle/full-mcast fast path로 유지한다.

### 단계 1 — max-4 segment primitive

- 지원 입력: full grid에서 **row 하나와 column 하나를 각각 선택적으로 rest**로 둔다.
- active region은 최대 네 개의 서로 겹치지 않는 rectangle으로 분해한다. middle row + middle column에서만 최대 네 개다.
- sender는 active bounding box top-left다.
- sender만 포함하는 rectangle은 destination이 없으므로 mcast issue 목록에서 생략한다.
- 각 data segment와 semaphore segment는 같은 rectangle 목록을 사용하며, destination count 합은 `active_count - 1`이다.
- 임의 hole 또는 4개 초과 rectangle은 이번 primitive의 범위 밖이다.

### 단계 2 — Conv 첫 통합: rectangle-preserving edge shift

처음에는 disconnected rest를 conv에 바로 넣지 않는다. 기존 block-sharded rectangle을 edge 방향으로 평행 이동해, MemoryConfig·ShardSpec·halo·output mapping의 기존 rectangle 가정을 유지한다. 이 경로에서 only-origin/sender/mcast anchor 이동의 correctness와 latency를 먼저 확인한다.

### 2026-08-12 보정 — Conv 2D mcast는 global 2D rectangle broadcast가 아니다

block-sharded Conv의 2D mcast는 일반적으로 두 축의 1D flow 조합이다.

- weight: top-row sender들이 각 column 방향으로 전송
- activation: left-column sender들이 각 row 방향으로 전송

따라서 generic max-4 rectangle planner는 one-sender primitive 검증에는 유효하지만 Conv mapping의 직접 모델은 아니다. Conv용 microbenchmark는 다음 셋을 분리한다.

1. vertical weight axis만 (NoC0/RISCV0)
2. horizontal activation axis만 (NoC1/RISCV1)
3. 둘을 하나의 Program에서 동시에 실행하는 dual-axis case

각 line sender는 자기 축에서 최대 두 segment만 발행한다. middle row+column rest의 전체 issue 증가는 sender 하나의 4-way split이 아니라, 여러 line sender의 `1 → 2` segment 분할이 누적된 결과다.

### 오늘의 device microbenchmark scaffold

`one_to_all/segmented_rest/test_conv_axis_segmented_mcast.cpp`는 Conv trace에 나타난 대표 transaction 크기 `8,160 B`로 16회 반복한다.

| Config | Weight issue 수 | Activation issue 수 |
|---|---:|---:|
| weight baseline | 5 | - |
| weight, middle row rest | 10 | - |
| activation baseline | - | 4 |
| activation, middle col rest | - | 8 |
| dual baseline | 5 | 4 |
| dual, middle row+column rest | 8 | 6 |

모든 config는 destination L1 readback으로 data correctness를 검증한다. sender kernel은 data barrier와 semaphore multicast를 발행한다. dual-axis에서는 RISCV1이 activation sender로 사용되므로 receiver semaphore wait은 이 첫 throughput scaffold에 넣지 않는다; 이후 Conv integration에서 실제 CB/semaphore dependency를 검증한다.

### 단계 3 — disconnected row/column rest 통합

단계 2가 통과한 뒤에만, max-4 segment plan을 Conv factory의 weight/activation mcast와 연결한다. 이 단계에서 MemoryConfig, halo neighbor, output writer mapping, CB/semaphore mapping을 함께 검증한다.

### 단계 4 — API 노출

처음에는 내부 planner로 유지한다. 이후 안정화되면 `ComputeOnly` (default full mcast + kernel skip)와 `ExcludeL1Writes` (opt-in segmented) 정책으로 일반화한다. segment 좌표나 ACK count는 사용자 API에 노출하지 않는다.

---

## 8. 다음 작업일 계획 — 정규화된 Conv-axis 결론과 첫 Conv 통합

### 오늘 확정된 사실

- 5×4 physical grid에서 axis별 segmented multicast는 16회 반복 전체에서 data readback과 barrier를 통과했다.
- `8,160 B` payload의 sender-kernel span(첫 sender 시작부터 마지막 sender barrier 종료)은 다음과 같았다.

| Config | Mean cycles | Mean time @650 MHz |
|---|---:|---:|
| weight baseline | 865.1 | 1.331 us |
| weight middle-row rest | 1022.5 | 1.573 us |
| activation baseline | 721.4 | 1.110 us |
| activation middle-column rest | 985.0 | 1.515 us |
| dual baseline | 754.2 | 1.160 us |
| dual middle row+column rest | 1039.4 | 1.599 us |

이 숫자는 **rest policy의 절대 completion time**이다. baseline과 rest는 active core/destination 수가 다르므로, 이를 순수 segment overhead로 해석하지 않는다.

### 오전 — 공정한 microbenchmark control 추가

같은 payload, active core 수, sender 수, destination 수를 유지하고 hole/segment 여부만 바꾼 compact control을 추가한다.

| Segmented case | Compact single-segment control |
|---|---|
| 5×4 middle-row rest: 15 active, weight 5 senders/10 destinations | contiguous 5×3: 15 active, weight 5 senders/10 destinations |
| 5×4 middle-column rest: 16 active, activation 4 senders/12 destinations | contiguous 4×4: 16 active, activation 4 senders/12 destinations |
| 5×4 row+column rest: 12 active, weight 8 + activation 9 destinations | contiguous 4×3: 12 active, weight 8 + activation 9 destinations |

이 control과 segmented case의 차이를 “physical hole을 보존하면서 L1 write를 제외하기 위한 segment cost”로 보고한다. row/column 축의 5×4 비대칭은 서로 직접 비교하지 않고, 각 축 안에서 baseline/control/segmented만 비교한다.

### 오후 — 실제 Conv 첫 통합 범위 확정 및 correctness

1. 실제 Conv의 weight/activation mcast program factory에서 sender, rectangle endpoint, semaphore destination count가 정해지는 위치를 확정한다.
2. 기존 fast path는 byte-for-byte 동일하게 남긴다.
3. `ExcludeL1Writes` 내부 plan이 있을 때만 line별 1–2 segment issue 목록을 사용하게 한다.
4. 우선 rectangle-preserving edge shift로 기존 Conv correctness를 확인한다. disconnected middle rest, MemoryConfig/halo/output mapping 변경은 그 다음 단계로 분리한다.
5. Conv correctness가 통과한 뒤에만 end-to-end profiler를 찍는다.
