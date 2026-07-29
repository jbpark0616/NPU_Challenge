# NPU Challenge — Never Pause Unit 발표 대본
**2026-07-28 미팅용** | 24 slides | 목표 시간 ~15분

---

## Slide 1 — Title
> "안녕하세요. 저희 팀 Never Pause Unit 의 6주차 연구 진행 상황 발표하겠습니다. 지난 미팅 이후 새 논문 방향으로 재정비했고, 오늘은 그 결과를 공유드리겠습니다."

---

## Slide 2 — Overview
> "발표는 네 부분입니다. Introduction 과 Background 로 문제 배경, Motivation 에서 저희가 왜 이 주제를 잡았는지, Proposed Approach 에서 접근 방법, 마지막에 Evaluation Plan 과 예상 결과를 말씀드리겠습니다."

---

## Slide 3 — Introduction (기존 연구 요약)
> "먼저 지난 미팅에서 다뤘던 두 방향을 되짚어봅니다. **DMR** 은 코어 두 개를 짝지어 결과를 비교하는 방식인데, 4:16 구성으로는 순간 커버리지가 25% 밖에 안 됩니다. 반면 **ABFT** 는 checksum 기반이라 이론상 100% 커버가 가능하지만, op 마다 연산 구조가 선형/비선형이 섞여 있어서 하나의 알고리즘으로 다 잡을 수 없다는 게 문제였습니다.
> 
> 오른쪽 timeline 은 실제 ResNet50 실행 중 op 종류를 색으로 표시한 건데, fused (빨강) 과 unfused (파랑) 이 반반 정도 섞여 있음을 보여줍니다. 즉 **결함 검출은 op 유형별로 다른 알고리즘이 필요**하다는 것이 지난 결론이었습니다."

**Bridge → 다음 슬라이드:**
> "그래서 저희는 순간 결함이 아니라 **누적 마모, aging** 쪽으로 방향을 튼 게 이번 발표의 핵심입니다."

---

## Slide 4 — Background: NBTI & HCI
> "NPU 하드웨어가 시간이 지나면서 늙는 두 메커니즘이 있습니다.
> 
> **NBTI** 는 PMOS 트랜지스터에 음의 바이어스가 계속 걸리면서 Si-SiO2 계면에 결함이 쌓여 문턱전압 Vth 가 올라가는, 정적 열화입니다. 코어가 켜져 있기만 해도 늙어요.
> 
> **HCI** 는 반대로 트랜지스터가 자주 스위칭할 때 고에너지 캐리어가 게이트 산화막에 주입되는 동적 열화입니다. **활동량에 선형 비례**해서 늙는 게 특징이고, 이게 나중에 중요한 포인트로 다시 나옵니다.
> 
> 오른쪽 그림 (b) 를 보시면 시간에 대해 로그 스케일로 완만하게 증가하는 걸 확인할 수 있습니다."

---

## Slide 5 — Motivation: 기존 Aging 완화 연구
> "멀티코어 가속기에서 이 aging 을 완화하려는 두 선행 연구가 있습니다.
> 
> 왼쪽 Chen 그룹의 Systolic Array 연구는 InceptionV4 를 10년 돌리면 top-1 정확도가 90% 대에서 20%대로 붕괴한다는 걸 보였습니다. 이유는 **PE 별 utilization 편중** 때문에 몇몇 코어만 빨리 늙어서 그 코어 결과가 오염되기 때문이죠.
> 
> 오른쪽 Brandalero 의 CGRA 연구는 실제 utilization heatmap 을 보여줍니다. 왼쪽 위 코어들은 100% 인데 오른쪽 아래는 1% — **극심한 편중**입니다.
> 
> **저희 발표의 첫 메시지**: 멀티코어 배열 아키텍처에서 utilization imbalance 는 곧 시스템 장기 신뢰성 파괴로 이어진다는 것이 이미 학계에서 입증됐다는 겁니다."

---

## Slide 6 — Motivation: HW 시뮬레이터 기반 검증
> "이 두 논문은 실제 aging 방정식을 하드웨어 시뮬레이터에 넣어서 AI 모델 accuracy 저하를 정량화합니다. 아래 두 수식이 Chen 이 사용한 NBTI/HCI 방정식이고, 저희도 이 방정식을 그대로 씁니다.
> 
> **핵심 관찰**: NBTI 는 `t^(1/6) × u^(1/6)` — activity 영향이 6제곱근으로 매우 약하고, HCI 는 `u × f × t^(1/6)` — activity 에 **선형 비례**. 이 두 항의 차이가 나중에 저희 분석의 방향을 정하게 됩니다."

---

## Slide 7 — Motivation: Research Gap
> "지금까지 aging 완화 연구는 **Systolic Array** 나 **CGRA** 같은 고정형 아키텍처에 집중돼 있었습니다. **NoC 기반 프로그래머블 NPU** 에서 aging-aware workload 를 어떻게 다룰지는 아직 빈 영역입니다. 이게 저희가 파고들 research gap 입니다.
> 
> 오른쪽 두 heatmap 은 Eagle-N 에서 ResNet50 batch=1 을 실제 실행한 실측 데이터입니다.
> - 왼쪽 injected NoC bytes 를 보면 (0,0) 코어가 8.34MB 로 나머지 대비 극단적으로 편중돼 있고
> - 가운데 consumed NoC bytes 는 (2,1), (3,1) 두 코어가 12.65MB 로 튀어나옵니다
> - 오른쪽 compute utilization 은 x=4 열 전체가 배제되는 30배 편중이 있습니다
> 
> **세 축, 세 편향** — 이게 저희가 마주한 실제 문제입니다."

---

## Slide 8 — Motivation: NIU 축 aging
> "이 세 실측 데이터를 Chen 방정식에 넣어서 시간 진행 시 코어별 aging 편중을 시뮬레이션했습니다.
> 
> 먼저 **NIU 축**입니다. NIU 는 Network Interface Unit — mcast sender/receiver 역할하는 트랜지스터입니다. Tensix 블록도에서 좌우의 ROUTER 부분 (빨간 박스) 이 여기 해당합니다.
> 
> 그래프의 빨간 선은 가장 마모가 심한 코어, 초록 선은 가장 덜 마모된 코어입니다. NBTI 는 편중이 약해 보이는데 HCI 는 격차가 확 벌어집니다.
> 
> **주의**: 이 y값은 K 상수를 1로 놓은 symbolic 값입니다. 실제 mV 로 환산하려면 chip-specific K 캘리브레이션이 필요하고, 이건 future work 로 남겼습니다. **비율만 defensible 하게 봐야 합니다**."

---

## Slide 9 — Motivation: SRAM 축 aging
> "**SRAM 축**입니다. 이건 각 코어의 L1 internal memory 셀 마모입니다. 코어가 output shard 를 자기 L1 에 쓰는 양이 activity 로 잡힙니다.
> 
> 여기서 편중이 세 축 중 가장 심합니다. HCI 그래프에서 worst 와 best 의 격차가 눈에 띄게 크죠."

---

## Slide 10 — Motivation: Compute 축 aging
> "**Compute 축** — Tensix Engine 내 matmul unit 마모입니다. NBTI 는 여전히 완만하지만 HCI 격차가 심합니다.
> 
> 세 그래프의 y값 스케일이 비슷해 보이는 건 각 축의 activity 를 [0,1] 정규화 (worst=1) 했기 때문이고, 실제 편중의 크기 차이는 다음 슬라이드 표에서 정확한 숫자로 보여드리겠습니다."

---

## Slide 11 — Worst/best aging ratio table
> "이 표가 세 축의 편중을 정량화한 겁니다.
> 
> **첫 열 raw activity 비율**은 실측 traffic 의 worst 대비 best 비율입니다. NIU 13배, SRAM **74배**, Compute 30배.
> 
> 이 raw 비율이 NBTI 와 HCI 에 다르게 들어갑니다:
> - **NBTI aging 비율** = `raw^(1/6)` → 6제곱근으로 압축돼서 1.5~2배 정도로 보임
> - **HCI aging 비율** = `raw` 그대로 → 원본 편중이 그대로 반영
> 
> **핵심 메시지**: NBTI 만 보면 심각해 보이지 않지만 **HCI 를 봐야 진짜 aging 편중이 드러납니다**. 특히 SRAM 축은 worst core 가 best core 대비 **74배 빨리 마모**됩니다. 시스템 수명은 worst core 로 결정되니까, 이 격차를 낮추는 게 저희 연구의 목표입니다."

**교수님 예상 질문 대비:**
- Q: "y값은 뭘 기준으로?"  
  A: Chen 방정식에 K=1 symbolic, t=년수×연-초, f=650MHz 측정치, u=raw activity 를 max=1 로 정규화한 값 넣어 산출. **K 상수 미지수라 절대값 대신 비율만 유효.**
- Q: "그래프에서 74배 안 보이는데?"  
  A: 그건 NBTI subplot 얘기고, HCI subplot 에서 74배가 드러납니다. (그래서 이 표가 필요)

---

## Slide 12 — 3 접근축 (Component-Aware)
> "저희 접근은 세 갈래입니다:
> 
> 1. **Component Stress Analysis** — 방금 보여드린 NIU/SRAM/Compute 별 편중 실측
> 2. **Component-Aware Aging Simulation** — 각 컴포넌트별 열화 특성 반영한 aging model
> 3. **Lifetime & Accuracy-Aware Mapping** — 이 데이터를 바탕으로 워크로드를 어디에 배치할지 정하는 rotation 알고리즘
> 
> 다음 슬라이드부터는 어떻게 이 매핑을 바꿀지 실제 접근을 보여드리겠습니다."

---

## Slide 13 — Proposed Approach: Asymmetric Stress
> "먼저 **하드웨어 자체의 물리적 비대칭성**을 짚고 갑니다. Eagle-N 은 6×5 grid 인데, 왼쪽 YAML 을 보시면 **y=3 행 전체가 DRAM row** 입니다. 즉 y=0,1,2 는 compute 연속인데 y=4 는 DRAM 을 건너뛰어야 도달하는 위치예요.
> 
> 오른쪽 injected NoC bytes heatmap 을 보시면 y=3 이 완전히 비어 있고 (DRAM row), 그 결과 y=4 코어들도 compute 이긴 하지만 **NoC hop 이 다른 코어보다 멀어서** 트래픽 패턴이 비대칭적으로 나옵니다.
> 
> **핵심**: 이 물리적 제약은 저희가 못 바꿉니다. 그러니 **소프트웨어 매핑 레이어에서 rotation** 으로 극복해야 합니다."

---

## Slide 14 — Proposed Approach: Anchor 하드코딩
> "이제 tt-metal 소스를 파고들어 갑니다. 문제 원인 첫 번째: **anchor 하드코딩**입니다.
> 
> 위쪽 코드는 `conv2d_op_sharded_program_factory.cpp` 에서 op 을 어디에 배치할지 결정하는 부분인데, `top_left_core = {0, 0}` 로 **명시적으로 (0,0) 좌표에 고정**돼 있습니다.
> 
> 아래쪽 코드는 all_cores 를 결정하는 로직인데 역시 CoreCoord(0, 0) 부터 시작하는 사각형만 허용합니다.
> 
> **결과**: 모든 op 이 항상 (0,0) 부터 시작하니 그 주변에 aging hotspot 이 극단적으로 집중됩니다. 오른쪽 그림에서 Step 1 의 빨간 셀이 (0,0) 위치예요.
> 
> 세 가지 한계: 시작 좌표 하드코딩, 사각형 강제, 그리고 동적 shift 자체가 불가능."

---

## Slide 15 — Proposed Approach: Mcast sender 하드코딩
> "문제 원인 두 번째: **Mcast sender 위치 하드코딩** 입니다.
> 
> 같은 파일 596~612 라인을 보시면, block-sharded op 의 경우 
> - `mcast_sender_cores` 를 top row (또는 left col) 로 결정하고
> - `mcast_receiver_cores` 는 그 외 사각형
> 
> 로 강제합니다. 즉 **top row 와 left col 이 항상 sender 역할** 을 맡는데, 이 두 축의 교집합인 **(0,0) 이 이중 부담** 을 지게 됩니다. 오른쪽 그림에서 (0,0) 이 진한 빨강인 게 그 이유예요 — weight sender + activation sender 겹침.
> 
> **결과**: NoC 트래픽이 top row/left col 로 집중되면서 이 엣지 코어들이 극단적으로 빨리 늙습니다."

---

## Slide 16 — Sharding scheme comparison
> "저희가 다루려는 스코프를 명확히 하기 위한 표입니다. tt-metal 은 세 가지 sharding scheme 을 지원합니다.
> 
> - **Height**: 1D weight mcast, sender 는 top-left 단일 코어
> - **Block**: 2D mcast, sender 는 top row + left col
> - **Width**: broadcast only, sender 개념 자체가 없고 별도 factory 사용
> 
> 아래 그림은 5×4 grid 에서 각 sharding 이 어떤 sender/receiver 구조를 만드는지 시각화한 겁니다. 특히 두 번째 panel 은 Height 도 noop 셀을 두면 **L-shape** 이 가능하다는 걸 보여줍니다 — CORE_SET_14 패턴이 이 방식입니다.
> 
> **결론**: Height 와 Block 은 같은 factory 를 공유하고 rotation framework 하나로 커버 가능. Width 는 별도 factory + 소수 op 이라 이번 스코프 밖입니다."

---

## Slide 17 — Evaluation Plan section divider
> "이제 실제 저희 rotation 알고리즘 세 tier 를 순차적으로 소개하고, 그 개선 효과를 정량화한 결과를 보여드리겠습니다."

---

## Slide 18 — Sharding distribution per stage
> "먼저 workload 분포입니다. 이 그래프는 ResNet50 batch=1 을 stage 별로 나눠 실제 어떤 sharding 이 몇 번 나오는지 실측한 결과입니다.
> 
> - **Layer 1-2** (Stage 0-2): Height sharded 지배
> - **Layer 3** (Stage 3): Block sharded 지배
> - **Layer 4 + Head** (Stage 4): Block 44개 + Width 27개 혼재
> 
> 오른쪽 빨간 주석: Width 는 Stage 4 Head 부근에만 등장하고 별도 factory 를 씁니다 → **rotation 스코프에서 제외**한 이유입니다.
> 
> **핵심 메시지**: 단일 매핑 규칙 하나로는 전체 칩 aging 을 제어할 수 없고, sharding 유형별로 다른 rotation 정책이 필요합니다."

---

## Slide 19 — Tier 1: Cyclic anchor rotation
> "**Tier 1** 은 가장 단순한 정책입니다. 왼쪽 shape reference 를 보시면 4×2 op 이 6개의 유효한 anchor 위치를 가집니다.
> 
> Tier 1 은 이 6개 anchor 를 **순서대로 순환** 하는 방식입니다. Op #k → anchor[k mod 6]. 사이클 카운터 하나만 있으면 되고 상태 유지가 필요 없어서 구현이 제일 쉽습니다.
> 
> 오른쪽 6-step 그림이 순환하는 모습을 보여줍니다. Hot corner (빨강) 가 6 스텝에 걸쳐 grid 를 훑고 지나가는 게 보이시죠. 여러 op 을 걸쳐 평균 내면 원래 (0,0) 에 집중되던 부담이 6개 위치로 분산됩니다."

---

## Slide 20 — Tier 2: + Reflection (block-sharded)
> "**Tier 2** 는 Tier 1 에 **반사** 를 추가합니다. Block-sharded op 에만 적용되는데요.
> 
> 위쪽 4개 panel 이 4가지 반사 모드입니다: TL/TR/BL/BR. 반사에 따라 hot corner 위치가 이동하고 weight sender/activation sender 가 어느 edge 로 갈지가 바뀝니다.
> 
> 아래 8-step 이 실제 배치인데, 2 anchor × 4 reflection = 8 configuration 이 나옵니다. 같은 op 을 여러 방향으로 뒤집으니 특정 edge 에만 부담이 몰리는 걸 훨씬 더 잘 분산시킬 수 있습니다.
> 
> Height-sharded 는 sender 가 단일 코어라 반사 개념이 없어서 Tier 1 만 적용됩니다."

---

## Slide 21 — Tier 3: Aging-aware greedy
> "**Tier 3** 는 상태 반응형입니다. Brandalero DAC'20 의 utilization-aware allocation 을 저희 NPU 컨텍스트에 맞게 이식한 정책입니다.
> 
> 왼쪽 (a) 는 지금까지 누적된 per-core Vth 상태 heatmap 입니다. Worst core 는 25 mV (보라 박스).
> 
> 중앙에 다음 op 을 놓을 8 개 candidate 를 모두 스코어링 합니다: 각 configuration 이 새로 만들어낼 worst-core Vth 를 예측하고, **argmin** 을 골라요. 여기서는 `(1,0) + BR` 이 winner — hot corner 가 이미 지친 코어를 피해서 가장 덜 마모된 (4,4) 근처에 오도록 배치됩니다.
> 
> 아래 8 candidate 그림에서 초록 박스가 선택된 config 입니다.
> 
> 오른쪽 (c) 는 이 config 적용 후 상태. Worst core 가 여전히 25 mV — 즉 이 op 은 worst core 를 아예 안 건드리는 방식으로 배치됐다는 뜻입니다. **탄력적으로 부담을 회피** 하는 게 Tier 3 의 핵심."

---

## Slide 22 — Rotation heatmap + tier summary
> "이제 세 tier 의 실제 개선 효과입니다.
> 
> 위쪽 heatmap 은 NIU (injected + consumed) 축, 아래는 SRAM (consumed only) 축. Baseline → Tier 1 → Tier 2 순서로 편중이 완화되는 게 시각적으로 보입니다.
> 
> 아래 표가 정량 결과인데, **HCI 지배 축 기준** (raw 편중 그대로 반영):
> - **Tier 1**: NIU 13~21% 감소, SRAM 20% 감소 — 무료로 얻는 이득
> - **Tier 2**: NIU 35~57%, SRAM 45% — 반사 추가로 더 확보
> - **Tier 3**: NIU **54~60%**, SRAM **62%** — 상태 반응형이 최고 성능
> 
> **핵심 청구**: Tier 3 는 worst core 부담을 baseline 대비 62% 감소시킵니다. 시스템 수명은 worst core 로 결정되니, **`t^(1/6)` 역산하면 lifetime 은 이론상 약 100배 연장** 됩니다 (HCI 지배 가정).
> 
> ⚠️ **정직한 caveat**: 이 결과는 실측이 아니라 **permutation 시뮬레이션** 입니다. Measured baseline traffic 을 rotation 정책에 따라 다른 코어로 재매핑한 값이고, 세 가지 가정 위에 얻은 결과입니다: (1) translation symmetry — 소스에서 확인, (2) reflection role preservation, (3) byte-linear aging. **T1 실측 검증이 다음 스텝** 입니다."

---

## Slide 23 — Next Week Plan
> "다음 주 계획입니다.
> 
> 1. **Aging 시뮬레이터 연동** — 지금까지의 rotation 결과가 실제 AI accuracy 저하를 얼마나 방어하는지 정량화
> 2. **TT-Metalium 코어 매핑 API 수정** — Tier 1 anchor shift 부터 실제 커널 패치하고 재실행 검증
> 3. **Sharding 특성 반영한 rotation 알고리즘 설계** — Height/Block 별 분기 로직 정교화
> 
> 왼쪽 그림은 추가로 저희가 관찰한 **NoC0 (read) vs NoC1 (write) 채널 비대칭성** 입니다. 지금까지는 두 채널을 합쳐서 봤는데, 분리해서 보니 NoC1 이 훨씬 바쁩니다. 이 비대칭도 aging 관점에서 추가 분석 대상으로 남겨두겠습니다."

---

## Slide 24 — 향후 계획 (전체 프로젝트 timeline)
> "마지막으로 전체 5주 로드맵입니다.
> 
> 1~3주차는 완료: HW 아키텍쳐 학습, SW 스택 이해, AI 모델 프로파일링. 4주차 여유 코어 확보와 SubDevice 탐구도 마쳤습니다 — 현재 16+4 core 구성 실장 완료 상태입니다.
> 
> **5주차 계획**:
> - **여유 코어에 SubDevice 적용** — fault detection logic 을 이 4 core 에서 실험
>   - 어떤 연산 결과가 downstream 에 치명적인지 우선 탐색
> - **코어 제한 모델 최적화** — 12+8 목표 구성에서도 baseline (16+4) 수준 정확도/latency 유지되는지 검증
> 
> 이 두 가지가 이번 프로젝트의 최종 deliverable 이고, 오늘 발표드린 rotation framework 이 이 위에 올라가는 구조입니다.
> 
> 발표 마치겠습니다. 감사합니다."

---

# 📋 예상 질문 대비 (교수님/조교님)

### 1. "y축이 뭘 의미하는지 다시 설명해줄래요?"
- Chen NBTI/HCI 식에 K 상수 1, t=년수×연-초, f=650MHz 측정치, u=raw activity/max 넣은 산출값
- **K 미지수라 절대값 무의미, 비율만 defensible**
- 실제 mV 환산은 chip-specific K calibration 필요 → future work

### 2. "이 데이터는 어떻게 뽑았어요? 실측인가요?"
- Baseline traffic: **Tracy profiler 실측** (진짜 실행 결과)
- Rotation 결과: **permutation 시뮬레이션** (실행 재현 아님)
- 세 가정: translation symmetry (소스 확인), reflection role preservation, byte-linear aging
- **T1 실측 검증이 다음 스텝**

### 3. "왜 Width sharding 은 안 다뤘어요?"
- ResNet50 batch=1 trace 실측: Width 는 Stage 4 Head 부근 27개만 등장
- 별도 factory (`multi_core_conv2d_width_sharded`) — Height/Block 과 코드 경로 자체가 다름
- Broadcast 성격 (Matmul K-dim 파티션) → mcast 편중 mechanism 도 다름
- 스코프 관리 + 비중 낮음 → future work 로 남김

### 4. "74배 편중이라는데 그래프에서는 안 보이는데?"
- NBTI subplot 은 6제곱근으로 압축돼서 1.5~2배로 보임
- **HCI subplot 을 봐야 진짜 편중 (13~74×) 이 드러남**
- Slide 11 표 참조

### 5. "Wrap-around rotation 은 안 되나?"
- **불가능**. 이유:
  - NoC mcast API 가 single rectangle 만 지원
  - Factory 가 all_cores 를 rectangular bbox 로 강제
  - Blackhole HW 는 torus 라 wrap 배선은 있지만 API 미노출
- 지원하려면 factory + kernel 대공사 → 학부 프로젝트 밖 (future work)

### 6. "그럼 실제 수명이 몇 년이야?"
- 정직하게: **모릅니다**. K calibration 안 함 + duty cycle 미반영
- 말할 수 있는 것: **worst/best 비율, baseline/rotation 개선률** 만 defensible
- t^(1/6) 역산으로 **rotation → lifetime extension 배율** 는 유도 가능

---

# 🎯 발표 팁

- **Slide 11 표** 가 이번 발표의 하이라이트. 여기서 시간 넉넉히 쓰기.
- **Slide 22 tier summary** 는 impact 슬라이드. 62% / 100× 숫자 반드시 강조.
- **정직성**: "시뮬레이션이지 실측 아님" 을 슬라이드 22 caveat 에서 명시적으로 말하기 → 신뢰도 상승
- **속도**: 슬라이드 4-6 배경 부분은 빠르게, 슬라이드 8-11 우리 데이터 부분은 느리게
- 총 15분 목표 (질의응답 별도)
