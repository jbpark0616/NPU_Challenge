# 발표 대본 — NPU Challenge 진행 발표 (2026-07-15)

> 대본 원칙: 학술체 · 존댓말 · 불필요한 서술어 최소화. Week 1·2 는 슬라이드당 한 문장, Week 3 부터 상세.
> 대상 슬라이드: `study_note/Never_Pause_Unit_PROGRESS.pdf` (총 24 슬라이드)
> 관련: `PROGRESS.md`, `study_note/slide_07_profiling_toolchain.md` (참고용 초안)

---

## 슬라이드 1 (표지)
"안녕하십니까. 팀 Never Pause Unit 의 NPU Challenge 연구 진행 발표를 시작하겠습니다."

## 슬라이드 2 (Timeline)
"발표는 세 부분입니다. 1·2주차의 하드웨어·소프트웨어 학습, 3주차의 baseline 프로파일링과 연구 방향 탐색, 그리고 이후 계획입니다."

---

## [Week 1] — 슬라이드당 한 문장

### 슬라이드 3 (Week 1 divider)
"1주차, Tenstorrent Blackhole 아키텍처 학습입니다."

### 슬라이드 4 (Architecture 학습)
"Tensix 코어를 네 가지 축 — Dual NoC, L1 SRAM 과 Circular Buffer, Tensix 연산 엔진, 그리고 Eagle-N 이 채택한 LPDDR5 DRAM — 으로 나누어 학습했습니다."

### 슬라이드 5 (NoC)
"코어 간 통신은 두 개의 독립된 NoC 가 XY 라우팅으로 수행하며, 링크 경합이 성능 병목의 잠재 요인임을 확인했습니다."

### 슬라이드 6 (L1 SRAM + CB)
"각 코어의 L1 SRAM 은 상단부터 할당되는 일반 버퍼와 하단부터 할당되는 Circular Buffer 로 이원화되어, 데이터 이동 커널과 연산 커널이 tile 단위 파이프라인을 이루도록 설계되어 있습니다."

### 슬라이드 7 (Tensix Compute Engine)
"Tensix 엔진은 Unpacker 가 입력을 tile 레지스터에 적재하고, FPU 가 32×32 tile 연산을 수행하며, Packer 가 결과를 L1 로 되돌리는 구조입니다."

### 슬라이드 8 (DRAM)
"Eagle-N 은 LPDDR5 3 컨트롤러 × 4 채널 구성의 off-chip DRAM 을 갖습니다."

---

## [Week 2] — 슬라이드당 한 문장

### 슬라이드 9 (Week 2 divider)
"2주차는 AI 모델과 소프트웨어 스택 학습입니다."

### 슬라이드 10 (Tensix core Kernel)
"각 Tensix 코어는 Read · Compute · Write 세 종류의 커널로 프로그래밍되며, 이들이 Circular Buffer 를 매개로 파이프라인을 형성합니다."

### 슬라이드 11 (Baseline AI-model)
"연구 baseline 으로 이미지 분류의 ResNet50 과 객체 탐지의 YOLOv8s 를 선정했습니다."

### 슬라이드 12 (SW 스택)
"소프트웨어 스택은 상위 TT-NN, 중간 Metalium, 하위 LLK 로 계층화되어 있으며, TOp-down 방식으로 TT-NN 의 op 호출부와 Metalium 의 커널 구현까지 함께 추적했습니다."

### 슬라이드 13 (Profiling Tool)
"프로파일링에는 세 도구를 병용합니다. 메모리 배치 확인용 TT-Visualizer, host · device 이벤트 타임라인용인 Tracy, NoC 트래픽 시뮬레이션용 TT-npe 입니다."

---

## [Week 3] — 본론

### 슬라이드 14 (Week 3 divider)

"여기까지가 학습 단계였고, 지금부터가 이번 발표의 본론인 3주차입니다.

3주차는 두 축으로 진행했습니다. 첫째, 앞서 학습한 도구들로 두 baseline 모델의 성능 특성을 정량화하는 것. 둘째, 이 데이터를 근거로 연구 방향을 탐색하는 것입니다.

탐색 결과 두 개의 후보 방향을 도출했습니다. 하나는 baseline 자체의 성능 최적화, 다른 하나는 SubDevice 를 활용한 여유 코어 활용 연구입니다. 순서대로 설명드리겠습니다."

---

### 슬라이드 15 (ResNet50 op별 wall time)

"먼저 ResNet50 프로파일링 결과입니다. 좌측은 전체 123 개 연산 중 wall time 상위 30 개, 우측은 연산 종류별 집계입니다.

막대 색은 각 연산의 병목 성격을 나타냅니다. 붉은색은 compute-bound, 파란색은 memory-bound 입니다. 이 분류는 Tenstorrent 의 Performance Model 이 각 op 에 대해 이론 compute cycle 과 bandwidth cycle 중 더 큰 쪽을 근거로 판정한 값입니다.

두 가지 관찰이 있습니다. 첫째, 상위 30 개 중 대부분이 compute-bound 로 분류되었습니다. 둘째, 연산 종류별로 보면 conv2d 와 matmul 두 계열이 총 wall time 의 대부분을 차지하고, 나머지 halo, pool, reshard 등의 비중은 작습니다.

정리하면, ResNet50 은 이 하드웨어 위에서 연산량이 지배적인 워크로드로 나타났습니다."

---

### 슬라이드 16 (Yolo op별 wall time)

"YOLOv8s 도 동일한 방식으로 프로파일링했습니다. 결과의 성격은 ResNet50 과 다릅니다.

상위 op 대부분이 파란색, 즉 memory-bound 로 분류되었습니다. 연산 종류별 분포에서도 conv2d 다음으로 unary — 원소별 활성화·정규화 연산 — 가 두 번째 비중을 차지합니다. unary 는 산술 강도가 낮아 메모리 대역폭에 민감한 연산입니다.

두 모델을 비교하면 병목 성격이 서로 반대에 가깝습니다. ResNet50 은 compute-bound, YOLOv8s 는 memory-bound 경향입니다. 이 차이는 이후 최적화 전략을 결정할 때 각 모델별로 달리 접근해야 함을 의미합니다."

---

### 슬라이드 17 (Roofline Graph)

"앞선 wall time 분석을 산술 강도 축으로 정리한 것이 이 Roofline plot 입니다. X축은 op 의 산술 강도 — FLOP 당 DRAM 바이트 — 이고, Y축은 실측 처리량입니다.

 plot 에 남아 있는 데이터 포인트는 conv2d 와 matmul 두 종류뿐입니다. 나머지 연산은 실측 DRAM I/O 가 사실상 0 에 가까워 산술 강도가 정의되지 않았고, 이는 해당 op 들이 L1 상주 데이터로 처리되었음을 의미합니다. 즉 BOS 구현이 activation 을 L1 안에 잘 관리하고 있다는 정황 증거입니다.

---

### 슬라이드 18 (Roofline Graph 개선 방안)

"다만 지금의 Classic Roofline 은 한계가 있습니다. Tensix 처럼 각 코어에 큰 L1 SRAM 을 두는 spatial NPU 에서는, 데이터 재사용이 L1 안에서 이루어지는 op 의 산술 강도가 매우 높아 X축 오른쪽으로 벗어나거나 하나의 슬로프로 표현되지 않습니다.

따라서 DRAM 슬로프 하나만 그리는 대신, L1 broadcast · L1 local · DRAM 세 계층의 슬로프를 각각 그려, 각 op 가 어느 계층에서 성능이 제한되는지 계층별로 분리하는 접근이 필요합니다.


---

### 슬라이드 19 (TT-npe 측정 작업 환경 구축 — 문제 3종)

"프로파일링 결과를 얻는 과정에서 세 가지 도구 성숙도 문제를 확인했습니다.

첫째, TT-metal 과 TT-npe 사이의 API 시그니처 불일치입니다. 하단 에러 메시지에서 보시는 것처럼 `getDatapointByID` 함수가 TT-npe 최신 버전에서 인자 세 개를 요구하도록 변경되었으나, TT-metal 은 여전히 하나만 넘겨 TypeError 가 발생합니다. 이 예외가 잡히지 않고 report 생성 전체가 중단되어, NoC 트레이스와 무관한 compute 데이터까지 손실됩니다.

둘째, TT-npe 가 제공하는 device model 이 Blackhole 표준 그리드 기준입니다. 좌측 시각화에서 17×12 로 표시되는 것이 이 때문입니다. Eagle-N 은 6×4 그리드의 파생 제품이므로, 링크 대역폭과 DRAM 대역폭의 절대 수치는 별도 검증이 필요합니다.

셋째, kernel profiler 의 device 측 이벤트 버퍼가 고정 크기여서 한 실행에서 약 65 개 op 를 넘기면 이후 이벤트가 소실됩니다. ResNet50 은 batch=1 기준 123 개 op 이므로 한 번의 실행으로는 profiling 이 완결되지 않습니다.

이 세 문제를 우회하기 위해, 모델 실행 자체를 stage 단위로 분할하는 작업이 필요했습니다."

---

### 슬라이드 20 (코드 split 구현)

"우회 구현입니다. ResNet50 의 forward pass 에 (RESNET_STAGE 라는) 환경 변수 게이트를 삽입하여, stem 과 layer1 부터 layer4 까지를 다섯 개의 subgraph 로 분리 실행할 수 있게 했습니다.


각 subgraph 를 개별 pytest 실행으로 profiling 하고, `run_host_id` 를 조인 키로 하여 오프라인에서 병합해, 원본 모델과 동등한 op 단위 데이터를 확보하는 워크플로우를 구축했습니다."

---

### 슬라이드 21 (Sub-device Partition — 유휴 코어 확인)

"두 번째 연구 방향인 SubDevice 활용으로 넘어가겠습니다. 먼저 하드웨어 자원에 대한 실측 확인입니다.

Metalium API 의 `compute_with_storage_grid_size` 를 호출한 결과, ttnn 이 실제 연산과 저장에 사용 가능한 core grid 는 5×4, 총 20 개로 확인되었습니다. Eagle-N 물리 그리드는 6×4=24 이지만 그 중 20 개가 사용 가능합니다.

우측 tensor 배치 시각화에서 확인할 수 있듯, ResNet50 의 대부분 layer 는 이 20 개 중 다시 14 개에서 16 개만 사용합니다. 즉 매 layer 실행 시점에 최소 4 개 core 가 유휴 상태입니다.

이 유휴 자원을 어떻게 활용할 것인가가 두 번째 연구 방향의 출발점입니다."

---

### 슬라이드 22 (Sub-device Partition — 개념)

"Tenstorrent 는 이 문제에 대해 abstraction 으로 SubDevice 를 제공합니다.

그림에서 보시는 것처럼 전체 코어 그리드를 복수 그룹으로 파티셔닝하고, 각 그룹이 독립된 local allocator 와 프로그램 실행 상태를 갖도록 격리하는 방식입니다. 우측의 allocator 도식은 파티션 도입 전후를 대비한 것입니다.

파티션 간 데이터 교환이 필요한 경우에는 Global Semaphore 와 Global Circular Buffer 두 종류의 cross-partition 통신 primitive 가 함께 제공됩니다.

이 abstraction 을 사용하면 메인 모델을 실행하는 파티션과, 유휴 코어에서 별도 워크로드를 실행하는 파티션을 하드웨어 수준에서 격리된 채로 동시에 운용할 수 있습니다."

이 와 같이 Sub-Device를 운영하였을때 생기는 추가적인 NoC Traffic을 추적하여 최적의 코어 배치, 할당 등을 설정할 필요성이 존재합니다.

---

### 슬라이드 23 (Sub-device Partition — 실측)

"실제로 TT-NN 계층에서 ResNet50 을 16 개 core 로 제한하여 실행한 로그입니다.

FC layer 를 제외한 모든 layer 에서 정상 실행이 확인되었습니다. FC layer 는 사용 가능한 core 수가 줄어들 때 memory 배치 최적화를 별도로 재구성해야 하는데, 이 조정은 상위 TT-NN 계층에서는 표현할 수 없고 하위 Metalium 계층의 커널 팩토리에서 해결해야 합니다.

따라서 다음 단계 작업은 Metalium 수준에서의 메모리 배치 최적화가 됩니다."

---

### 슬라이드 24 (향후 계획)

"마지막으로 향후 계획입니다. 지금까지의 학습과 프로파일링을 기반으로, 4주차에는 여유 코어 확보와 SubDevice 파티셔닝 실측 확대를 수행하고, 5·6주차 본격 연구 활동, 7·8주차 데이터 확보와 최적화로 이어질 예정입니다. 이상으로 발표를 마치겠습니다. 감사합니다."
