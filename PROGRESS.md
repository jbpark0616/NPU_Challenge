# NPU Challenge — 진행 상태

> 세션 시작 시 Claude가 먼저 읽고, 세션 끝날 때 함께 업데이트할 파일.
> 목적: 대화가 끊겨도 "지금 어디까지 왔고, 다음에 뭐 할지" 바로 복원.

---

## 프로젝트 전체 목표
`bos_model_copy`의 ResNet50 / YOLOv8s를 돌려보고, 이걸 **baseline으로 삼아 연구 주제를 잡는 것**.

## 진행 단계
- [x] Blackhole 아키텍처 공부
- [x] TT 소프트웨어 스택 공부
- [x] tt-npe 설치 및 프로파일링 워크플로우 정리 → `tutorials/guide/tt-npe_intro.md`
- [x] ResNet50 / YOLOv8s baseline 분석 — **PM(Performance Model) 기준**으로 완료 (tt-npe 실측은 커버리지 부족)
- [x] Roofline intensity 정공화 (bytes_total → bytes_dram) → [[roofline debt]] 문제 1 부분 해소
- [ ] 병목 지점 파악 → 연구 주제 확정
- [ ] 개선 아이디어 실험

## 지금 집중하고 있는 것
- **연구 방향 (2026-07-28 재정비)**: **Aging-aware core rotation** — SDC detection 에서 aging mitigation 으로 pivot. 6주차 미팅 발표 자료 완성 (2026-07-29 발표 예정). Rotation Tier 1/2/3 sim + 시각화 + PPT 대본 완료.
- **연구 방향 (이전 트랙, 유지)**: SDC detection on Eagle-N — Safety Island 가 커버 안 하는 correctness FT gap. `study_note/sdc_research_notes.md` + memory [[project-detection-only-scope]], [[project-eagle-n-safety-context]]

### 2026-07-28 세션 산출물 (aging_simulation)
- **Rotation sim 3-tier 완성** (`rotation_sim_tier{1,2,3}.py`) — Tier 3 결과: NIU worst-core -54%, SRAM -62% (permutation sim 기반)
- **알고리즘 시각화 3장** (`rotation_algorithm_figure_tier{1,2,3}.py`)
  - Tier 1: cyclic anchor rotation (6-step sweep) — "round-robin" 용어 → "cyclic anchor rotation" 으로 통일
  - Tier 2: anchor shift + reflection (4-mode TL/TR/BL/BR)
  - Tier 3: aging-aware greedy (argmin worst-after Vth, 상단 state 흐름 + 하단 8 candidate 썸네일 배치)
- **비교 표 3장 (통일된 신중한 스타일)** — dark header + 흰 셀 + 최소 색상
  - `mcast_comparison_table.py` — Height/Block/Width 4행 요약
  - `rotation_tier_summary_table.py` — 4 metric × 4 tier 개선률
  - `aging_ratio_table.py` — worst/best raw vs NBTI vs HCI 비율 (HCI 열만 강조)
- **시각화 그림** — `sharding_distribution.py` (stage별 op count 바 차트), `mcast_comparison_figure.py` (Height rect / Height L-shape / Block / Width 4-panel, CORE_SET_14 = 5×2+4×1 = 14 정정, y=0 top-left 라벨 수정)
- **발표 대본** — `aging_research/presentation_script_20260728.md` (24 슬라이드 × 30-60초 + 6개 예상 질문 답변)

### 다음 세션 (2026-07-29 발표 후) 착수 항목
- **T1 anchor shift 실측 검증** — 지금까지 permutation sim, 이제 실제 tt-metal 커널 패치 + Tracy 재측정으로 sim vs 실측 gap 확인
- **NoC0/1 채널 비대칭성 별도 분석** — Next Week Plan 슬라이드에 언급됨
- **AI accuracy 저하 정량화** — rotation 적용 후 실제 ResNet50 top-1/top-5 방어 정도
- **NoC 실측 파이프라인 완성 (2026-07-19)** — 5-stage 데이터 수집, aggregation 스크립트, tt-npe 시뮬 병행. Free-riding SubDevice thesis 근거 확보
- **시간축 파이프라인 완성 (2026-07-20)** — `analysis/outputs/time_window/` : ops.csv/timeline.csv/link_timeseries.csv/niu_timeseries.csv + 4 figures + summary.md. 관측 가능 창 **58.1% (idle≥4 & NOC0=0), 62.7% post-warmup**
- **SubDevice API 실증 시작 (2026-07-20)** — Step 1 성공, Step 2 진행 중. 저수준 tt_metal Program API 로 전환 결정
- **Core/Shard 이론 분석 (2026-07-21)** — `conv2d_utils.cpp` 정독. Auto-shard heuristic = L1 min. Halo 오버헤드가 코어 수에 비례해 증가 → CNN 이 20 코어 못 쓰는 아키텍처적 근거. `analysis/core_shard_study/` 폴더 생성.
- **다음**: (1) batch=1 baseline 재수집 (차량용 표준). (2) 저수준 verifier microkernel

## 활성 블로커 — 해결됨. 다음 단계로

### 이전 블로커 진단 정정
`profile_this.py deprecated` 경고는 **red herring**. 실제 원인:
1. `--collect-noc-traces` 켜면 tt-npe inline 분석 트리거
2. tt-npe 가 multicast op assertion 실패 (`MulticastCoordSet: start coord must be top-left of end coord`)
3. `process_ops_logs.py:436` 에서 `getDatapointByID(op_id)` — tt-metal ↔ tt-npe API skew (1 arg vs 3 arg 요구) → TypeError → 전체 report abort
4. 팀원은 `--collect-noc-traces` 안 켜서 이 경로 자체를 안 밟음 → CSV 정상

### 우회 방안 (확정)
**두 번의 별도 pytest 실행 + 오프라인 조인** (상세: `study_note/stage_split_profiling_workflow.md`)
- **Run A (compute pass)**: `python -m tracy -r -p -v -m pytest ...` — `--collect-noc-traces` 없음 → ops_perf_results.csv 정상 생성
- **Run B (NoC pass)**: `profile_this.py --collect-noc-traces` — noc_trace_ID*.json 확보, report_*/ 는 예상대로 크래시 (무시)
- **Run C (full calibration)**: `RESNET_STAGE` unset → 전체 모델 1회, stage-split 데이터의 캘리브레이션용
- **조인 키**: `run_host_id` (1024, 2048, ..., N × 1024). 같은 STAGE 로 두 번 실행하면 결정적으로 일치
- 총 11회 실행: Compute 5 + NoC 5 + Full 1. 사용자가 컨테이너에서 수동 실행

## Baseline 분석 산출물 위치
각 모델별 `study` 폴더에 CSV → plot / study note 정리:
- **ResNet50**: `bos_model_copy/resnet50/resnet_note/`
  - CSV: `ops_perf_results_2026_07_08_05_15_43.csv`
  - Plots: `plots/01_top_ops.png` ~ `08_achieved_pct.png`, `table_e2e.md` / `table_optype.md` / `table_hw_constants.md`
  - **[NEW] intensity 정공화 plot**: `plots_dram_intensity/07_real_roofline.png` — DRAM I/O 기준. 모든 점 ceiling 아래, matmul family(conv2d + matmul)만 남음
  - 분석 스크립트: `roofline_analysis.py` (수정됨: line ~340 `arith_intensity = flops / bytes_dram`)
- **YOLOv8s**: `bos_model_copy/yolov8s/yolo_study/`
  - CSV: `ops_perf_results_2026_07_09_06_40_40.csv` (+ 320x320 variant)
  - Plots: 같은 8종 세트 + tables
  - **[NEW] intensity 정공화 plot**: `plots_dram_intensity/07_real_roofline.png` — 대부분 개선, rogue 1점 (intensity ~2×10^4, ~150 TFLOPS conv2d) 남음
  - 분석 스크립트: `roofline_analysis.py` (수정됨)

**중요**: baseline 분석은 **PM(Performance Model) 기준**. tt-npe 실측이 커버리지 부족(op 실링, multicast assertion 등)으로 PM에 의존. [[roofline debt]]와 함께 재검증 대상. **오늘 intensity 정공화로 문제 1 (BW ceiling 위 점) 부분 해소**.

## Stage-split 데이터
- **로컬 확보**: `bos_model_copy/resnet50_jbpark/stage_split_20260712_173806/stage_0/logs/` — stem STAGE=0 실행 결과
  - `noc_trace_ID{1..7}_merged.json` (op별 raw NoC 트랜잭션)
  - `profile_log_device.csv` (5.3 MB, core별 RISC별 zone timing — CHIP_FREQ 650 MHz 확인)
  - `tracy_ops_data.csv`, `tracy_ops_times.csv`, `tracy_profile_log_host.tracy`
- **stage 1~4 미확보** — 아직 실행 안 함
- **검증 스크립트**: `stage_split_20260712_173806/verify_dram_bw.py` (오늘 작성) — noc_trace JSON 파싱 → 각 op 총 NoC bytes + tracy_ops_data 조인 → op_name 매핑
  - stem 실행 확인됨: 7 op (Halo/Move/Conv2d/Halo/Pool2D/Reshard/Tilize) 매핑 정확
  - **한계**: DRAM 좌표 식별은 heuristic (top-3 sourced coord). stem 은 순수 core-to-core 트래픽이 지배적이라 좌표 분리 잘 안 됨. Layer 3/4 데이터 나오면 개선 예상

## 최근 결정·전환점
- **2026-07-18**:
  - **연구 방향 SDC detection 으로 전환** — 팀 계획서 (SubDevice + NOC ATT 기반 재컴파일-없는 FT) 조사 결과 Blackhole 에 NOC ATT 없음 (Quasar 세대 기능) → 교수님 피드백 "detection 만 해도 의미 있는 연구" 활용 → SDC 방향 확정
  - **Safety Island 재해석** — liveness vs correctness 축 분리. Safety Island 는 liveness 만, correctness (SDC) 는 어느 계층도 커버 안 함 (자세한 근거: `study_note/safety_island_study.md` §7)
  - **tt-metal 팩트 체크** — 오늘 grep 결과:
    - **양성 근거**: NoC 헤더 SEC-DED, 라우터·NIU 내부 메모리 파리티, Ethernet CRC, RISC-V I-cache ECC
    - **음성 근거 (SW 인터페이스 부재)**: FPU/SFPU 결과 ECC, L1 SRAM ECC, NoC payload ECC — enable/counter/interrupt 3세트 모두 grep 실패
  - **Three SDC surfaces 도출**: compute-time (FPU/SFPU), storage-time (L1), transport-time (NoC payload)
  - **ECC_CTRL FORCE 비트** 확인 — `noc_parameters.h` `[5:3] = force ECC error` — bit 3 (MEM_PARITY), 4 (HEADER_1B), 5 (HEADER_2B). Fault injection lever 는 있지만 이건 헤더/NIU 만. L1/FPU 는 SW fault injection (커널 논리 오류) 필요
  - **조교님 sweep 방법론 확정**: SubDevice manager 를 매 실험마다 다른 위치에 로드 → 위치 sweep 데이터 수집. tt-metal 표준 패턴 (`tests/ttnn/unit_tests/base_functionality/test_sub_device.py`) 활용
  - **새 자료 만듦**: `study_note/sdc_research_notes.md` (문제정의·팩트체크·3 SDC surface·실험 로드맵 Phase 1~4)
  - **인식적 정직성 강화**: 팩트 체크 표에서 "없음" → "근거 없음 (SW 인터페이스 부재, 실측·벤더 문서 확인 전엔 확정 X)" 로 표현 완화
  - **BOS 벤더 문서 (`developers.bos-semi.com`) 접근 시도 → 403 Forbidden**. 사용자 로그인 필요할 것으로 추정. 내일 사용자가 필요 섹션 붙여넣기 예정
  - **미완**: 보드 뻑감 → Phase 1 실측 못 함. 커널 뼈대 미작성. BOS 문서 미확인
- **2026-07-14**:
  - **Roofline intensity fix** — `bytes_total` → `bytes_dram` 로 두 파일 모두 수정. 결과: 대부분 점이 ceiling 아래. Peak 값은 안 건드림 (여전히 64 TOPS / 150 GB/s 스펙 계산치)
  - 정공화 부작용: conv2d + matmul 만 남음. **해석**: 나머지 op (halo/pool/unary/reshard) 이 L1-resident → BOS 코드의 activation L1 관리 우수 증거. YoloV8s "Matmul" op 은 1×1 conv 가 matmul 로 auto-routing 된 것 (CSV attribute 확인, shape 6400×C₁ × C₁×C₂)
  - **PRoof (ICPP 2024)** 논문 발견 — DNN hierarchical roofline. NPU 명시 지원. 우리 방향의 결정적 선행 연구
  - **Simba (MICRO 2019 / CACM 2021)** 논문 검토 — MCM 계층 인터커넥트 + layer 별 BW 민감도. 우리 Big1 (예측 성능 모델) 방향의 motivation 근거
  - **여유 core 학술 용어 확정**: 주 = **spatial partition** (multi-tenancy 문헌 표준), 구현 = SubDevice, 안전 = FFI (Freedom from Interference)
  - **stage-split 워크플로우 확정**: compute/NoC pass 분리. 상세는 `study_note/stage_split_profiling_workflow.md` 저장
- **2026-07-13 (밤)**: ResNet50 stage-split 구현 및 첫 실행 (내용은 세션 로그)
- **2026-07-10 미팅** (요약, 상세는 이전 버전 참조)
  - subdevice 여유 코어 목표: 4 → **8**. 메인 모델: 16 → **12**
  - Roofline 재검증 필요 (peak / bound 경향성). 재검증 전 발표 인용 금지 → 오늘 부분 해소
  - 12 core 구성 자동차 스펙 대비 검증 요청
  - Fault-tolerant subdevice 구현 방식 조사 요청 (DRIVE Orin 벤치마킹)

## 미팅 발표 준비 상태 (2026-07-15 예정)

### 최종 슬라이드 (24장, PDF)
`study_note/Never_Pause_Unit_PROGRESS.pdf` — 팀명 "Never Pause Unit".
구조:
```
1. 표지
2. Timeline (Week 1-9)
[Week 1] 3. divider / 4. Architecture 학습 / 5. NoC / 6. L1+CB / 7. Tensix Engine / 8. DRAM
[Week 2] 9. divider / 10. Tensix core Kernel / 11. Baseline AI-model / 12. SW 스택 / 13. Profiling Tool
[Week 3] 14. divider (성능 최적화 · SubDevice 두 방향)
         15. ResNet50 wall time / 16. YOLOv8s wall time / 17. Roofline / 18. Roofline 개선 (hierarchical)
         19. TT-npe 환경 구축 문제 3종 / 20. 코드 split 구현
         21. SubDevice 유휴 코어 확인 / 22. SubDevice 개념 / 23. SubDevice 실측
24. 향후 계획
```

### 대본
`study_note/presentation_script.md` — Week 1·2 는 슬라이드당 한 문장, Week 3 부터 상세 (사용자 직접 수정 중).

### 미완 (내일 마감)
- 발표 리허설

## 다음에 할 일

### 내일 (2026-07-22) — batch=1 baseline + verifier 커널

**차량용은 batch=1 표준.** Baseline (batch=4) 을 batch=1 로 재수집.

- [ ] `bos_model_copy/resnet50/run_resnet50.py -b 1` 로 baseline CSV 재수집
- [ ] `analysis/core_shard_study/compare_csv.py` 확장:
  - PM IDEAL, PM COMPUTE, PM BANDWIDTH, PM FPU UTIL, NPE CONG IMPACT 컬럼 활용
  - Op 별 GLOBAL CALL COUNT 매치해서 same-op 직접 대조 (batch=1 vs batch=1)
  - Halo 오버헤드 실측 (Halo bytes/cycles 대 Conv bytes/cycles)
  - CORE COUNT 히스토그램 나란히
- [ ] Halo overhead 이론 검증 실측 — CSV 의 Halo op cycles 를 core count 별로 그룹핑 → 이론 예측 (14→50%, 20→67%) 과 비교

### 이후 — 저수준 Verifier 커널
어제 밤 SubDevice API 실증에서 얻은 결론 반영. **ttnn.add 같은 built-in op 는 verifier 시나리오에 안 맞음** — 프로그램-SubDevice 자동 결부 + 캐시 오염 이슈.

- [ ] **저수준 tt_metal Program API 로 verifier 프로토타입**
  - 커스텀 kernel.cpp 작성 (예: L1 checksum 계산 → 지정 주소에 결과 쓰기)
  - `CreateProgram()` + `CreateKernel(program, "kernel.cpp", core_range_set)` 로 우리 4 코어 (ttnn 좌표 (0,3)-(3,3)) 에 정확히 배치
  - `EnqueueProgram()` + sync
- [ ] **Global semaphore 도입** — verifier ↔ primary 완료 신호 (SubDevices.md §2)
- [ ] **Global CB 도입** — primary L1 → verifier 데이터 이동 (SubDevices.md §3)
- [ ] **Wall time 실측** — Verifier ON/OFF 로 ResNet-50 stem 오버헤드 측정
- [ ] **Phase 1 열린 질문 4개 결정** (`sdc_research_notes.md` §6): tile 데이터 타입 (int32 vs bf16), 주입 확률 (결정론 vs Poisson), baseline 비교, SubDevice 배치 (1+1 vs 12+8)

### 이후 — SDC Phase 진행
- [ ] Phase 1 실측 — L1 bit-flip 주입 후 카운터 안 오르는지 확인 (negative evidence 강화)
- [ ] Phase 2 sweep — 20 코어 × Verifier 위치 후보 조합
- [ ] Phase 3 — 3 SDC 종류 비교 (compute/storage/transport)
- [ ] Phase 4 — 실제 워크로드 (matmul, ResNet50 conv layer) 상향

### Big1 방향 관련 잔여 (후순위·후일 재활용용)
- [ ] **컨테이너에서 stage-split 데이터 수집 (11회)**: Compute pass 5회 + NoC pass 5회 + Full calibration 1회
- [ ] `verify_dram_bw.py` 로 stage 1~4 데이터 검증 (DRAM 좌표 정확 식별 기대)
- [ ] YoloV8s rogue point (intensity ~2×10^4) 조사

### 미팅 후 (Phase 2 — 근본 개선)
- [ ] PRoof 논문 정독 → 우리 스택 이식 방법론
- [ ] Multi-level roofline (L1 slope 추가) 구현
  - L1 peak BW: 20 core × 2 KB/cycle × 650 MHz ≈ 26 TB/s aggregate 예상
- [ ] `bytes_dram` 실측치 대체 (verify_dram_bw.py 데이터)
- [ ] tt-npe standalone 실행 → tt-npe assumed peak back-compute
- [ ] Peak BW 실측 검증 — GEMM microbench + DRAM saturation microbench (tt-metal tech_reports 참조)
- [ ] Achieved 기반 bound 분류 함수 추가 (PM 대비 비교)

### 미팅 후 (Phase 3 — 자동차 & 연구 방향)
- [ ] 12 core 구성 실측 → 자동차 스펙 대비 위치 (`study_note/automotive_specs.md` 비교표 채우기)
- [ ] Spatial partition (SubDevice) 실험 설계 — concurrent workload 후보 결정
- [ ] Simba 논문의 layer-sensitivity 분석 방법 우리 데이터에 이식

## 열린 질문·블로커
- **Multi-level roofline 정확한 L1 peak BW 값** — 20 core × 2 KB/cycle × 650 MHz 가 aggregate 인지 per-slot 인지 재확인
- **tt-npe multicast coord assertion 우회** — 우리 `verify_dram_bw.py` 로 로컬 대체 진행 중
- **Eagle-N 전용 device model 없음** → 절대 BW 수치는 발표 인용 여전히 조심
- **Fault-tolerance vs Concurrent workload 프레이밍** — Spatial partition 위에 뭘 돌릴지 결정 남음
- **차량용 FPS/latency 기준 수치** — Tesla FSD, Mobileye, NVIDIA DRIVE 공표 스펙 비교표는 있음 (`automotive_specs.md`), 12 core 실측 결과 대조 남음

## 참고 위치
- 메인 모델: `bos_model_copy/resnet50/`, `bos_model_copy/yolov8s/`
- 프로파일링 가이드: `tutorials/guide/tt-npe_intro.md`, `study_note/stage_split_profiling_workflow.md`
- 발표 자료: `study_note/Never_Pause_Unit_PROGRESS.pdf` (슬라이드), `study_note/presentation_script.md` (대본)
- 아키텍처 학습 소스:
  - **내 공부 노트 (Blackhole 정리)**: `study_note/Tenstorrent_Architecture_note.md`
  - Eagle-N 아키텍처 그림: `study_note/eagleN_Architecture.png`, `study_note/compute_core_size.png`
  - 상류 자료: `tt-metal/METALIUM_GUIDE.md`, `tt-metal/tech_reports/`, `tt-metal/docs/source/tech_reports/`
- 발표용 시각:
  - `tt-metal/tt_metal/tt-llk/docs/llk/l2/images/tensix_core.png` (총괄)
  - `tt-metal/docs/source/common/images/MfB-Fig5.png` (NoC 흐름)
  - `tt-metal/tech_reports/memory/images/sram-bank-cbs.png` (L1+CB)
  - `tt-metal/tt_metal/tt-llk/docs/llk/l2/images/tensix_engine.png` (Compute Engine)
  - `tt-metal/docs/source/common/images/MfB-Fig4.png` (DRAM)
- 참고 논문 (연구 방향):
  - **PRoof** (ICPP 2024) — DNN hierarchical roofline for CPU/GPU/**NPU** — 우리 방향 결정적 선행 연구
  - **Simba** (MICRO 2019 / CACM 2021) — MCM 계층 인터커넥트, layer-sensitivity — Big1 motivation
  - **Yang et al.** (2020) — Hierarchical Roofline for GPUs, arXiv:2009.02449
  - **Ilic et al.** (2014) — Cache-Aware Roofline — IEEE CAL 원조
  - **Williams et al.** (2009) — Roofline 원 논문 — CACM
  - **Jouppi et al.** (2017) — TPU ISCA — spatial NPU roofline 사례 (단, SRAM slope 없음)
- NPU 커맨드: `study_note/npu_command_set.md`, `bos_model_copy/npu_command_set.md`
- 환경: bos_host docker 진입 후 `source env_set.sh` 필수, 이후 `source /tt-npe/ENV_SETUP`

---

## 세션 로그
- **2026-07-21**: Core/Shard 이론 분석 + CSV 비교 초안
  - **`conv2d_utils.cpp` 4 개 핵심 함수 완전 이해**:
    - `find_closest_largest_divisor*` (line 54-90): padding-tolerant 약수 greedy
    - `determine_parallel_config` (line 169-237): sharding 별 grid 결정. BLOCK 은 각 축 상한이 grid.x/grid.y (4~5) 라 최대 20 못 씀
    - `calculate_L1_usage_for_conv_op` (line 948-1115): 코어당 L1 = CB + tensor + **halo output** (halo 명시적 modeling)
    - `determine_conv_config_for_auto_shard` (line 1117-1210): **auto-shard 실체 = `total_size (L1 min)` 최소인 후보 선택**. Wall time 아님.
  - **Halo 개념 확정**: conv 이 인접 픽셀 필요 → sharded 시 shard 경계에서 옆 코어 데이터 read = halo. 코어 늘림 → shard 얇아짐 → halo/shard 비율 커짐 (14 코어 50% vs 20 코어 67%). **CNN 이 20 코어 다 못 쓰는 아키텍처적 진짜 이유.**
  - **팀원 exp3 CSV 발견/분석** (`ops_perf_results_resnet50_exp3_2026_07_20_14_20_32.csv`):
    - Batch=1, 1 iteration, 115 op, 2.59 ms
    - vs baseline (batch=4, ~10 iter, 1260 op, 39.06 ms) 직접 비교 불가
    - **주의**: baseline 도 20 코어 op 이 25% 존재 → BOS `CORE_SET_14/16` 은 reshard 지점만 강제, 개별 op auto-shard
    - EXP3 는 batch=1 → auto-picker 가 작은 grid (4~9) 선택하는 op 도 등장. "16 코어 강제" 프레임 재검토 필요
  - **차량용 NPU 표준 = batch=1 확인** → memory 저장 [[project-automotive-batch]]
  - **미완 (내일)**: batch=1 baseline 재수집 후 exp3 와 정합 비교. Halo 오버헤드 실측치. 저수준 verifier microkernel.
  - **산출**: `bos_model_copy/resnet50_jbpark/analysis/core_shard_study/` — `compare_csv.py`, `README.md`

- **2026-07-20**: 시간축 파이프라인 완성 + SubDevice API 실증 시작
  - **시간축 데이터 지시서** (`analysis/outputs/time_window/시간축_데이터_추출_지시서.md`) 완전 이행:
    - **Op 겹침 없음 확인** (`_check_op_overlap.py`) — `profile_log_device.csv` 로 5 stage 검증, 순차 실행 확정
    - **Case A (sequential) 처리** — op별 bytes ÷ duration = rate. 파이프라인 단순화
    - Bin size: min(1μs, shortest_op/10) = 0.067 μs → 61,102 bins
  - **산출물** (전부 `outputs/time_window/`):
    - `ops.csv` (140 ops, per-op rate 통계)
    - `timeline.csv` (61k rows, real+total NOC0/1 rate, max link rate, active core count)
    - `link_timeseries.csv`, `niu_timeseries.csv`
    - `fig_timeline.png` (4-panel with 관측 가능 창 녹색 음영), `fig_link_heatmap.png` (NOC0 top15/NOC1 top40, 방향별 색 라벨), `fig_cdf.png` (log x + in-line labels), `fig_op_scatter.png`
    - `summary.md` (Section 1~8, warmup 제외 통계 포함)
  - **Priority 0 정합성 fix 성공**: total 을 link-byte 합으로 재정의 → NOC0 max==0 pct = total==0 pct = 95.61% 일치 검증. 9132 same-cell (src=dst) READ_SET_STATE 등 self-referencing txn 이 원인이었음.
  - **핵심 지표** (thesis 강화):
    - NOC 0 : NOC 1 = **3.9% : 96.1%** (24× 지배)
    - **NOC 0 완전 유휴 95.6% of run**
    - **관측 가능 창 (여유≥4 & NOC0 유휴) = 58.1%** (post-warmup **62.7%**)
    - 관측 가능 창 (여유≥8) = 26.1% (post-warmup 28.1%)
    - 활성 코어 mean 14.5 / median 16
    - Hot 링크 이동 안 함 (Row 1 WEST cluster 고정 hotspot) → 정적 배치 가능
  - **3-way 정합성 검증 스크립트** `_verify_consistency.py` — summary.md ↔ timeline.csv ↔ CDF 자동 대조. 모든 지표 반올림 오차 (<0.05%p) 내 일치
  - **좌표계 매핑 확정** — 우리 raw 관찰 (6×5) ↔ ttnn user grid (5×4) 명확:
    - x: 그대로 (0..4)
    - y: 우리 (0,1,2,4) → ttnn (0,1,2,3) — 우리 y=3 memory row 는 ttnn 이 숨김, 우리 y=4 = ttnn y=3
    - **top idle 후보 우리 (y=4, x=0..3) = ttnn (x=0..3, y=3)**
  - **SubDevice API 실증**:
    - `subdevice_test.md` (컨테이너 로그 저장) 에 전 과정 기록
    - 팀원(추정)이 남긴 `/tt-metal/inspect_subdevice_async_api.py` + `.log` 로 API 목록 확인:
      - ✅ `SubDevice`, `SubDeviceId`, `create_sub_device_manager`, `load_sub_device_manager`, `synchronize_device(sub_device_ids=)`, `record_event`, `wait_for_event`
      - ❌ `set_sub_device_stall_group`, `reset_sub_device_stall_group`, `create_event`
    - **표준 test `pytest tests/ttnn/unit_tests/test_sub_device.py` 실패** — 하지만 에러 메시지가 결정적 정보 제공:
      - `device grid [(x=0,y=0) - (x=4,y=3)]` = 20 compute cores 확증
      - 표준 test 가 (x=4, y=4) 를 예약하려다 격자 밖이라 실패한 것
    - **Step 1 성공** (`hello_kernel.py`) — SubDevice 로드 + `ttnn.add` 실행 + 결과 검증. 우리 4 코어 (ttnn (0,3)-(3,3)) 에서 커널 진짜 실행됨.
    - **Step 2 여러 번 실패 → 원인 확정**: `SubDevices.md` 확인 결과 —
      1. "Programs cannot be rerun on a different sub-device-manager configuration" — baseline compile 후 SubDevice 로드하면 프로그램 캐시 hit 되어 mismatch
      2. "Programs can only span one sub-device" — 2 SubDevice 로드시 자동 targeting 실패 (stall_group API 없어서 default 지정 불가)
      3. ttnn.add 는 임의 CoreRangeSet 에 자동 shard 안 함 — kernel group ≠ SubDevice cores 로 검증 실패
    - **결론**: **built-in ttnn op 는 verifier 시나리오에 안 맞음.** 저수준 `tt_metal Program` + `CreateKernel(program, "kernel.cpp", core_range_set)` 로 접근해야 함. Global semaphore/CB (SubDevices.md §2, §3) 도 필요.
  - **미완 (내일)**: 저수준 tt_metal verifier 커널 프로토타입 (커스텀 kernel.cpp 부터)

- **2026-07-19**: NoC 실측 파이프라인 완성 + SubDevice congestion thesis 근거 확보
  - **Stage-split 5개 다 수집 완료** — stage_0 (7 op, 재사용), stage_1 (18), stage_2 (29), stage_3 (54), stage_4 (32). 총 140 op, real 99 + boundary 41
    - 각 timestamp 폴더: `stage_split_20260712_173806` (stage_0), `20260718_143633` (stage_1), `20260718_144746` (stage_2), `20260718_151812` (stage_3), `20260718_155405` (stage_4)
  - **_synth_layer_input 3가지 fix** 적용해서 stage 1~4 실행 가능해짐:
    - (a) `dtype=dtype` → `dtype=ttnn.bfloat16` (bfloat8_b + ROW_MAJOR 불가 회피)
    - (b) stage 3/4 는 spatial 14×32 배수 아니라 DRAM interleaved 로 폴백
    - (c) `_forward_layer` shard cache 코드에 `shard_spec is not None` guard
  - **Fork ↔ 컨테이너 sync 이슈 재확인** — `bos_model_copy` 로컬 편집만 하고 `docker cp` 잊는 실수 반복 → memory 저장 ([[project-fork-container-sync]])
  - **Boundary artifact 정책 확정** — stage 첫 1~2 op (Interleaved→Sharded/Tilize/Reshard/Move) 은 synth setup, aggregation 에서 필터. bytes 18.7%, cycles 20.6% 손실 감내 ([[project-stage-split-boundary-artifact]])
  - **`analysis/` 폴더 생성** — `bos_model_copy/resnet50_jbpark/analysis/`:
    - `stage_paths.py`, `boundary_filter.py`, `aggregate_per_core.py`, `time_slice_heatmaps.py`, `README.md`
    - `outputs/` 하위에 `per_core.csv`, `per_op.csv`, `matrix.csv`, `per_core_heatmap.png`, `per_stage_heatmaps.png`, `cumulative_progression.png`, `per_op_top20_heatmaps.png`, `summary.txt`
  - **좌표계 파악** — Raw NoC trace 좌표는 6×5 translated 공간 (Blackhole 17×12 물리 subset). 29 좌표 관찰, (5,1) 만 미관찰. **(0..4)×(1..4)=20 좌표가 usable compute 와 정확히 대응**. Y=0 은 host-facing (LPDDR5 3 채널 + PCIe + ARC) ([[project-eagle-n-noc-coords]])
  - **tt-npe standalone 시뮬 5-stage 실행** — `npe_analyze_noc_trace_dir.py` 로 op별 NoC/DRAM util % + Cong Impact %. 결과:
    - Cycle-weighted DRAM util: stem 8.1%, layer1 1.2%, layer2 3.3%, layer3 3.8%, layer4 2.1%
    - **Layer3/4 (전체 68%) NoC 완전 idle** (Cong Impact 0%)
    - **Stem/layer2 short burst (전체 5%) 에서만 real congestion** — Halo 21%, layer2 op 6~7%
  - **신뢰도 매트릭스 확정** — 실측 vs tt-npe 시뮬(Blackhole ref) 분리:
    - **100% 실측**: raw JSON bytes/cycles/coords, 우리 aggregation 결과
    - **⚠️ Blackhole ref 기반**: tt-npe Util/Cong Impact 절대값. DRAM util 은 factor ~4.3배 정정 필요 (Blackhole 24 DRAM ↔ Eagle-N 3 LPDDR5)
    - **❌ 완전 아티팩트**: ETH BW util (P100 은 ETH disabled)
  - **SubDevice free-riding thesis 근거 확보** — wall time 최악 1-3%, 평균 <0.5% 예상. Layer3/4 시간대 (러너 68%) 완전 안전
  - **논문 각도 강화**: (1) "Eagle-N specific device model 부재로 인한 profiling 한계" 자체가 contribution 소재. (2) Spatial vs temporal partition tradeoff 를 discussion 에서. (3) Idle silicon 재활용 = SDC always-on 실용성 argument
  - **미완**: verifier microkernel 프로토타입, power 실측, tt-metal HAL coord translation 정확 매핑 (task #9)

- **2026-07-14**: 대량 진전.
  - **Stage-split 블로커 진단 정정**: `profile_this.py deprecated` 는 red herring. 실제 원인 = tt-metal(`process_ops_logs.py:436`) ↔ tt-npe API skew (`getDatapointByID` 1 arg vs 3 arg) + multicast assertion. 팀원 방식 (no `--collect-noc-traces`) 은 이 경로 자체를 안 밟아서 성공. → 두 번 나눠 실행 (compute pass + NoC pass) + 오프라인 조인 워크플로우로 확정, `study_note/stage_split_profiling_workflow.md` 저장.
  - **Roofline intensity 정공화**: `arith_intensity = flops / bytes_dram` (이전엔 bytes_total). YoloV8s + ResNet50 두 스크립트 모두 수정. 새 plot `plots_dram_intensity/` 폴더 생성. 결과: 대부분 점 ceiling 아래 (문제 1 해소). Peak 값 (150 GB/s / 64 TOPS) 은 안 건드림 — 여전히 debt.
  - **[[roofline debt]] 문제 1 부분 해소**: 원인이 peak 오계산이 아니라 **intensity 정의 (bytes_total 사용) 가 buggy** 였음. L1 재사용 op 이 낮은 intensity 로 몰려 낮은 ceiling 을 뚫었던 것.
  - **정공화 부작용 관찰**: conv2d + matmul 만 남음. 나머지 op (halo/pool/unary/reshard 등) 은 L1-resident → **BOS activation L1 관리의 최적화 증거**. YoloV8s "Matmul" = 1×1 conv 가 matmul 로 auto-routing (shape 6400×C×C 확인).
  - **verify_dram_bw.py 작성**: `stage_split_.../verify_dram_bw.py` — noc_trace_ID*_merged.json 파싱 → op별 NoC bytes + tracy_ops_data.csv 로 op_name 매핑. Stem 7 op 매핑 정확 확인. DRAM 좌표 식별은 heuristic 이라 layer 3/4 데이터 필요.
  - **논문 리서치 (연구 방향)**:
    - **PRoof (ICPP 2024)** — DNN hierarchical roofline for NPU. 우리 방향 결정적 선행 연구
    - **Simba (MICRO 2019 / CACM 2021)** — MCM 계층 + layer-sensitivity. Big1 motivation
    - **Yang 2020, Ilic 2014, Williams 2009, Jouppi 2017** — 부속 근거
  - **미팅 발표 준비 대량 진행**: 목차 확정 (학습 → 실측/방향 → 다음), 슬라이드 11장 계획, 각 슬라이드 시각/캡션 확정. 특히:
    - 슬라이드 6 (SW 스택 & 모델): `enable_act_double_buffer` 3-tier 캡처 결정 (ResNet50 Config.py + YoloV8s configs/utils.py + tt-metal conv2d_op_program_factory_common.cpp CB 로직)
    - 슬라이드 8 (Baseline 분석): "Classic Roofline" 명칭, 두 plot 비교 + 캡션 A₁
  - **여유 core 학술 용어**: 주 = **spatial partition** (Choi et al. HPDC 2022 등), 구현 = SubDevice, 안전 = FFI. 이걸로 슬라이드 9 (연구 방향) 씨앗
  - **미팅 일정**: 원래 화요일 → 수요일로 미룸. 남은 작업 = 슬라이드 실 제작 + 리허설
- **2026-07-13 (밤)**: ResNet50 stage-split 구현 및 첫 실행. Fork `resnet50_jbpark/` 로컬+컨테이너 세팅, `ttnn_functional_resnet50.py` 에 RESNET_STAGE 게이트 + fake input 합성 헬퍼 추가, `test_ttnn_functional_resnet50.py` 에 validate() 가드, `run_stage_split.sh` 신규. 변경 상세는 `bos_model_copy/resnet50_jbpark/CHANGES.md`. STAGE=0 첫 실행에서 `logs/` 생성, `report_*/` 미생성. 원인 진단은 다음 세션 (2026-07-14) 에서 완료.
- **2026-07-13**: 연구 주제 방향성 논의 심화. Big1(예측 성능 모델), Big2(taxonomy), Big3(새 abstraction). **데이터 더 뽑기로**. 상세는 memory `project-research-topic-deliberation`.
- **2026-07-12**: Safety Island 스터디 → `study_note/safety_island_study.md`. Eagle-N ASIL-B, NPU 는 인식 담당 / Safety Island 는 SoC FT.
- **2026-07-12**: 차량용 스펙 리서치 완료 → `study_note/automotive_specs.md`. 기준: ≥30 FPS + ≤30 ms.
- **2026-07-11~12**: tt-npe 설치, 프로파일링 워크플로우 정리 (`tt-npe_intro.md`).
- **2026-07-04~09**: ResNet50 / YOLOv8s baseline 프로파일링 CSV 수집 및 plot 8종 + tables 생성.
- **2026-07-12**: PROGRESS.md 초기 작성.


