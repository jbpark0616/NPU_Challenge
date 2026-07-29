# Stage-split Profiling Workflow (Compute + NoC 분리 수집)

> 목적: `resnet50_jbpark/` fork 를 활용해 ResNet50 을 layer 단위로 분할 프로파일링하되, tt-metal ↔ tt-npe 툴체인의 미숙성으로 인해 **compute pass 와 NoC pass 를 별도 실행**하고 오프라인에서 조인하는 실행 흐름.
> 관련: `PROGRESS.md`, `bos_model_copy/resnet50_jbpark/CHANGES.md`, `tutorials/guide/tt-npe_intro.md`.

날짜: 2026-07-13 정리

---

## TL;DR

- **원인**: `--collect-noc-traces` 를 켜면 tt-npe 의 inline 분석이 multicast op 에서 assertion 실패 → `process_ops_logs.py:436` 에서 `getDatapointByID` TypeError → **`report_*/` 폴더 자체가 생성 안 됨**. tt-metal ↔ tt-npe API 시그니처 drift 및 cascade fail 설계 결함.
- **우회**: Tracy 는 두 개의 독립 수집 경로 (op/kernel 프로파일, NoC 트레이스) 를 가짐. 이걸 **두 번의 별도 실행으로 분리**해서 각각 성공한 결과물만 얻은 뒤 로컬에서 `run_host_id` 기준으로 조인.
- **총 11회 pytest 실행** (Compute 5 + NoC 5 + Full calibration 1) → 오프라인 조인 스크립트 → 통합 DataFrame.
- **신뢰도**: Layer 내 per-op 지표는 원본 모델과 사실상 동일, layer 경계·end-to-end timeline 은 caveat 필요.

---

## 왜 두 번 나눠 뽑아야 하나

Tracy 내부 흐름을 보면:

```
   pytest 실행 → device profiling
        │
   ┌────┴────────────────────────┐
   ▼                             ▼
[기본 - 항상 켜짐]         [옵션 - --collect-noc-traces]
 op/kernel timing            NoC 트랜잭션 로깅
   │                           │
   ▼                           ▼
tracy_ops_data.csv       noc_trace_ID*_merged.json (raw)
tracy_ops_times.csv           │
profile_log_device.csv        ▼
   │                    tt-npe inline 분석  ← ★ 여기서 crash
   ▼                    (multicast assertion → TypeError)
process_ops              전체 report 생성 abort
   │
   ▼
report_*/ops_perf_results.csv   ← ★ 위 crash 로 인해 미생성
```

**노란 경로(NoC 분석)가 죽으면서 초록 경로(report CSV 생성)까지 같이 죽는 게 문제.** 두 경로는 원래 독립적이므로 실행을 분리하면 각각 성공.

---

## 실행 커맨드 (컨테이너 내부)

**세션 시작 시 (1회)**:
```bash
source /tt-metal/env_set.sh
source /tt-npe/ENV_SETUP
```

### ① Compute pass (Run A) — stage 별 5회

`--collect-noc-traces` 없음 → tt-npe 크래시 회피 → **`report_*/` 정상 생성** → `ops_perf_results_*.csv` 확보.

```bash
RESNET_STAGE=0 python -m tracy -r -p -v -m pytest \
    models/bos_model/resnet50_jbpark/tests/test_ttnn_functional_resnet50.py -k batch_1
```

`RESNET_STAGE=0..4` 로 5회 반복.

**얻는 것**: op 별 kernel duration, math fidelity, dtype, PM_COMPUTE / PM_BANDWIDTH bound, `run_host_id` (조인 키).

### ② NoC pass (Run B) — stage 별 5회 (stem 은 이미 확보됨)

`--collect-noc-traces` 켬 → NoC JSON 확보 → `report_*/` 는 예상대로 크래시.

```bash
RESNET_STAGE=0 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 \
    tools/tracy/profile_this.py --collect-noc-traces \
    -c "pytest models/bos_model/resnet50_jbpark/tests/test_ttnn_functional_resnet50.py -k batch_1"
```

**얻는 것**: `noc_trace_ID*_merged.json` per op — 모든 NoC 트랜잭션 (sx/sy/dx/dy/noc/num_bytes/type/vc/timestamp/run_host_id).

### ③ Full-model calibration (Run C) — 1회

`RESNET_STAGE` 미설정 = STAGE=-1 (원본 semantics 그대로).

```bash
python -m tracy -r -p -v -m pytest \
    models/bos_model/resnet50_jbpark/tests/test_ttnn_functional_resnet50.py -k batch_1
```

**얻는 것**: 전체 op ~123개 짜리 ops_perf_results.csv 하나. Stage-split 결과와 op-level 비교 (캘리브레이션) 및 FPS/latency ground truth.

---

## 파일 격리 (팀원과 공용 컨테이너 대응)

`/tt-metal/generated/profiler/` 하위는 팀원과 공용. 매 실행마다 새로 생긴 것만 골라 옮겨야 안 섞임.

### 각 러닝 직후

```bash
# 새로 생긴 report 확인
ls -lt generated/profiler/reports/ | head -3

# 원하는 것만 이동
mv generated/profiler/reports/<새 ts 폴더> stage_split_<ts>/stage_N_compute/
```

`.logs/` 는 매 실행 시 덮어써지는 파일도 있으니 **Run B 직후 바로** `noc_trace_ID*_merged.json` 등을 다른 폴더로 옮겨두는 게 안전.

### 권장 폴더 구조

```
generated/profiler/stage_split_<ts>/
├── full/
│   └── report/ops_perf_results_*.csv       ← Run C
├── stage_0/
│   ├── compute/report/ops_perf_results_*.csv   ← Run A, STAGE=0
│   └── noc/logs/noc_trace_ID*_merged.json      ← Run B, STAGE=0
├── stage_1/
│   ├── compute/report/...
│   └── noc/logs/...
├── stage_2/ ...
├── stage_3/ ...
└── stage_4/ ...
```

---

## 오프라인 조인 (로컬)

**조인 키**: `run_host_id` (1024, 2048, ..., N × 1024).

두 실행이 **같은 STAGE 값·같은 pytest 옵션**이면 op 순서와 `run_host_id` 할당이 결정적 → 같은 op 는 두 pass 에서 같은 ID 를 가짐.

### 통합 DataFrame 스키마 (예정)

```
per-op:
├── stage, op_name, run_host_id           (인덱스)
├── kernel_ns, cycles, math_fidelity      (Run A)
├── pm_compute, pm_bandwidth, dominant_bound  (Run A)
├── noc_bytes_ours, noc_txn_count_ours    (Run B, 로컬 파싱)
├── noc_multicast_flag                    (Run B, 로컬 파싱)
├── ttnpe_bw_estimate, ttnpe_cycle_estimate  (Run B, tt-npe standalone — 선택적)
├── ttnpe_coverage                        ("ok" / "multicast_skip" / "assertion_fail")
├── is_stage_artifact                     (reshard/synth 마킹 — 필터 컬럼)
└── source_of_kernel_time                 ("stage-split" / "full")
```

### tt-npe standalone (선택적, 컨테이너에서)

Run B 결과에 대해 별도로:
```bash
npe_analyze_noc_trace_dir.py stage_N/noc/logs -e -m 200 > stage_N/noc/summary.txt 2>&1
```
`-e` 로 multicast assertion 발생 op 는 스킵됨. Stem 의 ID3 같은 heavy multicast op 는 tt-npe 커버 못 하므로 우리 로컬 파싱 결과로만 채워짐.

---

## Stage-split 인위성 관리 (신뢰도 경계)

Fork 는 원본과 4곳에서 divergence 생김. 조인 후 후처리에서 대응.

| 항목 | 영향 대상 | 대응 |
|---|---|---|
| Fake input tensor | 값 무관 지표는 무영향 | 그대로 사용 |
| Reshard 오버헤드 (S3/S4) | Compute + NoC 실측 부풀림 | 조인 시 `is_stage_artifact=True` 마킹 후 aggregate 에서 제외 |
| LAYER_CONFIG mutation (S2) | S2 layer2 module1 first block 만 | S2 결과에 caveat 명시, aggregate 에서 별도 취급 |
| Cold cache 상태 | 미미 (2nd iter 측정) | 무시 |

### Full-model calibration 의 역할

- **캘리브레이션**: FULL 의 layer1~4 op timing ↔ stage-split 의 STAGE=1~4 op timing 을 op_name 기준으로 비교. 오차 <5% 나오면 stage-split 신뢰도 확보.
- **아티팩트 격리**: FULL 에 없고 stage-split 에만 있는 op 를 골라내면 그게 순수 fork 오버헤드.
- **최종 발표 인용**: FPS/latency 는 결국 FULL 러닝 기준. Stage-split 은 세부 분석 도구.

---

## 신뢰도 지도 (NoC 관점)

| 항목 | Stage-split 신뢰도 | 이유 |
|---|---|---|
| Layer 내 op 별 NoC bytes | ★★★★★ | 같은 shape/sharding → 같은 트랜잭션 |
| Layer 내 op 별 src/dst 패턴 | ★★★★★ | 값 무관, kernel dispatch 결정적 |
| Layer 내 hot link 분포 | ★★★★★ | 라우팅 결정적 |
| Layer 경계 handoff 트래픽 | ★★☆☆☆ | Fake input synth → 인공적 reshard 개입 |
| Reshard 오버헤드 (S3/S4) | ★★☆☆☆ | 원본에 없는 op → 필터링 필요 |
| End-to-end timeline congestion | 측정 불가 | Stage 별 독립 실행, 시간축 이어붙일 수 없음 |

---

## 왜 tt-npe 를 계속 봐야 하나 (로컬 파싱과의 역할 차이)

| 얻는 것 | tt-npe | 로컬 파싱 |
|---|:-:|:-:|
| Per-op 총 bytes | ✓ | ✓ |
| Src/dst 트랜잭션 분포 | ✓ | ✓ |
| NOC_0/1 imbalance | ✓ | ✓ |
| Multicast op 분석 | ✗ (crash) | ✓ |
| 시간축 링크 utilization | ✓ | ✗ |
| 동시성 contention (discrete-event sim) | ✓ | ✗ |
| NoC-bound cycle estimate | ✓ | ✗ |

**로컬 파싱은 정적 aggregate, tt-npe 는 동적 timeline sim** — 완전 대체 불가.

- Roofline 재검증 (`PROGRESS.md` 활성 debt): BW ceiling estimate 근거로 tt-npe 시뮬 결과 필요.
- Big1 (예측 성능 모델) 방향성: NoC congestion 이 latency 에 어떻게 기여하는지 정량화하려면 tt-npe 급 시뮬이 학습/검증 데이터.

**결론**: tt-npe 커버되는 op 는 tt-npe 로, 크래시하는 multicast op 는 로컬 파싱으로 공백 최소화. 논문에서는 "op N% 는 tt-npe 실측, 나머지는 aggregate proxy" 로 정직하게 명시.

---

## 다음 스텝

1. 컨테이너에서 위 3종 커맨드 실행하여 데이터 확보 (사용자 담당).
2. 로컬 조인 스크립트 초안 작성 (Claude 담당) — 폴더 구조만 표준화되어 있으면 자동 매칭.
3. Stem NoC 데이터 (이미 확보) + Compute pass STAGE=0 데이터로 조인 로직 검증.
4. Stage 1~4 데이터 도착 순으로 순차 조인 → 통합 DataFrame 완성.
5. Full calibration 결과와 stage-split 결과 op-level 비교 → 신뢰도 정량 검증.
