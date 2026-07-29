# NPU Command Set — Tracy 프로파일링 & 실행 명령 참고

자주 쓰는 Tracy 프로파일링, docker 서버 setup, CSV 처리 명령 모음.
소스 근거는 항목별로 명시 (`tt-metal/...`), 다음 세션에서 바로 복붙 가능.

관련 노트:
- `bos_model_copy/yolov8s/yolo_study/yolo_study_note.md` §12.2, §13.14
- `bos_model_copy/resnet50/resnet_note/` (batch 이슈 §)

---

## 1. Docker 서버 진입 후 필수 setup

```bash
cd /tt-metal
source env_set.sh   # ttnn, tracy 등 PYTHONPATH 세팅. 필수.
```

**확인**:
```bash
echo "PYTHONPATH=$PYTHONPATH"   # /tt-metal, bos-metal 경로 포함돼야 함
python -c "import tracy; import ttnn; print('OK')"
```

**증상: `env_set.sh` 안 하면 ttnn API 없다는 것처럼 보임** (auto-memory `feedback_env_setup`).

---

## 2. 기본 Tracy 프로파일링 — op-level CSV 생성

### YoloV8s
```bash
ENABLE_TRACY=1 TT_METAL_DEVICE_PROFILER=1 \
python -m tracy -r -p -v \
    models/bos_model/yolov8s/run_yolov8s.py --trace -i 320 -n 10
```

### ResNet50
```bash
ENABLE_TRACY=1 TT_METAL_DEVICE_PROFILER=1 \
python -m tracy -r -p -v \
    models/bos_model/resnet50/run_resnet50.py --trace -b 1 -n 2
```

**공통 flag 의미**:
- `-r` = tracy report (CSV 생성)
- `-p` = post-process (`process_ops_logs.py` 자동 실행)
- `-v` = verbose 출력
- `--trace` = tt-metal trace mode (host dispatch 최소화)
- `-n N` = iteration 수
- `-b B` = batch size (ResNet50만; YoloV8s는 config로 결정)
- `-i S` = input size (YoloV8s만; ex `-i 320`)

---

## 3. 큰 모델 / 많은 iter — 버퍼 오버플로 대처

**에러 (자주 만남)**:
```
Profiler DRAM buffers were full, markers were dropped
Please either decrease the number of ops being profiled
or run read device profiler more often
```
또는:
```
AssertionError: Unexpected FW start, core (4, 4), risc BRISC is reporting
a second start of FW for op 252928
```

**해법**: `TT_METAL_PROFILER_MID_RUN_DUMP=1` 추가 (실행 중 profile 버퍼 주기적 flush)
```bash
ENABLE_TRACY=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
python -m tracy -r -p -v \
    models/bos_model/resnet50/run_resnet50.py --trace -b 1 -n 2
```

**주의사항 1**: `TT_METAL_DEVICE_PROFILER=1`이 같이 있어야 활성화 (`rtoptions.cpp:937`).

**⚠️ 주의사항 2 — Trace 모드에서는 MID_RUN_DUMP 실질 무효**
`tt_metal_profiler.cpp:933`:
```cpp
return get_profiler_mid_run_dump() && state == ProfilerReadState::NORMAL;
```
Trace replay 중에는 `state = TRACE_REPLAY` 등이라 실제 dump 안 됨.
즉 **`--trace` + `-n ≥ 2` + MID_RUN_DUMP 조합은 여전히 오버플로 남**.

**Fallback = `-n 1` 강제**:
```bash
ENABLE_TRACY=1 TT_METAL_DEVICE_PROFILER=1 \
python -m tracy -r -p -v \
    models/bos_model/resnet50/run_resnet50.py --trace -b 1 -n 1
```
Trace mode에서 iter 1개 안에 warmup + capture + 1 replay 다 들어감. Baseline 분석 (op별 kernel time, 코어 분포)엔 iter 1개로 충분.

**대안 = Trace off + `-n` 늘리기**:
```bash
ENABLE_TRACY=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
python -m tracy -r -p -v \
    models/bos_model/resnet50/run_resnet50.py -b 1 -n 3
```
Trace 없으면 state가 NORMAL 유지 → MID_RUN_DUMP 실제 dump. 다만 host dispatch 시간이 정상 크기로 나옴 (kernel/host 비 다르게 나올 것).

---

## 4. Iteration 수 튜닝 가이드

| `-n` | 총 op 수 (ResNet50 ~130/iter 기준) | 성공 확률 (Eagle-N A0) |
|---|---:|---|
| 1 | ~130 | 항상 성공 (안전 fallback) |
| 2 | ~260 | MID_RUN_DUMP 필요 |
| 5 | ~650 | MID_RUN_DUMP 필수 |
| 10 | ~1300 | MID_RUN_DUMP 필수 |

Baseline 분석 목적이면 `-n 1~2`로 충분. 통계 노이즈 줄이려면 `-n 5+` (MID_RUN_DUMP와).

---

## 5. 결과 위치 & 확인

### 새 timestamp 폴더 찾기
```bash
# 최신순 정렬 (top 3)
ls -td /tt-metal/generated/profiler/reports/*/ | head -3

# 방금 실행한 폴더 자동 캐치
LATEST=$(ls -td /tt-metal/generated/profiler/reports/*/ | head -1)
echo "최신: $LATEST"
ls "$LATEST"
```

### 파일 3개 항상 같이 생성됨
| 파일 | 용도 |
|---|---|
| `ops_perf_results_<ts>.csv` | 분석용 (roofline_analysis.py 입력) |
| `profile_log_device.csv` | Device-side raw log (cycles·zones·source lines) |
| `tracy_profile_log_host.tracy` | Tracy GUI로 여는 host trace 파일 |

### CSV 정체 sanity check (yolo인지 resnet인지)
```bash
# OP CODE 컬럼의 unique 값 확인
LATEST=$(ls -td /tt-metal/generated/profiler/reports/*/ | head -1)
head -50 "$LATEST"ops_perf_results_*.csv | tail -49 | cut -d',' -f1 | sort -u
```

**판정 룰**:
- `UnaryDeviceOperation`, `ConcatDeviceOperation` 있음 → **YoloV8s** (SiLU + FPN concat)
- `BinaryDeviceOperation`, `MoveDeviceOperation` 있고 위 두 개 없음 → **ResNet50** (residual add + move)

### Attribute에서 batch/input 확인
```bash
python -c "
import csv, re
with open('$LATEST' + 'ops_perf_results_2026_XX_XX_XX_XX_XX.csv') as f:
    for r in csv.DictReader(f):
        if r['OP CODE'] == 'Conv2d':
            m = re.search(r'batch_size=(\d+)', r['ATTRIBUTES'])
            m2 = re.search(r'input_hw=\((\d+);\s*(\d+)\)', r['ATTRIBUTES'])
            print(f'batch={m.group(1)}, input={m2.group(1)}x{m2.group(2)}')
            break
"
```

---

## 6. 로컬로 CSV 다운로드

**Windows PowerShell**:
```powershell
# SSH 후 scp — 서버 주소는 실제 것으로
scp <server>:/tt-metal/generated/profiler/reports/<ts_folder>/ops_perf_results_*.csv `
    C:/NPU_Challenge/bos_model_copy/<model>/<study_folder>/
```

**팁**: 다운받은 후 노트 §13.14의 sanity 명령으로 batch/model 재확인. 파일명 timestamp만으로는 어떤 모델·조건인지 알 수 없음 (실수 방지).

---

## 7. HW 카운터 재취득 (Debt — 노트 §13.14.8)

현재 CSV의 `NOC UTIL`, `DRAM BW UTIL`, `NPE CONG IMPACT` 컬럼이 **empty**.
아래 flag 추가하면 채워질 것 (아직 Eagle-N에서 검증 안 됨):
```bash
ENABLE_TRACY=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
python -m tracy -r -p -v \
    --profiler-capture-perf-counters=fpu,pack,unpack,l1_0,instrn \
    models/bos_model/yolov8s/run_yolov8s.py --trace -i 320 -n 2
```

성공하면 Fig 4·5 (PM 예측)를 실측 카운터로 대체 가능.

---

## 8. 그 외 유용한 profiler env vars (`rtoptions.cpp:126-139`)

| Env var | 용도 |
|---|---|
| `TT_METAL_PROFILER_SYNC=1` | 동기 프로파일링 (op 겹침 없이 순차) |
| `TT_METAL_PROFILER_TRACE_TRACKING=1` | Trace mode 세부 추적 강화 |
| `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=<N>` | 지원 program 개수 상한 조정 |
| `TT_METAL_PROFILER_CPP_POST_PROCESS=1` | Python 대신 C++ post-process |
| `TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1` | 파일 dump off (tracy live만) |
| `TT_METAL_PROFILER_DIR=<path>` | 결과 저장 경로 커스텀 |

---

## 9. 흔한 에러 & 대처 (실전 요약)

| 에러 / 증상 | 원인 | 대처 |
|---|---|---|
| `ttnn 모듈 없음` / API 사라진 것처럼 보임 | env_set.sh 안 됨 | `source env_set.sh` |
| `Profiler DRAM buffers were full` | 버퍼 오버플로 | `TT_METAL_PROFILER_MID_RUN_DUMP=1` 추가 |
| `AssertionError: Unexpected FW start` | 위 오버플로의 변형 (op #252928 등) | 동 |
| Trace 모드 + MID_RUN_DUMP 붙였는데도 오버플로 | `state != NORMAL`이라 실제 dump 안 됨 (`tt_metal_profiler.cpp:933`) | **`-n 1`로 강제** or trace 빼고 재시도 |
| 새 timestamp 폴더 안 생김 | Tracy 명령 실패 or Ctrl+C | 좀비 프로세스 `pkill -f tracy` 후 재시도 |
| Tracy Tx 진행 안 함 (같은 MB에서 hang) | Tracy 서버 hang or 네트워크 | 5분 이상 안 움직이면 Ctrl+C |
| CSV의 `NOC/DRAM_BW UTIL` 컬럼 empty | HW counter capture flag 없음 | §7의 flag 추가 |

---

## 10. 명령 실행 후 즉시 확인 루틴 (복붙용)

Tracy 명령이 끝났을 때 이걸 그대로 붙여 넣기:

```bash
# 1) 새 폴더 나왔는가
NEW_FOLDER=$(ls -td /tt-metal/generated/profiler/reports/*/ | head -1)
echo "가장 최근 폴더: $NEW_FOLDER"

# 2) 파일 3종 세트 다 있는가
ls -la "$NEW_FOLDER"

# 3) CSV row 수 sanity
CSV=$(ls "$NEW_FOLDER"/ops_perf_results_*.csv | head -1)
echo "총 op 수: $(($(wc -l < "$CSV") - 1))"

# 4) 어떤 모델인지 (OP CODE unique)
tail -n +2 "$CSV" | cut -d',' -f1 | sort -u | head -20
```

---

## 11. 소스 참조

| 파일 | 내용 |
|---|---|
| `tt-metal/tt_metal/llrt/rtoptions.cpp:126-139` | 모든 profiler env var 목록 |
| `tt-metal/tt_metal/llrt/rtoptions.cpp:931-941` | `MID_RUN_DUMP` 동작 |
| `tt-metal/tt_metal/impl/profiler/profiler.cpp:1563-1578` | 버퍼 오버플로 에러 발생 위치 |
| `tt-metal/tools/tracy/process_ops_logs.py` | CSV 생성 & PM 값 굽기 |
| `tt-metal/docs/source/ttnn/ttnn/profiling_ttnn_operations.rst` | CSV 컬럼 스펙 |
| `tt-metal/tech_reports/MetalProfiler/metal-profiler.md` | Tracy·Metal 통합 아키텍처 |
| `tutorials/guide/Tracy_intro.md` | Tracy 사용 튜토리얼 |
| `tt-metal/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md` | Peak TFLOPS 벤치마크 (§13.14 근거) |
