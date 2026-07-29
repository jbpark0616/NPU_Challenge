# resnet50_jbpark — Fork Change Log

> 원본 `bos_model/resnet50/` 에서 fork. **tt-npe stage-split profiling** 을 위해 forward pass 를 stage 0~4 로 분할할 수 있게 확장.
> 원본 리서치 코드(Bottleneck, `_forward_layer`, Config, Conv, Linear 등)는 손대지 않았음.

Fork 시각: 2026-07-13

---

## 목적

tt-npe 는 per-run op count 실링(~65)이 있어 ResNet50 전체(123 ops @ batch=1)를 한 번에 프로파일할 수 없음. 이 fork 는 `RESNET_STAGE` env-var 로 forward pass 를 layer 단위로 잘라서 각 stage 를 실링 밑에서 개별 프로파일할 수 있게 함.

- **STAGE=-1 (default, unset)**: full run — **원본과 의미상 동일**
- **STAGE=0**: stem 만 실행 후 조기 반환
- **STAGE=1..4**: fake tensor 를 합성해서 layerN 진입점에 주입, 해당 layer 만 실행

관련 문서: `tutorials/guide/tt-npe_intro.md`, `PROGRESS.md`.

---

## 변경 요약

| 파일 | 변경 종류 | 목적 |
|---|---|---|
| `tests/test_ttnn_functional_resnet50.py` | Import 경로 + `validate()` 가드 | fork path + fake input 은 PCC 실패 확정이라 스킵 |
| `tests/resnet50_test_infra.py` | Import 경로 3줄 | fork path |
| `tt/ttnn_functional_resnet50.py` | Import + 헬퍼 함수 + `__call__` 재구성 + 내부 lazy import 1개 | stage 게이트 + fake input 합성 |
| `run_stage_split.sh` | **신규 파일** | 5-stage 순차 실행 자동화 |

### 손 안 댄 파일 (원본과 diff 없음)

- `tt/Config.py` — `LAYER_CONFIG` 값 자체는 그대로. STAGE=2 는 런타임에 in-place mutation 만 함 (아래 caveat 참고).
- `tt/Conv.py`, `tt/Linear.py`, `tt/Utils.py`, `tt/custom_preprocessing.py`
- `run_resnet50.py`, `README.md`

---

## 파일별 상세 변경

### 1. `tests/test_ttnn_functional_resnet50.py`

**변경 A — Import 추가**:
```python
import os
```

**변경 B — Fork path**:
```python
# before
from models.bos_model.resnet50.tests.resnet50_test_infra import create_test_infra
# after
from models.bos_model.resnet50_jbpark.tests.resnet50_test_infra import create_test_infra
```

**변경 C — `validate()` 조건부 실행**:
```python
# before
test_infra.validate()

# after
# Skip PCC validation for stage-split runs (STAGE >= 0) — synthesized inputs
# produce garbage outputs by design (see ttnn_functional_resnet50.__call__).
if int(os.environ.get("RESNET_STAGE", "-1")) < 0:
    test_infra.validate()
```

이유: fake input 은 랜덤 값이라 최종 output 이 원본 ResNet50 과 다름. PCC 체크 걸리면 profiling 중단됨. STAGE=-1 (원본 동작) 에서는 그대로 validate.

<br>

### 2. `tests/resnet50_test_infra.py`

**변경 — Import 경로 3줄**:
```python
# before → after
from models.bos_model.resnet50.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.bos_model.resnet50.tt.ttnn_functional_resnet50 import resnet50
from models.bos_model.resnet50.tt.Utils import fold_like_ttnn, get_conv_input_memory_config

# →

from models.bos_model.resnet50_jbpark.tt.custom_preprocessing import create_custom_mesh_preprocessor
from models.bos_model.resnet50_jbpark.tt.ttnn_functional_resnet50 import resnet50
from models.bos_model.resnet50_jbpark.tt.Utils import fold_like_ttnn, get_conv_input_memory_config
```

**로직 변경 없음**.

<br>

### 3. `tt/ttnn_functional_resnet50.py`

로직 수정이 몰려있는 핵심 파일. 변경 지점 4곳.

#### 3-1. Top-of-file imports

```python
import math
import os                # 추가
from typing import List

import torch             # 추가
from loguru import logger

import ttnn
from .Config import CORE_SET_14, CORE_SET_16, LAYER_CONFIG
...
```

#### 3-2. 신규 헬퍼 함수 `_synth_layer_input()` (Bottleneck 클래스 앞에 위치)

Fake 텐서 합성:
- 랜덤 값 (tt-npe 는 shape/layout 만 관심, 값 무관)
- `HEIGHT_SHARDED @ CORE_SET_14`, `ROW_MAJOR`, tile layout — stem 의 layer1 handoff recipe 와 동일
- STAGE=3(layer3.m1 auto-reshard) / STAGE=4(`_forward_layer("layer4")` explicit reshard) 는 sharding mismatch 를 자동 흡수

#### 3-3. `__call__` 재구성

원본 구조:
```
fold → conv1 → maxpool → reshard → layer1 → layer2 → layer3 → layer4 → head
```

Fork 구조:
```
[STAGE parse + LAYER_CONFIG mutate if STAGE=2]
[STAGE<=0]  fold → conv1 → maxpool → reshard  → (STAGE=0 early return)
[STAGE=1]   synth 64ch → layer1  → (STAGE=1 early return)
[STAGE=2]   synth 256ch → layer2 → (STAGE=2 early return)
[STAGE=3]   synth 512ch → layer3 → (STAGE=3 early return)
[STAGE=4]   synth 1024ch → layer4 → head → return
```

**의미상 원본과 100% 동일한 조건**: `STAGE == -1` (env-var 미설정 시 기본값).

**추가된 것**:
- `x_height, x_width = 56, 56` 를 `__call__` 시작 부분에서 pre-init (stage=1..4 에서 stem 을 건너뛰기 때문에 필요)
- 각 layer 호출을 `if STAGE in (-1, N):` 로 감쌈
- 각 stage 별 fake input 합성 블록
- 각 stage 별 조기 반환

**stem/layer4/head 코드 자체는 재정렬만 됐지 내부 로직 안 바뀜.**

#### 3-4. `ModelRunnerWrapper` 내부 lazy import (원래 line 730)

```python
# before → after
from models.bos_model.resnet50.tests.resnet50_test_infra import create_test_infra
# →
from models.bos_model.resnet50_jbpark.tests.resnet50_test_infra import create_test_infra
```

<br>

### 4. `run_stage_split.sh` (신규)

컨테이너 안에서 stage 0~4 순차 실행. **팀원 다른 모델 프로파일링과 공존 안전.**

매 stage 마다:
1. `mktemp` marker 로 시작 시각 기록, `reports/` 폴더명 스냅샷
2. `RESNET_STAGE=N tools/tracy/profile_this.py ... pytest -k batch_1` 실행
3. **`.logs/` 는 삭제하지 않고** — marker 시각 이후 생성된 파일만 이동 (`find -newer $MARKER`)
4. **`reports/` 는 스냅샷과 diff** 해서 새로 생긴 폴더만 이동 (`comm -13`)
5. 결과: `stage_split_<ts>/stage_N/{logs, report_<report_ts>}/`

이렇게 하면 옆에서 팀원이 다른 모델 프로파일링 중이어도 서로 안 침범.

단일 stage 실행: `bash run_stage_split.sh 3`

---

## 동작 보장 (원본과 관계)

| STAGE 값 | 동작 |
|---|---|
| `-1` (기본, env unset) | **원본과 의미상 동일** 한 full forward pass |
| `0` | stem 만 실행 |
| `1..4` | fake input 주입 → layerN 만 실행 (S4 는 layer4+head) |

**side effect 하나 (STAGE=2 전용)**:
- `LAYER_CONFIG["layer2"]["module1"][0]` 을 런타임에 `True` 로 in-place mutation
- 각 pytest 는 새 프로세스라 다음 실행에 새로 로드됨 → 실무상 문제 없음
- 다만 같은 프로세스에서 STAGE=2 한 번 실행 후 STAGE=-1 을 이어서 실행하면 원본과 다른 동작이 될 수 있음 → **stage split runner 는 항상 새 프로세스로 실행**

---

## 원본과 diff 확인 (참고)

원본은 컨테이너에 그대로 있음: `/tt-metal/models/bos_model/resnet50/`

핵심 파일만 diff:
```bash
diff /tt-metal/models/bos_model/resnet50/tt/ttnn_functional_resnet50.py \
     /tt-metal/models/bos_model/resnet50_jbpark/tt/ttnn_functional_resnet50.py
```

기대되는 diff 크기: ~100 줄 (import 2, 헬퍼 함수 ~30, `__call__` 재구성 ~70)

---

## 정리

- **알고리즘 파트 (Bottleneck, `_forward_layer`, Config values) 는 손대지 않음** — 논문 인용 시 "원본과 동일 인식 파이프라인, profiling 편의를 위한 gate 만 추가" 라고 명시 가능.
- **STAGE=-1 default 는 원본 semantics 보존** → 회귀 테스트로 언제든 검증 가능.
- **STAGE 별 side effect (S2 reshard override, fake input) 는 caveat 로 명시할 것** — 실측 인용 시 이 오버헤드가 원본 대비 몇 op 나오는지 계산해서 붙일 예정.

python_env) root@7575fef6ac31:/tt-metal# bash /tt-metal/models/bos_model/resnet50_jbpark/run_stage_split.sh 0
================================================================
==== Running RESNET_STAGE=0
================================================================
2026-07-12 17:38:07.262 | WARNING  | __main__:main:44 - profile_this.py is getting deprecated soon. Please use the tracy.py module with -r option to obtain op reports.
2026-07-12 17:38:07.262 | INFO     | __main__:main:48 - profile_this.py is running pytest models/bos_model/resnet50_jbpark/tests/test_ttnn_functional_resnet50.py -k batch_1
2026-07-12 17:38:08.409 | INFO     | __main__:main:186 - Using port 8086
2026-07-12 17:38:08.409 | INFO     | tracy:run_report_setup:88 - Verifying tracy profiling tools
2026-07-12 17:38:08.420 | INFO     | tracy:run_report_setup:108 - Capture command: ('/tt-metal/build/tools/profiler/bin/capture-release -o /tt-metal/generated/profiler/.logs/tracy_profile_log_host.tracy -f -p 8086',)
Connecting to 127.0.0.1:8086...2026-07-12 17:38:08.427 | INFO     | __main__:main:249 - Test process started
2026-07-12 17:38:09.907 | DEBUG    | ttnn:<module>:77 - Initial ttnn.CONFIG:
Config{cache_path=/root/.cache/ttnn,model_cache_path=/root/.cache/ttnn/models,tmp_dir=/tmp/ttnn,enable_model_cache=false,enable_fast_runtime_mode=true,throw_exception_on_fallback=false,enable_logging=false,enable_graph_report=false,enable_detailed_buffer_report=false,enable_detailed_tensor_report=false,enable_comparison_mode=false,comparison_mode_should_raise_exception=false,comparison_mode_pcc=0.9999,root_report_path=generated/ttnn/reports,report_name=std::nullopt,std::nullopt}

Queue delay: 15 ns
Timer resolution: 5 ns
   5.83 Mbps / 40.7% =  14.32 Mbps | Tx: 2235.96 KB | 67.07 MB | 2.91 s============================================================ test session starts ============================================================
platform linux -- Python 3.10.12, pytest-7.2.2, pluggy-1.6.0 -- /tt-metal/python_env/bin/python3
cachedir: .pytest_cache
benchmark: 4.0.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /tt-metal, configfile: pytest.ini
plugins: split-0.8.2, anyio-4.13.0, github-actions-annotate-failures-0.3.0, timeout-2.2.0, xdist-3.6.1, dash-2.15.0, benchmark-4.0.0
timeout: 300.0s
timeout method: signal
timeout func_only: False
collected 3 items / 2 deselected / 1 selected                                                                                               

models/bos_model/resnet50_jbpark/tests/test_ttnn_functional_resnet50.py::test_resnet_50[pretrained_weight_false-act_dtype0-weight_dtype0-math
   9.67 Mbps / 11.2% =  86.16 Mbps | Tx: 2477.97 KB | 67.16 MB | 3.12 s2026-07-12 17:38:12.870 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-12 17:38:12.904 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-12 17:38:12.904 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-12 17:38:12.905 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:202)
   9.67 Mbps / 11.2% =  86.16 Mbps | Tx: 2477.97 KB | 67.19 MB | 3.27 s2026-07-12 17:38:12.938 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-12 17:38:12.956 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-12 17:38:12.957 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-12 17:38:12.986 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-12 17:38:13.020 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-12 17:38:13.022 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-12 17:38:13.023 | info     |             UMD | Harvesting mask for chip 0 is 0x0 (NOC0: 0x0, simulated harvesting mask: 0x0). (cluster.cpp:399)
  13.06 Mbps / 19.1% =  68.42 Mbps | Tx: 2804.71 KB | 67.21 MB | 3.32 s2026-07-12 17:38:13.054 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-12 17:38:13.054 | info     |             UMD | Initializing iommu for sysmem (size: 0x40000000). (sysmem_manager.cpp:300)
2026-07-12 17:38:13.054 | info     |             UMD | Allocating sysmem without hugepages (size: 0x40000000). (sysmem_manager.cpp:306)
   6.68 Mbps / 21.2% =  31.44 Mbps | Tx: 2971.86 KB | 67.27 MB | 3.53 s2026-07-12 17:38:13.255 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-12 17:38:13.255 | info     |             UMD | Opening local chip ids/PCIe ids: {0}/[0] and remote chip ids {} (cluster.cpp:251)
2026-07-12 17:38:13.255 | info     |             UMD | IOMMU: enabled (cluster.cpp:173)
2026-07-12 17:38:13.255 | info     |             UMD | KMD version: 2.3.0 (cluster.cpp:176)
2026-07-12 17:38:13.256 | info     |          Device | start_device in llrt (tt_cluster.cpp:415)
2026-07-12 17:38:13.277 | info     |             UMD | Pinning pages for DMA: virtual address 0x765824a00000 and size 0x40000000 pinned to physical address 0x7fc0000000 without mapping to noc (pci_device.cpp:618)
2026-07-12 17:38:13.277 | info     |             UMD | Mapped sysmem without hugepages to IOVA 0x7fc0000000; NOC address 0x0 (sysmem_manager.cpp:354)
2026-07-12 17:38:13.321 | info     |             UMD | Initializing memory barriers (local_chip.cpp:171)
   6.68 Mbps / 21.2% =  31.44 Mbps | Tx: 2971.86 KB | 67.28 MB | 3.63 s2026-07-12 17:38:13.405 | DEBUG    | ttnn.device:__init__:150 - Using default dispatch core type for this system: DispatchCoreType.WORKER
2026-07-12 17:38:13.405 | DEBUG    | ttnn.device:__init__:152 - Using default dispatch core axis for this system: DispatchCoreAxis.COL
  16.31 Mbps / 17.8% =  91.81 Mbps | Tx: 8903.36 KB | 75.46 MB | 5.42 s2026-07-12 17:38:15.211 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
   6.37 Mbps / 20.4% =  31.30 Mbps | Tx: 11.83 MB | 77.95 MB | 8.29 s2026-07-12 17:38:18.030 | DEBUG    | ttnn:manage_config:85 - Set ttnn.CONFIG.enable_logging to False
2026-07-12 17:38:18.030 | DEBUG    | ttnn:manage_config:85 - Set ttnn.CONFIG.enable_comparison_mode to False
2026-07-12 17:38:18.030 | WARNING  | ttnn.model_preprocessing:from_torch:553 - ttnn: model cache can be enabled by passing model_name argument to preprocess_model[_parameters] and setting env variable TTNN_CONFIG_OVERRIDES='{"enable_model_cache": true}'
   5.92 Mbps / 14.3% =  41.34 Mbps | Tx: 11.97 MB | 78.01 MB | 8.39 s2026-07-12 17:38:18.082 | DEBUG    | ttnn:manage_config:88 - Restored ttnn.CONFIG.enable_comparison_mode to False
2026-07-12 17:38:18.082 | DEBUG    | ttnn:manage_config:88 - Restored ttnn.CONFIG.enable_logging to False
   6.94 Mbps / 22.3% =  31.09 Mbps | Tx: 15.07 MB | 80.64 MB | 11.76 s2026-07-12 17:38:21.529 | DEBUG    | models.bos_model.resnet50_jbpark.tt.ttnn_functional_resnet50:__call__:524 - ==== First run
2026-07-12 17:38:21.529 | DEBUG    | models.bos_model.resnet50_jbpark.tt.ttnn_functional_resnet50:__call__:533 - ==== fold on host
2026-07-12 17:38:21.529 | DEBUG    | models.bos_model.resnet50_jbpark.tt.ttnn_functional_resnet50:__call__:539 - ==== first conv
  14.97 Mbps / 35.2% =  42.50 Mbps | Tx: 19.55 MB | 86 MB | 14.55 s2026-07-12 17:38:24.336 | DEBUG    | models.bos_model.resnet50_jbpark.tt.ttnn_functional_resnet50:__call__:562 - ==== maxpool
  20.13 Mbps / 38.5% =  52.25 Mbps | Tx: 25.61 MB | 160.02 MB | 17.97 sPASSED2026-07-12 17:38:27.715 | warning  |          Always | Profiler NoC events are enabled; this can add 1-15% cycle overhead to typical operations! (profiler.cpp:1754)
   6.82 Mbps / 11.9% =  57.50 Mbps | Tx: 27.45 MB | 160.63 MB | 24.15 s

================================================================== PASSES ===================================================================
-------------------------------- generated xml file: /tt-metal/generated/test_reports/most_recent_tests.xml ---------------------------------
============================================================= slowest durations =============================================================
9.95s call     models/bos_model/resnet50_jbpark/tests/test_ttnn_functional_resnet50.py::test_resnet_50[pretrained_weight_false-act_dtype0-weight_dtype0-math_fidelity0-batch_1-device_params0]
5.01s setup    models/bos_model/resnet50_jbpark/tests/test_ttnn_functional_resnet50.py::test_resnet_50[pretrained_weight_false-act_dtype0-weight_dtype0-math_fidelity0-batch_1-device_params0]
1.20s teardown models/bos_model/resnet50_jbpark/tests/test_ttnn_functional_resnet50.py::test_resnet_50[pretrained_weight_false-act_dtype0-weight_dtype0-math_fidelity0-batch_1-device_params0]
========================================================== short test summary info ==========================================================
PASSED models/bos_model/resnet50_jbpark/tests/test_ttnn_functional_resnet50.py::test_resnet_50[pretrained_weight_false-act_dtype0-weight_dtype0-math_fidelity0-batch_1-device_params0]
===================================================== 1 passed, 2 deselected in 16.20s ======================================================
  12.84 Mbps / 19.6% =  65.66 Mbps | Tx: 28.19 MB | 160.88 MB | 24.15 s2026-07-12 17:38:29.486 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:427)
2026-07-12 17:38:29.500 | info     |             UMD | Unpinning pages for DMA: virtual address 0x765824a00000 and size 0x40000000 (pci_device.cpp:644)
  59.39 Kbps /  6.4% =   0.92 Mbps | Tx: 55.15 MB | 367.25 MB | 24.15 s
Frames: 2
Time span: 24.15 s
Zones: 79,106
Elapsed time: 16:32.5
Saving trace... done!
Trace size 92.85 MB (31.20% ratio)
2026-07-12 17:54:44.109 | INFO     | tracy:generate_report:156 - Host side ops time report generated at /tt-metal/generated/profiler/.logs/tracy_ops_times.csv
2026-07-12 17:54:44.941 | INFO     | tracy:generate_report:167 - Host side ops data report generated at /tt-metal/generated/profiler/.logs/tracy_ops_data.csv
2026-07-12 17:54:44.942 | INFO     | tracy.process_ops_logs:import_tracy_op_logs:111 - Importing ops logs
2026-07-12 17:54:45.174 | INFO     | tracy.process_ops_logs:get_device_op_data:255 - Getting device ops
2026-07-12 17:54:45.174 | INFO     | tracy.process_ops_logs:append_device_data:292 - Appending device data
2026-07-12 17:54:45.549 | INFO     | tracy.process_device_log:core_to_device_timeseries:398 - Importing Data For Device Number : 0
2026-07-12 17:54:47.215 | INFO     | tracy.process_ops_logs:analyzeNoCTraces:919 - tt-npe module imported successfully; analyzing noc traces ... 
W: Device sync isn't turned on. This will lead to inaccurate traces for mesh device workloads! If you are tracing a workload on a mesh device, please turn on device sync by setting TT_METAL_DEVICE_PROFILER=1 and TT_METAL_PROFILER_SYNC=1 then re-collect traces
MulticastCoordSet: start coord must be to the top-left of end coord
E: UNDEF - TT_ASSERT @ /tt-npe/tt_npe/cpp/include/npeUtil.hpp:194: start.row <= end.row || start.col <= end.col
info:
MulticastCoordSet: start coord must be to the top-left of end coord
backtrace:
 --- /tt-npe/install/lib/../lib/libtt_npe.so.0(+0x30f88) [0x79d40bad6f88]
 --- /tt-npe/install/lib/../lib/libtt_npe.so.0(+0xa0f66) [0x79d40bb46f66]
 --- tt_npe::convertNocTracesToNpeWorkload(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, bool)
 --- tt_npe::createWorkloadFromJSON(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, bool, bool)
 --- /tt-npe/install/lib/tt_npe_pybind.cpython-310-x86_64-linux-gnu.so(+0x29879) [0x79d44ec5c879]
 --- /tt-npe/install/lib/tt_npe_pybind.cpython-310-x86_64-linux-gnu.so(+0x14146) [0x79d44ec47146]
 --- python3(+0x18ae32) [0x632ce9181e32]
 --- python3(_PyObject_MakeTpCall+0x25b) [0x632ce917889b]
 --- python3(_PyEval_EvalFrameDefault+0x6a5b) [0x632ce9172f6b]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x6c0) [0x632ce916cbd0]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(PyObject_Call+0x122) [0x632ce9190392]
 --- python3(+0x2942dc) [0x632ce928b2dc]
 --- python3(_PyObject_MakeTpCall+0x25b) [0x632ce917889b]
 --- python3(_PyEval_EvalFrameDefault+0x286b) [0x632ce916ed7b]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x286b) [0x632ce916ed7b]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x807) [0x632ce916cd17]
 --- python3(+0x1986f1) [0x632ce918f6f1]
 --- python3(_PyEval_EvalFrameDefault+0x1992) [0x632ce916dea2]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x807) [0x632ce916cd17]
 --- python3(_PyObject_FastCallDictTstate+0xc4) [0x632ce9177a64]
 --- python3(+0x194a64) [0x632ce918ba64]
 --- python3(_PyObject_MakeTpCall+0x1fc) [0x632ce917883c]
 --- python3(_PyEval_EvalFrameDefault+0x5a87) [0x632ce9171f97]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x56cb) [0x632ce9171bdb]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x807) [0x632ce916cd17]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x56cb) [0x632ce9171bdb]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x807) [0x632ce916cd17]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyObject_FastCallDictTstate+0x16d) [0x632ce9177b0d]
 --- python3(+0x194a64) [0x632ce918ba64]
 --- python3(_PyObject_MakeTpCall+0x1fc) [0x632ce917883c]
 --- python3(_PyEval_EvalFrameDefault+0x6a5b) [0x632ce9172f6b]
 --- python3(+0x1986f1) [0x632ce918f6f1]
 --- python3(_PyEval_EvalFrameDefault+0x1992) [0x632ce916dea2]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x1992) [0x632ce916dea2]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x6c0) [0x632ce916cbd0]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x6c0) [0x632ce916cbd0]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x1992) [0x632ce916dea2]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x6c0) [0x632ce916cbd0]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x6c0) [0x632ce916cbd0]
 --- python3(+0x259d36) [0x632ce9250d36]
 --- python3(PyEval_EvalCode+0x86) [0x632ce9250c06]
 --- python3(+0x25f34d) [0x632ce925634d]
 --- python3(+0x18b8e9) [0x632ce91828e9]
 --- python3(_PyEval_EvalFrameDefault+0x6c0) [0x632ce916cbd0]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(_PyEval_EvalFrameDefault+0x6c0) [0x632ce916cbd0]
 --- python3(_PyFunction_Vectorcall+0x7c) [0x632ce918268c]
 --- python3(+0x274cad) [0x632ce926bcad]
 --- python3(Py_RunMain+0x128) [0x632ce926aa68]
 --- python3(Py_BytesMain+0x2d) [0x632ce9244c4d]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x29d90) [0x79d454d45d90]
 --- /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80) [0x79d454d45e40]
 --- python3(_start+0x25) [0x632ce9244b45]

E: Error processing /tt-metal/generated/profiler/.logs/noc_trace_ID3_merged.json: Could not create tt-npe workload from file '/tt-metal/generated/profiler/.logs/noc_trace_ID3_merged.json'; aborting ... 

E: Timeline Consistency Check Failed: Transfer group ID 280 is defined in noc_transfers but never appears in active_transfers of any timestep.
E: Timeline Consistency Check Failed: Transfer group ID 2353 is defined in noc_transfers but never appears in active_transfers of any timestep.
E: Timeline Consistency Check Failed: Transfer group ID 470 is defined in noc_transfers but never appears in active_transfers of any timestep.
E: Timeline Consistency Check Failed: Transfer group ID 2354 is defined in noc_transfers but never appears in active_transfers of any timestep.
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/tt-metal/tools/tracy/__main__.py", line 290, in <module>
    main()
  File "/tt-metal/tools/tracy/__main__.py", line 267, in main
    generate_report(
  File "/tt-metal/tools/tracy/__init__.py", line 169, in generate_report
    process_ops(
  File "/tt-metal/tools/tracy/process_ops_logs.py", line 946, in process_ops
    deviceOps, traceOps = append_device_data(
  File "/tt-metal/tools/tracy/process_ops_logs.py", line 436, in append_device_data
    op_npe_stats = npe_stats.getDatapointByID(op_id)
TypeError: Stats.getDatapointByID() missing 2 required positional arguments: 'metal_trace_id' and 'metal_trace_replay_session_id'
==== Moved 22 new .logs files → generated/profiler/stage_split_20260712_173806/stage_0/logs/
WARN: no new report dir found for STAGE=0 (op ceiling exceeded? Tracy stitching may have failed)
==== STAGE=0 outputs → generated/profiler/stage_split_20260712_173806/stage_0

================================================================
All stages done.
Output root: generated/profiler/stage_split_20260712_173806

To analyze raw traces per stage:
  for s in 0; do
    npe_analyze_noc_trace_dir.py generated/profiler/stage_split_20260712_173806/stage_$s/logs -e -m 200 \
      > generated/profiler/stage_split_20260712_173806/stage_$s/summary.txt 2>&1
  done
================================================================
