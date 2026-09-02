# TTNN Conv2D and Core Replacement Notes

이 문서는 selective core replacement를 TTNN Conv2D에 연결하기 위해 확인한
구조, 현재 코드의 제약, 수정 순서를 기록한다. 구현하면서 확인한 사실은 계속
이 문서에 누적한다.

## 1. 소스 기준

현재 작업에서 가장 우선하는 소스는 별도로 내려받은 다음 디렉터리다.

```text
operations/conv/conv2d/
operations/sliding_window/
```

로컬의 `tt-metal/ttnn/cpp/...`는 구조 비교용으로만 사용한다. 컨테이너에서
실제로 빌드하는 브랜치와 로컬 upstream 사이에 차이가 있을 수 있으므로,
수정 전에는 항상 컨테이너의 대응 파일과 다시 비교한다.

## 2. 좌표계와 용어

Tenstorrent 코드에는 이미 logical worker core와 physical NoC coordinate라는
용어가 존재한다. Conv workload의 좌표를 단순히 `logical`, replacement된
좌표를 `physical`이라고 부르면 기존 의미와 충돌한다. 이 프로젝트에서는
다음 세 좌표계를 구분한다.

```text
ShardCoord
    | CorePlacement::worker_core()
    v
WorkerCoord
    | device->worker_core_from_logical_core()
    v
NocCoord
```

| 용어 | 의미 | 예시 |
|---|---|---|
| `ShardCoord` | Conv workload 안의 조밀한 shard 위치 | `(2,1)` in 3x3 |
| `WorkerCoord` | TT-Metal worker grid에서 선택된 core 위치 | `(3,1)` in 5x4 |
| `NocCoord` | NoC command와 routing에 사용하는 hardware 좌표 | device mapping 결과 |

기존 TT API의 `worker_core_from_logical_core()`에 들어가는 `logical_core`는 이
문서의 `WorkerCoord`에 해당한다. 함수 이름은 기존 API이므로 바꾸지 않는다.

core replacement가 새로 담당하는 변환은 다음 하나다.

```text
ShardCoord -> WorkerCoord
```

기존 TT-Metal이 다음 변환을 담당한다.

```text
WorkerCoord -> NocCoord
```

코드와 로그에서는 단독 `logical`, `physical`이라는 이름을 새로 만들지 않는다.
다음 이름을 사용한다.

```text
shard_grid_size
worker_grid_size
worker_x
worker_y
worker_cores
excluded_worker_rows
excluded_worker_columns
worker_range
noc_start / noc_end
```

예시:

```text
shard_grid=3x3
worker_grid=5x4
worker_x=[0,1,3]
worker_y=[0,1,3]
excluded_worker_rows=[2]
excluded_worker_columns=[2]
mapping: shard=(2,1) worker=(3,1) noc=(...)
```

## 3. 이번 구현의 범위

첫 번째 end-to-end 목표는 다음 한 가지 구성이다.

```text
Conv:             3x3, stride 1, padding 1
Input:            N=1, H=24, W=24, C=32
Output channels:  96
Shard layout:     BLOCK_SHARDED
Orientation:      ROW_MAJOR
Logical grid:     3x3
Physical grid:    5x4
Replacement:      zero or more worker rows/columns, within spare-grid capacity
```

이 shape는 3x3 shard 분할과 잘 맞는다.

```text
output NHW tiles = 24 * 24 / 32 = 18
NHW cores         = 3
tiles/NHW core    = 18 / 3 = 6

output C tiles    = 96 / 32 = 3
channel cores     = 3
tiles/C core      = 3 / 3 = 1
```

첫 구현에서는 다음 항목을 다루지 않는다.

- HEIGHT_SHARDED와 WIDTH_SHARDED replacement
- 여러 행 또는 여러 열 제외
- 임의의 비-Cartesian core set
- 자동 placement 최적화
- ResNet50 전체 모델에 replacement 적용
- public Python API의 최종 형태

## 4. 전체 호출 흐름

사용자 호출부터 device kernel 생성까지의 흐름은 다음과 같다.

```text
ttnn.conv2d(...)
    |
    +-- DRAM input/slicing 여부 결정
    |
    +-- conv2d_L1(...)
          |
          +-- fold_input_tensor_if_required()
          |
          +-- output H/W 계산
          |
          +-- 필요하면 determine_conv_config_for_auto_shard()
          |     `-- HEIGHT/BLOCK/WIDTH 중 layout 선택
          |
          +-- shard_or_reshard_tensor_if_required()
          |     +-- get_conv_padded_input_shape_and_mem_config()
          |     +-- determine_parallel_config()
          |     +-- create_sharded_memory_config_from_parallel_config()
          |     `-- input tensor를 L1 shard/reshard
          |
          +-- get_conv_configs()
          |     +-- Conv2dParallelizationConfig
          |     +-- Conv2dBlockConfig
          |     `-- output MemoryConfig
          |
          +-- weight/bias preprocessing
          |
          +-- halo(...)
          |
          `-- Conv2d device operation
                `-- multi_core_conv2d_sharded() program factory
                      +-- circular buffers
                      +-- semaphores
                      +-- reader/writer/compute kernels
                      `-- per-core runtime arguments
```

중요한 점은 `ttnn.conv2d()`가 반환되었다고 device program이 끝난 것이 아닐
수 있다는 것이다. Fast dispatch에서는 작업이 command queue에 enqueue되고,
`from_device(..., blocking=true)` 같은 동기화 지점에서 실제 미완료가 드러날
수 있다. 따라서 host read에서 멈췄다고 host read가 원인인 것은 아니다.

## 5. 주요 class와 struct

### 5.1 `Conv2dConfig`

위치:

```text
operations/conv/conv2d/device/conv2d_op.hpp
```

사용자가 Conv 동작을 제어하는 상위 설정이다. core replacement와 직접 관련된
필드는 다음과 같다.

```cpp
bool reshard_if_not_optimal;
bool override_sharding_config;
std::optional<TensorMemoryLayout> shard_layout;
std::optional<CoreRangeSet> core_grid;
bool transpose_shards;
```

관계:

```text
override_sharding_config=true
    -> core_grid 필수
    -> shard_layout 필수
    -> reshard_if_not_optimal=true와 동시 사용 불가
```

`core_grid`를 직접 지정할 수 있다는 사실과, 그 topology를 device factory가
지원한다는 것은 별개의 문제다. 현재 API는 `CoreRangeSet`을 받을 수 있지만
BLOCK_SHARDED factory는 내부 hole을 지원하지 않는다.

### 5.2 `sliding_window::ParallelConfig`

위치:

```text
operations/sliding_window/sliding_window.hpp
```

정의:

```cpp
struct ParallelConfig {
    CoreRangeSet grid;
    TensorMemoryLayout shard_scheme;
    ShardOrientation shard_orientation;
};
```

이 구조체는 현재 다음 두 의미를 동시에 담는다.

1. shard workload의 병렬화 구조
2. 실제 tensor가 배치될 worker core set

연속 사각형에서는 두 의미가 같아 문제가 없다. 중간 hole이 생기면 서로
달라지므로 core replacement에서 가장 먼저 분리해야 할 경계다.

### 5.3 `MemoryConfig`와 `ShardSpec`

`create_sharded_memory_config_from_parallel_config()`가 `ParallelConfig`를 실제
tensor memory 배치로 변환한다.

개념적인 결과는 다음과 같다.

```cpp
ShardSpec {
    grid,          // tensor pages가 존재할 cores
    shard_shape,   // core 하나가 가지는 [NHW, C]
    orientation,
};

MemoryConfig {
    shard_scheme,
    BufferType::L1,
    shard_spec,
};
```

BLOCK_SHARDED + ROW_MAJOR에서는 원래 다음 관계를 기대한다.

```text
grid x dimension -> channel partition
grid y dimension -> NHW partition
```

replacement 후에도 shard shape는 worker bounding box가 아니라 shard 3x3을
기준으로 계산되어야 한다.

### 5.4 `SlidingWindowConfig`

Conv의 공간 연산과 halo 생성을 설명한다.

주요 정보:

```text
batch size
input H/W
kernel H/W
stride
padding
dilation
num_cores_nhw
core_range_set
```

`conv2d_L1()`에서 input sharding이 결정된 뒤 생성되며 `ttnn::halo()`와 Conv
micro op 양쪽에 전달된다. 따라서 replacement는 Conv multicast뿐 아니라 halo의
core ordering과 shard boundary에도 영향을 줄 수 있다.

### 5.5 `Conv2dParallelizationConfig`

위치:

```text
operations/conv/conv2d/device/conv2d_op.hpp
```

정의되는 값:

```text
grid_size
num_cores_nhw
num_cores_c_in
num_cores_c_out
per-core output matrix height tiles
per-core output matrix width tiles
```

`ParallelConfig`가 tensor sharding topology를 표현한다면, 이 구조체는 Conv
matrix work가 core별로 얼마나 배분되는지를 표현한다.

현재 `grid_size`가 `ShardSpec.grid.bounding_box().grid_size()`로 만들어지는
경로가 있다. 내부 hole이 있는 3x3 placement의 bounding box가 4x4라면 여기서
shard work 크기가 오염된다.

### 5.6 `Conv2dBlockConfig`

core 내부의 matrix multiplication block 크기를 담는다.

```text
activation block height/width in tiles
output subblock height/width in tiles
```

이 값은 L1/CB 사용량과 compute kernel loop에 영향을 준다. 첫 replacement
구현에서는 shard 3x3과 per-core shard shape를 유지하여 기존 block config가
변하지 않게 하는 것이 목표다.

### 5.7 device `Conv2d` operation

상위 단계에서 계산한 설정과 tensor를 받아 program factory를 호출하는 micro
operation이다. 이 단계부터는 이미 다음 값이 확정되어 있다.

- input/output `MemoryConfig`
- `ParallelConfig`
- `Conv2dParallelizationConfig`
- `Conv2dBlockConfig`
- `SlidingWindowConfig`
- processed weights and bias

core replacement 구현은 이 경계까지 shard/worker mapping을 잃지 않고 전달해야 한다.

## 6. sharding과 core 선택의 세 경로

### 6.1 완전 자동

조건:

```text
input이 sharded가 아님
conv_config.shard_layout이 없음
```

`determine_conv_config_for_auto_shard()`가 HEIGHT, BLOCK, WIDTH 각각에 대해
대략적인 L1 사용량을 계산한다.

포함되는 비용:

- input/output/weight tensor allocation
- circular buffers
- halo output
- core count

선택한 layout에 대해 `determine_parallel_config()`가 core 수와 grid를 만든다.
자동 생성 grid는 `(0,0)`에서 시작하는 연속 배치다.

### 6.2 기존 input shard 유지

input tensor가 이미 sharded이고 `reshard_if_not_optimal=false`이면 가능한 경우
input의 `ShardSpec.grid`, layout, orientation을 그대로 사용한다.

BOS ResNet50이 자주 사용하는 방식이다. 모델 코드가 먼저
`ttnn.to_memory_config()`로 tensor를 특정 grid에 배치한 뒤 Conv가 이를 이어서
사용한다.

### 6.3 사용자 override

```python
ttnn.Conv2dConfig(
    override_sharding_config=True,
    shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    core_grid=selected_core_range_set,
    transpose_shards=False,
)
```

이 경우 `get_conv_padded_input_shape_and_mem_config()`가 다음처럼 직접
`ParallelConfig`를 만든다.

```cpp
parallel_config = {
    .grid = conv_config.core_grid.value(),
    .shard_scheme = shard_layout,
    .shard_orientation = shard_orientation,
};
```

첫 end-to-end replacement 실험은 이 경로만 사용한다. 자동 선택기와의 결합은
replacement correctness가 확인된 뒤 추가한다.

## 7. 자동 `determine_parallel_config()`의 계산

### HEIGHT_SHARDED

NHW tile 수를 device 전체 core 수 이하의 적절한 divisor로 나누고
`num_cores_to_corerangeset()`으로 grid를 만든다.

### BLOCK_SHARDED

NHW와 channel을 각각 나눈다.

```text
num_cores_nhw <- output NHW tiles와 grid dimension
num_cores_c   <- input/output channel blocks와 grid dimension
```

ROW_MAJOR인 경우 개념적으로:

```text
x axis = channels
y axis = NHW
```

현재 자동 grid는 다음처럼 원점 기준 단일 사각형이다.

```cpp
CoreRange({0, 0}, {cores_x - 1, cores_y - 1});
```

### WIDTH_SHARDED

channel tile 수를 device core 수 이하의 적절한 divisor로 나눈다.

## 8. 현재 BLOCK_SHARDED factory의 가정

위치:

```text
operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp
```

### 8.1 단일 `CoreRange`

현재 코드에는 다음 assertion이 있다.

```cpp
TT_ASSERT(
    input_cores.size() == 1,
    "Block sharded convs should have only one input core range!");
```

여기서 `size()`는 core 수가 아니라 `CoreRange`의 개수다. 중간 row/column을
제외한 `CoreRangeSet`은 여러 range가 되므로 현재 경로를 통과할 수 없다.

### 8.2 원점과 bounding box

factory는 여러 곳에서 다음을 가정한다.

```text
top-left worker core = (0,0)
grid width            = bounding_box.end.x + 1
grid height           = bounding_box.end.y + 1
bottom-right          = (num_cores_x-1, num_cores_y-1)
```

중간 hole이 있는 placement에서는 bounding box 면적과 실제 active core 수가
다르다.

### 8.3 multicast sender/receiver

BLOCK_SHARDED는 `transpose_mcast`에 따라 top row 또는 left column을 sender
set으로 만들고, 반대편 직사각형을 receiver set으로 만든다. runtime args도
연속된 x/y 좌표를 순회하여 NoC endpoint를 구성한다.

따라서 `CoreRangeSet`만 바꿔서는 충분하지 않다. 다음을 함께 변경해야 한다.

- sender worker coordinates
- receiver worker coordinates
- rectangle segments
- destination count
- semaphore 참여 cores
- core별 shard work index
- runtime argument의 NoC coordinates

## 9. shard grid와 worker placement

### shard 배치

```text
L(0,0) L(1,0) L(2,0)
L(0,1) L(1,1) L(2,1)
L(0,2) L(1,2) L(2,2)
```

### cross replacement 예시

```text
worker_x = {0, 1, 3}
worker_y = {0, 1, 3}
```

```text
L(0,0)->P(0,0)  L(1,0)->P(1,0)  L(2,0)->P(3,0)
L(0,1)->P(0,1)  L(1,1)->P(1,1)  L(2,1)->P(3,1)
L(0,2)->P(0,3)  L(1,2)->P(1,3)  L(2,2)->P(3,3)
```

필요한 내부 표현의 초안:

```cpp
struct ShardCoord {
    uint32_t x;
    uint32_t y;
};

struct CorePlacement {
    CoreCoord shard_grid_size;
    CoreCoord worker_grid_size;
    std::vector<uint32_t> worker_x;
    std::vector<uint32_t> worker_y;
    CoreRangeSet worker_cores;

    CoreCoord worker_core(ShardCoord shard) const;
};
```

첫 구현의 worker core set은 반드시 Cartesian product여야 한다.

```text
worker_cores == worker_x x worker_y
worker_cores.num_cores() == shard_grid_size.x * shard_grid_size.y
```

## 10. core replacement 삽입 위치

장기적인 연결 위치는 shard parallel config가 결정된 직후, tensor memory config가
최종 확정되기 전이다.

```text
determine_parallel_config()
        |
        v
shard ParallelConfig
        |
        v
apply_core_replacement()
        |
        +-- shard topology 유지
        `-- worker placement 추가
        |
        v
MemoryConfig / halo / Conv program factory
```

그러나 첫 실험에서는 자동 선택기를 수정하지 않고 사용자가 3x3 grid를 직접
override한다. 이로써 sharding heuristic 변화와 replacement 변화를 분리한다.

## 11. 첫 수정 순서

### 단계 A: host-only placement

기존 `segmented_rest_planner.hpp`의 다음 항목을 검증한다.

- active core 수 9개
- shard/worker 순서 보존
- 중복 worker core 없음
- excluded worker rows/columns 미포함
- Cartesian product 성립
- shard-to-worker mapping 검증

### 단계 B: shard dimension 계산

`conv2d_utils.cpp`에서 BLOCK_SHARDED의 다음 값을 worker bounding box가 아닌
선택된 shard 축 수로 계산한다.

```text
num_cores_nhw
num_cores_channels
Conv2dParallelizationConfig.grid_size
per-core shard shape
```

ROW_MAJOR 3x3 replacement에서는:

```text
num_cores_channels = worker_x.size() = 3
num_cores_nhw      = worker_y.size() = 3
shard grid_size    = (3,3)
```

### 단계 C: compute placement

shard work index를 보존하면서 kernel을 대응 worker core에 생성한다. 이
단계에서는 multicast 변경 전에 core별 고유 값 또는 status를 기록하여 mapping을
검증한다.

### 단계 D: 한 축 multicast

row gap과 column gap을 따로 처리한다.

- 선택된 축 좌표를 contiguous intervals로 분리
- interval마다 NoC rectangle 발행
- 실제 receiver 수로 destination count 계산
- 모든 segment write가 끝난 뒤 semaphore release

### 단계 E: cross replacement

weight와 activation 양쪽의 segmented route를 동시에 활성화한다.

### 단계 F: Conv correctness

- PyTorch golden과 PCC 비교
- output shape 확인
- skipped/inactive core sentinel 확인
- receiver가 semaphore wakeup 후 최신 data를 관측했는지 확인

## 12. BOS ResNet50의 역할

`bos_model_copy/resnet50`은 첫 replacement 대상이 아니라 known-good environment
baseline이다.

이 모델은 다음과 같이 수동 최적화를 포함한다.

- host input fold
- 첫 Conv HEIGHT_SHARDED
- 14-core HEIGHT_SHARDED intermediate placement
- 후반 4x4 BLOCK_SHARDED placement
- layer별 `reshard_if_not_optimal` 정책
- layer별 double buffer와 split reader 정책

따라서 ResNet50 통과가 의미하는 것은 다음과 같다.

```text
board + driver + stock TTNN + stock Conv path가 정상
```

작은 Conv replacement가 통과한 후에만 ResNet50의 BLOCK_SHARDED layer 하나에
적용한다.

## 13. 테스트 계층

```text
1. planner unit test              (host only)
2. shard/worker mapping test      (host only)
3. shard math test                (host only)
4. kernel placement test          (device)
5. one-axis mcast test            (device)
6. cross mcast test               (device)
7. small Conv PCC test            (device)
8. one ResNet50 block test        (device)
9. full ResNet50 test             (device)
```

device test는 이전 단계가 통과한 경우에만 실행한다.

## 14. 안전 규칙

- 보드 상태가 의심되면 먼저 unmodified BOS baseline을 실행한다.
- replacement 실험과 baseline을 같은 프로세스에서 섞지 않는다.
- device test는 명시적 환경변수 opt-in을 사용한다.
- command timeout은 host process만 끝낼 뿐 device recovery를 보장하지 않는다.
- `ttnn.conv2d()` 반환 로그만으로 성공을 판단하지 않는다.
- 반드시 blocking read 또는 queue finish까지 확인한다.
- hang 후 다른 실험을 연속 실행하여 결과를 해석하지 않는다.

## 15. 현재 확인된 사실과 미확인 항목

### 확인됨

- 자동 layout/core 선택 경로가 존재한다.
- 기존 input shard 유지 경로가 존재한다.
- `Conv2dConfig.core_grid` 직접 override 경로가 존재한다.
- BLOCK_SHARDED factory는 한 개의 `CoreRange`를 요구한다.
- factory는 여러 곳에서 origin과 bounding box를 사용한다.
- microbenchmark의 1/2/4 segment data 및 receiver ordering은 통과했다.
- `CorePlacement`의 ShardCoord/WorkerCoord 양방향 mapping을 구현했다.
- 연속 3x3, worker row/column 경계 이동, 가운데 cross gap의 host-only
  mapping test를 추가했다.
- worker replacement 여유가 없거나 좌표가 범위를 벗어나면 device enqueue
  전에 host exception으로 거부한다.
- BLOCK_SHARDED Conv에 early preflight를 추가했다. non-contiguous grid는
  reshard/halo/Conv enqueue 전에 `TT_FATAL`로 거부한다.
- contiguous grid translation은 output placement, multicast endpoints,
  sender ID, config-reader index, weight-slice index가 translated origin 기준
  상대 좌표를 사용하도록 production path에 연결했다.
- contiguous remapping lifetime 결정을 `conv_core_remapping_planner.hpp`로
  분리했다. stock output topology는 기존 heuristic으로 먼저 계산하고,
  planner는 same-or-smaller topology만 origin을 전파하며 한 축이라도
  커지면 `ExitToStock`과 input reshard를 선택한다.
- BOS layer4 downsample의 translated 4x4 input -> stock 5x4 output 경로가
  device에서 완료되었고 PCC 0.999930으로 통과했다.
- `CorePlacement`를 production header `conv_core_placement.hpp`로 승격하고,
  각 axis의 `logical_start/worker_start/length` segment 및 그 Cartesian
  product인 rectangle route metadata를 추가했다. row/column/cross gap은
  각각 2/2/4개 rectangle으로 표현되며 아직 device enqueue에는 연결하지
  않았다.

### 아직 확인 필요

- translated 4x4 `CORE_SET_16`을 사용하는 BOS layer4/full-model device PCC
- halo가 non-contiguous worker ordering을 처리하는 범위
- production Conv reader/writer kernel에 필요한 최소 segment metadata
- program cache key에 placement를 추가해야 하는 정확한 위치
- output sharding이 input과 다른 경우의 replacement mapping
- split reader와 replacement의 상호작용

## 16. 구현 중 지켜야 할 원칙

1. shard work partition과 worker core placement를 혼합하지 않는다.
2. core 수 대신 bounding-box 면적을 사용하지 않는다.
3. segment destination count에는 sender를 포함하지 않는다.
4. WorkerCoord는 NoC endpoint 생성 직전에 NocCoord로 변환한다.
5. shard 크기와 tile 수는 shard topology로 계산한다.
6. semaphore 참여자는 실제 active worker cores로 계산한다.
7. 한 번에 sharding mode 하나만 변경한다.
8. 새로운 단계마다 host-only invariant를 먼저 추가한다.
