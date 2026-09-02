# Conv Core Remapping Design

Last updated: 2026-08-18

이 문서는 Conv core replacement/remapping 작업의 현재 설계 결정과 검증 상태를 보존한다. 구현 가능성을
보여 주는 host-only 코드와 실제 device에서 지원된 경로를 구분하며, 아직 열지 않은 segmented device
경로를 지원 완료로 간주하지 않는다.

## 1. 목표와 비목표

목표는 특정 worker row/column/core의 사용을 피하면서 logical Conv workload를 다른 worker 좌표에
결정론적으로 배치하는 것이다. 주요 사용 목적은 장시간 실행 시 코어 사용 불균형과 aging을 완화하는
것이다.

Remapping은 모델 전체에 영구적으로 붙는 속성이 아니다. 특정 tensor/branch 진입점에 명시적으로 부여되는
placement이며, 이후 Conv가 호환되는 topology를 유지하는 동안에만 전파된다.

현재 단계에서 다음 항목은 목표가 아니다.

- stock Conv가 선택한 output core 개수를 강제로 input core 개수와 같게 만들기
- 모든 sharding 방식과 모든 non-Cartesian core set을 한 번에 지원하기
- runtime마다 무거운 최적화 planner를 실행하기
- 검증되지 않은 segmented grid를 device에 제출하기

## 2. Remapping lifetime

`plan_conv_core_remapping()`은 stock output topology의 크기를 변경하지 않고 placement의 수명만 결정한다.

```text
explicitly remapped input
        |
        +-- current operation supports the placement
        |   and output.x <= input.x and output.y <= input.y
        |       -> Propagate
        |       -> input worker axes의 logical prefix를 output에 적용
        |
        +-- current operation does not support the placement
        |   or output.x > input.x or output.y > input.y
                -> ExitToStock
                -> input을 origin-anchored stock placement로 reshard
                -> output도 stock placement 사용
```

Action은 다음 세 가지다.

- `NotActive`: 명시적 remapping이 활성화되지 않음
- `Propagate`: 실행 가능한 placement에서 same-or-smaller output이 input placement의 logical prefix를 상속
- `ExitToStock`: 어느 한 축이라도 확장되거나 다음 operation이 현재 placement를 실행할 수 없으면 remapping을
  종료하고 stock origin으로 복귀

Remapping이 종료된 뒤 사용자가 다른 tensor 진입점에서 다시 명시적으로 remapping하는 것은 허용한다.

### 2.1 Placement provenance

`CoreRangeSet`의 origin, bounding box, core count만으로는 remapped placement의 원래 stock topology를
유일하게 복원할 수 없다. 예를 들어 5-wide stock prefix에서 column 0을 제외한 physical `x1..4`는 bounding
width가 4지만 원래 row-scan width는 5다. 따라서 HEIGHT consumer는
`Conv2dConfig.core_remapping_input_stock_grid`에 현재 입력의 stock grid를 명시한다.

- exclusion을 직접 요청한 Conv: preparation이 exclusion 전 `ParallelConfig`를 provenance로 사용
- remapped output을 받는 다음 Conv: caller/model planner가 producer의 stock output grid를 config에 전달
- lifetime planner: actual input grid와 stock provenance를 별도로 받아 `Propagate`/`ExitToStock` 결정
- `ExitToStock`: output topology나 remapped bounding box로 input stock grid를 재구성하지 않고 provenance 사용
- provenance 없이 non-canonical HEIGHT input이 들어오면 device enqueue 전에 host fail-fast

이 metadata는 Conv 전용 정책이므로 전역 `ShardSpec`/Tensor ABI에는 넣지 않는다. BLOCK_SHARDED의 기존
구조적 compatibility adapter는 유지하지만 HEIGHT production lifetime은 explicit provenance 경로를 사용한다.

## 3. Logical shard와 worker 좌표 분리

`ShardCoord`는 dense logical workload 좌표이고 `CoreCoord`는 TT-Metal worker 좌표다.

```text
ShardCoord
    -> CorePlacement::worker_core()
Worker CoreCoord
    -> IDevice::worker_core_from_logical_core()
Physical NoC coordinate
```

역변환은 `CorePlacement::shard_core()`가 담당한다. Conv의 sender ID, weight slice, config index와 compute
skip 여부는 가능한 한 logical 좌표로 계산하고, 실제 NoC endpoint를 만들 때만 worker/NoC 좌표로 변환한다.

예를 들어 worker column 2를 제외하면 다음 매핑이 가능하다.

```text
logical x: 0  1  2
worker  x: 0  1  3
```

Tensor shard ordering도 logical order를 따라야 한다. TT-Metal은 `CoreRangeSet`의 range 순서대로 각 range를
전개하므로, segmented placement는 ROW_MAJOR 또는 COL_MAJOR logical 순서의 singleton `CoreRange` 목록으로
직렬화한다. 큰 물리 rectangle 두 개만 넘기면 shard 순서가 rectangle 단위로 바뀔 수 있으므로 사용하지
않는다.

Conv/Halo config tensor는 data tensor와 반대 orientation으로 shard되는 경로가 있다. Dense rectangle에서는
orientation이 range 내부 순서를 정하지만 singleton segmented grid에서는 range 순서가 우선한다. 따라서
config tensor upload 시 동일 worker set을 config orientation 순서로 다시 직렬화한다. Data grid 자체의
placement는 바꾸지 않는다.

Python `create_sharded_memory_config_`는 BLOCK_SHARDED grid를 physical bounding rectangle로 해석하고
`bounding_box area == num_cores`를 요구한다. Isolated segmented device test는 이 helper를 우회해 logical 4x4
topology로 shard shape를 계산하고 `ShardSpec`/`MemoryConfig`를 직접 생성한다. Production full-model 경로는
사용자가 segmented `MemoryConfig`를 직접 만들지 않는다. `Conv2dConfig`의 exclusion 요청을 Conv preparation이
해석하고 stock input을 선택된 segmented placement로 내부 reshard한다.

Legacy `BLOCK_SHARDED` tensor는 내부적으로 `ShardDistributionStrategy::GRID_2D`를 사용한다. 기존
`BufferDistributionSpec`은 여기서 단일 `CoreRange`만 허용했기 때문에 Python helper를 우회해도 segmented
input allocation 단계에서 거부됐다. Multi-range `GRID_2D` 지원은 임의의 core set을 허용하지 않는다.
정렬된 selected X/Y axis의 Cartesian product인지 검증한 뒤, physical bounding box가 아니라 logical axis
순서로 page-to-core list를 생성한다. 따라서 `{0,1,3,4} x {0,1,2,3}`은 논리 4x4로 배치되고 skipped column은
buffer shard를 받지 않는다.

## 4. Axis segmentation

각 ordered worker axis에서 물리 좌표가 `+1`로 이어지지 않는 지점을 segment boundary로 간주한다.

```text
worker_x = {0, 1, 3, 4}

segment 0: logical [0..1] -> worker [0..1]
segment 1: logical [2..3] -> worker [3..4]
```

X segment와 Y segment의 Cartesian product가 rectangular route unit을 만든다.

- X만 끊김: rectangle 2개
- Y만 끊김: rectangle 2개
- X와 Y 모두 끊김: rectangle 4개
- 여러 좌표 제외: N개의 ordered segment/rectangle

exclusion vector의 길이 자체는 제한하지 않는다. 결과 placement가 축별 최대 2 segments, Cartesian 최대
4 rectangles인지만 제한한다. 따라서 경계의 연속된 여러 좌표 제외는 1-segment translation이 될 수 있다.

## 5. Conv activation multicast

BLOCK_SHARDED 2D Conv의 activation sender는 하나의 고정 core가 아니다. input channel shard를 담당하는
logical X core들이 round-robin으로 sender가 된다.

```text
act_mcast_sender_id = logical_core.x        // non-transposed ROW_MAJOR path

for act_w_outer_i in input channel shards:
    if act_w_outer_i == act_mcast_sender_id:
        this core is the sender
```

Column gap이 있으면 host factory가 output logical row를 물리적으로 연속된 여러 segment로 분해한다. Device
kernel은 excluded worker를 직접 탐색하지 않고 runtime arguments에 들어 있는 segment만 전송한다.

Runtime format:

```text
segment_count,
start_x, start_y, end_x, end_y, destination_count, contains_sender,
...
```

각 logical activation block의 protocol은 다음과 같다.

1. 모든 receiver가 rotating sender semaphore를 증가시킨다.
2. sender가 필요한 receiver 수를 기다린다.
3. sender가 자신의 compute CB로 local copy를 한 번 수행한다.
4. sender를 제외한 remote-only data segment를 모두 issue한다.
5. 모든 data segment 뒤에 write barrier를 정확히 한 번 수행한다.
6. sender가 각 remote segment에 VALID semaphore를 전송하고 local VALID를 설정한다.
7. receiver가 local VALID를 기다린 뒤 compute CB를 진행한다.

Host planner는 sender가 들어 있는 물리 segment를 sender 앞/뒤의 remote-only rectangle로 다시 분할한다.
따라서 device multicast rectangle에는 sender가 절대 포함되지 않으며 loopback-source multicast를 사용하지
않는다. 각 rotating sender의 remote destination 합은 항상 `logical_width - 1`이어야 한다. 최초 device
시도에서 sender-containing loopback multicast가 deadlock을 일으켰고, 이 remote-only protocol로 교체한 뒤
동일 case가 정상 완료됐다.

NOC1에서는 기존 Conv convention과 동일하게 rectangle endpoint 순서를 반대로 직렬화한다.

## 6. Conv weight multicast

Non-transposed ROW_MAJOR BLOCK_SHARDED Conv에서 weight sender는 output placement의 첫 logical row에 있는
각 logical X core다. 각 sender는 자신의 logical X에 해당하는 weight slice를 동일 X열의 receiver core들에
전송한다.

Column-gap 첫 지원 범위에서는 다음이 성립한다.

- selected X마다 top-row sender가 하나 존재
- Y axis는 물리적으로 연속
- weight multicast column은 각각 하나의 연속 segment
- bounding rectangle의 빈 column에 kernel을 생성하지 않음

따라서 factory는 `output_placement.worker_x`의 실제 top-row worker들만 `mcast_sender_cores`에 넣고,
`output_cores - mcast_sender_cores`를 weight receiver set으로 사용한다.

Row gap에서는 weight multicast column 자체가 여러 segment로 나뉘므로 별도 segmented weight protocol이
필요하다. 아직 production 구현 범위가 아니다.

## 7. Halo는 sharding 방식별로 다르다

현재 실제 Halo 경로는 다음과 같다.

```text
HaloDeviceOperation
    -> untilize_with_halo_multi_core()
    -> halo_gather.cpp
```

호출되지 않는 `inplace_untilize_with_halo_multi_core()`와 존재하지 않는 `halo_gather_in_place.cpp` 문자열은
현재 Conv 실행 경로가 아니다. 이를 segmented 지원의 필수 경로로 간주하지 않는다.

`halo_gather.cpp`는 대표 sender가 rectangle 전체로 multicast하는 구조가 아니다. 각 source core가
`gather_config`를 읽고 필요한 stick을 destination core로 unicast한다. 따라서 activation mcast처럼 모든
halo를 물리 segment별 multicast로 바꾸지 않는다. 대신 logical destination을 올바른 worker/NoC 좌표로
변환해야 한다.

### Height sharding

NHW가 linear core sequence에 분할된다. Destination X와 Y를 모두 config가 지정하므로 arbitrary gap 지원 시
전체 logical core ID에서 worker coordinate로의 변환이 필요하다.

### BLOCK_SHARDED ROW_MAJOR

- X axis: channel shards
- Y axis: NHW shards
- halo destination X: `my_noc_x`
- halo destination Y: gather config

따라서 column gap은 각 선택된 X열 안에서 Y방향 unicast가 독립적으로 수행되어 새로운 segmented halo
protocol이 필요하지 않다. 다만 halo/config 생성에 사용하는 channel-column cardinality는 physical bounding
box 폭이 아니라 selected logical X 개수여야 한다. Row gap은 gather config의 logical Y를 selected physical
worker Y로 변환해야 한다.

### BLOCK_SHARDED COL_MAJOR

- Y axis: channel shards
- X axis: NHW shards
- halo destination X: gather config
- halo destination Y: `my_noc_y`

따라서 row gap은 추가 변환 없이 유지될 수 있지만 column gap은 gather config의 logical X를 selected physical
worker X로 변환해야 한다.

### Width sharding

Channel만 분할되고 각 core가 전체 spatial extent를 보유하므로 일반적으로 remote halo 교환이 없다. Halo
관점에서 placement gap의 영향은 제한적이지만 별도 device 검증 전에는 지원 완료로 표시하지 않는다.

## 8. 현재 production 지원 경계

| Case | Host representation | Production code | Device validation |
|---|---:|---:|---:|
| Stock contiguous origin | Yes | Yes | Yes |
| Contiguous translated rectangle | Yes | Yes | Yes |
| Planner propagation on same/smaller topology | Yes | Yes | Yes |
| Planner ExitToStock on expansion | Yes | Yes | Yes |
| Planner ExitToStock at unsupported operation boundary | Yes | Yes | Yes; segmented 3x3 Conv to stock 1x1 Conv chain |
| ROW_MAJOR block-sharded single column gap | Yes | Config-driven 3x3 input selection, preflight/factory/kernel, and Cartesian buffer distribution | Yes; isolated and full-model BOS paths |
| ROW_MAJOR block-sharded single row gap | Yes | Config-driven 3x3 input selection, segmented weight/bias/semaphore multicast, halo Y translation, and Cartesian buffer distribution | Yes; isolated 4x3 Conv2 and Conv2-to-stock-Conv3 chain |
| ROW_MAJOR block-sharded row+column gap (four rectangles) | Yes | Combined segmented activation and weight multicast | Yes; isolated Conv2, Conv2-to-stock-Conv3 chain, and BOS full model |
| COL_MAJOR segmented block sharding | Yes | Incomplete | No |
| ROW_MAJOR height-sharded ordered gaps | Yes | 1D placement/lifetime, halo mapping, segmented weight multicast, and explicit stock-grid provenance implemented | Guarded column/row/four-rectangle/boundary slices and full-model lifetimes passed |
| ROW_MAJOR width-sharded ordered gaps | No | Current kernel assumes flattened bounding-box core IDs and one loopback multicast rectangle | Host fail-fast retained pending ordered-rank and segmented-loopback implementation |

현재 segmented production slice의 조건은 다음과 같다.

```text
BLOCK_SHARDED
ROW_MAJOR input and output
non-transposed multicast
input placement == output placement
X axis has at most two contiguous segments
Y axis has at most two contiguous segments
X gaps: segmented activation multicast
Y gaps: segmented weight/bias/semaphore multicast
Cartesian product has at most four rectangles
```

HEIGHT_SHARDED의 첫 slice는 같은 ROW_MAJOR 3x3 capability gate를 사용한다. Stock core cardinality는 유지하면서
제외된 physical row/column을 건너뛴 row-major worker sequence를 선택한다. Halo와 Conv reader는 이 순서를 logical
NHW core id로 사용하고, weight/bias/semaphore multicast는 실제 receiver만 포함하는 maximal rectangles로
분할한다. 다음 operation이 이 placement를 실행할 수 없으면 explicit stock-grid provenance를 사용해 1D stock
placement로 ExitToStock한다. Column gap, row gap, four rectangles, boundary translation의 guarded
isolated/lifetime/full-model 검증이 완료됐다.

Factory와 Conv preparation preflight는 ROW_MAJOR 3x3 Conv에서 축별 최대 2 segments를 허용한다. 경계
exclusion으로 생기는 contiguous translation과 row+column 동시 gap도 표현할 수 있다. 3 segments 이상을
만드는 exclusion, COL_MAJOR 및 다른 kernel size는 host/factory에서 거부한다. Four-rectangle isolated Conv2와
Conv2-to-stock-Conv3 chain은 모두 guarded device test를 통과했다.
Guarded 검증은 buffer distribution과 TensorSpec의 기존 contiguous-only gate를 단계적으로 열고, Halo-only
동기화로 Halo 완료를 확인한 뒤 Conv를 실행했다. 최초 Conv 실행은 sender-containing loopback multicast에서
deadlock됐지만, sender local copy와 remote-only rectangles로 교체한 뒤 Conv와 output readback이 완료됐다.

현재 사용자-facing 선택 API는 다음과 같다.

```python
config = ttnn.Conv2dConfig(
    enable_core_remapping=True,
    excluded_worker_columns=[2],
)

row_config = ttnn.Conv2dConfig(
    enable_core_remapping=True,
    excluded_worker_rows=[2],
)

# Consumer of a tensor whose producer kept a remapped HEIGHT placement.
consumer_config = ttnn.Conv2dConfig(
    enable_core_remapping=True,
    core_remapping_input_stock_grid=producer_stock_grid,
)
```

`excluded_worker_rows`, `excluded_worker_columns`, `core_remapping_input_stock_grid`는 Python/C++ config 및
repr에 노출된다. 현재 production
slice에서는 `enable_core_remapping=True`와 함께 두 축의 exclusion을 동시에 지정할 수 있다. Exclusion 개수
자체가 아니라 선택 결과가 축별 최대 두 contiguous segments인지 검사하며, 따라서 boundary/연속 exclusion은
하나의 translated segment로 자연스럽게 허용된다. 어느 축이든 세 segments 이상이면 device enqueue 전에
거부한다. Exclusion이 비어 있고 remapping이 비활성화된 기본 설정은 기존 stock Conv 동작을 유지한다.

## 9. 검증 현황

Host unit tests:

- total: 36/36 passed on 2026-08-18

검증 내용:

- contiguous baseline 및 boundary shift
- middle row/column exclusion
- logical↔worker round trip
- multiple exclusions 및 bounds validation
- Cartesian user-grid normalization
- segmented Cartesian `GRID_2D` ROW_MAJOR/COL_MAJOR page-to-core ordering
- unused logical-axis trimming and non-Cartesian buffer-grid rejection
- segmented Cartesian legacy/ND TensorSpec validation and selected-axis capacity checks
- ROW_MAJOR/COL_MAJOR tensor shard order
- rotating sender별 segment ownership/destination count
- singleton sender zero-destination 방지
- sender outside receiver placement
- NOC0/NOC1 runtime serialization
- planner segmented logical prefix propagation
- expansion 시 ExitToStock
- segmented 실행을 지원하지 않는 다음 operation 진입 시 ExitToStock
- explicit column exclusion의 segmented input 선택 및 unsupported exclusion fail-fast
- sliding-window replication factor가 bounding box가 아닌 selected axis size를 사용
- HEIGHT_SHARDED explicit stock-grid provenance validation 및 boundary ExitToStock
- HEIGHT_SHARDED non-loopback weight multicast rectangle의 exact remote receiver coverage 및 NOC0/NOC1 serialization
- WIDTH_SHARDED exclusion 요청의 host fail-fast

HEIGHT_SHARDED logical order/lifetime, exact multicast coverage, boundary provenance와 WIDTH_SHARDED fail-fast를
포함한 전체 host suite가 통과했다.

Device에서 검증 완료된 contiguous cases:

- translated layer4 1x1 Matmul-lowered Conv
- translated layer4 3x3 Conv
- translated downsample의 ExitToStock
- translated layer4/full BOS ResNet50

Device에서 검증 완료된 segmented case:

- layer4 module1 3x3 Conv의 logical 4x4 ROW_MAJOR placement
- physical X `{0,1,3,4}`, physical Y `{0,1,2,3}`
- segmented input upload/readback PCC 1.0
- segmented Halo synchronization 완료
- segmented Conv synchronization 및 output readback 완료
- stock vs segmented output PCC 1.0
- layer4 module1 Conv2의 segmented 16-core output 유지
- 다음 matmul-lowered Conv3 진입 전 stock contiguous 4x2(8-core)로 reshard 후 완료
- Conv2-to-Conv3 chained output vs PyTorch golden PCC `0.9998673514960605`
- `Conv2dConfig(enable_core_remapping=True, excluded_worker_columns=[2])`로 layer4 module1 Conv2만 선택한
  full BOS ResNet50 완료
- public Conv2 input은 stock 16-core, Conv preparation 내부 reshard 후 output은 segmented 16-core 유지,
  Conv3 output은 stock contiguous 8-core로 복귀
- config-driven segmented full-model 최종 PCC `0.999603608935384`
- 동일 build에서 remapping 기본값이 비활성화된 unmodified BOS environment baseline 재검증 완료
- logical 4x3 ROW_MAJOR Conv2에서 physical X `{0,1,2,3}`, physical Y `{0,1,3}` row gap 완료
- explicit segmented grid와 `excluded_worker_rows=[2]` config-driven 경로가 stock Conv2와 일치
- row-gap 12-core Conv2 output을 stock contiguous 4x2(8-core) Conv3로 reshard한 chain과 PyTorch golden PCC 통과
- logical 4x3 placement에서 physical X `{0,1,3,4}`, physical Y `{0,1,3}`인 four-rectangle Conv2 통과
- explicit four-rectangle grid와 row+column config-driven exclusion이 stock Conv2 결과와 일치
- four-rectangle 12-core Conv2 output을 stock contiguous 4x2(8-core) Conv3로 reshard한 chain과 PyTorch golden
  PCC 통과
- full BOS ResNet50의 실제 layer4 module2 Conv2에서 logical 4x2(8-core)를 physical X `{0,1,3,4}`, Y
  `{0,2}`의 four-rectangle placement로 실행하고 다음 Conv3에서 stock 4x2로 복귀
- config-driven row+column full-model 최종 PCC `0.999603608935384`
- Eagle-N은 physical worker row가 4개이므로 logical 4x4의 16-core full-model Conv2 row skip은 core 수를 유지할
  여분 row가 없다. 이 때문에 row path는 4x3 isolated/chain으로, column path는 full model로 검증했다.
- 2026-08-18 최신 build에서 host 36/36 및 HEIGHT guarded slices 재검증 완료

HEIGHT_SHARDED 첫 device 시도는 14-core stock case의 input-upload 로그 이후 watchdog가 발동했다. 정적 감사에서
factory가 BOS의 5-wide stock topology를 전체 Blackhole worker-grid 폭과 비교하여 stock을 remapped로 오판한 것이
확인되었다. 중간 수정은 physical row-scan boundary를 사용했지만 boundary exclusion은 bounding geometry 자체가
충분한 provenance가 아님을 다시 드러냈다. 최종 구현은 stock grid를 config/planner 경계에 명시적으로 전달한다.

clean-device 재시도에서 stock Conv는 completion까지 통과했고 segmented input도 readback PCC 1.0을
통과했지만, segmented Conv submit 이후 synchronize에서 watchdog가 발동했다. 따라서 문제 범위는
device program 내부로 한정된다. 초기 HEIGHT translation은 receiver를 행별 horizontal run으로만
나누어 production column gap을 7 rectangles로 만들었다. 인접 row에서 X bounds가 같은 run을 vertical로
합쳐 exact cover를 만들도록 수정했다. non-loopback multicast는 rectangle에 sender가 포함되어도 self-write하지
않으므로, sender를 geometry에 복원하고 해당 segment의 destination count에서만 1을 뺀다. 그 결과
planner의 두 physical band `(0,0)-(1,3)`(7 remote) + `(3,0)-(4,2)`(6 remote)를 그대로 유지하여
7→2 rectangles로 최종 압축했다. 또한 1D weight sender는 서로 독립적인 rectangle을 모두 `linked=true`로
보내고 있었다. 이미 실기 검증된 segmented activation protocol처럼 각 rectangle을 독립적인
`linked=false` multicast로 변경했고, 기존 data barrier 후 semaphore를 publish하는 순서는 유지했다.
이 수정의 guarded isolated 재검증은 stock/golden 및 segmented/golden PCC
`0.9999153194491188`, stock/segmented PCC `1.0`으로 통과했고 driver도 정상 종료했다. 이어 실제 full
ResNet50에서도 layer1 module1 Conv2가 segmented placement를 출력한 뒤 matmul-lowered Conv3가 stock
14-core placement로 `ExitToStock`하는 lifetime 경계와 최종 correctness validation이 통과했다. full-model
call은 45.71s, 전체 실행은 50.89s였고 driver close 및 DMA unpin까지 정상 완료됐다.

## 10. 안전 정책

- Host validation에서 지원하지 않는 grid는 device enqueue 전에 실패해야 한다.
- Preflight가 열려도 guarded device correctness test 전에는 지원 완료로 간주하지 않는다.
- Device test는 module-level device gate와 high-risk gate를 모두 사용한다.
- High-risk test는 `run_bos_resnet50_full_model_guarded.sh`를 통해서만 실행한다.
- Host watchdog은 host process를 종료할 수 있지만 deadlocked device를 복구하지는 못한다.
- 새로운 segmented case는 isolated Conv에서 stock/golden/translated 비교를 통과하기 전에 full model에 적용하지
  않는다.

## 11. 다음 단계

HEIGHT patch의 현재 종료 조건은 explicit provenance가 포함된 전체 host regression과 기존 guarded HEIGHT
regression의 재통과다. 이후 확장은 별도 작업으로 취급한다.

1. workload별 logical topology에서 row/column rotation 후보를 자동 선택하는 상위 planner 설계
2. WIDTH_SHARDED는 ordered worker-rank map, exact active-core launch, segmented loopback data/semaphore multicast의
   비용 모델이 성립할 때만 별도 slice로 개방
3. COL_MAJOR remapping은 별도 kernel capability와 guarded correctness slice로 확장

아래는 완료된 HEIGHT 검증 기록이다. BOS 5x4 topology에서 physical row 2를 제외하는 14-core placement의
logical order는 `y0: x0..4`, `y1: x0..4`, `y3: x0..3`이며, weight/bias multicast remote cover는
`(0,0)-(4,1)`의 9 destinations와 `(0,3)-(3,3)`의 4 destinations로 정확히 두 rectangle이다. Host
planner/rectangle test와 guarded isolated correctness를 고정했다.
해당 isolated test는 stock/golden 및 row-gap/golden PCC `0.9999153194491188`, stock/row-gap PCC `1.0`으로
통과했다. 이어 같은 row-gap을 실제 layer1 module1 Conv2에 주입해 Conv3의 stock 14-core
`ExitToStock`과 최종 ResNet50 correctness를 검증했다. 이 full-model test도 PCC `0.999603608935384`로
통과했고, call 45.47s/전체 50.75s 뒤 driver close와 DMA unpin까지 정상 완료됐다.

row와 column을 동시에 제외하면 BOS 5x4 grid에는 12 workers만 남으므로 기존 14-core full-model topology를
그대로 유지할 수 없다. 대신 동일한 layer1 `56x56, 64→64, 3x3` Conv2를 12-core stock prefix와 physical
X `{0,1,3,4}`, Y `{0,1,3}` four-rectangle placement에서 비교한다. 네 multicast rectangles의 remote
destination counts는 `3,4,2,2`이고 총 11 receivers를 정확히 덮는다. Host test는 ordered workers,
rectangle geometry, NOC0/NOC1 serialization, Propagate를 고정한다. guarded isolated test는
stock/golden 및 segmented/golden PCC `0.9999153194491188`, stock/segmented PCC
`1.0`으로 통과했다. 이어진 chain은 같은 config-driven four-rectangle Conv2 출력이 유지되는지 확인한 뒤
matmul-lowered Conv3에서 canonical stock placement로 `ExitToStock`하고 두 Conv의 PyTorch golden과 비교한다.
첫 실행에서 두 device program은 모두 completion까지 도달했지만 테스트가 Conv3 output을 12 cores로
잘못 가정해 assertion이 실패했다. 실제 stock optimizer는 98 NHW tiles에 대해 11-core prefix
`(0,0)-(4,1) + (0,2)`를 선택한다. planner host test와 Python expected grid를 이 12→11 transition으로
수정했다. corrected chain은 PCC `0.9998905649290319`로 통과했고, call 7.00s/전체 12.38s 뒤 driver
close와 DMA unpin까지 정상 완료됐다.

HEIGHT의 마지막 placement slice는 column 0을 제외하는 14-core boundary translation이다. ordered workers는
physical `x1..4 × y0..2` 뒤 `x1..2 × y3`이고 logical sender 0은 physical `(1,0)`으로 이동한다. remote
multicast는 `(1,0)-(4,2)`의 11 destinations와 `(1,3)-(2,3)`의 2 destinations로 압축된다. Host test는
ordered workers, sender-containing geometry, NOC0/NOC1 endpoint serialization, Propagate를 고정한다. Guarded
isolated test는 stock/golden 및 translated/golden PCC `0.9999153194491188`, stock/translated PCC `1.0`으로
통과했고 driver close와 DMA unpin도 정상 완료됐다. 첫 guarded full-model run은 Conv3 matmul host validation에서
translated `x1..4` bounding width를 4-wide stock topology로 오인한 것을 드러냈다. 최종 수정은 stock grid를
`core_remapping_input_stock_grid`로 명시해 원래 5-wide prefix를 보존한다. 수정된
full-model test는 Conv2의 translated output 전파, Conv3의 canonical stock grid 복귀, 최종 ResNet50 correctness를
함께 검증해 PCC `0.999603608935384`로 통과했다. Call 45.37s/전체 50.91s 뒤 driver close와 DMA unpin도
정상 완료됐다.

## 12. 관련 구현 파일

```text
ttnn/cpp/ttnn/operations/conv/conv2d/conv_core_placement.hpp
ttnn/cpp/ttnn/operations/conv/conv2d/conv_core_remapping_planner.hpp
ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp
ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp
ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/
    reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp
ttnn/cpp/ttnn/operations/sliding_window/sliding_window.cpp
tests/tt_metal/tt_metal/data_movement/conv_core_replacement/
```
