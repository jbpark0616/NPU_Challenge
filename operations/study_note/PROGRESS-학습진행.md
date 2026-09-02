# tt-metal conv 코어 앵커링 학습 — 진행 상황

> **새 대화에서 이어갈 때:** 이 파일 하나만 읽으면 어디까지 왔는지 파악됨. `study_note/` 폴더의 다른 노트들과 함께 보면 됨.

---

## 최종 목표

**tt-metal conv2d의 "코어 앵커링"을 이해하고, 코어 배치(위치/시프팅)를 수정할 수 있게 되기.**
- 배경: HCI/NBTI 열화 시뮬레이터 아이디어의 밑돌 (코어별 활동량이 열화 입력) — 장기 목표
- 사용자: C++ 입문자 (Python 배경). 문법을 필요할 때만 보고 코드 독해 위주로 진행 중
- 레포: `C:\NPU_Challenge\operations` (보스 버전 tt-metal의 operations 폴더만 떼온 것. tt-metal 본체 .cpp는 없음 → api는 사용법만 알면 됨)

---

## 지금까지 이해한 큰 그림

### 9단계 실행 흐름 (host가 디바이스에 일 시키는 파이프라인)
```
① 디바이스 열기  ② 큐  ③ 프로그램  ④ 버퍼
⑤ 코어 그리드 파악 ★  ⑥ 일 나누기 ★  ⑦ 커널 배치 ★  ⑧ 코어별 인자 ★  ⑨ 실행/회수
```
- **코어 앵커링은 ⑤⑥⑦⑧ 네 지점에만 산다.** 나머지는 코어와 무관한 배관.
- 상세: `tt-metal-flow-map.md` (작업폴더에 있음, 아직 study_note로 안 옮김)

### conv 코어 앵커링 지점 (실제 파일)
```
determine_parallel_config (conv2d_utils.cpp)  → 코어 결정, ParallelConfig 반환  ⑤⑥
        ↓
conv2d_op_sharded_program_factory.cpp         → all_cores 확정 + CreateKernel + SetRuntimeArgs 루프  ⑦⑧
```
- **코어와 데이터(샤딩)가 얽혀 있음.** 특히 mcast 물리좌표(`worker_core_from_logical_core`)가 코어 배치에 묶여서, 코어 시프팅 시 데이터 경로도 같이 바꿔야 함 → conv가 vecadd보다 까다로운 이유

---

## 코드 독해 진행 (재료 → 조립 순서)

### ✅ 완료: 재료 1 — core_coord.hpp (코어 위치 타입)
- **CoreCoord** = 코어 1개 (x,y 점) / **CoreRange** = 사각형 1개 (시작·끝 좌표쌍) / **CoreRangeSet** = 안 겹치는 사각형 집합 (vector에 담음)
- 하는 일: "어느 코어 쓸지" **위치 표현·관리만**. 실제 배치는 CreateKernel이 함
- 시프팅에 쓸 무기: `select_from_corerangeset`, `select_contiguous_range_from_corerangeset`, `subtract`
- 상세: **`core_coord-정리노트.md`**

### ✅ 완료: 재료 2 — ParallelConfig (sliding_window.hpp 16줄)
```cpp
struct ParallelConfig {
    CoreRangeSet grid;                  // 어느 코어 ← 재료1이 담김
    TensorMemoryLayout shard_scheme;    // 데이터 쪼개는 법 (HEIGHT/BLOCK/WIDTH)
    ShardOrientation shard_orientation; // 채우는 방향
};
```
- 코어(grid) + 데이터분배(shard_scheme)가 **한 그릇** → "코어·데이터 얽힘"의 증거
- `determine_parallel_config`가 이 그릇을 채워서 반환함

### ✅ 완료: 조립 — determine_parallel_config (conv2d_utils.cpp 120줄)
- 재료들이 조립돼 **코어가 실제로 정해지는 곳** = 목표의 핵심. 완독함
- 흐름: 텐서 크기 + 코어격자(compute_grid_size) 받음 → 일감(타일) 셈 → shard_layout 3갈래로 grid(CoreRangeSet) 채움 → ParallelConfig 그릇에 담아 반환
- **compute_grid_size** = 디바이스에 `compute_with_storage_grid_size()`로 물어본 "실제 쓸 수 있는 코어 격자" (harvesting 반영, 코드에 숫자 없음). conv2d.cpp가 넣어줌

### ✅ 완료: 3가지 샤딩 이해 (tensor_sharding.md 문서로 확인)
- 텐서를 2D(height×width)로 눌러서 봄. **nhw=height(공간), c=width(채널)**
- **HEIGHT** = 공간(행)만 자름 → 코어 **1D** (num_cores_to_corerangeset, row-major로 순번대로 채움)
- **WIDTH** = 채널(열)만 자름 → 코어 **1D** (HEIGHT와 방향만 반대)
- **BLOCK** = 공간+채널 둘 다 자름 → 코어 **2D 사각형**. nhw가 한 변, c가 다른 변 → 항상 직사각형
- **셋 다 (0,0)부터 시작** 고정
- 코드: HEIGHT/WIDTH는 `num_cores_to_corerangeset(개수, 격자, true)`, BLOCK은 `CoreRange((0,0),(cores_x-1,cores_y-1))` → `CoreRangeSet({core_range})`
- 1D vs 2D 차이: 1D는 "개수"만 주고 순번 채움 / 2D는 "가로×세로 크기"로 사각형 영역 지정

### ★★ 코어 시프팅 실마리 (연구 핵심 아이디어) ★★
- **grid는 결국 CoreRangeSet.** BLOCK은 사각형(CoreRange) 1개를 담은 것뿐
- **꽉 찬 사각형 → 쪼개려면: CoreRangeSet에 여러 조각(CoreRange)을 담으면 됨**
  ```cpp
  grid = CoreRangeSet({CoreRange({0,0},{1,3}), CoreRange({2,0},{3,3})});  // 쪼개거나 구멍
  ```
- CoreRangeSet 타입은 그대로 쓰되, "안에 뭘 담느냐"를 바꾸는 것 (CoreRangeSet은 원래 흩어진 코어용)
- 목표: **"가상 사각형(논리 배치)"과 "실제 코어에 올라가는 위치"를 분리해서, 실제 배치를 내 마음대로 자르고 싶음**
- 무기 (core_coord.hpp): `subtract()`, `select_from_corerangeset()`, `select_contiguous_range_from_corerangeset()` — 사각형에서 부분만 골라내기
- 손댈 후보 지점: determine_parallel_config의 grid 생성부 (169줄 BLOCK, num_cores_to_corerangeset 호출부)

### ✅ 완료: num_cores_to_corerangeset (work_split.cpp) — 1D 배치 실제 구현
- **핵심 발견: (0,0)은 하드코딩 제약이 아니라 편의 버전의 기본값!**
- 오버로딩 3개:
  - **77줄** = 진짜 함수. `start_core`를 인자로 받아 **아무 데서나 시작 가능** ⭐
  - **145줄** = 편의 버전. 시작점 없이 부르면 147줄에서 `{0,0}` 넣어 77줄 호출 (conv가 지금 쓰는 것)
  - **150줄** = `..._in_subcoregrids`. 특정 코어 집합 안에서만 배치 (아마 SubDevice용)
- 77줄 함수 동작:
  - 파트1: 시작점부터 쓸 코어 충분한지 검사 (TT_FATAL)
  - 파트2: N개를 **최대 3조각**으로 채움 = 조각A(시작 줄 나머지) + 조각B(꽉 찬 줄들, 직사각형 하나) + 조각C(끝 줄 나머지). `reserve(3)` 이유
  - 파트3: `CoreRangeSet(std::move(all_cores))` 반환
- **임의 시작점 이미 완벽 지원.** 시작점이 줄 중간이어도, 격자 경계 넘어가도 처리됨
- → **시프팅의 "배치" 부분은 로직 수정 불필요. conv가 77줄(start_core 지정)을 부르게 연결만 하면 됨**

### ✅ 완료: mcast 개념 + 코드 분석 (핵심 결론 다수)

**mcast(멀티캐스트)란:**
- 여러 코어가 같은 데이터(weight, activation) 필요 → sender 1개가 DRAM에서 한 번 읽고 → NoC로 나머지(receiver)에 뿌림 (DRAM 중복읽기 방지)
- 멀티캐스트 = 지정된 그룹에만 (브로드캐스트=전체와 다름)
- conv엔 sender/receiver 커널 따로, weights_mcast / act_mcast 두 종류

**하드웨어 제약 (중요):**
- **NoC mcast는 직사각형 하나에만 뿌릴 수 있음.** 각 라우터가 "내 좌표가 x범위 AND y범위 안?"으로 판정 → 두 범위 교집합 = 항상 직사각형
- 구멍/L자/흩어진 코어 = 단일 mcast로 **하드웨어가 못 함**. 우회하려면 mcast 여러 번 쪼개거나 유니캐스트 (복잡·느림)
- NoC 2개(NOC_0/NOC_1), 방향 반대 → setup_mcast_args가 좌표 순서 뒤집음

**샤딩별 mcast (데이터 공유 방식이 달라서):**
- HEIGHT/WIDTH (1D) = 단순. `mcast_receiver_cores = all_cores.subtract(mcast_sender_cores)` 한 줄 (621줄)
- BLOCK (2D) = 복잡. transpose_mcast에 따라 sender가 한 행 or 한 열, 나머지에 뿌림 (596~609줄). act+weights 둘 다

**★★ 코어 시프팅 핵심 결론 ★★**
- mcast 계산은 두 부류:
  - **유연(grid 따라옴):** `output_cores.contains(core)`/`input_cores.contains(core)` (sender/receiver 판정), `core_physical` (코어별 좌표), `all_cores.ranges()` 순회, `subtract()`
  - **하드코딩 (고쳐야 함):** ⬇
- **수정 타깃 목록 (내일 재확인):**
  1. **576줄** `top_left_core = {0,0}` → mcast 범위 **기준점**. 이게 `top_left_core_physical`로 변환돼 1070/1081줄 등 mcast 범위 시작점으로 박힘 ★최우선
  2. **600, 605, 602, 607줄** BLOCK sender/receiver 좌표 `(0,0)`,`(1,0)`,`(0,1)` 원점 기준
  3. **1042~1049줄** 물리좌표 목록 루프가 `{0, y}`,`{x, 0}` = 0번 줄부터 가정
- **해결 방향:** `top_left`를 `grid.bounding_box().start_coord`로 (bottom_right는 이미 `bounding_box().end_coord`로 동적 → 절반은 이미 유연!)
- **업스트림 main도 (0,0) 하드코딩 동일** (697줄) → 아무도 안 풀었음 = 내 기여 지점. backport 불가
- **결론:** 사각형째 시프팅 = top_left만 고치면 됨(할 만함) / 흩뿌리기 = mcast 하드웨어 한계(재설계 필요)

**보너스 아이디어 (열화 연구용):**
- mcast sender는 혼자 DRAM 읽고 뿌려서 부담 큼 → 더 빨리 열화될 수 있음
- **sender 위치 로테이션 = 열화 분산** 아이디어. 단, top_left=sender로 묶여있어 분리 + NoC 라우팅 재계산 필요
- 디버깅 도구: 627~628줄 `log_debug`로 sender/receiver 코어 실제값 로그 (`TT_METAL_LOGGER_LEVEL=Debug`)

### ✅ 완료: 샤딩 ↔ 가중치 ↔ mcast 관계 (헷갈렸던 핵심, 확실히 정리)

**conv = 입력 텐서 × 가중치 텐서 (행렬곱)**
- 입력(activation) 텐서 = [N,H,W,C]. nhwc **전부 입력의 축** (c도 입력 채널, 가중치 아님!)
- 가중치(weight) 텐서 = 별개. 너비 = 출력 채널

**입력 샤딩 (어제 배운 것) = 입력 텐서를 2D로 눌러 나눔**
- 2D = [nhw, c] : 세로축 nhw(공간 N·H·W), 가로축 c(채널)
- HEIGHT=nhw만 / WIDTH=c만 / BLOCK=둘 다 (2D 격자)
- **입력 배치가 기준.** transpose_mcast = 채널 축이 격자의 x냐 y냐를 반영

**가중치 나눔 = 무조건 c(출력채널)로만 자름**
- 가중치엔 nhw(공간) 개념이 없음 → 채널로만 나뉨
- `num_weight_slices_width = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles`
  = 가중치를 채널(너비)로 몇 조각 내나
- weight_matrix_width = b.padded_shape()[3] = 출력 채널

**mcast는 "누가 같은 데이터 쓰나"로 결정 (값은 같고, 대상 그룹이 다름)**
- 같은 nhw(공간) 줄 코어들 → 같은 입력 공유 → **activation mcast**
- 같은 c(채널) 줄 코어들 → 같은 가중치 공유 → **weights mcast**
- 두 mcast는 서로 **수직 방향** (BLOCK이 2방향인 이유)
- **가중치 mcast 방향(행/열)은 입력 배치(transpose_mcast)를 따라감**
  - `transpose_mcast=false`: sender=가로 한 행 `(0,0)~(x,0)`, 세로로 뿌림 (596~608줄)
  - `transpose_mcast=true`: sender=세로 한 열 `(0,0)~(0,y)`, 가로로 뿌림
- 핵심: 가중치를 능동적으로 자르는 게 아니라, **입력 나눔이 "누가 같은 가중치 쓰는지"를 정하고 그 방향으로 뿌림**

> ⚠️ 헷갈림 주의: "c = 가중치"가 아님. c는 입력의 채널 축. 입력·가중치 둘 다 텐서고 각각 나뉨.

### ⬜ 다음: program_factory 순서대로 계속 읽기
- 현재 위치: 490줄대(weight slice/코어 분배) 지나 573줄~ mcast 코어 확정으로 가는 중
- 남은 핵심: 573~614 mcast sender/receiver 확정(어제 봄) → 985~1018 CreateKernel → 1029~1090 mcast 좌표 계산(어제 봄) → SetRuntimeArgs로 커널 전달
- 학습법: 파라미터/검증 구간은 ③넘김, 코어·mcast·커널만 정독

---

## C++ 문법 — 배운 것 (필요시 복습)
클래스/생성자/소멸자/복사생성자/깊은복사, 포인터·참조·`->`·new/delete, const/constexpr, enum, static, this, const 멤버함수, 메서드체이닝, explicit/mutable, 상속(개념), std::string/vector/optional, 템플릿(제네릭), 연산자오버로딩, `=default`/`=delete`, 구조적바인딩
- conv 독해용 5핵심: **참조인자 `const T&` / 구조적바인딩 `auto[]` / optional / 메서드체이닝 / 범위기반for**
- 상세: **`cpp-guide-for-conv-code.md`**

## 학습 원칙 (합의된 것)
- 문법 하나씩 다 안 봄 → 코드 읽다 막히면 그때 봄
- api는 **사용법(선언)만** 알면 됨. 내부 .cpp는 동작을 바꿀 때만
- 모르는 함수는 이름으로 뜻만 잡고 넘어감 (진도 우선)
- 코어 **개수 계산**(split_work 등)은 최적화 결과라 존중, **배치/위치**만 손댈 계획

---

## study_note 폴더 파일들
- `PROGRESS-학습진행.md` (이 파일 — 항상 먼저 읽기)
- `core_coord-정리노트.md` (재료1 완독 정리)
- `cpp-guide-for-conv-code.md` (C++ 문법 가이드)

---

## [최신] program_factory 완독 + 시프팅 실행 계획

### 커널 4개 + 프로세서
- writer_mcast_sender/receiver (RISCV_0), reader (RISCV_1), compute (컴퓨트 유닛)
- read/write는 다른 RISC-V(병렬). config는 DataMovement/Compute 2종뿐 (프로세서가 그것뿐이라)
- **커널은 mcast 좌표·세마포어를 인자(get_arg_val)로 받음 → 커널 수정 불필요, host만 고치면 됨** ★

### mcast 개념 완전 정리 (sender/receiver/read/write)
- read(RISCV_1)=입력 들여옴 / write(RISCV_0)=내보냄. 방향 반대라 다른 프로세서로 병렬
- **write = "메모리에 데이터 쓴다"** (내보내기 아님). sender가 receiver 메모리에 씀, receiver는 당함
- receiver는 데이터 안 옮김 — "다 받았다" 세마포어 신호만 (인자도 sender보다 훨씬 적음)
- activation mcast=reader 담당 / weights mcast=writer 담당 (다른 프로세서라 병렬)
- mcast는 NoC의 한 방향 순환으로 사각형 덮음 → sender를 코너(top_left)에 두면 한 방향으로 다 도달
- **왜 mcast 필요:** 행렬곱에서 같은 A행/B열을 여러 코어가 재사용 → 그 겹치는 입력을 한 번 읽어 나눠줌
  - HEIGHT(공간만 나눔)=공유 적음, BLOCK(2D)=공유 많음 (act+weights 둘 다)
  - 샤딩 ≠ 데이터플로우 (샤딩=코어에 나누기/분산, 데이터플로우=연산유닛 내부). 둘 다 텐스토런트 전용 아님

### 세마포어 = L1(SRAM)의 카운터 하나
- 코어들이 비동기(제각각 속도) → "다 준비될 때까지 기다리는" rendezvous 신호
- receiver +1(준비), sender가 num_dests 찰 때까지 wait → 그 후 mcast
- CB(코어 안 데이터 파이프라인)와 다름 — 세마포어는 코어 사이 순수 신호
- 소프트웨어로 짜는 이유: 데이터무브가 프로그래머블 RISC-V라서 (텐스토런트 특유)

### ★ 부하 불균형의 코드 근거
- **`weight_slice_i = core.x 또는 core.y` (1131줄)** → 코어 위치가 일감을 직접 결정
- `skip_compute` (1286줄): 격자 밖 코어 계산 스킵 = 어제 "안 쓰이는 코어" 근거
- → (0,0)부터 채운 배치 → 앞쪽 코어에 일감 고정 → 누적 시 위→아래 그라데이션(30배)

### ★★★ 시프팅 실행 계획
**핵심: Python(grid 지정) + C++(top_left 수정) 둘 다 필요**
- Python(`ttnn_functional_resnet50.py`)에서 op별 grid 지정 가능:
  - `create_sharded_memory_config_(grid, ...)`, `ops_parallel_config[...]`, `to_memory_config`/`reshard`
- **근데 Python만으론 안 됨** — C++ mcast `top_left={0,0}`(576줄)이 grid 무시하고 하드코딩
  - end_coord(끝)는 grid 따라가는데 start_coord(시작)만 안 씀 → 비대칭
  - 수정: `top_left = input_cores.bounding_box().start_coord` (+ 1042~1049 루프 0시작도)
  - **C++ 이해 덕에 이걸 앎** — Python만 만졌으면 디버깅 지옥

**"넉넉하게 mcast + skip" 아이디어 (구현 간단화):**
- conv는 이미 "격자 폭(num_dests=num_cores_x-1)으로 mcast + 짜투리 skip(non_meaningful_work)" 함
- 내 시프팅 = 이 기존 skip 패턴 확장 (밑바닥 아님, 세마포어/skip 이미 검증됨)
- **20코어 소규모라 skip 오버헤드 감당 가능** → 넉넉하게 잡기 실용적
- 격자 안 코어 선택은 거의 바로, 격자 밖/흩뿌리기는 num_dests 조정 필요

**난이도 전략:** HEIGHT/WIDTH(1D)=쉬움/임팩트작을수도, BLOCK(2D 행/열)=복잡/임팩트큼(Stage3)
- → 1주일: HEIGHT/WIDTH로 시프팅+spread감소 증명, BLOCK은 향후 과제

### 데이터 결론 (실물 보드 측정)
- 코어 컴퓨트 불균형: **모델 전체 누적 30배 spread, 위→아래 1.5배 그라데이션** (유의미)
- sender 자체 효과는 작음. NoC 0/1 불균형은 배선이라 HCI/NBTI엔 덜 크리티컬
- **진짜 원인 = (0,0) 고정 배치로 앞쪽 코어 부하 누적** → 시프팅이 해결책

### ⬜ 다음 (내일)
- **width_sharded program factory** (conv2d_op_width_sharded_program_factory.cpp)
- **다른 op들**(add, maxpool, reshape 등) 코어 배치 부분만 훑기 → 같은 (0,0) 패턴인지 확인
- **목표: 불균형 원인을 op 전반에서 잡기** (conv뿐 아니라 모델 전체 관점)

_마지막 업데이트: sharded program_factory 완독 + Python 레벨(ttnn_functional_resnet50) 시프팅 제어점 발견._

---

## [최신] width factory 완독 + 샤딩별 mcast 본질 (핵심 개념 정리)

### ★★★ 가장 중요한 개념: mcast가 왜 필요한가 = "채널 합산(reduction)"
- **일반 conv 출력 = Σ(모든 입력채널 × 가중치)** → 출력 하나에 전체 입력채널이 필요 (합산!)
- 곱셈만 보면 "자기 것만 곱하면" 되지만, **합산(reduction)** 때문에 남의 채널도 필요
- **예외 (합산 없음 → mcast 불필요):** depthwise conv, element-wise(add) — 채널 독립
- **ResNet Bottleneck(1x1→3x3→1x1)은 전부 일반 conv (채널 합산 O).** depthwise 아님!
  - 1x1 conv = 채널 섞기/합산 그 자체가 본질
- 다 matmul이지만 "채널을 합치냐"가 연산마다 달라서 mcast 필요 여부가 갈림

### 샤딩별 mcast 성격 (완전히 다름)
| 샤딩 | 채널 처리 | mcast 성격 | 통신량 |
|---|---|---|---|
| HEIGHT | 공간만 나눔, 채널 통째 | **재사용**(공통 weight) | 적음 |
| BLOCK | 공간+채널 2D | 재사용(weight+act) + 부분 채널통신 | 중간 |
| WIDTH | 채널을 주로 쪼갬 | **reduction 필수** (채널 다시 모음) | 많음 ★ |
- **HEIGHT/BLOCK mcast = "중복 읽기 피하는 재사용 최적화"** (같은 weight/act 공유)
- **WIDTH mcast = "채널 쪼갠 걸 다시 합치는 필수 통신"** → 성격이 근본적으로 다름
- width 트래픽 폭발(NoC 데이터의 Stage4)이 이 reduction 통신 때문

### width mcast 실제 동작 (activation_reader_width_sharded.cpp)
- **라운드 로빈 브로드캐스트:** `for(act_w_outer_i < num_input_cores)` 루프로 코어들이 **차례차례 한 명씩 sender**가 됨
- 내 차례(`act_w_outer_i == this_core_id`)면 → 내 채널 조각을 나머지 전부에게 `noc_async_write_multicast_loopback_src`로 뿌림
- 나머지 차례면 → 받음. 세마포어로 동기화(`noc_semaphore_wait_min`)
- 반복 끝나면 → 모든 코어가 전체 채널 확보 → 각자 계산 완성
- **그래서 전체 좌표 테이블 필요** (라운드 로빈이라 모두가 서로 위치 알아야) — sharded는 범위(코너2개)면 충분

### width factory 구조 (sharded와 뼈대 같음)
- 커널 3개: activation_reader / weights_reader / compute (sharded는 4개, width가 더 단순)
- 코어 선정: input/output_cores(grid) → all_cores = 많은 쪽 (동적, HEIGHT처럼 (0,0) 하드코딩 아님)
- **mcast 시작점 320줄 `act_mcast_start(0,0)` = 하드코딩** (sharded top_left와 같은 패턴), 끝은 bounding_box 동적
- 573~588줄: 각 코어에 "내 위치(core_x,y) + 전체 좌표 테이블" rt_args → 커널이 스스로 mcast 계산
- 590~600줄: weights는 활성 코어만 (skip)

### ★ 시프팅 관점 재정리 (통신 복잡도 반영)
- 며칠전 "HEIGHT/WIDTH 1D라 쉽다"고 했는데 → **통신 관점에선 width가 복잡** (채널 reduction 통신 폭발)
- HEIGHT = mcast 단순(재사용), 시프팅 영향 적음 → 제일 만만
- WIDTH = 코어배치는 1D로 단순하지만 통신 많음 → 조심
- **당신 직관 검증됨: mcast (0,0) 시작 자체는 문제 작을 수 있음** — width는 "전체 좌표 테이블 + 각 코어가 자기 위치 기반 계산 + skip" 구조라 코어 배치 바꿔도 커널이 따라올 여지

### ⬜ 다음
- width SetRuntimeArgs 이후 마무리 + halo/output 확인 (이미 halo 586줄 (0,0) 확인됨, BLOCK은 grid 씀)
- 다른 op들(add/maxpool) 코어 배치 패턴 확인
- 그 후 시프팅 수정 착수 (HEIGHT부터 — mcast 제일 단순)

_최신 업데이트: width factory 완독 + 채널 reduction이 mcast의 본질임을 파악._

---

## [최신] 연구 방향 확정 + 선행논문 근거

### ★★★ 방향 확정: "sender 로테이션" 버리고 "코어 균등 사용"에 집중
**sender 부담이 열화에 무관한 이유 (3논리, 스스로 검증):**
1. mcast 비중이 전체 conv 대비 작음 (계산이 통신을 압도)
2. mcast 복사는 NoC 일 → 컴퓨트 열화(HCI/NBTI)와 결이 다름 (데이터무브/배선)
3. mcast 끝나면 모든 코어가 같은 양 데이터로 같은 계산 → 계산 부담 동일
→ **sender는 컴퓨트 열화로 더 늙지 않음.** (NoC/데이터무브만 조금 더 쓰이나 부차적)
→ **진짜 문제 = 코어 배치 편중** (bottom_right 저활용, 어제 30배 spread)
→ **연구 = "코어 균등 사용"만 집중하면 됨** (sender 신경 끔)

### ★ 선행논문 근거 (Numerical Kernels on Spatial Accelerator: Wormhole, 2026, §5.2)
- naive(top_left) vs center 라우팅 비교 실측
- 결과: **큰 문제에선 성능 차이 거의 0** (NoC가 low latency라). top_left가 단순해서 실용적 최선
- → **conv가 top_left 박은 이유 = NoC 빨라서 어차피 차이 없으니 단순한 것 택함** (합리적)
- **★ 내 연구에 결정적:** "배치 바꿔도 성능 차이 미미" = **시프팅해도 성능 손해 거의 없음**
  → "성능 유지하면서 배치를 자유롭게 열화 분산에 활용" 논리 성립 (선행논문이 뒷받침)

### 연구 포지셔닝 (논문 검색 결과)
- 텐스토런트 성능/포팅 논문 여럿 (stencil, numerical kernels, N-body) = 성능 관점
- **"코어 배치 + 열화" 논문은 안 보임** → 블루오션, 내 기여 지점
- 포지셔닝: "성능은 선행연구, 열화는 내가 처음" + 차량용(batch=1, 고온) 맥락
- 연구 성립 조건(스스로 도출): 코어 덜 쓰는 경우(작은 feature map/batch 작음)에 편중 큼 → **차량 batch=1 추론이 딱 이 조건**

### 텐스토런트 아키텍처 위치 (vs GPU/TPU)
- 핵심 = spatial dataflow: SRAM 상주(샤딩) + 코어 간 NoC 직접 통신(mcast) = DRAM 회피
- 샤딩 = DRAM→각 코어 SRAM 쪼개 올림 (conv 전, 한 번). mcast = SRAM↔SRAM 조각 공유 (계산 중)
- vs GPU(캐시+HBM 자동, 편하나 전력↑) / vs TPU(고정 systolic, 효율↑ 유연성↓)
- 텐스토런트 = 유연+효율 절충, 대가는 SW 복잡 (내가 메우는 지점)

_최신: 방향 "코어 균등 사용"으로 확정. 선행논문(§5.2)이 "배치 바꿔도 성능 손해 없음" 뒷받침. 다음: 논문 더 읽고 → 다른 op 확인 → 시프팅 착수._
