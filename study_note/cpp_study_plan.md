# C++ 학습 플랜 (다음 미팅까지)

> 2026-07-29 세션에서 세운 플랜. **이번엔 반드시 파일로 남겨서 세션 넘겨 유지.**
> 목적: TT-Metal 커널 코드 읽고 쓰기 위한 C++ 습득. 최종 목표는 aging rotation T1 anchor shift 커널 패치.

---

## 학습 방식
**Top-down**: 실제 커널 코드 읽으면서 나오는 C++ 문법을 그때그때 짚기. 별도 문법 교재 안 씀. C 대비 차이 위주 (참조/네임스페이스/클래스/템플릿/RAII/auto/std::vector).

## Target 폴더 (중요: upstream 아님)
`C:\NPU_Challenge\operations\` — **컨테이너에서 pull 해 온 BOS fork copy** (2026-07-26).
- ❌ tt-metal upstream (`C:\NPU_Challenge\tt-metal\ttnn\cpp\ttnn\operations\`) 아님. 파일명이 다름 (upstream: `*_nanobind.cpp`, `conv2d_device_operation.hpp/cpp`; BOS fork: `*_pybind.cpp`, `conv2d_op.hpp/cpp`)
- **혼동 방지**: 앞으로 conv 언급 시 항상 `C:\NPU_Challenge\operations\conv\` 로 시작

## Plan

### Step 1 — 폴더 계층 이해
`operations/conv/conv2d/` 안의 파일 역할 파악.
- `conv2d.hpp/cpp` (5KB header, 43KB source) — user-facing 래퍼
- `conv2d_op_program_factory_common.hpp/cpp` (31KB) — host-side program factory 공통 코드
- `conv2d_pybind.hpp/cpp` — Python 바인딩
- `conv2d_utils.hpp/cpp` — parallel_config, L1 계산 (이미 정독함, 2026-07-21)
- `prepare_conv2d_weights.hpp/cpp` — weight prep
- `device/` — 실제 device operation
  - `conv2d_op.hpp/cpp` — device op 정의
  - `conv2d_op_sharded_program_factory.cpp`
  - `conv2d_op_width_sharded_program_factory.cpp`
  - `kernels/` — RISC-V 디바이스 커널

### Step 2 — `conv2d_op.hpp/cpp` 정독
파일 하나씩, 섹션 단위로. 나오는 C++ 문법 개념 마다 C 대비 설명.
- 예상 개념: namespace, class, template, references, const-correctness, override, std::vector, initializer list, structured bindings, RAII, smart pointers 등.
- 세션 로그: 학습한 개념·해당 코드 라인 이 파일에 계속 append.

### Step 3 — (이후) program factory 확장
`conv2d_op_sharded_program_factory.cpp` 등으로 확장. CoreRangeSet, CreateProgram, CreateKernel 실제 사용법 학습 → aging T1 patch 지점 특정 가능해짐.

---

## 세션 로그 (append-only)

<!-- 세션 끝날 때 마다 "무슨 파일 어디까지, 어떤 개념 학습" 한 줄 추가. -->

### 2026-07-30 세션 — Step 1 완료 (호출 경로 검증)

**검증된 호출 경로** (file:line 근거):
1. `conv2d_pybind.cpp:33` — `ttnn.conv2d` Python API 등록
2. `conv2d_pybind.cpp:69` (lambda) — `ttnn::conv2d` object 로 dispatch
3. `conv2d.cpp:964` `Conv2dOperation::invoke(...)` — 자유함수 `conv2d(...)` 로 전달
4. `conv2d.cpp:61` 자유함수 `conv2d(...)` — shape/config 정리
5. `device/conv2d_op.cpp:65-83` — `Conv2d` struct 인스턴스 생성
6. `device/conv2d_op.cpp:88` — `operation::run_without_autoformat(conv_op, ...)` — framework 가 validate/create_program 호출
7. `device/conv2d_op.cpp:171` **`Conv2d::create_program(...)`** — ★ dispatcher
   - :188 `WIDTH_SHARDED` → `multi_core_conv2d_width_sharded(...)`
   - :238 else → `multi_core_conv2d_sharded(...)`
8. `device/conv2d_op_*_program_factory.cpp` — `CreateProgram + CoreRangeSet + CreateKernel`
9. `device/kernels/*.cpp` — RISC-V 커널

**T1 patch 관점 발견** (Step 3 로 이관):
`Conv2d::create_program:193/229` 에서 `ParallelConfig.grid = input_tensor_a.shard_spec().value().grid`. **CoreRangeSet 은 conv 호출 전에 결정됨** (input tensor 의 shard_spec). Patch 지점 후보:
- (A) 상류 sharding 결정 (`conv2d_utils.cpp::determine_parallel_config`)
- (B) `multi_core_conv2d_sharded` 안 kernel 배치 전 remap

**다음 세션 (Step 2)**: `device/conv2d_op.hpp` 부터 정독 — struct 정의, Conv2dConfig 필드, initializer list, const member function, `std::optional`, `std::make_tuple` attribute_names 등 C++ 문법 노트.

---

### 2026-07-30 세션 (오후) — Step 2 시작, 개념 훑기

**정독 진행도**: `conv2d_op.hpp` line 1-20 (Section 1 — includes + namespace) 완료. 이후 실제 코드는 line 22 (`struct Conv2dConfig`) 부터 시작 예정.

**커버된 C++ 개념** (지도 대비):
- ✅ **Tier 1**
  - #1 `#pragma once`
  - #2 `<...>` vs `"..."` include
  - #3 C++ 표준 헤더 (`<optional>` etc, `.h` 없음)
  - #4 **namespace** (선언, 중첩, `::` 접근, 전통 vs C++17 축약)
  - #6 **Reference** `T&`, `const T&` (사용자가 별도로 이미 학습)
- ✅ **Tier 2 (개념만, 실제 코드 정독 전)**
  - #10-11 **struct/class 차이**: C vs C++ struct, C++ struct vs class, default access
  - #12 **접근 지정자** `public:` `private:` `protected:`
  - #14 **멤버 함수** (this 포인터 원리)
  - #15 **`const` 멤버 함수** (뒤에 붙는 `const` = self 수정 안 함)
  - Class 실전 예: `class WeightLayoutThreader` in prepare_conv2d_weights.cpp:39 (encapsulation 대조 예시)
- ✅ **Tier 3 (개념만)**
  - #23 함수 오버로딩 (5분, 개념만)
  - Constructor + initializer list 원리 (다음 세션 Section 4 에서 실전)
  - Default constructor (자동 생성 규칙)
- ✅ **Tier 4 (개념만)**
  - #34 RAII 원리
  - #35 Destructor (자동생성 이해)
  - #37-38 Move semantics + Copy vs Move (`std::move` 배경)
- ❌ **확정 skip**: #40 new/delete (우리 파일에 실제 할당 문법 0개)

**미커버 (다음 세션 이후)**:
- Tier 2 나머지: `std::optional`, `std::array`, `std::vector`, `std::pair/tuple`, `std::variant` — Section 2/4 에서 실전
- Tier 3 나머지: default 인자, `static`, `constexpr`, `inline`, `enum class`, `using` 별칭, `decltype`, structured bindings, range-for, lambda
- Tier 5: 템플릿 (사용 위주 → 정의)
- Tier 6: 커널 코드 특유 API

**다음 세션 착수**: `struct Conv2dConfig` (line 22) 실제 정독. 여기서 `std::optional`, member default value, `std::array` 등 Tier 2 실전 시작.
