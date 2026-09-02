# conv 코드 독해용 C++ 문법 가이드 (입문자용)

목표: C++ 교재를 처음부터 다 떼는 게 아니라, **당신의 conv2d 코드를 읽는 데 실제로 나오는 문법만** 골라 순서대로. 모든 예시는 당신 레포(`operations/conv/conv2d/...`)의 실제 코드에서 뽑았어요. 각 항목 = "이 문법을 알면 이 줄이 읽힌다".

읽는 법: 위에서부터 내려가되, **이미 아는 건 건너뛰기.** 지난 대화에서 배운 것(클래스·포인터·new/delete·`->`·const·auto·enum·template 기초)은 ✅로 표시했어요.

---

## 0단계 — 이미 배운 것 (복습만) ✅

이건 며칠간 다뤘으니 넘어가도 돼요. 코드에서 이렇게 나와요:

```cpp
device->compute_with_storage_grid_size();   // -> (포인터로 멤버 접근) ✅
const uint32_t num_cores_x = ...;           // const, 고정폭 정수 ✅
auto core_grid = ...;                        // auto (타입 추론) ✅
enum class TensorMemoryLayout { ... };       // enum class ✅
CoreRange(CoreCoord({0,0}), ...);            // 생성자, 클래스 ✅
```

---

## 1단계 — 참조 인자 `const T&` (최우선) ⭐

conv 코드 함수는 **거의 다** 인자를 `const 타입&`로 받아요. 이거 하나 알면 함수 시그니처의 절반이 읽혀요.

당신 코드 (`conv2d_utils.hpp`):
```cpp
const std::optional<MemoryConfig>& input_memory_config
const std::array<uint32_t, 2>& kernel_size
```

읽는 법: **`const T&` = "T를 복사 없이(참조 &), 수정 안 하고(const) 받는다".** 왜? conv가 다루는 텐서·설정은 크고 무거워서 복사하면 느려요. 그래서 "원본을 그대로 빌려 읽기만" 하는 거예요. (며칠 전 배운 참조 + const가 합쳐진 것.)

기억할 것: 함수 인자에서 `const ...&`를 보면 **"아, 이건 그냥 읽기용으로 넘기는구나"** 하고 넘어가면 돼요. 깊게 안 봐도 됨.

---

## 2단계 — 구조적 바인딩 `auto [a, b] = ...` ⭐

conv 코드에 정말 자주 나와요. 함수가 **여러 값을 한 번에 반환**할 때, 그걸 여러 변수로 한 방에 받는 문법.

당신 코드 (`conv2d.cpp`):
```cpp
auto [output_height, output_width] = get_output_dim(...);
auto [input_tensor_post_tm, parallel_config, output_parallel_config] =
    shard_or_reshard_tensor_if_required(...);
```

읽는 법: 오른쪽 함수가 값 2개(또는 3개)를 묶어서 돌려주면, 왼쪽 `[...]`가 그걸 **각각 이름 붙여 풀어받아요.** 며칠 전 vecadd에서 본 `auto [num_cores, all_cores, ...] = split_work_to_cores(...)`가 바로 이거였어요.

파이썬의 `a, b = func()`와 똑같은 개념이에요. 익숙할 거예요.

---

## 3단계 — `std::optional` 과 `.value()` / `.has_value()` ⭐

"값이 있을 수도, 없을 수도" 있는 걸 표현. conv 설정은 "지정 안 하면 자동" 옵션이 많아서 이게 흔해요.

당신 코드 (`conv2d_utils.cpp`):
```cpp
const auto& shard_spec = conv_output_mem_config.shard_spec().value();
conv_config.shard_layout.has_value() &&
    conv_config.shard_layout.value() == TensorMemoryLayout::WIDTH_SHARDED
```

읽는 법:
- `optional<T>` = "T가 들어있거나, 비어있거나(`nullopt`)"
- `.has_value()` = "값 있어?" (true/false)
- `.value()` = "그 값 꺼내줘"

며칠 전 Marine의 `name = NULL`로 "이름 없음"을 표현했죠? `optional`은 그걸 **안전하게** 하는 현대식 도구예요. 포인터·NULL 대신 "있음/없음"을 명확히 표현.

---

## 4단계 — 메서드 체이닝 `a.b().c().d()` ⭐

점(`.`)으로 함수를 줄줄이 잇는 것. conv 코드가 이걸로 도배돼 있어요. 무섭지만 원리는 단순.

당신 코드 (`conv2d_op_sharded_program_factory.cpp`):
```cpp
a.memory_config().is_sharded()
a.shard_spec().value().shape
a.memory_config().shard_spec().value().grid
```

읽는 법: **왼쪽부터 한 단계씩** 읽어요.
```
a                      → 텐서 a
 .memory_config()      → a의 메모리 설정을 꺼냄
 .shard_spec()         → 그 설정의 샤드 스펙을 꺼냄 (optional)
 .value()              → optional에서 값 꺼냄
 .grid                 → 그 안의 grid (코어들)
```
각 함수가 결과를 돌려주면, 그 결과에 또 함수를 부르는 거예요. "a에서 시작해 점점 안으로 파고든다"고 보면 돼요. 한 줄에 다 있어서 그렇지, 사실 4단계를 이어붙인 것뿐이에요.

---

## 5단계 — `using` 별명 ⭐

긴 타입에 짧은 이름을 붙이는 것. 코드 위쪽에 자주 모여 있어요.

당신 코드 (`conv2d.cpp`):
```cpp
using OutputHeight = uint32_t;                          // OutputHeight = uint32_t의 별명
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth,
                          ttnn::Tensor, std::optional<ttnn::Tensor>>;
using namespace conv2d;                                  // conv2d:: 를 생략하게 해줌
```

읽는 법:
- `using A = B;` = "A는 B의 다른 이름" (며칠 전 `using CoreCoord = tt_xy_pair`에서 봄)
- `using namespace X;` = "X:: 접두사 생략 허용"

`Result` 같은 별명을 보면, 그 정의를 찾아가서 "아 이건 튜플이구나" 한 번만 확인하면 돼요.

---

## 6단계 — `std::vector`, `std::array`, `std::set` (컨테이너) ⭐

여러 값을 담는 상자들. conv는 코어 인자를 `vector`에 담아 넘기는 게 기본.

당신 코드:
```cpp
std::vector<uint32_t> reader_rt_args{core_index};   // 크기 가변 배열 (런타임 인자 담기)
const std::array<uint32_t, 2>& kernel_size;         // 고정 크기 2짜리 배열
std::set<CoreCoord> cores_with_non_meaningful_work; // 중복 없는 집합
```

읽는 법:
- `vector<T>` = 크기 변하는 배열 (며칠 전 배운 그 vector)
- `array<T, N>` = 크기 N 고정 배열 (kernel_size는 항상 [높이, 너비] 2개니까)
- `set<T>` = 중복 없는 모음

⑧단계 코어 루프에서 `reader_rt_args`(vector)에 값 채워 `SetRuntimeArgs`로 넘기는 게 핵심 패턴이에요.

---

## 7단계 — 범위 기반 for 루프 ⭐

컨테이너를 하나씩 도는 것. 코어 순회 루프가 전부 이거예요.

당신 코드 (`conv2d_op_sharded_program_factory.cpp`):
```cpp
for (const CoreRange& core_range : input_cores.ranges()) {
    for (const CoreCoord& core : core_range) {     // 코어 하나하나
        SetRuntimeArgs(program, reader_id, core, reader_rt_args);
    }
}
```

읽는 법: `for (원소 : 컨테이너)` = "컨테이너의 원소를 하나씩 꺼내 돈다". 여기선 "코어 범위들을 돌고, 각 범위 안 코어를 하나씩" 이중으로. 며칠 전 vecadd에서 본 그 이중 for와 똑같아요. **이게 ⑧ 코어별 인자 세팅의 뼈대**예요.

`const CoreCoord& core` = 각 코어를 참조로(복사 없이) 받음 (1단계와 연결).

---

## 8단계 — `std::variant` (여러 타입 중 하나)

"이 자리엔 A거나 B거나 C가 올 수 있다"를 표현. 코어 지정에 나와요.

당신 코드 (`conv2d_op_program_factory_common.hpp`):
```cpp
const std::variant<CoreCoord, CoreRange, CoreRangeSet>& all_cores
```

읽는 법: `variant<A, B, C>` = "A, B, C 중 하나가 담긴다". 코어를 지정할 때 **한 개(`CoreCoord`)든, 사각형(`CoreRange`)든, 집합(`CoreRangeSet`)든** 다 받겠다는 뜻. 그래서 `CreateKernel`에 셋 중 뭘 넘겨도 동작했던 거예요.

지금은 "여러 형태를 다 받는 유연한 타입"으로만 이해하면 충분. 깊게 안 봐도 됨.

---

## 9단계 — 매크로 `TT_FATAL` / `TT_ASSERT` (그냥 넘기기)

대문자 함수처럼 생긴 것들. 검증/에러 처리용이에요.

당신 코드:
```cpp
TT_FATAL(a.memory_config().is_sharded(), "Conv activation must be sharded.");
TT_ASSERT(input_cores == output_cores || block_sharded, "...");
```

읽는 법: **"이 조건이 참이어야 한다, 아니면 에러 내고 멈춤".** 로직의 본체가 아니라 "안전장치"라, 흐름 읽을 땐 **대체로 건너뛰어도** 돼요. 첫 인자(조건)만 슬쩍 보면 "아 이게 참이라고 가정하는구나" 정도.

---

## 학습 순서 요약 (우선순위)

이 코드를 읽는 데 중요한 순서:

1. **`const T&` 참조 인자** — 함수 시그니처 절반이 이거
2. **`auto [a,b] = ...` 구조적 바인딩** — 다중 반환 받기
3. **`optional` / `.value()` / `.has_value()`** — 있음/없음
4. **메서드 체이닝 `a.b().c()`** — 왼쪽부터 파고들기
5. **범위 기반 for** — 코어 순회 (⑧의 뼈대)
6. **`vector`/`array`/`set`** — 인자 담는 상자
7. `using` 별명 / `variant` / 매크로 — 보이면 넘기는 수준

**1~5만 확실히 잡으면 conv host 코드의 80%가 읽혀요.** 6~7은 보일 때마다 이 문서 다시 보면 돼요.

---

## 실전 팁

- **모르는 함수(`shard_or_reshard_tensor_if_required` 같은)는 이름으로 대충 뜻만** 잡고 넘어가세요. "샤딩 다시 하는 함수구나" 정도. 정의까지 다 파면 진도 안 나가요.
- **읽다 막히면 "이 줄은 위 9개 중 어느 문법이지?"로 되돌아오세요.** 대부분 이 목록 안에 있어요.
- 학습 자료: learncpp.com에서 위 항목 이름으로 검색 (특히 "references", "structured bindings", "std::optional", "range-based for").

---

## 다음 단계

이 문법들이 손에 익으면, `determine_parallel_config`(코어 결정) → `all_cores` 확정 → `CreateKernel`(배치) → `SetRuntimeArgs` 루프(코어별 인자) 흐름을 **한 줄씩** 스스로 읽을 수 있어요. 그때 막히는 줄을 가져오면 같이 풀어요.
