# core_coord.hpp 완독 정리 노트

tt-metal의 **코어 좌표 타입**을 정의한 헤더. 코어 앵커링의 "재료(부품) 정의서". 이 파일 = **코어 위치를 표현·관리하는 도구들의 모음**이다. (실제 커널 올리기·일 시키기는 다른 곳 → `CreateKernel`/`SetRuntimeArgs`.)

---

## 파일 지도 (한눈에)

| 줄 | 내용 | 중요도 |
|---|---|---|
| 1~27 | include, json 전방선언 | ③ 무시 |
| 29~35 | **CoreCoord** (좌표 = x,y 쌍) | ① 핵심 |
| 37~103 | **CoreRange** (사각형 = 시작·끝) | ① 핵심 |
| 105~171 | **CoreRangeSet** (사각형들의 집합) | ① 핵심 |
| 173 | 비교 연산자 | ③ 넘김 |
| 175~206 | **자유 함수** (코어 변환/선택 도구) | ① 핵심 |
| 210~260 | deprecated 별명·래퍼 | ③ 완전 무시 |
| 262~330 | fmt / hash / json 부가장치 | ③ 무시 |

**핵심은 딱 두 덩어리: 타입 3형제 + 자유 함수 6개.** 나머지 절반은 부가 장치라 무시 OK.

---

## 코어 타입 3형제

```
CoreCoord    = 코어 1개         {3, 5}                       → 점
CoreRange    = 사각형 1개       (0,0)~(3,3)                  → 시작·끝 좌표 한 쌍
CoreRangeSet = 사각형 여러 개   [(0,0)~(2,2), (4,0)~(5,2)]   → 겹치지 않는 사각형들의 집합
```

**공통 성격:** 셋 다 "어느 코어를 쓸지" **위치를 표현·관리만** 한다. 실제 코어를 움직이거나 일 시키는 게 아니다. 지도 위에 "우리 구역" 표시하는 것과 같다.

### 1. CoreCoord
```cpp
using CoreCoord = tt_xy_pair;   // (x, y) 좌표 한 쌍의 별명
```
- `tt_xy_pair`(UMD 소속, x·y 담는 단순 구조체)에 붙인 별명
- 실체는 `umd/device/types/xy_pair.hpp` → 안 봐도 됨 ("x,y 쌍"이면 충분)

### 2. CoreRange — 사각형 하나
```cpp
CoreCoord start_coord;   // 시작 코너
CoreCoord end_coord;     // 끝 코너
```
- **멤버는 시작·끝 좌표 2개뿐.** 사이 코어는 저장 X → 필요할 때 계산 ("1~100"에서 양 끝만 적는 것)
- 생성자: `CoreRange(point)` (1×1), `CoreRange(start, end)` (사각형)
- 복사/이동: 다 `= default` (포인터 없이 좌표뿐이라 기본 복사로 안전)
- 주요 함수(다 `const`, 읽기 전용):
  - `contains(...)` 내 안에 있나? (코어/범위/집합 오버로딩)
  - `intersects(other)` 겹치나? (예/아니오, bool)
  - `intersection(other)` 겹친 영역 줘 (없으면 `nullopt` → optional)
  - `merge(cr)` 합쳐줘 (안 되면 nullopt)
  - `size()` 총 코어 개수(16) / `grid_size()` 모양({4,4}) / `str()` 문자열
- `CoreIterator` (중첩 클래스) + `begin()`/`end()` → `for (auto& core : range)` 순회 가능하게 함

### 3. CoreRangeSet — 사각형들의 집합
```cpp
private:
    std::vector<CoreRange> ranges_;   // CoreRange(사각형)들을 담은 벡터
```
- **핵심 = `vector<CoreRange>`.** 서로 안 겹치는 사각형들의 모음 (`validate_no_overlap`이 겹침 검사)
- 왜 필요? 흩어지거나 구멍 뚫린 코어 배치는 사각형 하나로 표현 불가 → 여러 조각으로
- 생성자 5개: 사각형 여럿 / set / 사각형 하나 / 좌표들 / 빈 것
- 주요 함수:
  - `ranges()` → 담긴 사각형 목록 꺼내기 (순회용, getter). `for (range : cores.ranges())`
  - `contains(...)` 포함? / `intersects(...)` 겹치나? / `intersection(...)` 겹친 영역
  - `merge(other)` 합치기 (`template<T>` → 코어/사각형/집합 뭐든 받음)
  - `subtract(other)` 겹치는 부분 빼기 (A - A∩B) ← 특정 코어 제외에 유용
  - `merge_ranges()` 같은 코어 덮되 사각형 개수 최소화
  - `num_cores()` **총 코어 개수** (≠ `size()`는 조각 개수!) / `bounding_box()` 전체 감싸는 최소 사각형

> **주의: `size()` vs `num_cores()`** — 사각형 2개에 각 9코어면 `size()=2`(조각 수), `num_cores()=18`(코어 수)

---

## 자유 함수 6개 (175~206줄) — 코어 변환/선택 도구 ⭐

클래스 밖 함수들. "개수/범위/집합 ↔ 개별 좌표"를 변환하고 부분 선택. **코어 시프팅에 직접 쓸 무기.**

| 함수 | 넣는 것 | 나오는 것 | 용도 |
|---|---|---|---|
| `grid_to_cores` | 개수 or 범위 | 좌표 리스트 | 범위를 개별 코어로 펼침 |
| `grid_to_cores_with_noop` | 위 + | 좌표 리스트(노는 코어 뒤에) | 일하는/노는 코어 구분 |
| `corerange_to_cores` | CoreRangeSet | 좌표 리스트 | 집합을 개별 코어로 펼침 |
| `select_from_corerangeset` ⭐ | 집합 + 인덱스 범위 | 부분 집합 | **N번째~M번째만 뽑기** |
| `select_contiguous_range...` ⭐ | 집합 + x,y | 연속 블록 (없으면 nullopt) | **x×y 블록 확보** |

⭐ 두 개(`select_*`)가 "전체 코어에서 원하는 부분만 골라내기" = 코어 시프팅/부분 배치의 핵심 도구.

---

## 무시해도 되는 부분 (③)

- **deprecated** (210~260): `[[deprecated]]` = "옛날 거, 쓰지 마". 옛 이름 → 새 `tt::tt_metal::` 로 넘기는 래퍼. 완전 무시
- **fmt::formatter** (262~282): `fmt::print("Core {}", core)` 출력 포맷. 무시
- **std::hash** (284~296): map/set 키로 쓸 수 있게. 무시
- **ttsl::json** (298~330): JSON 저장/복원. 무시

---

## 이 파일에서 복습된 C++ 문법

- **참조 인자** `const T& other` — 복사 없이 읽기전용으로 받기
- **const 멤버 함수** `... () const` — 객체 안 바꿈(읽기만)
- **optional** `std::optional<CoreRange>` — 결과 없을 수 있음 (intersection/merge)
- **연산자 오버로딩** `operator==`, `operator*`, `operator++` — 기호를 클래스에 정의
- **템플릿(제네릭)** `template<typename T>` — 여러 타입에 재사용 (merge)
- **vector** `std::vector<CoreRange>` — 크기 자유로운 배열
- **`= default` / `= delete`** — 복사/이동 허용(기본)/금지 선언
- **중첩 클래스** — CoreRange 안의 CoreIterator
- **선언 vs 정의** — 헤더(.hpp)엔 `;`로 선언만, 동작은 .cpp에

---

## 한 줄 요약

**core_coord.hpp = 코어 위치를 표현·관리하는 도구 정의서.** 타입 3형제(점/사각형/집합)로 "어느 코어를 쓸지"를 표현하고, 자유 함수로 변환·선택한다. 이 위치 정보를 `CreateKernel`/`SetRuntimeArgs`에 넘겨야 실제 코어에 커널·일이 올라간다. 코어 앵커링의 "재료" 파트.

## 다음 단계
이 재료(타입들)가 conv에서 실제로 어떻게 쓰여 코어가 정해지는지 → `conv2d_utils.cpp`의 `determine_parallel_config`부터.
