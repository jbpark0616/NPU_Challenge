# Conv 실행 흐름 + 코어 시프팅 수정 체크리스트

> ttnn(Python) 노출부 → device op → program factory → kernel 까지 전체 흐름.
> 각 단계의 (0,0) origin 하드코딩을 **체크리스트**로 정리. 수정 시 이 문서를 따라가며 체크.
> 모든 line 번호는 실측 (BOS operations 복사본 기준).

---

## 0. 큰 그림 (한 장)

```
Python: ttnn.conv2d(...)  /  ttnn.linear(...)  ← op 노출부
   ↓
【conv2d.cpp】 호스트 오케스트레이션 (지휘자)
   ├─ (A) 1x1 conv? → use_matmul_for_1x1_conv (line 216)
   │        YES → ttnn::linear(=matmul)로 우회 (line 288-310)  ★1x1은 conv factory 안 거침
   ├─ (B) shard_layout 자동추론 (line 528-532 determine_conv_config_for_auto_shard)
   ├─ (C) shard_or_reshard_tensor_if_required (line 738)  ★origin 결정 지점
   ├─ (D) get_conv_configs (line 759)
   ├─ (E) prepare weights/bias (line 800-841)
   ├─ (F) halo 호출 (line 846-884, bypass_halo 아니면)
   └─ (G) Conv2d device op 호출 (line 899+)
        ↓
【conv2d_op.cpp】 device op
   ├─ validate (line 96: activation sharded)
   └─ create_program (line 171) → factory 선택 (line 185-238)
        ├─ WIDTH_SHARDED → multi_core_conv2d_width_sharded  【width factory】
        └─ HEIGHT/BLOCK  → multi_core_conv2d_sharded        【sharded factory】
             ↓
【program factory】 CB + kernel(reader/compute/writer) + runtime args
             ↓
【kernels】 코어에서 실행 (reader→compute→writer)
```

**병렬 경로:** 1x1 conv → matmul factory (`matmul_op_multi_core_reuse_mcast_1d/2d_program_factory.cpp`).
conv와 **동일 메커니즘**(sharding+mcast+(0,0)). 별도 공부 불필요, fix 이식만.

---

## 1. origin이 흐르는 경로 (핵심 이해)

```
(C) shard_or_reshard → determine_parallel_config → num_cores_to_corerangeset
       → grid를 (0,0)부터 row-major 로 생성  ★근원
       ↓ (이 grid가 shard_spec에 담겨 아래로 전파)
(F) halo   → noc_TL/rectangular/cast_core 가 (0,0) 가정
(G) factory → mcast top_left 이 (0,0) 가정
```
**즉 origin은 (C)에서 정해지고, halo·factory가 그걸 따라야 함.**
**(C)만 바꾸면 halo·factory 하드코딩과 어긋남 → 전부 동기화 필요.**

---

## 2. ★ 수정 체크리스트 (파일:라인)

### [ ] A. Python 레벨 (데이터 위치의 손잡이)
- **`ttnn_functional_resnet50.py` line 557**
  `CoreRange(CoreCoord(0,0), CoreCoord(gx-1, gy-1))` → 모양만 지정, **origin (0,0) 고정**
  - [ ] `CoreCoord(0,0)` → `CoreCoord(ox, oy)` 로 파라미터화
  - [ ] line 207/256/708 `create_sharded_memory_config_(...grid...)` = 이전 grid 전파 → 명시적 (0,0) 재지정 지점 전수조사

### [ ] B. 코어 grid 생성 (C++ 자동 경로)
- **`conv2d_utils.cpp`**
  - line 120 `determine_parallel_config` — grid 계산 진입
  - [ ] line 153 `num_cores_to_corerangeset(num_cores_nhw, compute_grid_size, true)` (HEIGHT)
  - [ ] line 176 `num_cores_to_corerangeset(num_cores_c, ...)` (WIDTH)
  - [ ] line 201 output_parallel_config grid
  - → `num_cores_to_corerangeset`의 **start_core 버전(work_split.cpp line 77)** 사용하도록 변경
- **`conv2d.cpp` line 738** `shard_or_reshard_tensor_if_required` — 여기서 origin을 받도록 확장

### [ ] C. Sharded factory (HEIGHT/BLOCK) — `conv2d_op_sharded_program_factory.cpp`
- [ ] **line 576** `top_left_core = {0,0}` → `all_cores.bounding_box().start_coord`
- [ ] line 577 `top_left_core_plus_one = {1,1}` → origin+1 상대화
- [ ] line 600 `CoreRange(top_left, CoreCoord(0, num_cores_y-1))` → origin 상대
- [ ] line 605 `CoreRange(top_left, CoreCoord(num_cores_x-1, 0))` → origin 상대
- [ ] line 607 receiver `CoreRange(CoreCoord(0,1), ...)` → origin 상대
- [ ] line 1070~1259 `top_left_core_physical.x/y` = runtime arg → 위 고치면 **자동 반영** (확인만)

### [ ] D. Width factory — `conv2d_op_width_sharded_program_factory.cpp`
- [ ] **line 320** `act_mcast_start_core_logical(0,0)` → origin
- [ ] line 321 end는 이미 bounding_box (동적) → 확인만

### [ ] E. Halo factory — `untilize_with_halo_program_factory.cpp`
⚠️ (0,0) 3곳. block이라도 자동 안전 아님
- [ ] **line 554-561** `core_id_to_noc_coords(0)` → `noc_TL` = 항상 logical(0,0). **block 포함 전부.** origin 반영
- [ ] **line 586** height/width rectangular `CoreRange(CoreCoord(0,0), ...)` → origin
- [ ] **line 689** `cast_core = core_i == 0` → grid의 첫 코어로
- [ ] halo origin = conv origin **동일값** 확인 (안 맞으면 결과값 오류)

### [ ] F. Matmul factory (1x1 conv 경로) — 나중, conv fix 이식
- [ ] `matmul_op_multi_core_reuse_mcast_1d_program_factory.cpp` — (0,0)/top_left 찾아 수정
- [ ] `matmul_op_multi_core_reuse_mcast_2d_program_factory.cpp` — 동일
- [ ] 정확한 라인은 착수 시 grep (conv와 동일 패턴 확인됨)

---

## 3. 수정 전략 요약

- **원칙:** grid **모양 유지 + origin만 오프셋** (Fig 4.1/4.2: 모양 바꾸면 성능 영향)
- **origin 값:** `all_cores.bounding_box().start_coord` (Python에서 지정한 grid 좌상단)
- **커널 무변경:** 좌표는 runtime arg로 전달됨. host에서 origin 반영하면 자동
- **동기화 필수:** Python(A) + conv factory(C/D) + halo(E) origin이 **전부 같은 값**
- **wrap은 phase 2:** 경계 넘는 배치는 mcast 재작업(CoreRangeSet/exclude) 필요 → 지금 X

---

## 4. 작업 순서 (권장)

1. [ ] **Width factory(320) 최소 수정** → 빌드 → 정합성 확인 = **파이프라인 검증** (aging 판단 X)
2. [ ] Sharded factory(576+) → Stage 3(block) 대응
3. [ ] Halo(554-561/586/689) 동기화
4. [ ] Python(557) origin 파라미터화 + 전파 지점 정리
5. [ ] Matmul 1d/2d 이식 (1x1 = 절반이라 필수)
6. [ ] **conv+matmul 둘 다 시프팅된 상태에서** before/after 측정 (Tracy: 코어별 duty spread + 성능)

---

## 5. 검증 체크 (수정 후 반드시)

- [ ] **결과값 정합성** — 시프팅 후 conv 출력이 baseline과 동일 (수치 오차 내)
      ※ halo origin 불일치는 성능 아니라 **결과 오류**로 나타남 → 최우선 확인
- [ ] **커널 절대좌표 가정 없나** — `grep "noc_x == 0\|== 0" kernels/` 확인
- [ ] **grid 모양 유지** — origin만 옮기고 aspect ratio 안 바뀌었나
- [ ] **보드 경계 초과 없나** — origin+shape가 20코어 범위 내 (bounding 체크)
- [ ] **1x1 포함 여부** — 측정 시 matmul도 시프팅됐는지 (안 됐으면 결과 muddy)

---

## 부록: 갈림길 두 개 (놓치기 쉬움)

1. **1x1 conv → matmul 우회** (conv2d.cpp line 216, 288): ResNet bottleneck 1x1들이 conv factory 안 거침. **matmul factory도 고쳐야 model-wide.**
2. **factory 선택** (conv2d_op.cpp line 185): width→width factory, height/block→sharded factory. **Stage 3(block)=sharded factory 경로.**
