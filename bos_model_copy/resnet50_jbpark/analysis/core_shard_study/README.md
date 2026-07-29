# Core / Shard Study

## 목표
"AI 모델 특성상 20 코어 다 사용하는 건 비효율적이다" 라는 thesis 를 정직하게 검증.

## 오늘까지 (2026-07-21) 진행

### tt-metal 소스 분석 완료
`tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp`

1. **`find_closest_largest_divisor*` 계열** (line 54-90)
   - Padding-tolerant 약수 탐색. 최대 코어 수부터 감소하며 "코어 하나 통째 낭비 되지 않는 padding" 조건 만족하는 값 반환.

2. **`determine_parallel_config`** (line 169-237)
   - Sharding scheme (HEIGHT/BLOCK/WIDTH) 별로 위 함수들 조합해서 grid 결정.
   - HEIGHT: nhw 축 하나만 나눔.
   - BLOCK: nhw + channels 두 축. `start_divisor` 가 grid.x/grid.y (5 or 4). BLOCK 은 최대 20 못 씀 (각 축 상한 4~5).
   - WIDTH: channels 만.

3. **`calculate_L1_usage_for_conv_op`** (line 948-1115)
   - 코어당 L1 총합 = CB + tensor + halo output.
   - Halo 오버헤드가 `precise_input_size_per_core` 로 명시적 modeling.

4. **`determine_conv_config_for_auto_shard`** (line 1117-1210)
   - 진짜 sharding 선택 로직: **`total_size (L1) 최소` 인 후보 선택**.
   - Wall time 아님. L1 footprint heuristic.

### Halo 이해
- Conv 는 인접 픽셀 필요 → sharded 시 shard 경계에서 옆 코어 데이터 가져와야 함 = **halo**.
- `HaloDeviceOperation` 이 tt-metal 에 별도 op 로 존재 (우리 데이터에서도 확인됨).
- 코어 늘림 → shard 얇아짐 → halo/shard 비율 커짐. 
- 예: 3×3 conv on 56×56, kernel=3 → halo 는 상수 2 rows.
  - 14 코어 (4 rows/core): halo/shard = 50%
  - 20 코어 (2.8 rows/core → 3 padded): halo/shard = 67%
- → **CNN 이 20 코어 다 못 쓰는 진짜 아키텍처적 이유**.

### 오늘 발견 (CSV 비교 시도)

팀원 exp3 CSV (`ops_perf_results_resnet50_exp3_2026_07_20_14_20_32.csv`) 분석:

| 항목 | Baseline (2026_07_08) | EXP3 |
|---|---|---|
| batch_size | 4 | **1** |
| iteration | ~10 | 1 |
| total ops | 1260 | 115 |
| wall time | 39.06 ms | 2.59 ms |
| core count 분포 | 14 (20%), 15 (6%), 16 (48%), **20 (25%)** | 4, 8, 9, 13, 14, **16 (47%)**, **20 (14%)** |

**관찰**:
- **Baseline 도 25% op 이 20 코어 사용** — Config.py CORE_SET_14/16 은 reshard 지점만 강제. 개별 op 은 auto-shard.
- EXP3 는 batch=1 → auto-picker 가 작은 grid (4~9) 선택하는 op 도 등장. "16 코어 강제" 라는 프레임과 다름.

**비교의 함정**:
- Batch 4 vs 1 이라 per-op wall time 직접 비교 불가.
- Per-op 당 처리 데이터당 시간 지표가 필요한데, batch 도 core 도 다 다르니 confound.

## 내일 (2026-07-22) 할 것

**차량용 NPU 는 batch=1 이 표준.** Baseline 을 batch=1 로 다시 뽑고 exp3 와 정합 비교.

- [ ] `bos_model_copy/resnet50/run_resnet50.py -b 1` 로 baseline 재수집 → CSV
- [ ] `compare_csv.py` 확장:
  - PM IDEAL, PM COMPUTE, PM BANDWIDTH, PM FPU UTIL, NPE CONG IMPACT 컬럼 활용
  - Op 별 GLOBAL CALL COUNT 매치해서 same-op 직접 대조
  - CORE COUNT 히스토그램 나란히
  - Halo 오버헤드 실측치 (Halo bytes/Halo cycles 대 Conv bytes/Conv cycles)
- [ ] 저수준 verifier microkernel 진도

## 파일
- `compare_csv.py` — 오늘 만든 초기 스크립트. batch 다른 상태로 비교. 내일 확장 대상.
