# tt-metal NoC 테스트 파일 경로 맵

Aging rotation / NoC mcast 실험 설계를 위한 tt-metal 기존 자산 인벤토리.
전체 경로는 `C:\NPU_Challenge\tt-metal\` 기준.

---

## 1) NoC 마이크로벤치 (single-chip, on-die NoC)

### `tests/tt_metal/tt_metal/perf_microbenchmark/`

| 경로 | 내용 |
|---|---|
| `noc/test_noc_unicast_vs_multicast_to_single_core_latency.cpp` | mcast vs unicast 1-core latency |
| `noc/kernels/multicast_to_single_core.cpp` | mcast 커널 |
| `noc/kernels/unicast_to_single_core.cpp` | unicast 커널 |
| `2_noc_adjacent/test_noc_adjacent.cpp` | grid 전체 인접-코어 read/write BW |
| `2_noc_adjacent/kernels/noc_read.cpp`, `noc_write.cpp` | |
| `2_noc_rtor/test_noc_rtor.cpp` | reader-to-reader 버전 |
| `2_noc_rtor/kernels/noc_read.cpp`, `noc_write.cpp` | |
| `old/noc/test_noc_read_global_l1.cpp` | (legacy) global L1 read |
| `old/noc/test_noc_read_local_l1.cpp` | (legacy) local L1 read |
| `old/noc/kernels/reader_global_l1.cpp`, `reader_local_l1.cpp` | |

### `tests/tt_metal/microbenchmarks/noc/` (파이썬 러너)

- `test_noc_unicast_vs_multicast_to_single_core_latency.py` — 위 C++ 벤치 실행 + tracy 로그 파싱

---

## 2) Data-movement 프레임워크 ★ (aging 연구에 가장 유용)

### `tests/tt_metal/tt_metal/data_movement/`

| 경로 | 패턴 | 관련성 |
|---|---|---|
| `one_to_all/test_one_to_all.cpp` | 1 sender → N receivers | ★★★ mcast rectangle 실험 base |
| `one_to_all/test_multicast_schemes.cpp` | 여러 mcast scheme 비교 | ★★★ |
| `one_to_all/kernels/sender_multicast.cpp` | | |
| `one_to_all/kernels/sender_multicast_2_0.cpp` | | |
| `one_to_all/kernels/sender_multicast_sem.cpp` | semaphore 버전 | |
| `one_from_all/test_one_from_all.cpp` | N senders → 1 receiver | ★★ gather 패턴 |
| `all_to_all/test_all_to_all.cpp` | full mesh | ★★ |
| `all_from_all/test_all_from_all.cpp` | | ★ |
| `one_to_one/test_one_to_one.cpp` | 단일 페어 | baseline |
| `one_from_one/test_one_from_one.cpp` | 단일 페어 | baseline |
| `multicast_atomics/test_multicast_atomic_semaphore.cpp` | mcast + atomic sem | |
| `multicast_atomics/kernels/multicast_atomic_sender.cpp`, `_receiver.cpp` (+`_2_0`) | | |
| `noc_api_latency/test_noc_api_latency.cpp` | NoC API별 latency | ★ |
| `noc_api_latency/kernels/multicast_write_2_0.cpp` | | |
| `noc_estimator/test_noc_estimator.cpp` | tt-npe 검증용? | ★★ 확인 필요 |
| `noc_estimator_tests/test_noc_estimator.cpp` | | ★★ |
| `atomics/test_atomic_semaphore_bandwidth.cpp` | atomic BW | |
| `core_bidirectional/test_core_bidirectional.cpp` | 양방향 | |
| `direct_write/test_direct_write.cpp` | | |
| `loopback/test_loopback.cpp` | | |
| `one_packet/test_one_packet.cpp` | 최소 payload | |
| `transaction_id/test_transaction_id.cpp` | | |
| `matmul/test_matmul_1d.cpp`, `test_matmul_1d_v2.cpp`, `test_matmul_2d.cpp` | matmul 트래픽 | |
| `conv_hardcoded/test_conv_hardcoded.cpp` | conv 트래픽 | ★ ResNet 관련 |
| `interleaved/test_interleaved.cpp` | | |
| `interleaved_to_sharded_hardcoded/test_interleaved_to_sharded_hardcoded.cpp` | | ★ resharding cost |
| `reshard_hardcoded/test_reshard_hardcoded.cpp` | | ★ |
| `multi_interleaved/test_multi_interleaved.cpp` | | |
| `dram_neighbour/test_dram_neighbour.cpp` | DRAM→인접 L1 | |
| `dram_sharded/test_dram_sharded.cpp` | | |
| `dram_unary/test_unary_dram.cpp` | | |
| `deinterleave_hardcoded/test_deinterleave_hardcoded.cpp` | | |
| `pcie_read_bw/test_pcie_read_bw.cpp`, `pcie_write_bw/test_pcie_write_bw.cpp` | | |
| `quasar_*` | Quasar (차세대) 전용 — 무시 | |

### `tests/tt_metal/tt_metal/data_movement/python/` ★

| 파일 | 용도 |
|---|---|
| `test_data_movement.py` | pytest 러너 |
| `config.py`, `constants.py`, `conftest.py` | 설정 |
| `stats_collector.py`, `stats_reporter.py` | 통계 수집 |
| `performance_checker.py` | 성능 체크 |
| `plotter.py`, `heatmap.py` | ★★ **NoC 히트맵 시각화 유틸** |
| `test_metadata.py` | |

---

## 3) Stress / mcast integration

| 경로 | 내용 |
|---|---|
| `tests/tt_metal/tt_metal/test_stress_noc_mcast.cpp` | stress mcast 러너 |
| `tests/tt_metal/tt_metal/test_kernels/stress_noc_mcast.cpp` | stress 커널 |
| `tests/tt_metal/tt_metal/api/test_dram_to_l1_multicast.cpp` | DRAM→L1 mcast API |
| `tests/tt_metal/tt_metal/test_dram_to_l1_multicast.cpp` | |
| `tests/tt_metal/tt_metal/test_dram_to_l1_multicast_loopback_src.cpp` | |
| `tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_multicast.cpp` | |
| `tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_multicast_include_src.cpp` | |
| `tests/tt_metal/tt_metal/integration/matmul/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast.cpp` | matmul dual mcast |
| `tests/tt_metal/tt_metal/integration/matmul/test_matmul_multi_core_multi_dram_inX_mcast.cpp` | |
| `tests/tt_metal/tt_metal/test_matmul_multi_core_multi_dram_in0_mcast.cpp` | |
| `tests/tt_metal/tt_metal/test_matmul_multi_core_multi_dram_in1_mcast.cpp` | |
| `tests/tt_metal/tt_metal/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast.cpp` | |

---

## 4) API · Dynamic NoC unit tests

| 경로 | 내용 |
|---|---|
| `tests/tt_metal/tt_metal/api/test_noc.cpp` | NoC API gtest |
| `tests/tt_metal/tt_metal/noc/test_dynamic_noc.cpp` | dynamic NoC 할당 |
| `tests/tt_metal/tt_metal/noc/CMakeLists.txt`, `sources.cmake` | 빌드 |
| `tests/tt_metal/tt_metal/debug_tools/watcher/test_noc_sanitize_delays.cpp` | watcher 검증 |
| `tests/tt_metal/tt_metal/debug_tools/watcher/test_mcast_wrap_around.cpp` | mcast 좌표 wrap |
| `tests/tt_metal/tt_metal/debug_tools/watcher/test_mcast_wrap_around_device.cpp` | |

---

## 5) Ethernet / Fabric mcast (chip-to-chip, **Eagle-N 연구 스코프 아님**)

| 경로 | 비고 |
|---|---|
| `tests/tt_metal/tt_fabric/**` | 전부 다중 chip fabric — 스킵 |
| `tests/tt_metal/microbenchmarks/ethernet/**` | Ethernet BW — 스킵 |
| `perf_microbenchmark/routing/kernels/tt_fabric_2d_mcast_tx.cpp`, `tt_fabric_line_mcast_tx.cpp` | fabric mcast — 스킵 |

---

## 6) 참고용 (예제 · 문서)

| 경로 | 내용 |
|---|---|
| `tt_metal/programming_examples/NoC_tile_transfer/noc_tile_transfer.cpp` | 학습용 예제 |
| `tt_metal/programming_examples/contributed/multicast/multicast.cpp` | 예제 |
| `tt_metal/programming_examples/profiler/test_noc_event_profiler/test_noc_event_profiler.cpp` | ★ NoC event profiler 사용법 |
| `ttnn/examples/lab_multicast/lab_multicast.cpp` | ttnn mcast 예제 |
| `tech_reports/prog_examples/multicast/multicast.md` | mcast 문서 |
| `tt_metal/impl/experimental/noc_estimator/noc_estimator.cpp` | tt-npe 내장 estimator 소스 |
| `tt_metal/impl/debug/noc_logging.cpp`, `noc_debugging.cpp` | NoC 로깅 |
| `tools/triage/check_noc_locations.py`, `check_noc_status.py` | NoC 진단 스크립트 |

---

## 아키텍처별 하드웨어 헤더 (참고)

- `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_overlay_parameters.cpp` — **Eagle-N (Blackhole)**
- `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_overlay_parameters.cpp`
- `tt_metal/hw/firmware/src/tt-2xx/quasar/noc_address_translation_tables.cpp`

---

## 추천 진입점 (aging rotation 실험)

1. **`data_movement/one_to_all/test_multicast_schemes.cpp`** + **`python/test_data_movement.py`** + **`python/heatmap.py`**
   → 이 세트가 이미 "mcast scheme 바꿔가며 실측 + 히트맵"을 하고 있음.
   Rotation policy만 축으로 추가하면 됨.
2. **`data_movement/noc_estimator/`** — tt-npe 결과 vs 실측 diff 확인용
3. **`perf_microbenchmark/noc/`** — 순수 latency 검증용 sanity

---

## Next actions

- [ ] `test_multicast_schemes.cpp` 열어서 mcast scheme 정의 방식 확인
- [ ] `python/heatmap.py` 로직 확인 (per-link vs per-core 어떤 축인지)
- [ ] `noc_estimator` 실측 커버리지 확인 (npe와의 차이)
- [ ] rotation policy 축 추가 지점 식별
