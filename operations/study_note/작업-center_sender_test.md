# 작업 지시 — sender가 grid 중앙(interior)인 mcast 정합성 테스트 (Claude Code 실행용)

> **목표:** sender가 receiver grid **안쪽 중앙**에 있을 때 mcast가 정합하게 동작하는지 격리 검증.
> 기존은 in-grid **모서리 4개만** 테스트됨 → **center 추가.**
> 배경: `실험설계-mcast위치벤치.md` 참고 (임의 sender mcast는 코너/외부 10위치 PASS 완료).

---

## 0. 왜

- in-grid sender가 지금 **코너만** 검증됨. **중앙(interior)**은 미검증.
- 핵심 질문: **단일 mcast 호출(`noc_async_write_multicast`)이 내부-중앙 source에서도 전 방향 커버하나?**
  - 코너 source = fabric이 한 방향으로 커버 (쉬움, 검증됨)
  - **중앙 source = 양방향 커버 필요** → fabric이 단일 호출로 처리하나? = 이 테스트가 답함
- **mcast 로직(커널 코드)은 안 바뀜.** 같은 호출, sender 위치만 런타임 파라미터. 이 테스트는 "같은 로직이 중앙 source서도 정합하나"를 검증.

---

## 1. 실행 규칙

1. **컨테이너 안 + 보드에서** 실행 (`docker exec -it bos_host bash`).
2. **공유 컨테이너.** 소스 수정 전 **`.orig` 백업 필수.** **`git add` 금지.** 남의 파일·실험 폴더 건드리지 말 것.
3. **기존 sweep(Loopback/NoLoopback) 재실행 금지** — 이미 검증됨(~7분). 필터로 제외.
4. 새 코드 최소화: enum 1개 + case 1개 + SchemeSingle 수정만.

---

## 2. 대상 파일

`/tt-metal/tests/tt_metal/tt_metal/data_movement/one_to_all/test_multicast_schemes.cpp`

---

## 3. 수정 단계

### Step 0 — 백업
```bash
cp /tt-metal/tests/tt_metal/tt_metal/data_movement/one_to_all/test_multicast_schemes.cpp{,.orig}
```

### Step 1 — enum에 center 추가
`enum class MulticastSchemeType { ... }`의 **`End` 바로 앞**에 추가:
```cpp
    SenderInGridCenter,
    End
```
(기존 스킴 번호 유지 위해 맨 끝에 추가 → center = 11번)

### Step 2 — get_coordinates에 case 추가
`get_coordinates(...)`의 `switch` 안에 (default 앞) 추가:
```cpp
    case MulticastSchemeType::SenderInGridCenter:
        mst_core_coord = {sub_grid_dimension_size / 2, sub_grid_dimension_size / 2};
        sub_start_core_coord = {0, 0};
        break;
```

### Step 3 — `TensixDataMovementOneToAllMulticastSchemeSingle` 수정
이 TEST_F 안에서:
- **`GTEST_SKIP() << "Skipping test";` 줄 삭제** (un-skip)
- **`sub_grid_dimension_size`를 `3`으로** 고정
  ⚠️ **중요:** 기존은 `arch==WORMHOLE_B0 ? 7 : 9`인데, **Eagle-N(Blackhole)은 9 → 5×4 칩 범위 초과 = 가짜 FAIL.** 반드시 `3`으로.
- **`multicast_scheme`을 `MulticastSchemeType::SenderInGridCenter`로**
- `loopback = false`, `noc_id = NOC::NOC_0` 유지

### Step 4 — 재빌드
```bash
cd /tt-metal
cmake --build build --target unit_tests_data_movement
```
✅ 정상 = `[N/M] Linking CXX executable ...` (여러 컴파일 태스크). `[1/1]`만 뜨면 소스 반영 안 된 것 → Step 1-3 재확인.

### Step 5 — 실행 (center만)
```bash
./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*SchemeSingle*"
```
→ center(3×3, sender=(1,1)) 한 config만 실행 (수 초). 기존 sweep은 필터로 제외됨.

---

## 4. 보고 (PASS/FAIL 기준)

- **PASS** = `[ OK ] ...SchemeSingle` → 내부-중앙 source에서 mcast 정합.
  - 의미: **단일 mcast 호출이 어느 위치서든(중앙 포함) 커버** → conv 자유배치 코드 그대로 가능
- **FAIL** = 데이터 불일치 → 단일 호출로 center 커버 못 함.
  - 의미: center엔 별도 로직(2조각 mcast / exclude) 필요 = 제약 발견 (중요)
- **함께 보고:** latency(cycles). 코너(3×3 기준 ~68000cyc)와 비교 → center가 더 빠른가(중앙 hop 짧음).

---

## 5. 하지 말 것

- ❌ 기존 Loopback/NoLoopback TEST_F 수정·실행 (필터로 제외)
- ❌ 다른 소스 파일 수정
- ❌ `git add` / 커밋
- ❌ SchemeSingle의 grid를 3보다 크게 (Eagle-N 범위 초과 위험; 먼저 3으로 검증)

---

## 6. 성공 후 (다음, 참고)

- center PASS면 → **큰·비정사각형 grid(4×4, 5×4, 4×3)** + **변 중점 sender**로 확장 (별도 지시)
- 그 다음 → **exclude(중간 코어 rest = rotation 핵심)** 검증
- 최종 → conv factory 수정


(python_env) root@7575fef6ac31:/tt-metal# cmake --build build --target unit_tests_data_movement
[0/2] Re-checking globbed directories...
[1/3] Generating RPC stub for /tt-metal/tt_metal/impl/debug/inspector/rpc.capnp
Cannot generate stub file for Inspector. If you need python autocomplete/static analysis, install with: pip install pycapnp
[2/3] Building CXX object tests/tt_metal/tt_metal/data_movement/CMakeFiles/unit_tests_data_movement.dir/Unity/unity_0_cxx.cxx.o
FAILED: tests/tt_metal/tt_metal/data_movement/CMakeFiles/unit_tests_data_movement.dir/Unity/unity_0_cxx.cxx.o
/usr/bin/clang++-17 -DEAGLE_AS_TT -DENCHANTUM_ENABLE_MSVC_SPEEDUP=1 -DFMT_HEADER_ONLY=1 -DSPDLOG_FMT_EXTERNAL -DSPDLOG_FWRITE_UNLOCKED -DTRACY_ENABLE -DTRACY_IMPORTS -DTT_ENABLE_LIGHT_METAL_TRACE=1 -I/tt-metal/tt_metal/api -I/tt-metal/build_Release_tracy/tt_metal/api -I/tt-metal/tt_metal/api/tt-metalium -I/tt-metal/tt_metal/hostdevcommon/api -I/tt-metal/tt_stl/. -I/tt-metal/tt_metal -I/tt-metal/tt_metal/impl -I/tt-metal/tt_metal/impl/debug -I/tt-metal/tt_metal/impl/profiler -I/tt-metal/tt_metal/include -I/tt-metal/build_Release_tracy/tt_metal/impl -I/tt-metal -I/tt-metal/tt_metal/common -I/tt-metal/tt_metal/third_party/umd/src/firmware/riscv -I/tt-metal/.cpmcache/simd-everywhere/b3b426f78574ef837b17f42e86bab88314c5e4db -I/tt-metal/tt_metal/llrt -I/tt-metal/tt_metal/fabric -I/tt-metal/tt_metal/lite_fabric -I/tt-metal/.cpmcache/flatbuffers/2c4062bffa52fa4157b1b4deeae73395df475fda/include -I/tt-metal/tests -I/tt-metal/tests/tt_metal/tt_metal/common -I/tt-metal/tests/tt_metal/tt_metal/data_movement -I/tt-metal/tests/tt_metal/tt_metal/data_movement/common -isystem /tt-metal/tt_metal/third_party/umd/device/api -isystem /tt-metal/.cpmcache/enchantum/2fb7ab238e36c101b9848892ddb6382276b65837/enchantum/include -isystem /tt-metal/.cpmcache/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6/include -isystem /tt-metal/tt_metal/third_party/tracy/public -isystem /tt-metal/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include -isystem /tt-metal/.cpmcache/reflect/f93e77475670eaeacf332927dfe8b50e3f3812e0 -isystem /tt-metal/.cpmcache/tt-logger/5d45c661affa374c5656335a77010ab503f96a99/include -isystem /tt-metal/.cpmcache/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include -isystem /tt-metal/.cpmcache/taskflow/52063f60902bfeb362fa4616b1394ab5efe30994 -isystem /tt-metal/.cpmcache/googletest/96129d89f45386492ae46d6bb8c027bc3df5f949/googlemock/include -isystem /tt-metal/.cpmcache/googletest/96129d89f45386492ae46d6bb8c027bc3df5f949/googlemock -isystem /tt-metal/.cpmcache/googletest/96129d89f45386492ae46d6bb8c027bc3df5f949/googletest/include -isystem /tt-metal/.cpmcache/googletest/96129d89f45386492ae46d6bb8c027bc3df5f949/googletest -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/algorithm/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/array/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/assert/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/config/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/core/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/static_assert/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/throw_exception/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/bind/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/concept_check/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/preprocessor/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/type_traits/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/exception/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/smart_ptr/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/move/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/tuple/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/function/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/iterator/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/detail/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/function_types/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/mpl/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/predef/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/utility/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/io/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/fusion/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/container_hash/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/describe/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/mp11/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/typeof/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/functional/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/optional/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/range/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/conversion/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/regex/include -isystem /tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/unordered/include -O3 -std=c++20 -fPIE -fsized-deallocation -march=x86-64-v3 -fPIC -fvisibility-inlines-hidden -Wall -Werror -Wconditional-uninitialized -Wno-deprecated-declarations -Wno-unused-but-set-variable -Wno-c++11-narrowing -Wno-deprecated-volatile -Wno-logical-op-parentheses -Wno-reorder-ctor -Wno-unused-lambda-capture -Wno-unused-private-field -Wno-unknown-pragmas -Xclang -fno-pch-timestamp -Wno-int-to-pointer-cast -fno-omit-frame-pointer -isystem /tt-metal/.cpmcache/taskflow/52063f60902bfeb362fa4616b1394ab5efe30994 -MD -MT tests/tt_metal/tt_metal/data_movement/CMakeFiles/unit_tests_data_movement.dir/Unity/unity_0_cxx.cxx.o -MF tests/tt_metal/tt_metal/data_movement/CMakeFiles/unit_tests_data_movement.dir/Unity/unity_0_cxx.cxx.o.d -o tests/tt_metal/tt_metal/data_movement/CMakeFiles/unit_tests_data_movement.dir/Unity/unity_0_cxx.cxx.o -c /tt-metal/build_Release_tracy/tests/tt_metal/tt_metal/data_movement/CMakeFiles/unit_tests_data_movement.dir/Unity/unity_0_cxx.cxx
In file included from /tt-metal/build_Release_tracy/tests/tt_metal/tt_metal/data_movement/CMakeFiles/unit_tests_data_movement.dir/Unity/unity_0_cxx.cxx:25:
/tt-metal/tests/tt_metal/tt_metal/data_movement/one_to_all/test_multicast_schemes.cpp:11:10: fatal error: 'distributed/mesh_device_impl.hpp' file not found
   11 | #include <distributed/mesh_device_impl.hpp>
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1 error generated.
ninja: build stopped: subcommand failed.

(python_env) root@7575fef6ac31:/tt-metal# ./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*SchemeSingle*"
Running main() from gmock_main.cc
Note: Google Test filter = *SchemeSingle*
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from GenericMeshDeviceFixture
[ RUN      ] GenericMeshDeviceFixture.TensixDataMovementOneToAllMulticastSchemeSingle
2026-08-10 18:06:45.457 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-10 18:06:45.491 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-10 18:06:45.491 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-08-10 18:06:45.493 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:202)
2026-08-10 18:06:45.515 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-10 18:06:45.549 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-10 18:06:45.549 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-08-10 18:06:45.573 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-10 18:06:45.607 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-10 18:06:45.610 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-08-10 18:06:45.611 | info     |             UMD | Harvesting mask for chip 0 is 0x0 (NOC0: 0x0, simulated harvesting mask: 0x0). (cluster.cpp:399)
2026-08-10 18:06:45.642 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-10 18:06:45.642 | info     |             UMD | Initializing iommu for sysmem (size: 0x40000000). (sysmem_manager.cpp:300)
2026-08-10 18:06:45.642 | info     |             UMD | Allocating sysmem without hugepages (size: 0x40000000). (sysmem_manager.cpp:306)
2026-08-10 18:06:45.848 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-08-10 18:06:45.848 | info     |             UMD | Opening local chip ids/PCIe ids: {0}/[0] and remote chip ids {} (cluster.cpp:251)
2026-08-10 18:06:45.848 | info     |             UMD | IOMMU: enabled (cluster.cpp:173)
2026-08-10 18:06:45.848 | info     |             UMD | KMD version: 2.3.0 (cluster.cpp:176)
2026-08-10 18:06:45.848 | info     |          Device | start_device in llrt (tt_cluster.cpp:415)
2026-08-10 18:06:45.871 | info     |             UMD | Pinning pages for DMA: virtual address 0x7aec83000000 and size 0x40000000 pinned to physical address 0x40000000 without mapping to noc (pci_device.cpp:618)
2026-08-10 18:06:45.871 | info     |             UMD | Mapped sysmem without hugepages to IOVA 0x40000000; NOC address 0x0 (sysmem_manager.cpp:354)
2026-08-10 18:06:45.915 | info     |             UMD | Initializing memory barriers (local_chip.cpp:171)
2026-08-10 18:06:46.020 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-10 18:06:46.054 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-10 18:06:46.054 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-08-10 18:06:47.471 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
2026-08-10 18:06:49.891 | info     |            Test | Running Test ID: 102, Run ID: 0 (test_one_to_all.cpp:189)
[       OK ] GenericMeshDeviceFixture.TensixDataMovementOneToAllMulticastSchemeSingle (5077 ms)
[----------] 1 test from GenericMeshDeviceFixture (5077 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (5077 ms total)
[  PASSED  ] 1 test.
2026-08-10 18:06:50.511 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:427)
2026-08-10 18:06:50.525 | info     |             UMD | Unpinning pages for DMA: virtual address 0x7aec83000000 and size 0x40000000 (pci_device.cpp:644)



python_env) root@7575fef6ac31:/tt-metal# cmake --build build --target unit_tests_data_movement
[0/2] Re-checking globbed directories...
[1/3] Generating RPC stub for /tt-metal/tt_metal/impl/debug/inspector/rpc.capnp
Cannot generate stub file for Inspector. If you need python autocomplete/static analysis, install with: pip install pycapnp
[3/3] Linking CXX executable test/tt_metal/unit_tests_data_movement
(python_env) root@7575fef6ac31:/tt-metal# ./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*ArrivalHeatmap*"
Running main() from gmock_main.cc
Note: Google Test filter = *ArrivalHeatmap*
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from GenericMeshDeviceFixture
[ RUN      ] GenericMeshDeviceFixture.TensixDataMovementMulticastArrivalHeatmap
2026-08-11 05:31:41.037 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-11 05:31:41.071 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-11 05:31:41.071 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-08-11 05:31:41.072 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:202)
2026-08-11 05:31:41.105 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-11 05:31:41.124 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-11 05:31:41.124 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-08-11 05:31:41.148 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-11 05:31:41.169 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-11 05:31:41.171 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-08-11 05:31:41.172 | info     |             UMD | Harvesting mask for chip 0 is 0x0 (NOC0: 0x0, simulated harvesting mask: 0x0). (cluster.cpp:399)
2026-08-11 05:31:41.204 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-11 05:31:41.204 | info     |             UMD | Initializing iommu for sysmem (size: 0x40000000). (sysmem_manager.cpp:300)
2026-08-11 05:31:41.204 | info     |             UMD | Allocating sysmem without hugepages (size: 0x40000000). (sysmem_manager.cpp:306)
2026-08-11 05:31:41.395 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-08-11 05:31:41.395 | info     |             UMD | Opening local chip ids/PCIe ids: {0}/[0] and remote chip ids {} (cluster.cpp:251)
2026-08-11 05:31:41.395 | info     |             UMD | IOMMU: enabled (cluster.cpp:173)
2026-08-11 05:31:41.395 | info     |             UMD | KMD version: 2.3.0 (cluster.cpp:176)
2026-08-11 05:31:41.396 | info     |          Device | start_device in llrt (tt_cluster.cpp:415)
2026-08-11 05:31:41.427 | info     |             UMD | Pinning pages for DMA: virtual address 0x741620000000 and size 0x40000000 pinned to physical address 0x40000000 without mapping to noc (pci_device.cpp:618)
2026-08-11 05:31:41.427 | info     |             UMD | Mapped sysmem without hugepages to IOVA 0x40000000; NOC address 0x0 (sysmem_manager.cpp:354)
2026-08-11 05:31:41.483 | info     |             UMD | Initializing memory barriers (local_chip.cpp:171)
2026-08-11 05:31:41.576 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-11 05:31:41.610 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-08-11 05:31:41.610 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-08-11 05:31:43.053 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
2026-08-11 05:31:45.454 | info     |            Test | ARRIVAL: 5x4 sender(0,0) TL grid=5x4 sender=(0,0) NoC0 (test_multicast_schemes.cpp:298)
2026-08-11 05:31:45.454 | critical |          Always | Cannot create CoreRangeSet with specified core ranges because core ranges [(x=0,y=0) - (x=0,y=0)] and [(x=0,y=0) - (x=4,y=3)] overlap! (assert.hpp:103)
unknown file: Failure
C++ exception with description "TT_THROW @ /tt-metal/tt_metal/common/core_coord.cpp:420: tt::exception
info:
Cannot create CoreRangeSet with specified core ranges because core ranges [(x=0,y=0) - (x=0,y=0)] and [(x=0,y=0) - (x=4,y=3)] overlap!
backtrace:
 --- /tt-metal/build_Release_tracy/tt_metal/libtt_metal.so(+0x4e3c28) [0x74168c268c28]
 --- CoreRangeSet::validate_no_overlap()
 --- CoreRangeSet::CoreRangeSet(std::set<CoreRange, std::less<CoreRange>, std::allocator<CoreRange> > const&)
 --- tt::tt_metal::GenericMeshDeviceFixture_TensixDataMovementMulticastArrivalHeatmap_Test::TestBody()
 --- void testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>(testing::Test*, void (testing::Test::*)(), char const*)
 --- testing::Test::Run()
 --- testing::TestInfo::Run()
 --- testing::TestSuite::Run()
 --- testing::internal::UnitTestImpl::RunAllTests()
 --- bool testing::internal::HandleExceptionsInMethodIfSupported<testing::internal::UnitTestImpl, bool>(testing::internal::UnitTestImpl*, bool (testing::internal::UnitTestImpl::*)(), char const*)
 --- testing::UnitTest::Run()
 --- ./build/test/tt_metal/unit_tests_data_movement(main+0x3e) [0x6136c4e87ede]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x29d90) [0x74168b2f9d90]
 --- /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80) [0x74168b2f9e40]
 --- ./build/test/tt_metal/unit_tests_data_movement(_start+0x25) [0x6136c4dff065]
" thrown in the test body.
[  FAILED  ] GenericMeshDeviceFixture.TensixDataMovementMulticastArrivalHeatmap (4460 ms)
[----------] 1 test from GenericMeshDeviceFixture (4460 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (4460 ms total)
[  PASSED  ] 0 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] GenericMeshDeviceFixture.TensixDataMovementMulticastArrivalHeatmap

 1 FAILED TEST
2026-08-11 05:31:45.474 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:427)
2026-08-11 05:31:45.489 | info     |             UMD | Unpinning pages for DMA: virtual address 0x741620000000 and size 0x40000000 (pci_device.cpp:644)
