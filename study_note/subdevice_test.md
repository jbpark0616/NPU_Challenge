(python_env) root@7575fef6ac31:/tt-metal# find /tt-metal -name "*sub_device*" -o -name "*subdevice*" 2>/dev/null | head -20
/tt-metal/tests/tt_metal/tt_metal/dispatch/dispatch_trace/test_sub_device.cpp
/tt-metal/tests/tt_metal/tt_metal/dispatch/sub_device_test_utils.hpp
/tt-metal/tests/tt_metal/tt_metal/dispatch/dispatch_program/test_sub_device.cpp
/tt-metal/tests/tt_metal/tt_metal/dispatch/dispatch_buffer/test_sub_device.cpp
/tt-metal/tests/tt_metal/tt_metal/test_kernels/misc/sub_device
/tt-metal/tests/tt_metal/distributed/test_mesh_sub_device.cpp
/tt-metal/tests/ttnn/unit_tests/test_sub_device.py
/tt-metal/inspect_subdevice_async_api.log
/tt-metal/models/demos/llama3_subdevices
/tt-metal/models/demos/llama3_70b_galaxy/tests/unit_tests/test_subdevice_switching.py
/tt-metal/inspect_subdevice_async_api.py
/tt-metal/.build/default/tt_metal/impl/CMakeFiles/impl.dir/ASan/sub_device
/tt-metal/.build/default/tt_metal/impl/CMakeFiles/impl.dir/Release/sub_device
/tt-metal/.build/default/tt_metal/impl/CMakeFiles/impl.dir/Debug/sub_device
/tt-metal/.build/default/tt_metal/impl/CMakeFiles/impl.dir/RelWithDebInfo/sub_device
/tt-metal/.build/default/tt_metal/impl/CMakeFiles/impl.dir/TSan/sub_device
/tt-metal/build_Release_tracy/tests/tt_metal/distributed/CMakeFiles/distributed_unit_tests.dir/test_mesh_sub_device.cpp.o
/tt-metal/build_Release_tracy/tt_metal/impl/CMakeFiles/impl.dir/sub_device
/tt-metal/build_Release_tracy/tt_metal/impl/CMakeFiles/impl.dir/sub_device/sub_device_manager_tracker.cpp.o
/tt-metal/build_Release_tracy/tt_metal/impl/CMakeFiles/impl.dir/sub_device/sub_device.cpp.o
(python_env) root@7575fef6ac31:/tt-metal# find /tt-metal/tests -name "test_*.py" | xargs grep -l "SubDevice\|create_sub_device" 2>/dev/null | head -10
/tt-metal/tests/tt_metal/tools/profiler/test_device_profiler.py
/tt-metal/tests/ttnn/unit_tests/test_sub_device.py
/tt-metal/tests/ttnn/unit_tests/operations/ccl/test_all_reduce_async.py
/tt-metal/tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py
/tt-metal/tests/ttnn/unit_tests/operations/ccl/test_reduce_scatter_TG_nightly.py
/tt-metal/tests/ttnn/unit_tests/operations/ccl/test_all_to_all_dispatch_t3000.py
/tt-metal/tests/ttnn/unit_tests/operations/ccl/test_llama_prefill_ccl_ops.py
/tt-metal/tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py
/tt-metal/tests/ttnn/unit_tests/operations/ccl/test_new_all_broadcast.py
/tt-metal/tests/ttnn/unit_tests/operations/ccl/test_llama_reduce_scatter_async_TG.py
(python_env) root@7575fef6ac31:/tt-metal# cd /tt-metal && git log --oneline -5
git rev-parse HEAD
a035c886ba (HEAD -> release/A0_FC_RC5.1, origin/release/A0_FC_RC5.1, origin/release/A0_FCRC5_ORIN) add execution permission to env_set script
cd268cc7d6 modify python package dependency for aarch64
7be5af7847 Update cmake toolchain file to support ARM build (cmake will automatically detect system arch for native build)
d2be9ca9ae (origin/main) [Hotfix] Update submodule url to github and fix umd build issue (#135)
6165fdcc93 (tag: v0.64_e2_1.0) Merge pull request #98 from bos-semi/release/A0_FC_RC5
a035c886ba841831104e597b563826b06fe96c81

(python_env) root@7575fef6ac31:/tt-metal# cat /tt-metal/inspect_subdevice_async_api.py | head -80
echo "===="
cat /tt-metal/inspect_subdevice_async_api.log | head -100
import inspect
import pathlib
import re

import ttnn


def print_signature(name, obj):
    """Print the callable signature exposed by the installed Python binding."""
    print(f"\n{name}")
    print("-" * len(name))

    try:
        print("signature:", inspect.signature(obj))
    except Exception as error:
        print("signature unavailable:", error)

    print("docstring:")
    print(inspect.getdoc(obj) or "<no docstring>")


def search_source(root, patterns):
    """Search local TT-Metal source files for relevant API usage."""
    suffixes = {".py", ".cpp", ".cc", ".cxx", ".hpp", ".h", ".md"}

    for pattern in patterns:
        regex = re.compile(pattern)
        matches = []

        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue

            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append((path, line_number, line.strip()))

                    if len(matches) >= 15:
                        break

            if len(matches) >= 15:
                break

        print(f"\nSOURCE SEARCH: {pattern}")
        print("-" * 72)

        if not matches:
            print("No matches found.")
            continue

        for path, line_number, line in matches:
            print(f"{path}:{line_number}")
            print(f"    {line}")


def main():
    print("=" * 80)
    print("TTNN SubDevice / Command Queue / Synchronization API inspection")
    print("=" * 80)

    print("\n[Top-level API availability]")

    names = [
        "open_device",
        "close_device",
        "synchronize_device",
        "SubDevice",
        "SubDeviceId",
        "QueueId",
        "create_event",
        "record_event",
        "wait_for_event",
        "event_synchronize",
        "set_sub_device_stall_group",
        "reset_sub_device_stall_group",
====
================================================================================
TTNN SubDevice / Command Queue / Synchronization API inspection
================================================================================

[Top-level API availability]
open_device                        : True
close_device                       : True
synchronize_device                 : True
SubDevice                          : True
SubDeviceId                        : True
QueueId                            : True
create_event                       : False
record_event                       : True
wait_for_event                     : True
event_synchronize                  : True
set_sub_device_stall_group         : False
reset_sub_device_stall_group       : False

ttnn.open_device
----------------
signature unavailable: no signature found for builtin <built-in method open_device of PyCapsule object at 0x7bf5358d1f80>
docstring:
open_device(*, device_id: int, l1_small_size: int = 0, trace_region_size: int = 0, dispatch_core_config: ttnn._ttnn.device.DispatchCoreConfig = <ttnn._ttnn.device.DispatchCoreConfig object at 0x7bf5356f93b0>, worker_l1_size: int = 0) -> ttnn._ttnn.multi_device.MeshDevice


Open a device with the given device_id. If the device is already open, return the existing device.

Keyword Args:
    device_id (int): The device ID to open.
    l1_small_size (int, optional): The size of the L1 small buffer. Defaults to `ttnn.device.DEFAULT_L1_SMALL_SIZE`.
    trace_region_size (int, optional): The size of the trace region. Defaults to `ttnn.device.DEFAULT_TRACE_REGION_SIZE`.
    worker_l1_size (int, optional): The size of the user-allocatable L1 buffer. Defaults to a hardware-specific value.
    dispatch_core_type (ttnn.device.DispatchCoreType, optional): The type of dispatch core to use. Defaults to `ttnn.device.DispatchCoreType.WORKER`.

Returns:
    ttnn.Device: The device with the given device_id.

Example:
    >>> device_id = 0
    >>> device = ttnn.open_device(device_id=device_id)
    >>> print(device)
    <ttnn._ttnn.device.Device object at 0x7fbac5bfc1b0>

ttnn.synchronize_device
-----------------------
signature unavailable: no signature found for builtin <built-in method synchronize_device of PyCapsule object at 0x7bf5358d2580>
docstring:
synchronize_device(device: ttnn._ttnn.multi_device.MeshDevice, cq_id: Optional[ttnn._ttnn.types.QueueId] = None, sub_device_ids: list[ttnn._ttnn.device.SubDeviceId] = []) -> None


Synchronize the device with host by waiting for all operations to complete.
If cq_id is provided then only the operations associated with that cq_id are waited for,
otherwise operations for all command queues are waited on.
If the device has been configured with sub-devices, then sub_device_ids can be provided to only wait
for the operations that ran on the specified sub-devices, otherwise all sub-devices (the entire chip) are waited on.

Args:
    device (ttnn.device.Device): The device to synchronize with.
    cq_id (int, optional): The command queue ID to synchronize. Defaults to `None`.
    sub_device_ids (List[ttnn.SubDeviceId], optional): The sub-device IDs to synchronize. Defaults to sub-devices set by set_sub_device_stall_group.

Returns:
    `None`: The op ensures that all operations are completed.

Example:
    >>> device_id = 0
    >>> device = ttnn.open_device(device_id=device_id)
    >>> # Assume some operations are queued on the device
    >>> ttnn.synchronize_device(device)

ttnn.record_event
-----------------
signature unavailable: no signature found for builtin <built-in method record_event of PyCapsule object at 0x7bf5357084b0>
docstring:
record_event(mesh_device: ttnn._ttnn.multi_device.MeshDevice, cq_id: ttnn._ttnn.types.QueueId, sub_device_ids: list[ttnn._ttnn.device.SubDeviceId] = [], device_range: Optional[ttnn._ttnn.multi_device.MeshCoordinateRange] = None) -> ttnn._ttnn.events.MeshEvent


Records the completion of commands on this CQ, preceeding this call.

Args:
    mesh_device (ttnn.MeshDevice*): The device on which the event is being recorded.
    cq_id (int): The Command Queue on which event completion will be recorded.
    sub_device_ids (List[ttnn.SubDeviceId], optional): The sub-device IDs to record completion for. Defaults to sub-devices set by set_sub_device_stall_group.
    device_range (ttnn.MeshCoordinateRange, optional): The range of devices to record completion for. Defaults to all devices.

Returns:
    MeshEvent: The event used to record completion of preceeding commands.

ttnn.wait_for_event
-------------------
signature unavailable: no signature found for builtin <built-in method wait_for_event of PyCapsule object at 0x7bf5357084e0>
docstring:
wait_for_event(cq_id: ttnn._ttnn.types.QueueId, mesh_event: ttnn._ttnn.events.MeshEvent) -> None


Inserts a barrier - makes a CQ wait until an event is recorded.

Args:
    cq_id (int): The Command Queue on which the barrier is being issued.
    mesh_event (MeshEvent): The Command Queue will stall until this event is completed.
(python_env) root@7575fef6ac31:/tt-metal#   



(python_env) root@7575fef6ac31:/tt-metal# cd /tt-metal
pytest tests/ttnn/unit_tests/test_sub_device.py -v -s
2026-07-19 19:52:11.456 | DEBUG    | ttnn:<module>:77 - Initial ttnn.CONFIG:
Config{cache_path=/root/.cache/ttnn,model_cache_path=/root/.cache/ttnn/models,tmp_dir=/tmp/ttnn,enable_model_cache=false,enable_fast_runtime_mode=true,throw_exception_on_fallback=false,enable_logging=false,enable_graph_report=false,enable_detailed_buffer_report=false,enable_detailed_tensor_report=false,enable_comparison_mode=false,comparison_mode_should_raise_exception=false,comparison_mode_pcc=0.9999,root_report_path=generated/ttnn/reports,report_name=std::nullopt,std::nullopt}
================================================================== test session starts ===================================================================
platform linux -- Python 3.10.12, pytest-7.2.2, pluggy-1.6.0 -- /tt-metal/python_env/bin/python3
cachedir: .pytest_cache
benchmark: 4.0.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /tt-metal, configfile: pytest.ini
plugins: split-0.8.2, anyio-4.13.0, github-actions-annotate-failures-0.3.0, timeout-2.2.0, xdist-3.6.1, dash-2.15.0, benchmark-4.0.0
timeout: 300.0s
timeout method: signal
timeout func_only: False
collecting ... 2026-07-19 19:52:11.572 | WARNING  | tests.ttnn.conftest:pytest_collection_modifyitems:39 - Fast Runtime Mode is ON. Skipping tests tagged with @pytest.mark.requires_fast_runtime_mode_off
collected 4 items                                                                                                                                        

tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices 2026-07-19 19:52:11.573 | DEBUG    | ttnn:manage_config:85 - Set ttnn.CONFIG.report_name to tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices: 2026-07-19 19:52:11 (UTC)
2026-07-19 19:52:11.598 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 19:52:11.632 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 19:52:11.632 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 19:52:11.633 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:202)
2026-07-19 19:52:11.656 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 19:52:11.690 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 19:52:11.690 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 19:52:11.724 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 19:52:11.742 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 19:52:11.745 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 19:52:11.746 | info     |             UMD | Harvesting mask for chip 0 is 0x0 (NOC0: 0x0, simulated harvesting mask: 0x0). (cluster.cpp:399)
2026-07-19 19:52:11.766 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 19:52:11.767 | info     |             UMD | Initializing iommu for sysmem (size: 0x40000000). (sysmem_manager.cpp:300)
2026-07-19 19:52:11.767 | info     |             UMD | Allocating sysmem without hugepages (size: 0x40000000). (sysmem_manager.cpp:306)
2026-07-19 19:52:11.943 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 19:52:11.944 | info     |             UMD | Opening local chip ids/PCIe ids: {0}/[0] and remote chip ids {} (cluster.cpp:251)
2026-07-19 19:52:11.944 | info     |             UMD | IOMMU: enabled (cluster.cpp:173)
2026-07-19 19:52:11.944 | info     |             UMD | KMD version: 2.3.0 (cluster.cpp:176)
2026-07-19 19:52:11.944 | info     |          Device | start_device in llrt (tt_cluster.cpp:415)
2026-07-19 19:52:11.989 | info     |             UMD | Pinning pages for DMA: virtual address 0x74fea8a00000 and size 0x40000000 pinned to physical address 0x7fc0000000 without mapping to noc (pci_device.cpp:618)
2026-07-19 19:52:11.989 | info     |             UMD | Mapped sysmem without hugepages to IOVA 0x7fc0000000; NOC address 0x0 (sysmem_manager.cpp:354)
2026-07-19 19:52:12.046 | info     |             UMD | Initializing memory barriers (local_chip.cpp:171)
2026-07-19 19:52:12.120 | DEBUG    | ttnn.device:__init__:150 - Using default dispatch core type for this system: DispatchCoreType.WORKER
2026-07-19 19:52:12.120 | DEBUG    | ttnn.device:__init__:152 - Using default dispatch core axis for this system: DispatchCoreAxis.COL
2026-07-19 19:52:13.465 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
2026-07-19 19:52:15.795 | critical |          Always | Tensix cores {[(x=4,y=4) - (x=4,y=4)]} specified in sub device must be within device grid [(x=0,y=0) - (x=4,y=3)] (assert.hpp:103)
FAILED2026-07-19 19:52:15.858 | DEBUG    | ttnn:manage_config:88 - Restored ttnn.CONFIG.report_name to None

tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices_mesh[silicon_arch_name=blackhole] 2026-07-19 19:52:15.860 | DEBUG    | ttnn:manage_config:85 - Set ttnn.CONFIG.report_name to tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices_mesh[silicon_arch_name=blackhole]: 2026-07-19 19:52:15 (UTC)
2026-07-19 19:52:15.860 | DEBUG    | ttnn.device:__init__:150 - Using default dispatch core type for this system: DispatchCoreType.WORKER
2026-07-19 19:52:15.860 | DEBUG    | ttnn.device:__init__:152 - Using default dispatch core axis for this system: DispatchCoreAxis.COL
2026-07-19 19:52:16.193 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
2026-07-19 19:52:16.199 | DEBUG    | conftest:mesh_device:436 - multidevice with 1 devices is created
2026-07-19 19:52:16.200 | critical |          Always | Tensix cores {[(x=4,y=4) - (x=4,y=4)]} specified in sub device must be within device grid [(x=0,y=0) - (x=4,y=3)] (assert.hpp:103)
FAILED2026-07-19 19:52:16.207 | DEBUG    | ttnn:manage_config:88 - Restored ttnn.CONFIG.report_name to None

tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program_mesh[silicon_arch_name=blackhole] 2026-07-19 19:52:16.208 | DEBUG    | ttnn:manage_config:85 - Set ttnn.CONFIG.report_name to tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program_mesh[silicon_arch_name=blackhole]: 2026-07-19 19:52:16 (UTC)
2026-07-19 19:52:16.208 | DEBUG    | ttnn.device:__init__:150 - Using default dispatch core type for this system: DispatchCoreType.WORKER
2026-07-19 19:52:16.208 | DEBUG    | ttnn.device:__init__:152 - Using default dispatch core axis for this system: DispatchCoreAxis.COL
2026-07-19 19:52:16.554 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
2026-07-19 19:52:16.560 | DEBUG    | conftest:mesh_device:436 - multidevice with 1 devices is created
2026-07-19 19:52:16.560 | critical |          Always | Tensix cores {[(x=4,y=4) - (x=4,y=4)]} specified in sub device must be within device grid [(x=0,y=0) - (x=4,y=3)] (assert.hpp:103)
FAILED2026-07-19 19:52:16.567 | DEBUG    | ttnn:manage_config:88 - Restored ttnn.CONFIG.report_name to None

tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program 2026-07-19 19:52:16.568 | DEBUG    | ttnn:manage_config:85 - Set ttnn.CONFIG.report_name to tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program: 2026-07-19 19:52:16 (UTC)
2026-07-19 19:52:16.568 | DEBUG    | ttnn.device:__init__:150 - Using default dispatch core type for this system: DispatchCoreType.WORKER
2026-07-19 19:52:16.568 | DEBUG    | ttnn.device:__init__:152 - Using default dispatch core axis for this system: DispatchCoreAxis.COL
2026-07-19 19:52:16.889 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
2026-07-19 19:52:16.896 | critical |          Always | Tensix cores {[(x=4,y=4) - (x=4,y=4)]} specified in sub device must be within device grid [(x=0,y=0) - (x=4,y=3)] (assert.hpp:103)
FAILED2026-07-19 19:52:16.903 | DEBUG    | ttnn:manage_config:88 - Restored ttnn.CONFIG.report_name to None


======================================================================== FAILURES ========================================================================
____________________________________________________________________ test_sub_devices ____________________________________________________________________

device = MeshDevice(1x1 grid, 1 devices)

    def test_sub_devices(device):
>       run_sub_devices(device)

tests/ttnn/unit_tests/test_sub_device.py:131: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

device = MeshDevice(1x1 grid, 1 devices)

    def run_sub_devices(device):
        tensix_cores0 = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(3, 3),
                ),
            }
        )
        tensix_cores1 = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(4, 4),
                    ttnn.CoreCoord(4, 4),
                ),
            }
        )
        sub_device_1 = ttnn.SubDevice([tensix_cores0])
        sub_device_2 = ttnn.SubDevice([tensix_cores1])
        sub_devices_1 = [sub_device_1, sub_device_2]
        sub_devices_2 = [sub_device_2]
>       sub_device_manager1 = device.create_sub_device_manager(sub_devices_1, 3200)
E       RuntimeError: TT_FATAL @ /tt-metal/tt_metal/impl/sub_device/sub_device_manager.cpp:205: device_worker_cores.contains(worker_cores)
E       info:
E       Tensix cores {[(x=4,y=4) - (x=4,y=4)]} specified in sub device must be within device grid [(x=0,y=0) - (x=4,y=3)]
E       backtrace:
E        --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xfe11a8) [0x74ff421381a8]
E        --- tt::tt_metal::SubDeviceManager::validate_sub_devices() const
E        --- tt::tt_metal::SubDeviceManager::SubDeviceManager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long, tt::tt_metal::IDevice*)
E        --- tt::tt_metal::SubDeviceManagerTracker::create_sub_device_manager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long)
E        --- tt::tt_metal::distributed::MeshDevice::create_sub_device_manager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long)
E        --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xf4f852) [0x74ff4357d852]
E        --- /tt-metal/ttnn/ttnn/_ttnn.so(+0x586b5e) [0x74ff42bb4b5e]
E        --- /tt-metal/python_env/bin/python3(+0x18ae32) [0x598f24493e32]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(+0x1989cb) [0x598f244a19cb]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(PyObject_Call+0x122) [0x598f244a2392]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x807) [0x598f2447ed17]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(PyObject_Call+0xbb) [0x598f244a232b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x1992) [0x598f2447fea2]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x1992) [0x598f2447fea2]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(+0x259d36) [0x598f24562d36]

tests/ttnn/unit_tests/test_sub_device.py:31: RuntimeError
___________________________________________________ test_sub_devices_mesh[silicon_arch_name=blackhole] ___________________________________________________

mesh_device = MeshDevice(1x1 grid, 1 devices)

    def test_sub_devices_mesh(mesh_device):
>       run_sub_devices(mesh_device)

tests/ttnn/unit_tests/test_sub_device.py:135: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

device = MeshDevice(1x1 grid, 1 devices)

    def run_sub_devices(device):
        tensix_cores0 = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(3, 3),
                ),
            }
        )
        tensix_cores1 = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(4, 4),
                    ttnn.CoreCoord(4, 4),
                ),
            }
        )
        sub_device_1 = ttnn.SubDevice([tensix_cores0])
        sub_device_2 = ttnn.SubDevice([tensix_cores1])
        sub_devices_1 = [sub_device_1, sub_device_2]
        sub_devices_2 = [sub_device_2]
>       sub_device_manager1 = device.create_sub_device_manager(sub_devices_1, 3200)
E       RuntimeError: TT_FATAL @ /tt-metal/tt_metal/impl/sub_device/sub_device_manager.cpp:205: device_worker_cores.contains(worker_cores)
E       info:
E       Tensix cores {[(x=4,y=4) - (x=4,y=4)]} specified in sub device must be within device grid [(x=0,y=0) - (x=4,y=3)]
E       backtrace:
E        --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xfe11a8) [0x74ff421381a8]
E        --- tt::tt_metal::SubDeviceManager::validate_sub_devices() const
E        --- tt::tt_metal::SubDeviceManager::SubDeviceManager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long, tt::tt_metal::IDevice*)
E        --- tt::tt_metal::SubDeviceManagerTracker::create_sub_device_manager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long)
E        --- tt::tt_metal::distributed::MeshDevice::create_sub_device_manager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long)
E        --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xf4f852) [0x74ff4357d852]
E        --- /tt-metal/ttnn/ttnn/_ttnn.so(+0x586b5e) [0x74ff42bb4b5e]
E        --- /tt-metal/python_env/bin/python3(+0x18ae32) [0x598f24493e32]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(+0x1989cb) [0x598f244a19cb]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(PyObject_Call+0x122) [0x598f244a2392]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x807) [0x598f2447ed17]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(PyObject_Call+0xbb) [0x598f244a232b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x1992) [0x598f2447fea2]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x1992) [0x598f2447fea2]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(+0x259d36) [0x598f24562d36]

tests/ttnn/unit_tests/test_sub_device.py:31: RuntimeError
_______________________________________________ test_sub_device_program_mesh[silicon_arch_name=blackhole] ________________________________________________

mesh_device = MeshDevice(1x1 grid, 1 devices)

    def test_sub_device_program_mesh(mesh_device):
>       run_sub_devices_program(mesh_device)

tests/ttnn/unit_tests/test_sub_device.py:143: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

device = MeshDevice(1x1 grid, 1 devices)

    def run_sub_devices_program(device):
        is_mesh_device = isinstance(device, ttnn.MeshDevice)
        if is_mesh_device:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
            num_devices = device.get_num_devices()
        else:
            inputs_mesh_mapper = None
            output_mesh_composer = None
            num_devices = 1
        tensix_cores0 = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(4, 4),
                    ttnn.CoreCoord(4, 4),
                ),
            }
        )
        tensix_cores1 = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(3, 3),
                ),
            }
        )
        sub_device_1 = ttnn.SubDevice([tensix_cores0])
        sub_device_2 = ttnn.SubDevice([tensix_cores1])
        sub_devices = [sub_device_1, sub_device_2]
>       sub_device_manager = device.create_sub_device_manager(sub_devices, 3200)
E       RuntimeError: TT_FATAL @ /tt-metal/tt_metal/impl/sub_device/sub_device_manager.cpp:205: device_worker_cores.contains(worker_cores)
E       info:
E       Tensix cores {[(x=4,y=4) - (x=4,y=4)]} specified in sub device must be within device grid [(x=0,y=0) - (x=4,y=3)]
E       backtrace:
E        --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xfe11a8) [0x74ff421381a8]
E        --- tt::tt_metal::SubDeviceManager::validate_sub_devices() const
E        --- tt::tt_metal::SubDeviceManager::SubDeviceManager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long, tt::tt_metal::IDevice*)
E        --- tt::tt_metal::SubDeviceManagerTracker::create_sub_device_manager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long)
E        --- tt::tt_metal::distributed::MeshDevice::create_sub_device_manager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long)
E        --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xf4f852) [0x74ff4357d852]
E        --- /tt-metal/ttnn/ttnn/_ttnn.so(+0x586b5e) [0x74ff42bb4b5e]
E        --- /tt-metal/python_env/bin/python3(+0x18ae32) [0x598f24493e32]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(+0x1989cb) [0x598f244a19cb]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(PyObject_Call+0x122) [0x598f244a2392]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x807) [0x598f2447ed17]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(PyObject_Call+0xbb) [0x598f244a232b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x1992) [0x598f2447fea2]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x1992) [0x598f2447fea2]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(+0x259d36) [0x598f24562d36]

tests/ttnn/unit_tests/test_sub_device.py:76: RuntimeError
________________________________________________________________ test_sub_device_program _________________________________________________________________

device = MeshDevice(1x1 grid, 1 devices)

    def test_sub_device_program(device):
>       run_sub_devices_program(device)

tests/ttnn/unit_tests/test_sub_device.py:139: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

device = MeshDevice(1x1 grid, 1 devices)

    def run_sub_devices_program(device):
        is_mesh_device = isinstance(device, ttnn.MeshDevice)
        if is_mesh_device:
            inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
            num_devices = device.get_num_devices()
        else:
            inputs_mesh_mapper = None
            output_mesh_composer = None
            num_devices = 1
        tensix_cores0 = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(4, 4),
                    ttnn.CoreCoord(4, 4),
                ),
            }
        )
        tensix_cores1 = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(3, 3),
                ),
            }
        )
        sub_device_1 = ttnn.SubDevice([tensix_cores0])
        sub_device_2 = ttnn.SubDevice([tensix_cores1])
        sub_devices = [sub_device_1, sub_device_2]
>       sub_device_manager = device.create_sub_device_manager(sub_devices, 3200)
E       RuntimeError: TT_FATAL @ /tt-metal/tt_metal/impl/sub_device/sub_device_manager.cpp:205: device_worker_cores.contains(worker_cores)
E       info:
E       Tensix cores {[(x=4,y=4) - (x=4,y=4)]} specified in sub device must be within device grid [(x=0,y=0) - (x=4,y=3)]
E       backtrace:
E        --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xfe11a8) [0x74ff421381a8]
E        --- tt::tt_metal::SubDeviceManager::validate_sub_devices() const
E        --- tt::tt_metal::SubDeviceManager::SubDeviceManager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long, tt::tt_metal::IDevice*)
E        --- tt::tt_metal::SubDeviceManagerTracker::create_sub_device_manager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long)
E        --- tt::tt_metal::distributed::MeshDevice::create_sub_device_manager(std::span<tt::tt_metal::SubDevice const, 18446744073709551615ul>, unsigned long)
E        --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xf4f852) [0x74ff4357d852]
E        --- /tt-metal/ttnn/ttnn/_ttnn.so(+0x586b5e) [0x74ff42bb4b5e]
E        --- /tt-metal/python_env/bin/python3(+0x18ae32) [0x598f24493e32]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(+0x1989cb) [0x598f244a19cb]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(PyObject_Call+0x122) [0x598f244a2392]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x807) [0x598f2447ed17]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(PyObject_Call+0xbb) [0x598f244a232b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x1992) [0x598f2447fea2]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x1992) [0x598f2447fea2]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x286b) [0x598f24480d7b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(+0x1986f1) [0x598f244a16f1]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x56cb) [0x598f24483bdb]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyObject_FastCallDictTstate+0x16d) [0x598f24489b0d]
E        --- /tt-metal/python_env/bin/python3(_PyObject_Call_Prepend+0x5c) [0x598f2449e84c]
E        --- /tt-metal/python_env/bin/python3(+0x29cde4) [0x598f245a5de4]
E        --- /tt-metal/python_env/bin/python3(_PyObject_MakeTpCall+0x25b) [0x598f2448a89b]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6a5b) [0x598f24484f6b]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(_PyFunction_Vectorcall+0x7c) [0x598f2449468c]
E        --- /tt-metal/python_env/bin/python3(_PyEval_EvalFrameDefault+0x6c0) [0x598f2447ebd0]
E        --- /tt-metal/python_env/bin/python3(+0x259d36) [0x598f24562d36]

tests/ttnn/unit_tests/test_sub_device.py:76: RuntimeError
--------------------------------------- generated xml file: /tt-metal/generated/test_reports/most_recent_tests.xml ---------------------------------------
=================================================================== slowest durations ====================================================================
4.22s setup    tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices
0.35s setup    tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program_mesh[silicon_arch_name=blackhole]
0.34s setup    tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices_mesh[silicon_arch_name=blackhole]
0.33s setup    tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program
0.00s call     tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices
0.00s call     tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program_mesh[silicon_arch_name=blackhole]
0.00s call     tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program
0.00s call     tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices_mesh[silicon_arch_name=blackhole]
0.00s teardown tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program
0.00s teardown tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices
0.00s teardown tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices_mesh[silicon_arch_name=blackhole]
0.00s teardown tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program_mesh[silicon_arch_name=blackhole]
================================================================ short test summary info =================================================================
FAILED tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices - RuntimeError: TT_FATAL @ /tt-metal/tt_metal/impl/sub_device/sub_device_manager.cpp:205: device_worker_cores.contains(worker_cores)
FAILED tests/ttnn/unit_tests/test_sub_device.py::test_sub_devices_mesh[silicon_arch_name=blackhole] - RuntimeError: TT_FATAL @ /tt-metal/tt_metal/impl/sub_device/sub_device_manager.cpp:205: device_worker_cores.contains(worker_cores)
FAILED tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program_mesh[silicon_arch_name=blackhole] - RuntimeError: TT_FATAL @ /tt-metal/tt_metal/impl/sub_device/sub_device_manager.cpp:205: device_worker_cores.contains(worker_cores)
FAILED tests/ttnn/unit_tests/test_sub_device.py::test_sub_device_program - RuntimeError: TT_FATAL @ /tt-metal/tt_metal/impl/sub_device/sub_device_manager.cpp:205: device_worker_cores.contains(worker_cores)
=================================================================== 4 failed in 5.38s ====================================================================
2026-07-19 19:52:17.335 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:427)
2026-07-19 19:52:17.351 | info     |             UMD | Unpinning pages for DMA: virtual address 0x74fea8a00000 and size 0x40000000 (pci_device.cpp:644) 

026-07-19 20:52:08.399 | DEBUG    | ttnn:<module>:77 - Initial ttnn.CONFIG:
Config{cache_path=/root/.cache/ttnn,model_cache_path=/root/.cache/ttnn/models,tmp_dir=/tmp/ttnn,enable_model_cache=false,enable_fast_runtime_mode=true,throw_exception_on_fallback=false,enable_logging=false,enable_graph_report=false,enable_detailed_buffer_report=false,enable_detailed_tensor_report=false,enable_comparison_mode=false,comparison_mode_should_raise_exception=false,comparison_mode_pcc=0.9999,root_report_path=generated/ttnn/reports,report_name=std::nullopt,std::nullopt}
2026-07-19 20:52:09.527 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:52:09.561 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:52:09.561 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:52:09.562 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:202)
2026-07-19 20:52:09.585 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:52:09.619 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:52:09.619 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:52:09.653 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:52:09.684 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:52:09.686 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:52:09.687 | info     |             UMD | Harvesting mask for chip 0 is 0x0 (NOC0: 0x0, simulated harvesting mask: 0x0). (cluster.cpp:399)
2026-07-19 20:52:09.724 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:52:09.724 | info     |             UMD | Initializing iommu for sysmem (size: 0x40000000). (sysmem_manager.cpp:300)
2026-07-19 20:52:09.724 | info     |             UMD | Allocating sysmem without hugepages (size: 0x40000000). (sysmem_manager.cpp:306)
2026-07-19 20:52:09.898 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:52:09.898 | info     |             UMD | Opening local chip ids/PCIe ids: {0}/[0] and remote chip ids {} (cluster.cpp:251)
2026-07-19 20:52:09.898 | info     |             UMD | IOMMU: enabled (cluster.cpp:173)
2026-07-19 20:52:09.898 | info     |             UMD | KMD version: 2.3.0 (cluster.cpp:176)
2026-07-19 20:52:09.899 | info     |          Device | start_device in llrt (tt_cluster.cpp:415)
2026-07-19 20:52:09.922 | info     |             UMD | Pinning pages for DMA: virtual address 0x78f449000000 and size 0x40000000 pinned to physical address 0x7fc0000000 without mapping to noc (pci_device.cpp:618)
2026-07-19 20:52:09.922 | info     |             UMD | Mapped sysmem without hugepages to IOVA 0x7fc0000000; NOC address 0x0 (sysmem_manager.cpp:354)
2026-07-19 20:52:09.966 | info     |             UMD | Initializing memory barriers (local_chip.cpp:171)
2026-07-19 20:52:11.447 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
[baseline    ]    38.30 μs/iter  (1000 iters, add 32x32 tile)
2026-07-19 20:52:14.496 | critical |          Always | Programs must be executed on a single sub-device (assert.hpp:103)
Traceback (most recent call last):
  File "/tt-metal/measure_subdev_overhead.py", line 43, in <module>
    t_on = timed_workload(tag="SubDev ON   ")
  File "/tt-metal/measure_subdev_overhead.py", line 14, in timed_workload
    _ = ttnn.add(at, bt)
  File "/tt-metal/ttnn/ttnn/decorators.py", line 376, in __call__
    result = self.function(*function_args, **function_kwargs)
RuntimeError: TT_FATAL @ /tt-metal/tt_metal/distributed/fd_mesh_command_queue.cpp:222: sub_device_ids.size() == 1
info:
Programs must be executed on a single sub-device
backtrace:
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0x69a5a9) [0x78f51b6e35a9]
 --- tt::tt_metal::distributed::FDMeshCommandQueue::enqueue_mesh_workload(tt::tt_metal::distributed::MeshWorkload&, bool)
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail21enqueue_mesh_workloadINS0_26MeshDeviceOperationAdapterINS_10operations9binary_ng23BinaryNgDeviceOperationEEEEEvRKNT_22operation_attributes_tERKNS8_13tensor_args_tERNS8_21tensor_return_value_tEPN2tt8tt_metal11distributed10MeshDeviceERNSJ_12MeshWorkloadE+0xf1) [0x78f51a741341]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29handle_mesh_adapter_cache_hitINS0_26MeshDeviceOperationAdapterINS_10operations9binary_ng23BinaryNgDeviceOperationEEEEEvRKNT_22operation_attributes_tERKNS8_13tensor_args_tERNS8_21tensor_return_value_tEPN2tt8tt_metal11distributed10MeshDeviceERNSI_13program_cache6detail12ProgramCacheEm+0x19e) [0x78f51a73d01e]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterINS_10operations9binary_ng23BinaryNgDeviceOperationEEEEEvRKNT_22operation_attributes_tERKNS8_13tensor_args_tERNS8_21tensor_return_value_tEPN2tt8tt_metal11distributed10MeshDeviceE+0x1d9) [0x78f51a73c6a9]
 --- ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::launch_on_device<ttnn::operations::binary_ng::BinaryNgDeviceOperation>(ttnn::operations::binary_ng::BinaryNgDeviceOperation::operation_attributes_t const&, ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_args_t const&)
 --- ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::invoke<ttnn::operations::binary_ng::BinaryNgDeviceOperation>(ttnn::operations::binary_ng::BinaryNgDeviceOperation::operation_attributes_t const&, ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_args_t const&)
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xbc9aa8) [0x78f51a73baa8]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xbc967c) [0x78f51a73b67c]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xba914f) [0x78f51a71b14f]
 --- ttnn::operations::binary::BinaryOperation<(ttnn::operations::binary::BinaryOpType)0>::invoke(tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, std::optional<tt::tt_metal::DataType const> const&, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<tt::tt_metal::Tensor> const&, std::span<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int> const, 18446744073709551615ul>, std::span<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int> const, 18446744073709551615ul>, std::span<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int> const, 18446744073709551615ul>, std::optional<bool> const&)
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4c0cb) [0x78f51bd950cb]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4bd3d) [0x78f51bd94d3d]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4baf1) [0x78f51bd94af1]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4b5fa) [0x78f51bd945fa]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4b594) [0x78f51bd94594]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0x586b5e) [0x78f51b5cfb5e]
 --- python(+0x18ae32) [0x625fde4fee32]
 --- python(_PyObject_MakeTpCall+0x25b) [0x625fde4f589b]
 --- python(+0x198b20) [0x625fde50cb20]
 --- python(+0x29ce0f) [0x625fde610e0f]
 --- python(PyObject_Call+0xbb) [0x625fde50d32b]
 --- python(_PyEval_EvalFrameDefault+0x286b) [0x625fde4ebd7b]
 --- python(_PyObject_FastCallDictTstate+0xc4) [0x625fde4f4a64]
 --- python(_PyObject_Call_Prepend+0x5c) [0x625fde50984c]
 --- python(+0x29cde4) [0x625fde610de4]
 --- python(_PyObject_MakeTpCall+0x25b) [0x625fde4f589b]
 --- python(_PyEval_EvalFrameDefault+0x60be) [0x625fde4ef5ce]
 --- python(_PyFunction_Vectorcall+0x7c) [0x625fde4ff68c]
 --- python(_PyEval_EvalFrameDefault+0x1992) [0x625fde4eaea2]
 --- python(+0x259d36) [0x625fde5cdd36]
 --- python(PyEval_EvalCode+0x86) [0x625fde5cdc06]
 --- python(+0x280678) [0x625fde5f4678]
 --- python(+0x27accf) [0x625fde5eeccf]
 --- python(+0x280415) [0x625fde5f4415]
 --- python(_PyRun_SimpleFileObject+0x1a8) [0x625fde5f3958]
 --- python(_PyRun_AnyFileObject+0x47) [0x625fde5f3637]
 --- python(Py_RunMain+0x2be) [0x625fde5e7bfe]
 --- python(Py_BytesMain+0x2d) [0x625fde5c1c4d]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x29d90) [0x78f51cfc2d90]
 --- /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80) [0x78f51cfc2e40]
 --- python(_start+0x25) [0x625fde5c1b45]

2026-07-19 20:52:14.833 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:427)
2026-07-19 20:52:14.847 | info     |             UMD | Unpinning pages for DMA: virtual address 0x78f449000000 and size 0x40000000 (pci_device.cpp:644)
(python_env) root@7575fef6ac31:/tt-metal# 

2026-07-19 20:54:53.527 | DEBUG    | ttnn:<module>:77 - Initial ttnn.CONFIG:
Config{cache_path=/root/.cache/ttnn,model_cache_path=/root/.cache/ttnn/models,tmp_dir=/tmp/ttnn,enable_model_cache=false,enable_fast_runtime_mode=true,throw_exception_on_fallback=false,enable_logging=false,enable_graph_report=false,enable_detailed_buffer_report=false,enable_detailed_tensor_report=false,enable_comparison_mode=false,comparison_mode_should_raise_exception=false,comparison_mode_pcc=0.9999,root_report_path=generated/ttnn/reports,report_name=std::nullopt,std::nullopt}
2026-07-19 20:54:54.671 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:54:54.705 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:54:54.705 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:54:54.706 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:202)
2026-07-19 20:54:54.729 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:54:54.763 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:54:54.763 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:54:54.797 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:54:54.828 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:54:54.830 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:54:54.831 | info     |             UMD | Harvesting mask for chip 0 is 0x0 (NOC0: 0x0, simulated harvesting mask: 0x0). (cluster.cpp:399)
2026-07-19 20:54:54.868 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:54:54.868 | info     |             UMD | Initializing iommu for sysmem (size: 0x40000000). (sysmem_manager.cpp:300)
2026-07-19 20:54:54.868 | info     |             UMD | Allocating sysmem without hugepages (size: 0x40000000). (sysmem_manager.cpp:306)
2026-07-19 20:54:55.044 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:54:55.044 | info     |             UMD | Opening local chip ids/PCIe ids: {0}/[0] and remote chip ids {} (cluster.cpp:251)
2026-07-19 20:54:55.044 | info     |             UMD | IOMMU: enabled (cluster.cpp:173)
2026-07-19 20:54:55.044 | info     |             UMD | KMD version: 2.3.0 (cluster.cpp:176)
2026-07-19 20:54:55.045 | info     |          Device | start_device in llrt (tt_cluster.cpp:415)
2026-07-19 20:54:55.072 | info     |             UMD | Pinning pages for DMA: virtual address 0x74555e200000 and size 0x40000000 pinned to physical address 0x7fc0000000 without mapping to noc (pci_device.cpp:618)
2026-07-19 20:54:55.072 | info     |             UMD | Mapped sysmem without hugepages to IOVA 0x7fc0000000; NOC address 0x0 (sysmem_manager.cpp:354)
2026-07-19 20:54:55.116 | info     |             UMD | Initializing memory barriers (local_chip.cpp:171)
2026-07-19 20:54:56.530 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
[baseline     ]    38.01 us/iter
2026-07-19 20:54:59.581 | critical |          Always | Kernel group cores do not match sub device cores for programmable core type TENSIX (assert.hpp:103)
Traceback (most recent call last):
  File "/tt-metal/measure_subdev_overhead.py", line 36, in <module>
    t_on = timed_workload(tag="16-core only ")
  File "/tt-metal/measure_subdev_overhead.py", line 13, in timed_workload
    _ = ttnn.add(at, bt)
  File "/tt-metal/ttnn/ttnn/decorators.py", line 376, in __call__
    result = self.function(*function_args, **function_kwargs)
RuntimeError: TT_FATAL @ /tt-metal/tt_metal/impl/program/program.cpp:1267: num_intersections == num_cores
info:
Kernel group cores do not match sub device cores for programmable core type TENSIX
backtrace:
 --- void tt::assert::detail::tt_throw_impl<fmt::v11::fstring<std::basic_string_view<char, std::char_traits<char> > const&>, std::basic_string_view<char, std::char_traits<char> > >(char const*, int, char const*, char const*, fmt::v11::fstring<std::basic_string_view<char, std::char_traits<char> > const&> const&, std::basic_string_view<char, std::char_traits<char> > const&)
 --- /tt-metal/build_Release_tracy/lib/libtt_metal.so(+0x517b4c) [0x74562e6e8b4c]
 --- tt::tt_metal::detail::ProgramImpl::determine_sub_device_ids(tt::tt_metal::IDevice const*)
 --- tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region(tt::tt_metal::IDevice const*)
 --- tt::tt_metal::distributed::MeshWorkloadImpl::compile_program(tt::tt_metal::distributed::MeshCoordinateRange const&, tt::tt_metal::distributed::MeshDevice*)
 --- tt::tt_metal::distributed::MeshWorkloadImpl::compile(tt::tt_metal::distributed::MeshDevice*)
 --- tt::tt_metal::distributed::EnqueueMeshWorkload(tt::tt_metal::distributed::MeshCommandQueue&, tt::tt_metal::distributed::MeshWorkload&, bool)
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail21enqueue_mesh_workloadINS0_26MeshDeviceOperationAdapterINS_10operations9binary_ng23BinaryNgDeviceOperationEEEEEvRKNT_22operation_attributes_tERKNS8_13tensor_args_tERNS8_21tensor_return_value_tEPN2tt8tt_metal11distributed10MeshDeviceERNSJ_12MeshWorkloadE+0xf1) [0x74562f8d5341]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29handle_mesh_adapter_cache_hitINS0_26MeshDeviceOperationAdapterINS_10operations9binary_ng23BinaryNgDeviceOperationEEEEEvRKNT_22operation_attributes_tERKNS8_13tensor_args_tERNS8_21tensor_return_value_tEPN2tt8tt_metal11distributed10MeshDeviceERNSI_13program_cache6detail12ProgramCacheEm+0x19e) [0x74562f8d101e]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterINS_10operations9binary_ng23BinaryNgDeviceOperationEEEEEvRKNT_22operation_attributes_tERKNS8_13tensor_args_tERNS8_21tensor_return_value_tEPN2tt8tt_metal11distributed10MeshDeviceE+0x1d9) [0x74562f8d06a9]
 --- ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::launch_on_device<ttnn::operations::binary_ng::BinaryNgDeviceOperation>(ttnn::operations::binary_ng::BinaryNgDeviceOperation::operation_attributes_t const&, ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_args_t const&)
 --- ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::invoke<ttnn::operations::binary_ng::BinaryNgDeviceOperation>(ttnn::operations::binary_ng::BinaryNgDeviceOperation::operation_attributes_t const&, ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_args_t const&)
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xbc9aa8) [0x74562f8cfaa8]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xbc967c) [0x74562f8cf67c]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xba914f) [0x74562f8af14f]
 --- ttnn::operations::binary::BinaryOperation<(ttnn::operations::binary::BinaryOpType)0>::invoke(tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, std::optional<tt::tt_metal::DataType const> const&, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<tt::tt_metal::Tensor> const&, std::span<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int> const, 18446744073709551615ul>, std::span<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int> const, 18446744073709551615ul>, std::span<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int> const, 18446744073709551615ul>, std::optional<bool> const&)
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4c0cb) [0x745630f290cb]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4bd3d) [0x745630f28d3d]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4baf1) [0x745630f28af1]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4b5fa) [0x745630f285fa]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4b594) [0x745630f28594]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0x586b5e) [0x745630763b5e]
 --- python(+0x18ae32) [0x5d277459de32]
 --- python(_PyObject_MakeTpCall+0x25b) [0x5d277459489b]
 --- python(+0x198b20) [0x5d27745abb20]
 --- python(+0x29ce0f) [0x5d27746afe0f]
 --- python(PyObject_Call+0xbb) [0x5d27745ac32b]
 --- python(_PyEval_EvalFrameDefault+0x286b) [0x5d277458ad7b]
 --- python(_PyObject_FastCallDictTstate+0xc4) [0x5d2774593a64]
 --- python(_PyObject_Call_Prepend+0x5c) [0x5d27745a884c]
 --- python(+0x29cde4) [0x5d27746afde4]
 --- python(_PyObject_MakeTpCall+0x25b) [0x5d277459489b]
 --- python(_PyEval_EvalFrameDefault+0x60be) [0x5d277458e5ce]
 --- python(_PyFunction_Vectorcall+0x7c) [0x5d277459e68c]
 --- python(_PyEval_EvalFrameDefault+0x1992) [0x5d2774589ea2]
 --- python(+0x259d36) [0x5d277466cd36]
 --- python(PyEval_EvalCode+0x86) [0x5d277466cc06]
 --- python(+0x280678) [0x5d2774693678]
 --- python(+0x27accf) [0x5d277468dccf]
 --- python(+0x280415) [0x5d2774693415]
 --- python(_PyRun_SimpleFileObject+0x1a8) [0x5d2774692958]
 --- python(_PyRun_AnyFileObject+0x47) [0x5d2774692637]
 --- python(Py_RunMain+0x2be) [0x5d2774686bfe]
 --- python(Py_BytesMain+0x2d) [0x5d2774660c4d]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x29d90) [0x745632156d90]
 --- /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80) [0x745632156e40]
 --- python(_start+0x25) [0x5d2774660b45]

2026-07-19 20:54:59.920 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:427)
2026-07-19 20:54:59.934 | info     |             UMD | Unpinning pages for DMA: virtual address 0x74555e200000 and size 0x40000000 (pci_device.cpp:644)
(python_env) root@7575fef6ac31:/tt-metal# 

026-07-19 20:58:47.650 | DEBUG    | ttnn:<module>:77 - Initial ttnn.CONFIG:
Config{cache_path=/root/.cache/ttnn,model_cache_path=/root/.cache/ttnn/models,tmp_dir=/tmp/ttnn,enable_model_cache=false,enable_fast_runtime_mode=true,throw_exception_on_fallback=false,enable_logging=false,enable_graph_report=false,enable_detailed_buffer_report=false,enable_detailed_tensor_report=false,enable_comparison_mode=false,comparison_mode_should_raise_exception=false,comparison_mode_pcc=0.9999,root_report_path=generated/ttnn/reports,report_name=std::nullopt,std::nullopt}
2026-07-19 20:58:48.776 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:58:48.810 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:58:48.810 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:58:48.811 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:202)
2026-07-19 20:58:48.844 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:58:48.874 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:58:48.874 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:58:48.897 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:58:48.915 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:58:48.917 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:58:48.918 | info     |             UMD | Harvesting mask for chip 0 is 0x0 (NOC0: 0x0, simulated harvesting mask: 0x0). (cluster.cpp:399)
2026-07-19 20:58:48.944 | info     |             UMD | Initializing TLB mappings in BlackholeTTDevice (blackhole_tt_device.cpp:37)
2026-07-19 20:58:48.944 | info     |             UMD | Initializing iommu for sysmem (size: 0x40000000). (sysmem_manager.cpp:300)
2026-07-19 20:58:48.944 | info     |             UMD | Allocating sysmem without hugepages (size: 0x40000000). (sysmem_manager.cpp:306)
2026-07-19 20:58:49.118 | info     |             UMD | Initializing TLB manager (local_chip.cpp:126)
2026-07-19 20:58:49.118 | info     |             UMD | Opening local chip ids/PCIe ids: {0}/[0] and remote chip ids {} (cluster.cpp:251)
2026-07-19 20:58:49.119 | info     |             UMD | IOMMU: enabled (cluster.cpp:173)
2026-07-19 20:58:49.119 | info     |             UMD | KMD version: 2.3.0 (cluster.cpp:176)
2026-07-19 20:58:49.119 | info     |          Device | start_device in llrt (tt_cluster.cpp:415)
2026-07-19 20:58:49.146 | info     |             UMD | Pinning pages for DMA: virtual address 0x7237f4800000 and size 0x40000000 pinned to physical address 0x7fc0000000 without mapping to noc (pci_device.cpp:618)
2026-07-19 20:58:49.146 | info     |             UMD | Mapped sysmem without hugepages to IOVA 0x7fc0000000; NOC address 0x0 (sysmem_manager.cpp:354)
2026-07-19 20:58:49.190 | info     |             UMD | Initializing memory barriers (local_chip.cpp:171)
2026-07-19 20:58:50.596 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
[baseline    ]    37.58 us/iter
2026-07-19 20:58:53.629 | critical |          Always | Kernel group cores do not match sub device cores for programmable core type TENSIX (assert.hpp:103)
Traceback (most recent call last):
  File "/tt-metal/measure_subdev_overhead.py", line 35, in <module>
    t_on = timed_workload(tag="4-core SD   ")
  File "/tt-metal/measure_subdev_overhead.py", line 13, in timed_workload
    _ = ttnn.add(at, bt)
  File "/tt-metal/ttnn/ttnn/decorators.py", line 376, in __call__
    result = self.function(*function_args, **function_kwargs)
RuntimeError: TT_FATAL @ /tt-metal/tt_metal/impl/program/program.cpp:1267: num_intersections == num_cores
info:
Kernel group cores do not match sub device cores for programmable core type TENSIX
backtrace:
 --- void tt::assert::detail::tt_throw_impl<fmt::v11::fstring<std::basic_string_view<char, std::char_traits<char> > const&>, std::basic_string_view<char, std::char_traits<char> > >(char const*, int, char const*, char const*, fmt::v11::fstring<std::basic_string_view<char, std::char_traits<char> > const&> const&, std::basic_string_view<char, std::char_traits<char> > const&)
 --- /tt-metal/build_Release_tracy/lib/libtt_metal.so(+0x517b4c) [0x7238c4c2ab4c]
 --- tt::tt_metal::detail::ProgramImpl::determine_sub_device_ids(tt::tt_metal::IDevice const*)
 --- tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region(tt::tt_metal::IDevice const*)
 --- tt::tt_metal::distributed::MeshWorkloadImpl::compile_program(tt::tt_metal::distributed::MeshCoordinateRange const&, tt::tt_metal::distributed::MeshDevice*)
 --- tt::tt_metal::distributed::MeshWorkloadImpl::compile(tt::tt_metal::distributed::MeshDevice*)
 --- tt::tt_metal::distributed::EnqueueMeshWorkload(tt::tt_metal::distributed::MeshCommandQueue&, tt::tt_metal::distributed::MeshWorkload&, bool)
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail21enqueue_mesh_workloadINS0_26MeshDeviceOperationAdapterINS_10operations9binary_ng23BinaryNgDeviceOperationEEEEEvRKNT_22operation_attributes_tERKNS8_13tensor_args_tERNS8_21tensor_return_value_tEPN2tt8tt_metal11distributed10MeshDeviceERNSJ_12MeshWorkloadE+0xf1) [0x7238c5e17341]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29handle_mesh_adapter_cache_hitINS0_26MeshDeviceOperationAdapterINS_10operations9binary_ng23BinaryNgDeviceOperationEEEEEvRKNT_22operation_attributes_tERKNS8_13tensor_args_tERNS8_21tensor_return_value_tEPN2tt8tt_metal11distributed10MeshDeviceERNSI_13program_cache6detail12ProgramCacheEm+0x19e) [0x7238c5e1301e]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterINS_10operations9binary_ng23BinaryNgDeviceOperationEEEEEvRKNT_22operation_attributes_tERKNS8_13tensor_args_tERNS8_21tensor_return_value_tEPN2tt8tt_metal11distributed10MeshDeviceE+0x1d9) [0x7238c5e126a9]
 --- ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::launch_on_device<ttnn::operations::binary_ng::BinaryNgDeviceOperation>(ttnn::operations::binary_ng::BinaryNgDeviceOperation::operation_attributes_t const&, ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_args_t const&)
 --- ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::invoke<ttnn::operations::binary_ng::BinaryNgDeviceOperation>(ttnn::operations::binary_ng::BinaryNgDeviceOperation::operation_attributes_t const&, ttnn::operations::binary_ng::BinaryNgDeviceOperation::tensor_args_t const&)
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xbc9aa8) [0x7238c5e11aa8]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xbc967c) [0x7238c5e1167c]
 --- /tt-metal/build_Release_tracy/lib/_ttnncpp.so(+0xba914f) [0x7238c5df114f]
 --- ttnn::operations::binary::BinaryOperation<(ttnn::operations::binary::BinaryOpType)0>::invoke(tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, std::optional<tt::tt_metal::DataType const> const&, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<tt::tt_metal::Tensor> const&, std::span<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int> const, 18446744073709551615ul>, std::span<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int> const, 18446744073709551615ul>, std::span<ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int> const, 18446744073709551615ul>, std::optional<bool> const&)
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4c0cb) [0x7238c746b0cb]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4bd3d) [0x7238c746ad3d]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4baf1) [0x7238c746aaf1]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4b5fa) [0x7238c746a5fa]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0xd4b594) [0x7238c746a594]
 --- /tt-metal/ttnn/ttnn/_ttnn.so(+0x586b5e) [0x7238c6ca5b5e]
 --- python(+0x18ae32) [0x5d595d825e32]
 --- python(_PyObject_MakeTpCall+0x25b) [0x5d595d81c89b]
 --- python(+0x198b20) [0x5d595d833b20]
 --- python(+0x29ce0f) [0x5d595d937e0f]
 --- python(PyObject_Call+0xbb) [0x5d595d83432b]
 --- python(_PyEval_EvalFrameDefault+0x286b) [0x5d595d812d7b]
 --- python(_PyObject_FastCallDictTstate+0xc4) [0x5d595d81ba64]
 --- python(_PyObject_Call_Prepend+0x5c) [0x5d595d83084c]
 --- python(+0x29cde4) [0x5d595d937de4]
 --- python(_PyObject_MakeTpCall+0x25b) [0x5d595d81c89b]
 --- python(_PyEval_EvalFrameDefault+0x60be) [0x5d595d8165ce]
 --- python(_PyFunction_Vectorcall+0x7c) [0x5d595d82668c]
 --- python(_PyEval_EvalFrameDefault+0x1992) [0x5d595d811ea2]
 --- python(+0x259d36) [0x5d595d8f4d36]
 --- python(PyEval_EvalCode+0x86) [0x5d595d8f4c06]
 --- python(+0x280678) [0x5d595d91b678]
 --- python(+0x27accf) [0x5d595d915ccf]
 --- python(+0x280415) [0x5d595d91b415]
 --- python(_PyRun_SimpleFileObject+0x1a8) [0x5d595d91a958]
 --- python(_PyRun_AnyFileObject+0x47) [0x5d595d91a637]
 --- python(Py_RunMain+0x2be) [0x5d595d90ebfe]
 --- python(Py_BytesMain+0x2d) [0x5d595d8e8c4d]
 --- /lib/x86_64-linux-gnu/libc.so.6(+0x29d90) [0x7238c8698d90]
 --- /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80) [0x7238c8698e40]
 --- python(_start+0x25) [0x5d595d8e8b45]

2026-07-19 20:58:53.967 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:427)
2026-07-19 20:58:53.981 | info     |             UMD | Unpinning pages for DMA: virtual address 0x7237f4800000 and size 0x40000000 (pci_device.cpp:644)
(python_env) root@7575fef6ac31:/tt-metal# 