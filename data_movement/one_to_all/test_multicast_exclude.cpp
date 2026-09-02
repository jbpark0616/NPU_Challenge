// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"

#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "dm_common.hpp"

namespace tt::tt_metal {
namespace unit_tests::dm::core_to_all {
namespace {

constexpr uint32_t STATUS_MAGIC = 0x4558434Cu;  // "EXCL"
constexpr uint32_t SOURCE_WORD = 0x5A17C0DEu;
constexpr uint32_t SENTINEL_WORD = 0xC001CAFEu;
constexpr uint32_t PAYLOAD_BYTES = 64;
constexpr uint32_t SOURCE_L1_ADDR = 200 * 1024;
constexpr uint32_t DESTINATION_L1_ADDR = 201 * 1024;
constexpr uint32_t STATUS_L1_ADDR = 202 * 1024;
constexpr uint32_t READY_POLL_LIMIT = 10'000'000;
constexpr uint32_t DELIVERY_DELAY_CYCLES = 1'000'000;

bool exclude_microtests_enabled() {
    const char* value = std::getenv("RUN_NOC_EXCLUDE_MICROTESTS");
    return value != nullptr && std::string_view(value) == "1";
}

std::vector<CoreCoord> rectangle_cores(CoreCoord start, CoreCoord end) {
    std::vector<CoreCoord> cores;
    for (uint32_t y = start.y; y <= end.y; ++y) {
        for (uint32_t x = start.x; x <= end.x; ++x) {
            cores.push_back({x, y});
        }
    }
    return cores;
}

void run_posted_quadrant_exclude(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, NOC noc_id) {
    IDevice* device = mesh_device->get_device(0);
    ASSERT_EQ(device->arch(), ARCH::BLACKHOLE);

    const CoreCoord grid = device->compute_with_storage_grid_size();
    ASSERT_GE(grid.x, 4u);
    ASSERT_GE(grid.y, 4u);

    // Keep the sender outside the multicast rectangle. This isolates explicit
    // exclusion from the normal non-loopback "do not write self" behavior.
    const CoreCoord sender_logical{3, 3};
    const CoreCoord rectangle_start_logical{0, 0};
    const CoreCoord rectangle_end_logical{2, 2};
    const CoreCoord exclude_start_logical{1, 1};
    const CoreCoord exclude_direction{1, 1};
    const auto targets = rectangle_cores(rectangle_start_logical, rectangle_end_logical);

    const CoreCoord rectangle_start_physical =
        device->worker_core_from_logical_core(rectangle_start_logical);
    const CoreCoord rectangle_end_physical =
        device->worker_core_from_logical_core(rectangle_end_logical);
    const CoreCoord exclude_start_physical = device->worker_core_from_logical_core(exclude_start_logical);

    const auto is_excluded = [&](CoreCoord core) {
        const bool excluded_x = exclude_direction.x == 0 ? core.x <= exclude_start_logical.x
                                                         : core.x >= exclude_start_logical.x;
        const bool excluded_y = exclude_direction.y == 0 ? core.y <= exclude_start_logical.y
                                                         : core.y >= exclude_start_logical.y;
        return excluded_x && excluded_y;
    };
    uint32_t excluded_core_count = 0;
    for (const CoreCoord core : targets) {
        excluded_core_count += static_cast<uint32_t>(is_excluded(core));
    }

    std::vector<uint32_t> source(PAYLOAD_BYTES / sizeof(uint32_t), SOURCE_WORD);
    std::vector<uint32_t> sentinel(PAYLOAD_BYTES / sizeof(uint32_t), SENTINEL_WORD);
    for (const CoreCoord core : targets) {
        detail::WriteToDeviceL1(device, core, DESTINATION_L1_ADDR, sentinel);
    }
    detail::WriteToDeviceL1(device, sender_logical, SOURCE_L1_ADDR, source);
    std::vector<uint32_t> initial_status(7, 0);
    detail::WriteToDeviceL1(device, sender_logical, STATUS_L1_ADDR, initial_status);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    Program program = CreateProgram();
    const uint32_t actual_num_destinations = static_cast<uint32_t>(targets.size()) - excluded_core_count;
    const std::vector<uint32_t> compile_args = {
        SOURCE_L1_ADDR,
        DESTINATION_L1_ADDR,
        PAYLOAD_BYTES,
        static_cast<uint32_t>(rectangle_start_physical.x),
        static_cast<uint32_t>(rectangle_start_physical.y),
        static_cast<uint32_t>(rectangle_end_physical.x),
        static_cast<uint32_t>(rectangle_end_physical.y),
        static_cast<uint32_t>(exclude_start_physical.x),
        static_cast<uint32_t>(exclude_start_physical.y),
        static_cast<uint32_t>(exclude_direction.x),
        static_cast<uint32_t>(exclude_direction.y),
        actual_num_destinations,
        STATUS_L1_ADDR,
        READY_POLL_LIMIT,
        DELIVERY_DELAY_CYCLES,
    };

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/sender_multicast_exclude.cpp",
        CoreRangeSet({CoreRange(sender_logical)}),
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = noc_id,
            .compile_args = compile_args});

    auto workload = distributed::MeshWorkload();
    const auto device_range = distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0));
    workload.add_program(device_range, std::move(program));
    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    std::vector<uint32_t> status;
    detail::ReadFromDeviceL1(device, sender_logical, STATUS_L1_ADDR, 7 * sizeof(uint32_t), status);
    ASSERT_EQ(status.size(), 7u);
    ASSERT_EQ(status[0], STATUS_MAGIC);
    ASSERT_EQ(status[1], 1u) << "write command buffer was not initially ready";
    ASSERT_EQ(status[2], 1u) << "posted multicast command was not accepted before the bounded poll expired";
    ASSERT_LT(status[3], READY_POLL_LIMIT);
    ASSERT_LT(status[4], READY_POLL_LIMIT);
    ASSERT_EQ(status[6], 0u) << "NOC_BRCST_EXCLUDE was not cleared after the command completed";
    log_info(
        LogTest,
        "NoC{} exclude encoding=0x{:06x}, initial_polls={}, completion_polls={}",
        static_cast<uint32_t>(noc_id),
        status[5],
        status[3],
        status[4]);

    for (const CoreCoord core : targets) {
        std::vector<uint32_t> output;
        detail::ReadFromDeviceL1(device, core, DESTINATION_L1_ADDR, PAYLOAD_BYTES, output);
        if (is_excluded(core)) {
            ASSERT_EQ(output, sentinel)
                << "excluded quadrant core L1 was modified on NoC " << static_cast<uint32_t>(noc_id)
                << "; core=(" << core.x << "," << core.y << ")";
        } else {
            ASSERT_EQ(output, source)
                << "included core did not receive payload on NoC " << static_cast<uint32_t>(noc_id)
                << "; core=(" << core.x << "," << core.y << ")";
        }
    }
}

}  // namespace
}  // namespace unit_tests::dm::core_to_all

TEST_F(GenericMeshDeviceFixture, TensixDataMovementMulticastExcludePostedQuadrantNoc0) {
    if (!unit_tests::dm::core_to_all::exclude_microtests_enabled()) {
        GTEST_SKIP() << "Set RUN_NOC_EXCLUDE_MICROTESTS=1 and use the guarded runner on a clean device";
    }
    const auto mesh_device = get_mesh_device();
    if (mesh_device->get_device(0)->arch() != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "NOC_BRCST_EXCLUDE discovery test is Blackhole-only";
    }

    ASSERT_NO_FATAL_FAILURE(
        unit_tests::dm::core_to_all::run_posted_quadrant_exclude(mesh_device, NOC::NOC_0));
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementMulticastExcludePostedQuadrantNoc1) {
    if (!unit_tests::dm::core_to_all::exclude_microtests_enabled()) {
        GTEST_SKIP() << "Set RUN_NOC_EXCLUDE_MICROTESTS=1 and use the guarded runner on a clean device";
    }
    const auto mesh_device = get_mesh_device();
    if (mesh_device->get_device(0)->arch() != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "NOC_BRCST_EXCLUDE discovery test is Blackhole-only";
    }

    ASSERT_NO_FATAL_FAILURE(
        unit_tests::dm::core_to_all::run_posted_quadrant_exclude(mesh_device, NOC::NOC_1));
}

}  // namespace tt::tt_metal
