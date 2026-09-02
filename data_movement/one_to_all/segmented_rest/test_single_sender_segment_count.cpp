// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "dm_common.hpp"
#include "one_to_all/test_one_to_all.hpp"
#include "one_to_all/segmented_rest/segmented_rest_planner.hpp"

namespace tt::tt_metal::unit_tests::dm::core_to_all::segmented_rest {

namespace single_sender_segment_count {

constexpr CoreCoord kPhysicalGrid{5, 4};
constexpr CoreCoord kLogicalGrid{3, 3};
constexpr uint32_t kPayloadBytes = 8160;
constexpr uint32_t kSourceL1Addr = 200 * 1024;
constexpr uint32_t kDestinationL1Addr = 300 * 1024;
constexpr uint32_t kReceiverStatusL1Addr = 350 * 1024;
constexpr uint32_t kIterations = 16;
constexpr uint32_t kGoldenWord = 0x13579BDF;
constexpr uint32_t kSentinelWord = 0xDEADC0DE;

CoreRangeSet sender_set(const Plan& plan) {
    return CoreRangeSet(std::set<CoreRange>{CoreRange(plan.sender)});
}

}  // namespace single_sender_segment_count

// Transport-only control for a single multicast sender. It holds the 3x3
// logical workload, payload, sender, and destination count constant while
// varying only the physical rectangle decomposition: 1, 2, or 4 segments.
namespace single_sender_segment_count {

TEST_F(GenericMeshDeviceFixture, TensixDataMovementSingleSenderSegmentCount) {
    struct Config {
        const char* name;
        std::optional<uint32_t> excluded_row;
        std::optional<uint32_t> excluded_column;
    };
    const std::vector<Config> configs = {
        {"single_sender_3x3_1_segment", std::nullopt, std::nullopt},
        {"single_sender_3x3_2_segment_row_gap", 2, std::nullopt},
        {"single_sender_3x3_2_segment_col_gap", std::nullopt, 2},
        {"single_sender_3x3_4_segment_cross_gap", 2, 2},
    };

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);
    const auto device_grid = device->compute_with_storage_grid_size();
    ASSERT_GE(device_grid.x, kPhysicalGrid.x);
    ASSERT_GE(device_grid.y, kPhysicalGrid.y);

    const CoreRangeSet full_cores(
        std::set<CoreRange>{CoreRange({0, 0}, {kPhysicalGrid.x - 1, kPhysicalGrid.y - 1})});
    const auto full_core_list = corerange_to_cores(full_cores);
    std::vector<uint32_t> golden(kPayloadBytes / sizeof(uint32_t), kGoldenWord);
    std::vector<uint32_t> sentinel(kPayloadBytes / sizeof(uint32_t), kSentinelWord);
    std::vector<uint32_t> receiver_status_not_ready{0};
    std::vector<uint32_t> receiver_status_pass{1};

    for (const auto& cfg : configs) {
        const auto replacement = make_core_replacement_plan(
            kPhysicalGrid, kLogicalGrid, cfg.excluded_row, cfg.excluded_column);
        const auto plan = make_single_sender_mcast_plan(replacement);
        const auto sender_cores = sender_set(plan);
        const auto receiver_cores = replacement.active_cores.subtract(sender_cores);
        ASSERT_EQ(plan.destination_count, receiver_cores.num_cores());

        bool data_pass = true;
        bool inactive_sentinel = true;
        bool receiver_order_pass = true;
        for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
            Program program = CreateProgram();
            const uint32_t semaphore_id = CreateSemaphore(program, replacement.active_cores, 0);
            const std::vector<uint32_t> sender_compile_args = {
                kSourceL1Addr,
                kDestinationL1Addr,
                kPayloadBytes,
                static_cast<uint32_t>(plan.segments.size()),
                106,
                plan.destination_count};
            const auto sender_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/segmented_multicast_sender.cpp",
                sender_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::NOC_0,
                    .compile_args = sender_compile_args});
            std::vector<uint32_t> sender_runtime_args = {semaphore_id};
            for (const auto& segment : plan.segments) {
                const auto start = device->worker_core_from_logical_core(segment.logical_range.start_coord);
                const auto end = device->worker_core_from_logical_core(segment.logical_range.end_coord);
                sender_runtime_args.insert(
                    sender_runtime_args.end(), {start.x, start.y, end.x, end.y, segment.destination_count});
            }
            SetRuntimeArgs(program, sender_kernel, sender_cores, sender_runtime_args);

            const auto receiver_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/segmented_multicast_receiver_check.cpp",
                receiver_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::NOC_1,
                    .compile_args = {
                        kDestinationL1Addr,
                        kPayloadBytes,
                        kGoldenWord,
                        kReceiverStatusL1Addr,
                        106}});
            SetRuntimeArgs(program, receiver_kernel, receiver_cores, {semaphore_id});

            for (const auto& core : full_core_list) {
                detail::WriteToDeviceL1(device, core, kDestinationL1Addr, sentinel);
                detail::WriteToDeviceL1(device, core, kReceiverStatusL1Addr, receiver_status_not_ready);
            }
            detail::WriteToDeviceL1(device, plan.sender, kSourceL1Addr, golden);
            MetalContext::instance().get_cluster().l1_barrier(device->id());

            program.set_runtime_id(unit_tests::dm::runtime_host_id++);
            distributed::MeshWorkload workload;
            const auto mesh_zero = distributed::MeshCoordinate(0, 0);
            workload.add_program(distributed::MeshCoordinateRange(mesh_zero, mesh_zero), std::move(program));
            auto& cq = mesh_device->mesh_command_queue();
            distributed::EnqueueMeshWorkload(cq, workload, false);
            Finish(cq);

            std::vector<uint32_t> readback;
            for (const auto& core : full_core_list) {
                detail::ReadFromDeviceL1(device, core, kDestinationL1Addr, kPayloadBytes, readback);
                if (receiver_cores.contains(core)) {
                    data_pass &= (readback == golden);
                } else {
                    inactive_sentinel &= (readback == sentinel);
                }
            }
            detail::ReadFromDeviceL1(device, plan.sender, kSourceL1Addr, kPayloadBytes, readback);
            data_pass &= (readback == golden);
            for (const auto& core : corerange_to_cores(receiver_cores)) {
                detail::ReadFromDeviceL1(device, core, kReceiverStatusL1Addr, sizeof(uint32_t), readback);
                receiver_order_pass &= (readback == receiver_status_pass);
            }
        }

        log_info(
            tt::LogTest,
            "SINGLE_SENDER_SEGMENT_COUNT config={} logical_grid=3x3 active_cores={} sender=({},{}) "
            "segments={} destinations={} iterations={} data={} inactive_sentinel={} receiver_order={}",
            cfg.name,
            replacement.active_cores.num_cores(),
            plan.sender.x,
            plan.sender.y,
            plan.segments.size(),
            plan.destination_count,
            kIterations,
            data_pass ? "PASS" : "FAIL",
            inactive_sentinel ? "PASS" : "FAIL",
            receiver_order_pass ? "PASS" : "FAIL");
        EXPECT_TRUE(data_pass) << cfg.name << ": active receiver or source readback mismatch";
        EXPECT_TRUE(inactive_sentinel) << cfg.name << ": sender or inactive core received a multicast write";
        EXPECT_TRUE(receiver_order_pass) << cfg.name << ": receiver observed stale data after semaphore wakeup";
    }
}

}  // namespace single_sender_segment_count

}  // namespace tt::tt_metal::unit_tests::dm::core_to_all::segmented_rest
