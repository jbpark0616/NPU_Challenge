// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "dm_common.hpp"
#include "one_to_all/test_one_to_all.hpp"
#include "one_to_all/segmented_rest/segmented_rest_planner.hpp"

#include <numeric>

namespace tt::tt_metal::unit_tests::dm::core_to_all::segmented_rest {

namespace {

constexpr CoreCoord kPhysicalGrid{5, 4};
constexpr CoreCoord kLogicalGrid{3, 3};
constexpr uint32_t kPayloadBytes = 8160;  // representative Conv trace transaction size
constexpr uint32_t kWeightSource = 200 * 1024;
constexpr uint32_t kWeightDestination = 300 * 1024;
constexpr uint32_t kActivationSource = 400 * 1024;
constexpr uint32_t kActivationDestination = 500 * 1024;
constexpr uint32_t kIterations = 16;
constexpr uint32_t kGoldenWord = 0x13579BDF;
constexpr uint32_t kSentinelWord = 0xDEADC0DE;

CoreRangeSet sender_set(const LinePlan& line) {
    return CoreRangeSet(std::set<CoreRange>{CoreRange(line.sender)});
}

uint32_t issue_count(const AxisPlan& axis_plan) {
    return std::accumulate(
        axis_plan.lines.begin(), axis_plan.lines.end(), 0u,
        [](uint32_t total, const LinePlan& line) {
            return total + static_cast<uint32_t>(line.segments.size());
        });
}

void add_axis_senders(
    Program& program,
    IDevice* device,
    const AxisPlan& axis_plan,
    uint32_t source_l1_addr,
    uint32_t destination_l1_addr,
    uint32_t semaphore_id,
    NOC noc,
    DataMovementProcessor processor) {
    for (const auto& line : axis_plan.lines) {
        const std::vector<uint32_t> compile_args = {
            source_l1_addr,
            destination_l1_addr,
            kPayloadBytes,
            static_cast<uint32_t>(line.segments.size()),
            105,
            std::accumulate(
                line.segments.begin(), line.segments.end(), 0u,
                [](uint32_t sum, const Segment& segment) { return sum + segment.destination_count; }),
        };
        const auto kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/segmented_multicast_sender.cpp",
            sender_set(line),
            DataMovementConfig{
                .processor = processor,
                .noc = noc,
                .compile_args = compile_args});
        std::vector<uint32_t> runtime_args = {semaphore_id};
        for (const auto& segment : line.segments) {
            const auto start = device->worker_core_from_logical_core(segment.logical_range.start_coord);
            const auto end = device->worker_core_from_logical_core(segment.logical_range.end_coord);
            runtime_args.insert(
                runtime_args.end(), {start.x, start.y, end.x, end.y, segment.destination_count});
        }
        SetRuntimeArgs(program, kernel, sender_set(line), runtime_args);
    }
}

void initialize_axis_buffers(
    IDevice* device,
    const AxisPlan& axis_plan,
    uint32_t source_l1_addr,
    uint32_t destination_l1_addr,
    std::vector<uint32_t>& golden,
    std::vector<uint32_t>& sentinel) {
    const CoreRangeSet full_cores(
        std::set<CoreRange>{CoreRange({0, 0}, {kPhysicalGrid.x - 1, kPhysicalGrid.y - 1})});
    for (const auto& core : corerange_to_cores(full_cores)) {
        detail::WriteToDeviceL1(device, core, destination_l1_addr, sentinel);
    }
    for (const auto& line : axis_plan.lines) {
        detail::WriteToDeviceL1(device, line.sender, source_l1_addr, golden);
    }
}

bool check_axis_readback(
    IDevice* device,
    const AxisPlan& axis_plan,
    uint32_t source_l1_addr,
    uint32_t destination_l1_addr,
    const std::vector<uint32_t>& golden,
    const std::vector<uint32_t>& sentinel) {
    std::set<CoreCoord> senders;
    for (const auto& line : axis_plan.lines) {
        senders.insert(line.sender);
    }
    bool pass = true;
    std::vector<uint32_t> readback;
    const CoreRangeSet full_cores(
        std::set<CoreRange>{CoreRange({0, 0}, {kPhysicalGrid.x - 1, kPhysicalGrid.y - 1})});
    for (const auto& core : corerange_to_cores(full_cores)) {
        detail::ReadFromDeviceL1(device, core, destination_l1_addr, kPayloadBytes, readback);
        const bool is_active_receiver = axis_plan.active_cores.contains(core) && !senders.contains(core);
        pass &= (readback == (is_active_receiver ? golden : sentinel));
    }
    for (const auto& sender : senders) {
        detail::ReadFromDeviceL1(device, sender, source_l1_addr, kPayloadBytes, readback);
        pass &= (readback == golden);
    }
    return pass;
}

}  // namespace

// Conv-faithful microbenchmark: a fixed 3x3 logical workload is placed on a
// 5x4 physical grid. A middle physical row/column exclusion keeps the logical
// work constant while forcing the affected axis to issue two NoC segments.
TEST_F(GenericMeshDeviceFixture, TensixDataMovementConvAxisSegmentedMulticast) {
    struct Config {
        const char* name;
        std::optional<uint32_t> rest_row;
        std::optional<uint32_t> rest_column;
        bool run_weight;
        bool run_activation;
    };
    const std::vector<Config> configs = {
        {"weight_3x3_baseline", std::nullopt, std::nullopt, true, false},
        {"weight_3x3_row_gap", 2, std::nullopt, true, false},
        {"activation_3x3_baseline", std::nullopt, std::nullopt, false, true},
        {"activation_3x3_col_gap", std::nullopt, 2, false, true},
        {"dual_3x3_baseline", std::nullopt, std::nullopt, true, true},
        {"dual_3x3_row_col_gap", 2, 2, true, true},
    };

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);
    const auto device_grid = device->compute_with_storage_grid_size();
    ASSERT_GE(device_grid.x, kPhysicalGrid.x);
    ASSERT_GE(device_grid.y, kPhysicalGrid.y);
    std::vector<uint32_t> golden(kPayloadBytes / sizeof(uint32_t), kGoldenWord);
    std::vector<uint32_t> sentinel(kPayloadBytes / sizeof(uint32_t), kSentinelWord);

    for (const auto& cfg : configs) {
        const auto replacement = make_core_replacement_plan(
            kPhysicalGrid, kLogicalGrid, cfg.rest_row, cfg.rest_column);
        const auto weight_plan = make_conv_axis_plan(replacement, McastAxis::Column);
        const auto activation_plan = make_conv_axis_plan(replacement, McastAxis::Row);
        bool weight_pass = true;
        bool activation_pass = true;

        for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
            Program program = CreateProgram();
            const auto& active_cores = replacement.active_cores;
            const uint32_t weight_sem = CreateSemaphore(program, active_cores, 0);
            const uint32_t activation_sem = CreateSemaphore(program, active_cores, 0);
            if (cfg.run_weight) {
                add_axis_senders(
                    program, device, weight_plan, kWeightSource, kWeightDestination, weight_sem,
                    NOC::NOC_0, DataMovementProcessor::RISCV_0);
                initialize_axis_buffers(device, weight_plan, kWeightSource, kWeightDestination, golden, sentinel);
            }
            if (cfg.run_activation) {
                add_axis_senders(
                    program, device, activation_plan, kActivationSource, kActivationDestination, activation_sem,
                    NOC::NOC_1, DataMovementProcessor::RISCV_1);
                initialize_axis_buffers(device, activation_plan, kActivationSource, kActivationDestination, golden, sentinel);
            }
            MetalContext::instance().get_cluster().l1_barrier(device->id());

            program.set_runtime_id(unit_tests::dm::runtime_host_id++);
            distributed::MeshWorkload workload;
            const auto mesh_zero = distributed::MeshCoordinate(0, 0);
            workload.add_program(distributed::MeshCoordinateRange(mesh_zero, mesh_zero), std::move(program));
            auto& cq = mesh_device->mesh_command_queue();
            distributed::EnqueueMeshWorkload(cq, workload, false);
            Finish(cq);

            if (cfg.run_weight) {
                weight_pass &= check_axis_readback(
                    device, weight_plan, kWeightSource, kWeightDestination, golden, sentinel);
            }
            if (cfg.run_activation) {
                activation_pass &= check_axis_readback(
                    device, activation_plan, kActivationSource, kActivationDestination, golden, sentinel);
            }
        }

        log_info(
            tt::LogTest,
            "CONV_AXIS_MCAST config={} logical_grid=3x3 active_cores={} weight_lines={} weight_destinations={} activation_lines={} "
            "activation_destinations={} weight_issues={} activation_issues={} iterations={} weight={} activation={}",
            cfg.name, replacement.active_cores.num_cores(),
            weight_plan.lines.size(), weight_plan.total_destination_count,
            activation_plan.lines.size(), activation_plan.total_destination_count,
            issue_count(weight_plan), issue_count(activation_plan),
            kIterations,
            weight_pass ? "PASS" : "FAIL", activation_pass ? "PASS" : "FAIL");
        EXPECT_TRUE(weight_pass) << cfg.name << ": weight-axis data mismatch";
        EXPECT_TRUE(activation_pass) << cfg.name << ": activation-axis data mismatch";
    }
}

}  // namespace tt::tt_metal::unit_tests::dm::core_to_all::segmented_rest
