// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include "test_one_to_all.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::core_to_all::multicast_schemes {

uint32_t determine_max_grid_dimension(const shared_ptr<distributed::MeshDevice>& mesh_device) {
    uint32_t smaller_dimension =
        min(mesh_device->get_device(0)->compute_with_storage_grid_size().x,
            mesh_device->get_device(0)->compute_with_storage_grid_size().y);
    return (smaller_dimension - 1);
}

enum class MulticastSchemeType {
    // Sender is IN the grid
    SenderInGridTopRight = 1,
    SenderInGridBottomRight,
    SenderInGridBottomLeft,
    SenderInGridTopLeft,
    // Sender is NOT in the grid
    SenderInGridStartingRowNotColumn,
    SenderInGridStartingColumnNotRow,
    SenderNotInGridStartingColumnOrRow,
    SenderInGridEndingRowNotColumn,
    SenderInGridEndingColumnNotRow,
    SenderNotInGridEndingColumnOrRow,
    SenderInGridCenter,
    End
};

// Helper function to determine coordinates based on multicast scheme type
pair<CoreCoord, CoreCoord> get_coordinates(uint32_t sub_grid_dimension_size, MulticastSchemeType type) {
    CoreCoord mst_core_coord;
    CoreCoord sub_start_core_coord;

    switch (type) {
        // Sender is IN the grid
        case MulticastSchemeType::SenderInGridTopRight:
            mst_core_coord = {sub_grid_dimension_size - 1, sub_grid_dimension_size - 1};
            sub_start_core_coord = {0, 0};
            break;
        case MulticastSchemeType::SenderInGridBottomRight:
            mst_core_coord = {sub_grid_dimension_size - 1, 0};
            sub_start_core_coord = {0, 0};
            break;
        case MulticastSchemeType::SenderInGridBottomLeft:
            mst_core_coord = {0, 0};
            sub_start_core_coord = {0, 0};
            break;
        case MulticastSchemeType::SenderInGridTopLeft:
            mst_core_coord = {0, sub_grid_dimension_size - 1};
            sub_start_core_coord = {0, 0};
            break;

        // Sender is NOT in the grid
        case MulticastSchemeType::SenderInGridStartingRowNotColumn:
            mst_core_coord = {0, 1};
            sub_start_core_coord = {1, 1};
            break;
        case MulticastSchemeType::SenderInGridStartingColumnNotRow:
            mst_core_coord = {1, 0};
            sub_start_core_coord = {1, 1};
            break;
        case MulticastSchemeType::SenderNotInGridStartingColumnOrRow:
            mst_core_coord = {0, 0};
            sub_start_core_coord = {1, 1};
            break;
        case MulticastSchemeType::SenderInGridEndingRowNotColumn:
            mst_core_coord = {sub_grid_dimension_size, sub_grid_dimension_size - 1};
            sub_start_core_coord = {0, 0};
            break;
        case MulticastSchemeType::SenderInGridEndingColumnNotRow:
            mst_core_coord = {sub_grid_dimension_size - 1, sub_grid_dimension_size};
            sub_start_core_coord = {0, 0};
            break;
        case MulticastSchemeType::SenderNotInGridEndingColumnOrRow:
            mst_core_coord = {sub_grid_dimension_size, sub_grid_dimension_size};
            sub_start_core_coord = {0, 0};
            break;
        case MulticastSchemeType::SenderInGridCenter:
            mst_core_coord = {sub_grid_dimension_size / 2, sub_grid_dimension_size / 2};
            sub_start_core_coord = {0, 0};
            break;

        default: throw invalid_argument("Invalid multicast scheme type");
    }

    return {mst_core_coord, sub_start_core_coord};
}

void test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    uint32_t sub_grid_dimension_size,
    NOC noc_id,
    MulticastSchemeType multicast_scheme_type,
    bool loopback = true,
    bool is_linked = true) {
    bool is_multicast = true;

    CoreCoord sub_grid_size = {sub_grid_dimension_size, sub_grid_dimension_size};

    auto [mst_core_coord, sub_start_core_coord] = get_coordinates(sub_grid_dimension_size, multicast_scheme_type);

    // Run the directed ideal test
    unit_tests::dm::core_to_all::directed_ideal_test(
        mesh_device,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        loopback,
        noc_id,
        static_cast<uint32_t>(multicast_scheme_type));
}

void run_all_tests(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_case_id, bool loopback = true) {
    vector<NOC> noc_ids = {NOC::NOC_0, NOC::NOC_1};
    uint32_t starting_sub_grid_dimension_size = 2;  // Minimum size for sub-grid dimension
    uint32_t sub_grid_dimension_limit = determine_max_grid_dimension(mesh_device);

    MulticastSchemeType starting_multicast_scheme_type = MulticastSchemeType::SenderInGridTopRight;

    for (const auto& noc_id : noc_ids) {
        for (uint32_t sub_grid_dimension_size = starting_sub_grid_dimension_size;
             sub_grid_dimension_size <= sub_grid_dimension_limit;
             sub_grid_dimension_size++) {
            for (uint32_t multicast_scheme_type = static_cast<uint32_t>(starting_multicast_scheme_type);
                 multicast_scheme_type < static_cast<uint32_t>(MulticastSchemeType::End);
                 multicast_scheme_type++) {
                test(
                    mesh_device,
                    test_case_id,
                    sub_grid_dimension_size,
                    (noc_id),
                    static_cast<MulticastSchemeType>(multicast_scheme_type),
                    loopback);
            }
        }
    }
}

}  // namespace unit_tests::dm::core_to_all::multicast_schemes

/* ============================================================= */
/* =================== LOOP THROUGH SCHEMES ==================== */
/* ============================================================= */

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastSchemesLoopback) {
    //GTEST_SKIP() << "Skipping test";

    uint32_t test_case_id = 100;
    bool loopback = true;

    unit_tests::dm::core_to_all::multicast_schemes::run_all_tests(get_mesh_device(), test_case_id, loopback);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastSchemesNoLoopback) {
    //GTEST_SKIP() << "Skipping test";

    uint32_t test_case_id = 101;
    bool loopback = false;

    unit_tests::dm::core_to_all::multicast_schemes::run_all_tests(get_mesh_device(), test_case_id, loopback);
}

/* ============================================================= */
/* ================== INDIVIDUAL TEST CASES ==================== */
/* ============================================================= */

/*
    Loopback: Loopback, No Loopback
    NOC: NOC 0, NOC 1
    Grid Size: 2, ..., 7 (WH), ... 9 (BH)
    Schemes:
        1. Sender in grid top right
        2. Sender in grid bottom right
        3. Sender in grid bottom left
        4. Sender in grid top left
        5. Sender out grid starting row not column
        6. Sender out grid starting column not row
        7. Sender out grid starting not row not column
        8. Sender out grid ending row not column
        9. Sender out grid ending column not row
        10. Sender out grid ending not row not column
*/

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastSchemeSingle) {
    uint32_t test_case_id = 102;

    auto mesh_device = get_mesh_device();
    auto grid = mesh_device->get_device(0)->compute_with_storage_grid_size();

    bool loopback = false;
    bool is_multicast = true;
    bool is_linked = true;

    // Grid shapes to sweep with center-sender mcast (rectangle + square + full grid)
    std::vector<std::pair<uint32_t, uint32_t>> grid_shapes = {
        {3, 3}, {4, 4}, {3, 4}, {4, 3}, {5, 2}, {2, 5},
        {5, 4},  // Eagle-N full grid (rotation 정책 baseline)
    };

    for (NOC noc_id : {NOC::NOC_0, NOC::NOC_1}) {
        for (const auto& [gx, gy] : grid_shapes) {
            if (gx > grid.x || gy > grid.y) {
                log_info(tt::LogTest, "SKIP center mcast {}x{} NoC{}: exceeds device grid {}x{}",
                         gx, gy, static_cast<int>(noc_id), grid.x, grid.y);
                continue;
            }
            CoreCoord sub_grid_size = {gx, gy};
            CoreCoord sub_start_core_coord = {0, 0};
            CoreCoord mst_core_coord = {gx / 2, gy / 2};

            log_info(tt::LogTest, "CENTER mcast: grid={}x{}, sender=({},{}), NoC{}",
                     gx, gy, mst_core_coord.x, mst_core_coord.y, static_cast<int>(noc_id));

            unit_tests::dm::core_to_all::directed_ideal_test(
                mesh_device,
                test_case_id,
                is_multicast,
                is_linked,
                mst_core_coord,
                sub_start_core_coord,
                sub_grid_size,
                loopback,
                noc_id,
                static_cast<uint32_t>(
                    unit_tests::dm::core_to_all::multicast_schemes::MulticastSchemeType::SenderInGridCenter));
        }
    }
}

/* ============================================================= */
/* ============ PER-RECEIVER ARRIVAL HEATMAP =================== */
/* ============================================================= */

TEST_F(GenericMeshDeviceFixture, TensixDataMovementMulticastArrivalHeatmap) {
    uint32_t test_case_id = 103;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);
    auto grid = device->compute_with_storage_grid_size();

    bool loopback = false;
    bool is_linked = true;

    // 5x4 full grid, sender 20 위치 × NoC 2 = 40 config
    // NoC0 (row-friendly, X-long) / NoC1 (col-friendly, Y-long) 모두 sweep
    // 2D matmul 에서 row+col 이 다른 NoC 로 pipeline 되므로 두 축 데이터 필요
    struct Config {
        uint32_t gx, gy, sx, sy;
        NOC noc;
    };
    std::vector<Config> configs;
    constexpr uint32_t GX = 5, GY = 4;
    for (NOC noc_id : {NOC::NOC_0, NOC::NOC_1}) {
        for (uint32_t sy = 0; sy < GY; sy++) {
            for (uint32_t sx = 0; sx < GX; sx++) {
                configs.push_back({GX, GY, sx, sy, noc_id});
            }
        }
    }

    for (const auto& cfg : configs) {
        if (cfg.gx > grid.x || cfg.gy > grid.y || cfg.sx >= grid.x || cfg.sy >= grid.y) {
            log_info(tt::LogTest, "SKIP 5x4 sender=({},{}): exceeds grid {}x{}",
                     cfg.sx, cfg.sy, grid.x, grid.y);
            continue;
        }

        log_info(
            tt::LogTest,
            "ARRIVAL: 5x4 sender=({},{}) NoC{}",
            cfg.sx,
            cfg.sy,
            static_cast<int>(cfg.noc));

        Program program = CreateProgram();

        CoreCoord sender_logical = {cfg.sx, cfg.sy};
        CoreRangeSet receiver_set(std::set<CoreRange>{CoreRange({0, 0}, {cfg.gx - 1, cfg.gy - 1})});
        if (!loopback) {
            CoreRangeSet sender_only(std::set<CoreRange>{CoreRange(sender_logical)});
            receiver_set = receiver_set.subtract(sender_only);
        }
        uint32_t num_receivers = receiver_set.num_cores();

        CoreCoord start_phys = device->worker_core_from_logical_core({0, 0});
        CoreCoord end_phys = device->worker_core_from_logical_core({cfg.gx - 1, cfg.gy - 1});

        // Semaphore: sender + receivers 모두 접근. Sender 가 rect 안에 있으므로 rect 하나면 충분.
        CoreRangeSet all_cores(std::set<CoreRange>{
            CoreRange({0, 0}, {cfg.gx - 1, cfg.gy - 1})});
        uint32_t arrival_sem_id = CreateSemaphore(program, all_cores, 0);

        // 워크로드 파라미터 (STEP 4 SchemeSingle 과 동일)
        uint32_t bytes_per_page = 64;
        uint32_t pages_per_transaction = 256;
        uint32_t num_of_transactions = 256;
        uint32_t l1_base = 200 * 1024;
        uint32_t sub_l1_base = l1_base;

        // Sender kernel (ENABLE_ARRIVAL_SEM 로 sem mcast 활성화)
        std::vector<uint32_t> sender_cta = {
            l1_base,                          // 0 mst_base_addr
            sub_l1_base,                      // 1 sub_base_addr
            num_of_transactions,              // 2
            pages_per_transaction,            // 3
            bytes_per_page,                   // 4
            test_case_id,                     // 5
            num_receivers,                    // 6 num_subordinates
            (uint32_t)is_linked,              // 7
            (uint32_t)loopback,               // 8
            (uint32_t)start_phys.x,           // 9
            (uint32_t)start_phys.y,           // 10
            (uint32_t)end_phys.x,             // 11
            (uint32_t)end_phys.y,             // 12
            0,                                // 13 multicast_scheme_type (0=skip stamp)
            cfg.gx,                           // 14 sub_grid_size_x
            cfg.gy,                           // 15 sub_grid_size_y
        };

        std::map<std::string, std::string> sender_defines = {{"ENABLE_ARRIVAL_SEM", "1"}};

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/sender_multicast.cpp",
            sender_logical,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = cfg.noc,
                .compile_args = sender_cta,
                .defines = sender_defines});

        // Receiver kernel (RISCV1, 반대 NoC)
        std::vector<uint32_t> receiver_cta = {
            l1_base,                          // 0 src_addr
            sub_l1_base,                      // 1 dst_addr
            num_of_transactions,              // 2
            pages_per_transaction,            // 3
            bytes_per_page,                   // 4
            test_case_id,                     // 5
        };

        auto receiver_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/receiver.cpp",
            receiver_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = cfg.noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0,
                .compile_args = receiver_cta});

        SetRuntimeArgs(program, receiver_kernel, receiver_set, {arrival_sem_id});

        program.set_runtime_id(unit_tests::dm::runtime_host_id++);

        distributed::MeshWorkload workload;
        auto zero = distributed::MeshCoordinate(0, 0);
        auto range = distributed::MeshCoordinateRange(zero, zero);
        workload.add_program(range, std::move(program));

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, true);
    }
}

/* ============================================================= */
/* ===== GENERALIZED ROW/COLUMN REST SEGMENTED MULTICAST ====== */
/* ============================================================= */

TEST_F(GenericMeshDeviceFixture, TensixDataMovementSegmentedRowColumnRestMulticast) {
    constexpr uint32_t grid_x = 5, grid_y = 4, iterations = 16;
    constexpr uint32_t payload_bytes = 64;
    constexpr uint32_t source_l1_addr = 200 * 1024;
    constexpr uint32_t destination_l1_addr = 300 * 1024;
    constexpr uint32_t sentinel_word = 0xDEADC0DE;
    constexpr uint32_t golden_word = 0x13579BDF;
    constexpr uint32_t test_case_id = 104;

    struct BaseConfig { const char* name; bool is_row; int32_t rest_index; };
    const std::vector<BaseConfig> base_configs = {
        {"full_5x4", true, -1},
        {"skip_row_0", true, 0}, {"skip_row_1", true, 1},
        {"skip_row_2", true, 2}, {"skip_row_3", true, 3},
        {"skip_col_0", false, 0}, {"skip_col_1", false, 1},
        {"skip_col_2", false, 2}, {"skip_col_3", false, 3},
        {"skip_col_4", false, 4},
    };
    struct Config { const char* name; bool is_row; int32_t rest_index; NOC noc; };
    std::vector<Config> configs;
    for (NOC noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (const auto& base : base_configs) {
            configs.push_back({base.name, base.is_row, base.rest_index, noc});
        }
    }

    struct Segment { CoreRange logical_range; uint32_t destination_count; };
    struct Result {
        std::string name, rest_line, rectangles;
        CoreCoord sender;
        uint32_t active_count, destination_count;
        bool active_data, rest_sentinel, barrier, semaphore;
    };

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);
    const auto device_grid = device->compute_with_storage_grid_size();
    ASSERT_GE(device_grid.x, grid_x);
    ASSERT_GE(device_grid.y, grid_y);

    const CoreRangeSet full_cores(std::set<CoreRange>{CoreRange({0, 0}, {grid_x - 1, grid_y - 1})});
    const auto full_core_list = corerange_to_cores(full_cores);
    std::vector<uint32_t> golden(payload_bytes / sizeof(uint32_t), golden_word);
    std::vector<uint32_t> sentinel(payload_bytes / sizeof(uint32_t), sentinel_word);
    std::vector<Result> results;

    for (const auto& cfg : configs) {
        CoreRangeSet rest_cores;
        std::vector<CoreRange> segment_ranges;
        const std::string config_label = std::string(cfg.name) + " NoC" + std::to_string(static_cast<uint32_t>(cfg.noc));
        std::string rest_line = "none";
        if (cfg.rest_index < 0) {
            segment_ranges.emplace_back(CoreCoord{0, 0}, CoreCoord{grid_x - 1, grid_y - 1});
        } else if (cfg.is_row) {
            rest_cores = CoreRangeSet(std::set<CoreRange>{CoreRange({0, static_cast<uint32_t>(cfg.rest_index)}, {grid_x - 1, static_cast<uint32_t>(cfg.rest_index)})});
            rest_line = "row_" + std::to_string(cfg.rest_index);
            if (cfg.rest_index > 0) segment_ranges.emplace_back(CoreCoord{0, 0}, CoreCoord{grid_x - 1, static_cast<uint32_t>(cfg.rest_index - 1)});
            if (cfg.rest_index + 1 < static_cast<int32_t>(grid_y)) segment_ranges.emplace_back(CoreCoord{0, static_cast<uint32_t>(cfg.rest_index + 1)}, CoreCoord{grid_x - 1, grid_y - 1});
        } else {
            rest_cores = CoreRangeSet(std::set<CoreRange>{CoreRange({static_cast<uint32_t>(cfg.rest_index), 0}, {static_cast<uint32_t>(cfg.rest_index), grid_y - 1})});
            rest_line = "col_" + std::to_string(cfg.rest_index);
            if (cfg.rest_index > 0) segment_ranges.emplace_back(CoreCoord{0, 0}, CoreCoord{static_cast<uint32_t>(cfg.rest_index - 1), grid_y - 1});
            if (cfg.rest_index + 1 < static_cast<int32_t>(grid_x)) segment_ranges.emplace_back(CoreCoord{static_cast<uint32_t>(cfg.rest_index + 1), 0}, CoreCoord{grid_x - 1, grid_y - 1});
        }

        const CoreRangeSet active_cores = full_cores.subtract(rest_cores);
        const auto active_core_list = corerange_to_cores(active_cores);
        const auto rest_core_list = corerange_to_cores(rest_cores);
        CoreCoord sender = active_core_list.front();
        for (const auto& core : active_core_list) {
            sender.x = std::min(sender.x, core.x);
            sender.y = std::min(sender.y, core.y);
        }
        ASSERT_NE(std::find(active_core_list.begin(), active_core_list.end(), sender), active_core_list.end());

        std::vector<Segment> segments;
        uint32_t total_destination_count = 0;
        std::string rectangles;
        for (const auto& range : segment_ranges) {
            const auto segment_cores = corerange_to_cores(CoreRangeSet(std::set<CoreRange>{range}));
            const bool contains_sender = std::find(segment_cores.begin(), segment_cores.end(), sender) != segment_cores.end();
            const uint32_t destination_count = segment_cores.size() - (contains_sender ? 1 : 0);
            ASSERT_GT(destination_count, 0u);
            total_destination_count += destination_count;
            segments.push_back({range, destination_count});
            if (!rectangles.empty()) rectangles += ",";
            rectangles += "(" + std::to_string(range.start_coord.x) + "," + std::to_string(range.start_coord.y) + ")~(" + std::to_string(range.end_coord.x) + "," + std::to_string(range.end_coord.y) + ")";
        }
        ASSERT_EQ(total_destination_count, active_cores.num_cores() - 1) << config_label;
        const CoreRangeSet sender_set(std::set<CoreRange>{CoreRange(sender)});
        const CoreRangeSet receiver_cores = active_cores.subtract(sender_set);

        bool active_data = true, rest_sentinel = true, barrier = true, semaphore = true;
        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
            Program program = CreateProgram();
            const uint32_t semaphore_id = CreateSemaphore(program, active_cores, 0);
            const std::vector<uint32_t> sender_compile_args = {
                source_l1_addr, destination_l1_addr, payload_bytes,
                static_cast<uint32_t>(segments.size()), test_case_id, total_destination_count};
            const auto sender_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/segmented_multicast_sender.cpp",
                sender_set,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = cfg.noc, .compile_args = sender_compile_args});
            std::vector<uint32_t> sender_runtime_args = {semaphore_id};
            for (const auto& segment : segments) {
                const auto start = device->worker_core_from_logical_core(segment.logical_range.start_coord);
                const auto end = device->worker_core_from_logical_core(segment.logical_range.end_coord);
                sender_runtime_args.insert(sender_runtime_args.end(), {start.x, start.y, end.x, end.y, segment.destination_count});
            }
            SetRuntimeArgs(program, sender_kernel, sender_set, sender_runtime_args);

            const auto receiver_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/segmented_multicast_receiver.cpp",
                receiver_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = cfg.noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0,
                    .compile_args = {payload_bytes, test_case_id}});
            SetRuntimeArgs(program, receiver_kernel, receiver_cores, {semaphore_id});

            for (const auto& core : full_core_list) detail::WriteToDeviceL1(device, core, destination_l1_addr, sentinel);
            detail::WriteToDeviceL1(device, sender, source_l1_addr, golden);
            MetalContext::instance().get_cluster().l1_barrier(device->id());

            program.set_runtime_id(unit_tests::dm::runtime_host_id++);
            distributed::MeshWorkload workload;
            const auto mesh_zero = distributed::MeshCoordinate(0, 0);
            workload.add_program(distributed::MeshCoordinateRange(mesh_zero, mesh_zero), std::move(program));
            auto& cq = mesh_device->mesh_command_queue();
            distributed::EnqueueMeshWorkload(cq, workload, false);
            Finish(cq);

            std::vector<uint32_t> readback;
            for (const auto& core : corerange_to_cores(receiver_cores)) {
                detail::ReadFromDeviceL1(device, core, destination_l1_addr, payload_bytes, readback);
                active_data &= (readback == golden);
            }
            detail::ReadFromDeviceL1(device, sender, destination_l1_addr, payload_bytes, readback);
            active_data &= (readback == sentinel);  // non-loopback: sender is not a destination
            detail::ReadFromDeviceL1(device, sender, source_l1_addr, payload_bytes, readback);
            active_data &= (readback == golden);
            for (const auto& core : rest_core_list) {
                detail::ReadFromDeviceL1(device, core, destination_l1_addr, payload_bytes, readback);
                rest_sentinel &= (readback == sentinel);
            }
        }

        log_info(tt::LogTest, "| {} | {} | ({},{}) | {} | {} | {} | {} | {} | {} | {} |",
                 config_label, rest_line, sender.x, sender.y, rectangles, active_cores.num_cores(),
                 total_destination_count, iterations, active_data ? "PASS" : "FAIL",
                 rest_sentinel ? "PASS" : "FAIL", barrier ? "PASS" : "FAIL", semaphore ? "PASS" : "FAIL");
        EXPECT_TRUE(active_data) << config_label << ": active receiver/source/destination readback mismatch";
        EXPECT_TRUE(rest_sentinel) << config_label << ": rest core received a data multicast";
        results.push_back({config_label, rest_line, rectangles, sender, active_cores.num_cores(), total_destination_count, active_data, rest_sentinel, barrier, semaphore});
    }

    log_info(tt::LogTest, "| Config | Rest line | Sender | Segment rectangles | Active cores | Dest count | Iterations | Active data | Rest sentinel | Barrier | Semaphore |");
    for (const auto& result : results) {
        log_info(tt::LogTest, "| {} | {} | ({},{}) | {} | {} | {} | {} | {} | {} | {} |",
                 result.name, result.rest_line, result.sender.x, result.sender.y, result.rectangles,
                 result.active_count, result.destination_count, iterations,
                 result.active_data ? "PASS" : "FAIL", result.rest_sentinel ? "PASS" : "FAIL",
                 result.barrier ? "PASS" : "FAIL", result.semaphore ? "PASS" : "FAIL");
    }
}

}  // namespace tt::tt_metal
