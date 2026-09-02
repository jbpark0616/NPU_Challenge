// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/conv/conv2d/conv_core_placement.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::conv {

enum class ConvCoreRemappingAction {
    NotActive,
    Propagate,
    ExitToStock,
};

struct ConvCoreRemappingPlan {
    ConvCoreRemappingAction action;
    sliding_window::ParallelConfig input_parallel_config;
    sliding_window::ParallelConfig output_parallel_config;

    bool requires_input_reshard() const { return action == ConvCoreRemappingAction::ExitToStock; }
};

struct ConvCorePlacementProvenance {
    sliding_window::ParallelConfig stock_input_parallel_config;
};

inline bool block_sharded_override_matches_remapped_logical_grid(
    const tt::tt_metal::CoreCoord& compute_grid_size,
    const tt::tt_metal::CoreRangeSet& input_grid,
    const tt::tt_metal::CoreRangeSet& override_grid) {
    const auto input_placement = make_core_placement_from_worker_grid(compute_grid_size, input_grid);
    const auto override_placement = make_core_placement_from_worker_grid(compute_grid_size, override_grid);
    return input_placement.shard_grid_size == override_placement.shard_grid_size;
}

inline tt::tt_metal::CoreRangeSet select_equivalent_remapped_block_execution_grid(
    const tt::tt_metal::CoreCoord& compute_grid_size,
    const tt::tt_metal::CoreRangeSet& input_grid,
    const tt::tt_metal::CoreRangeSet& override_grid,
    bool retain_equivalent_remapped_input) {
    if (retain_equivalent_remapped_input &&
        block_sharded_override_matches_remapped_logical_grid(compute_grid_size, input_grid, override_grid)) {
        return input_grid;
    }
    return override_grid;
}

inline std::vector<tt::tt_metal::CoreCoord> ordered_conv_worker_cores(
    const tt::tt_metal::CoreRangeSet& grid) {
    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(grid.num_cores());
    for (const auto& range : grid.ranges()) {
        for (const auto& core : range) {
            cores.push_back(core);
        }
    }
    return cores;
}

// Reconstruct the legacy HEIGHT_SHARDED ROW_MAJOR prefix using the physical
// row-scan boundary. A boundary exclusion can move the first worker away from
// zero without changing the original scan width, so bounding_box().grid_size()
// is insufficient: for x1..4 it reports four even though the stock scan width
// is five. Conv may intentionally use less than the device-wide worker grid,
// therefore the physical end coordinate is the narrowest recoverable stock
// boundary at this planner ingress.
inline tt::tt_metal::CoreRangeSet make_stock_height_sharded_worker_grid(
    const tt::tt_metal::CoreRangeSet& grid) {
    if (grid.empty()) {
        throw std::invalid_argument("HEIGHT_SHARDED worker grid must not be empty");
    }
    const auto grid_box = grid.bounding_box();
    return tt::tt_metal::num_cores_to_corerangeset(
        grid.num_cores(),
        tt::tt_metal::CoreCoord{grid_box.end_coord.x + 1, grid_box.end_coord.y + 1},
        true);
}

inline bool is_remapped_height_sharded_worker_grid(const tt::tt_metal::CoreRangeSet& grid) {
    return grid.bounding_box().start_coord != tt::tt_metal::CoreCoord{0, 0} ||
           grid != make_stock_height_sharded_worker_grid(grid);
}

inline tt::tt_metal::CoreRangeSet make_ordered_height_sharded_worker_grid(
    const tt::tt_metal::CoreCoord& compute_grid_size,
    uint32_t num_cores,
    const std::vector<uint32_t>& excluded_worker_rows,
    const std::vector<uint32_t>& excluded_worker_columns,
    uint32_t row_scan_width = 0) {
    const std::set<uint32_t> excluded_rows(excluded_worker_rows.begin(), excluded_worker_rows.end());
    const std::set<uint32_t> excluded_columns(excluded_worker_columns.begin(), excluded_worker_columns.end());
    if (excluded_rows.size() != excluded_worker_rows.size() ||
        excluded_columns.size() != excluded_worker_columns.size()) {
        throw std::invalid_argument("Conv core-remapping exclusions must not contain duplicates");
    }
    if ((!excluded_rows.empty() && *excluded_rows.rbegin() >= compute_grid_size.y) ||
        (!excluded_columns.empty() && *excluded_columns.rbegin() >= compute_grid_size.x)) {
        throw std::invalid_argument("Conv core-remapping exclusion is outside the compute worker grid");
    }
    if (row_scan_width == 0) {
        row_scan_width = compute_grid_size.x;
    }
    if (row_scan_width > compute_grid_size.x) {
        throw std::invalid_argument("HEIGHT_SHARDED row scan width is outside the compute worker grid");
    }

    std::vector<tt::tt_metal::CoreRange> selected;
    selected.reserve(num_cores);
    for (uint32_t y = 0; y < compute_grid_size.y && selected.size() < num_cores; ++y) {
        if (excluded_rows.contains(y)) {
            continue;
        }
        for (uint32_t x = 0; x < row_scan_width && selected.size() < num_cores; ++x) {
            if (!excluded_columns.contains(x)) {
                selected.emplace_back(tt::tt_metal::CoreCoord{x, y});
            }
        }
    }
    if (selected.size() != num_cores) {
        throw std::invalid_argument("Conv core-remapping exclusions leave too few workers for HEIGHT_SHARDED");
    }
    return tt::tt_metal::CoreRangeSet(std::move(selected));
}

// Select the physical input placement requested by Conv2dConfig while preserving the input's logical shard-grid
// dimensions and ordering. Keep validation here independent of device execution so unsupported placement requests
// fail on the host and can be covered by unit tests.
inline sliding_window::ParallelConfig select_conv_core_remapping_input_placement(
    const sliding_window::ParallelConfig& input_parallel_config,
    const tt::tt_metal::CoreCoord& compute_grid_size,
    const std::vector<uint32_t>& excluded_worker_rows,
    const std::vector<uint32_t>& excluded_worker_columns) {
    if (excluded_worker_rows.empty() && excluded_worker_columns.empty()) {
        return input_parallel_config;
    }
    if (input_parallel_config.shard_orientation != tt::tt_metal::ShardOrientation::ROW_MAJOR) {
        throw std::invalid_argument("Conv core-remapping placement selection currently supports only ROW_MAJOR");
    }
    if (input_parallel_config.shard_scheme == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        auto selected_input = input_parallel_config;
        const auto input_box = input_parallel_config.grid.bounding_box();
        if (input_box.start_coord != tt::tt_metal::CoreCoord{0, 0}) {
            throw std::invalid_argument(
                "Explicit HEIGHT_SHARDED exclusions currently require a stock-origin input placement");
        }
        selected_input.grid = make_ordered_height_sharded_worker_grid(
            compute_grid_size,
            input_parallel_config.grid.num_cores(),
            excluded_worker_rows,
            excluded_worker_columns,
            input_box.grid_size().x);
        return selected_input;
    }
    if (input_parallel_config.shard_scheme != tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
        throw std::invalid_argument(
            "Conv core-remapping placement selection currently supports HEIGHT_SHARDED and BLOCK_SHARDED");
    }
    const CorePlacement input_placement =
        make_core_placement_from_worker_grid(compute_grid_size, input_parallel_config.grid);
    const CorePlacement selected_placement = make_core_placement(
        compute_grid_size,
        input_placement.shard_grid_size,
        excluded_worker_rows,
        excluded_worker_columns);
    const auto x_segments = selected_placement.x_segments();
    const auto y_segments = selected_placement.y_segments();
    if (x_segments.size() > 2 || y_segments.size() > 2) {
        throw std::invalid_argument(
            "Conv core-remapping exclusions must produce at most two segments per worker axis (four Cartesian "
            "rectangles)");
    }
    auto selected_input = input_parallel_config;
    selected_input.grid = selected_placement.ordered_worker_cores(CorePlacementOrder::RowMajor);
    return selected_input;
}

// Decide the lifetime of an explicitly remapped placement without changing TTNN's stock output topology.
// HEIGHT_SHARDED propagates an ordered 1D worker prefix. BLOCK_SHARDED propagates compatible logical axis prefixes.
// Expansion or entry into an unsupported execution path reanchors both sides to the stock origin.
inline ConvCoreRemappingPlan plan_conv_core_remapping(
    const sliding_window::ParallelConfig& input_parallel_config,
    const sliding_window::ParallelConfig& stock_output_parallel_config,
    const tt::tt_metal::CoreCoord& compute_grid_size,
    bool remapping_active,
    bool current_placement_is_executable = true,
    const std::optional<ConvCorePlacementProvenance>& placement_provenance = std::nullopt) {
    if (placement_provenance.has_value()) {
        const auto& stock_input = placement_provenance->stock_input_parallel_config;
        if (stock_input.shard_scheme != input_parallel_config.shard_scheme ||
            stock_input.shard_orientation != input_parallel_config.shard_orientation ||
            stock_input.grid.num_cores() != input_parallel_config.grid.num_cores()) {
            throw std::invalid_argument(
                "Conv core-remapping placement provenance must preserve input scheme, orientation, and core count");
        }
    }

    if (!remapping_active) {
        return {
            .action = ConvCoreRemappingAction::NotActive,
            .input_parallel_config = input_parallel_config,
            .output_parallel_config = stock_output_parallel_config,
        };
    }

    if (input_parallel_config.shard_scheme == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED &&
        stock_output_parallel_config.shard_scheme == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        auto selected_input = input_parallel_config;
        auto selected_output = stock_output_parallel_config;
        const auto input_workers = ordered_conv_worker_cores(input_parallel_config.grid);
        const uint32_t output_cores = stock_output_parallel_config.grid.num_cores();
        if (current_placement_is_executable && output_cores <= input_workers.size()) {
            std::vector<tt::tt_metal::CoreRange> output_workers;
            output_workers.reserve(output_cores);
            for (uint32_t index = 0; index < output_cores; ++index) {
                output_workers.emplace_back(input_workers[index]);
            }
            selected_output.grid = tt::tt_metal::CoreRangeSet(std::move(output_workers));
            return {
                .action = ConvCoreRemappingAction::Propagate,
                .input_parallel_config = std::move(selected_input),
                .output_parallel_config = std::move(selected_output),
            };
        }

        selected_input.grid = placement_provenance.has_value()
            ? placement_provenance->stock_input_parallel_config.grid
            : (output_cores == input_parallel_config.grid.num_cores()
                   ? stock_output_parallel_config.grid
                   : tt::tt_metal::num_cores_to_corerangeset(
                         input_parallel_config.grid.num_cores(), compute_grid_size, true));
        return {
            .action = ConvCoreRemappingAction::ExitToStock,
            .input_parallel_config = std::move(selected_input),
            .output_parallel_config = std::move(selected_output),
        };
    }

    if (input_parallel_config.shard_scheme != tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
        stock_output_parallel_config.shard_scheme != tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
        throw std::invalid_argument(
            "Conv core remapping propagation requires matching HEIGHT_SHARDED or BLOCK_SHARDED input and output");
    }

    const CorePlacement input_placement =
        make_core_placement_from_worker_grid(compute_grid_size, input_parallel_config.grid);
    const CorePlacement stock_output_placement =
        make_core_placement_from_worker_grid(compute_grid_size, stock_output_parallel_config.grid);
    const tt::tt_metal::CoreCoord input_size = input_placement.shard_grid_size;
    const tt::tt_metal::CoreCoord output_size = stock_output_placement.shard_grid_size;

    const bool output_does_not_expand = output_size.x <= input_size.x && output_size.y <= input_size.y;

    auto selected_input = input_parallel_config;
    auto selected_output = stock_output_parallel_config;
    // Placement lifetime also ends at an operation boundary that cannot execute the current physical placement.
    // This is distinct from validation: a recognized segmented tensor is safe to inspect and reshard even when the
    // next Conv implementation (for example a matmul-lowered 1x1 Conv) cannot consume it directly.
    if (current_placement_is_executable && output_does_not_expand) {
        std::vector<uint32_t> output_worker_x(
            input_placement.worker_x.begin(), input_placement.worker_x.begin() + output_size.x);
        std::vector<uint32_t> output_worker_y(
            input_placement.worker_y.begin(), input_placement.worker_y.begin() + output_size.y);
        selected_output.grid =
            make_core_placement_from_worker_axes(
                compute_grid_size, std::move(output_worker_x), std::move(output_worker_y))
                .ordered_worker_cores(
                    selected_output.shard_orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR
                        ? CorePlacementOrder::RowMajor
                        : CorePlacementOrder::ColumnMajor);
        return {
            .action = ConvCoreRemappingAction::Propagate,
            .input_parallel_config = std::move(selected_input),
            .output_parallel_config = std::move(selected_output),
        };
    }

    selected_input.grid = placement_provenance.has_value()
        ? placement_provenance->stock_input_parallel_config.grid
        : make_core_placement(compute_grid_size, input_size).worker_cores;
    selected_output.grid = make_core_placement(compute_grid_size, output_size).worker_cores;
    return {
        .action = ConvCoreRemappingAction::ExitToStock,
        .input_parallel_config = std::move(selected_input),
        .output_parallel_config = std::move(selected_output),
    };
}

}  // namespace ttnn::operations::conv
