// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tt_metal/api/tt-metalium/core_coord.hpp"

namespace tt::tt_metal::unit_tests::dm::core_to_all::segmented_rest {

// This is deliberately a constrained planner: one optional rest row and one
// optional rest column. Their Cartesian complement has at most four rectangles.
struct Segment {
    CoreRange logical_range;
    uint32_t destination_count;
};

struct Plan {
    CoreRangeSet active_cores;
    CoreCoord sender;
    std::vector<Segment> segments;
    uint32_t destination_count;
};

enum class McastAxis { Row, Column };

// A real block-sharded Conv 2D multicast is composed from 1D row and column
// flows. Each line has one sender and at most two destination segments when a
// perpendicular rest line cuts it.
struct LinePlan {
    CoreCoord sender;
    std::vector<Segment> segments;
};

struct AxisPlan {
    CoreRangeSet active_cores;
    std::vector<LinePlan> lines;
    uint32_t total_destination_count;
};

// A logical rectangle can be placed on a subset of physical rows/columns.
// The selected coordinates stay ordered, so logical row/column order is
// preserved even when a physical line is skipped between two coordinates.
struct CoreReplacementPlan {
    CoreCoord physical_grid;
    std::vector<uint32_t> physical_x;
    std::vector<uint32_t> physical_y;
    CoreRangeSet active_cores;
};

inline bool range_contains(const CoreRange& range, const CoreCoord& core) {
    return core.x >= range.start_coord.x && core.x <= range.end_coord.x &&
           core.y >= range.start_coord.y && core.y <= range.end_coord.y;
}

inline std::vector<std::pair<uint32_t, uint32_t>> active_intervals(
    uint32_t dimension, const std::optional<uint32_t>& rest_index) {
    if (!rest_index.has_value()) {
        return {{0, dimension - 1}};
    }
    if (*rest_index >= dimension) {
        throw std::invalid_argument("rest index is outside the grid");
    }

    std::vector<std::pair<uint32_t, uint32_t>> intervals;
    if (*rest_index > 0) {
        intervals.emplace_back(0, *rest_index - 1);
    }
    if (*rest_index + 1 < dimension) {
        intervals.emplace_back(*rest_index + 1, dimension - 1);
    }
    return intervals;
}

inline std::vector<std::pair<uint32_t, uint32_t>> contiguous_intervals(const std::vector<uint32_t>& coordinates) {
    if (coordinates.empty()) {
        throw std::invalid_argument("replacement coordinates must be non-empty");
    }
    std::vector<std::pair<uint32_t, uint32_t>> intervals;
    uint32_t start = coordinates.front();
    uint32_t previous = start;
    for (size_t i = 1; i < coordinates.size(); ++i) {
        if (coordinates[i] <= previous) {
            throw std::invalid_argument("replacement coordinates must be strictly ordered");
        }
        if (coordinates[i] != previous + 1) {
            intervals.emplace_back(start, previous);
            start = coordinates[i];
        }
        previous = coordinates[i];
    }
    intervals.emplace_back(start, previous);
    return intervals;
}

inline std::vector<uint32_t> select_replacement_coordinates(
    uint32_t physical_dimension,
    uint32_t logical_dimension,
    const std::optional<uint32_t>& excluded_coordinate) {
    if (logical_dimension == 0 || logical_dimension > physical_dimension) {
        throw std::invalid_argument("logical dimension must fit in the physical dimension");
    }
    if (excluded_coordinate.has_value() && *excluded_coordinate >= physical_dimension) {
        throw std::invalid_argument("excluded coordinate is outside the physical grid");
    }

    std::vector<uint32_t> selected;
    for (uint32_t coordinate = 0; coordinate < physical_dimension && selected.size() < logical_dimension; ++coordinate) {
        if (!excluded_coordinate.has_value() || coordinate != *excluded_coordinate) {
            selected.push_back(coordinate);
        }
    }
    if (selected.size() != logical_dimension) {
        throw std::invalid_argument("not enough physical coordinates after exclusion");
    }
    return selected;
}

inline CoreReplacementPlan make_core_replacement_plan(
    CoreCoord physical_grid,
    CoreCoord logical_grid,
    const std::optional<uint32_t>& excluded_row = std::nullopt,
    const std::optional<uint32_t>& excluded_column = std::nullopt) {
    auto physical_x = select_replacement_coordinates(physical_grid.x, logical_grid.x, excluded_column);
    auto physical_y = select_replacement_coordinates(physical_grid.y, logical_grid.y, excluded_row);

    std::set<CoreRange> active_ranges;
    for (uint32_t y : physical_y) {
        for (uint32_t x : physical_x) {
            active_ranges.emplace(CoreRange({x, y}));
        }
    }
    return {
        .physical_grid = physical_grid,
        .physical_x = std::move(physical_x),
        .physical_y = std::move(physical_y),
        .active_cores = CoreRangeSet(std::move(active_ranges)),
    };
}

// One sender broadcasts a dense logical rectangle that has been remapped to
// the selected physical coordinates. This is a transport-only control: unlike
// the Conv axis plan below, it has exactly one sender for the whole workload.
inline Plan make_single_sender_mcast_plan(const CoreReplacementPlan& replacement) {
    const auto x_intervals = contiguous_intervals(replacement.physical_x);
    const auto y_intervals = contiguous_intervals(replacement.physical_y);
    const CoreCoord sender{replacement.physical_x.front(), replacement.physical_y.front()};
    Plan plan{
        .active_cores = replacement.active_cores,
        .sender = sender,
        .segments = {},
        .destination_count = 0,
    };
    for (const auto& [start_x, end_x] : x_intervals) {
        for (const auto& [start_y, end_y] : y_intervals) {
            const CoreRange range({start_x, start_y}, {end_x, end_y});
            const uint32_t core_count = (end_x - start_x + 1) * (end_y - start_y + 1);
            const uint32_t destination_count = core_count - (range_contains(range, sender) ? 1 : 0);
            if (destination_count == 0) {
                continue;
            }
            plan.segments.push_back({range, destination_count});
            plan.destination_count += destination_count;
        }
    }
    if (plan.segments.empty() || plan.segments.size() > 4 ||
        plan.destination_count != replacement.active_cores.num_cores() - 1) {
        throw std::logic_error("invalid single-sender segmented multicast plan");
    }
    return plan;
}

inline Plan make_row_column_rest_plan(
    CoreCoord grid_size,
    const std::optional<uint32_t>& rest_row = std::nullopt,
    const std::optional<uint32_t>& rest_column = std::nullopt) {
    if (grid_size.x == 0 || grid_size.y == 0) {
        throw std::invalid_argument("grid dimensions must be non-zero");
    }

    const CoreRangeSet full_cores(std::set<CoreRange>{CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1})});
    CoreRangeSet active_cores = full_cores;
    if (rest_row.has_value()) {
        if (*rest_row >= grid_size.y) {
            throw std::invalid_argument("rest row is outside the grid");
        }
        active_cores = active_cores.subtract(
            CoreRangeSet(std::set<CoreRange>{CoreRange({0, *rest_row}, {grid_size.x - 1, *rest_row})}));
    }
    if (rest_column.has_value()) {
        if (*rest_column >= grid_size.x) {
            throw std::invalid_argument("rest column is outside the grid");
        }
        active_cores = active_cores.subtract(
            CoreRangeSet(std::set<CoreRange>{CoreRange({*rest_column, 0}, {*rest_column, grid_size.y - 1})}));
    }
    if (active_cores.num_cores() == 0) {
        throw std::invalid_argument("rest lines remove every core");
    }

    CoreCoord sender{grid_size.x - 1, grid_size.y - 1};
    for (uint32_t y = 0; y < grid_size.y; ++y) {
        for (uint32_t x = 0; x < grid_size.x; ++x) {
            const CoreCoord core{x, y};
            if (active_cores.contains(core)) {
                sender.x = std::min(sender.x, core.x);
                sender.y = std::min(sender.y, core.y);
            }
        }
    }

    Plan plan{.active_cores = active_cores, .sender = sender, .segments = {}, .destination_count = 0};
    for (const auto& [start_x, end_x] : active_intervals(grid_size.x, rest_column)) {
        for (const auto& [start_y, end_y] : active_intervals(grid_size.y, rest_row)) {
            const CoreRange range({start_x, start_y}, {end_x, end_y});
            const uint32_t core_count = (end_x - start_x + 1) * (end_y - start_y + 1);
            const uint32_t destination_count = core_count - (range_contains(range, sender) ? 1 : 0);
            // A singleton sender rectangle has no destination and must not turn
            // into a zero-destination NoC command.
            if (destination_count == 0) {
                continue;
            }
            plan.segments.push_back({range, destination_count});
            plan.destination_count += destination_count;
        }
    }

    if (plan.segments.empty() || plan.segments.size() > 4 ||
        plan.destination_count != active_cores.num_cores() - 1) {
        throw std::logic_error("invalid max-4 segmented multicast plan");
    }
    return plan;
}

inline AxisPlan make_conv_axis_plan(
    CoreCoord grid_size,
    McastAxis axis,
    const std::optional<uint32_t>& rest_row = std::nullopt,
    const std::optional<uint32_t>& rest_column = std::nullopt) {
    const auto global_plan = make_row_column_rest_plan(grid_size, rest_row, rest_column);
    const auto x_intervals = active_intervals(grid_size.x, rest_column);
    const auto y_intervals = active_intervals(grid_size.y, rest_row);

    AxisPlan plan{
        .active_cores = global_plan.active_cores,
        .lines = {},
        .total_destination_count = 0,
    };
    const bool is_row_axis = axis == McastAxis::Row;
    const auto& fixed_coordinates = is_row_axis ? y_intervals : x_intervals;
    const auto& varying_intervals = is_row_axis ? x_intervals : y_intervals;

    for (const auto& [fixed_start, fixed_end] : fixed_coordinates) {
        for (uint32_t fixed = fixed_start; fixed <= fixed_end; ++fixed) {
            LinePlan line{.sender = {}, .segments = {}};
            const auto& first_interval = varying_intervals.front();
            line.sender = is_row_axis ? CoreCoord{first_interval.first, fixed} : CoreCoord{fixed, first_interval.first};

            for (const auto& [start, end] : varying_intervals) {
                const CoreRange range = is_row_axis
                    ? CoreRange({start, fixed}, {end, fixed})
                    : CoreRange({fixed, start}, {fixed, end});
                const uint32_t core_count = end - start + 1;
                const uint32_t destination_count = core_count - (range_contains(range, line.sender) ? 1 : 0);
                if (destination_count == 0) {
                    continue;
                }
                line.segments.push_back({range, destination_count});
                plan.total_destination_count += destination_count;
            }
            if (!line.segments.empty()) {
                plan.lines.push_back(std::move(line));
            }
        }
    }
    return plan;
}

inline AxisPlan make_conv_axis_plan(const CoreReplacementPlan& replacement, McastAxis axis) {
    const auto x_intervals = contiguous_intervals(replacement.physical_x);
    const auto y_intervals = contiguous_intervals(replacement.physical_y);
    const bool is_row_axis = axis == McastAxis::Row;
    const auto& fixed_coordinates = is_row_axis ? replacement.physical_y : replacement.physical_x;
    const auto& varying_intervals = is_row_axis ? x_intervals : y_intervals;

    AxisPlan plan{
        .active_cores = replacement.active_cores,
        .lines = {},
        .total_destination_count = 0,
    };
    for (uint32_t fixed : fixed_coordinates) {
        LinePlan line{.sender = {}, .segments = {}};
        const auto& first_interval = varying_intervals.front();
        line.sender = is_row_axis ? CoreCoord{first_interval.first, fixed} : CoreCoord{fixed, first_interval.first};
        for (const auto& [start, end] : varying_intervals) {
            const CoreRange range = is_row_axis
                ? CoreRange({start, fixed}, {end, fixed})
                : CoreRange({fixed, start}, {fixed, end});
            const uint32_t core_count = end - start + 1;
            const uint32_t destination_count = core_count - (range_contains(range, line.sender) ? 1 : 0);
            if (destination_count == 0) {
                continue;
            }
            line.segments.push_back({range, destination_count});
            plan.total_destination_count += destination_count;
        }
        if (!line.segments.empty()) {
            plan.lines.push_back(std::move(line));
        }
    }
    return plan;
}

}  // namespace tt::tt_metal::unit_tests::dm::core_to_all::segmented_rest
