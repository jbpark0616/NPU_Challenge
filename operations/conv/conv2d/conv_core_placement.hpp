// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::conv {

// Dense workload coordinate. This is deliberately distinct from CoreCoord,
// which identifies a physical worker at the TT-Metal API boundary.
struct ShardCoord {
    uint32_t x;
    uint32_t y;

    bool operator==(const ShardCoord&) const = default;
};

// One ordered, physically contiguous interval from a logical shard axis.
// logical_start preserves the shard ordering when a skipped worker coordinate
// splits an axis into multiple physical multicast/halo segments.
struct CorePlacementAxisSegment {
    uint32_t logical_start;
    uint32_t worker_start;
    uint32_t length;

    uint32_t logical_end() const { return logical_start + length - 1; }
    uint32_t worker_end() const { return worker_start + length - 1; }

    bool operator==(const CorePlacementAxisSegment&) const = default;
};

// Cartesian product of one x-axis and one y-axis segment. This is the unit a
// rectangular NoC multicast can consume without crossing an excluded worker.
struct CorePlacementRectangleSegment {
    ShardCoord logical_start;
    tt::tt_metal::CoreRange workers;

    tt::tt_metal::CoreCoord logical_size() const { return workers.grid_size(); }
};

enum class CorePlacementMcastAxis {
    Row,
    Column,
};

enum class CorePlacementOrder {
    RowMajor,
    ColumnMajor,
};

struct CorePlacementMcastSegment {
    tt::tt_metal::CoreRange workers;
    uint32_t destination_count;
    bool contains_sender;
};

struct CorePlacementMcastLine {
    ShardCoord logical_line;
    tt::tt_metal::CoreCoord sender;
    std::vector<CorePlacementMcastSegment> segments;
};

struct CorePlacementMcastPlan {
    CorePlacementMcastAxis axis;
    std::vector<CorePlacementMcastLine> lines;
    uint32_t total_destination_count;
};

// Compact an arbitrary 1D HEIGHT_SHARDED receiver set into exact maximal
// rectangles. Horizontal runs with identical X bounds are merged across
// adjacent rows, so the physical placement is translated into NoC rectangles
// instead of issuing one multicast per row. Non-loopback multicast permits the
// sender to lie inside one rectangle: it is recorded by contains_sender and
// excluded from that rectangle's destination_count.
inline CorePlacementMcastLine make_remote_rectangle_mcast_line_for_core_set(
    const tt::tt_metal::CoreRangeSet& receiver_cores,
    tt::tt_metal::CoreCoord sender,
    tt::tt_metal::CoreCoord worker_grid_size) {
    if (sender.x >= worker_grid_size.x || sender.y >= worker_grid_size.y) {
        throw std::invalid_argument("multicast sender is outside the worker grid");
    }
    if (receiver_cores.contains(sender)) {
        throw std::invalid_argument("remote multicast receiver set must exclude the sender");
    }

    const tt::tt_metal::CoreRangeSet sender_core(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange(sender)});
    const tt::tt_metal::CoreRangeSet multicast_geometry = receiver_cores.merge(sender_core);

    CorePlacementMcastLine line{
        .logical_line = {0, 0},
        .sender = sender,
        .segments = {},
    };

    struct PendingRectangle {
        uint32_t start_x;
        uint32_t end_x;
        uint32_t start_y;
        uint32_t end_y;
    };
    std::vector<PendingRectangle> rectangles;

    for (uint32_t worker_y = 0; worker_y < worker_grid_size.y; ++worker_y) {
        std::optional<uint32_t> run_start_x;
        for (uint32_t worker_x = 0; worker_x <= worker_grid_size.x; ++worker_x) {
            const bool is_receiver =
                worker_x < worker_grid_size.x && multicast_geometry.contains({worker_x, worker_y});
            if (is_receiver && !run_start_x.has_value()) {
                run_start_x = worker_x;
            } else if (!is_receiver && run_start_x.has_value()) {
                const uint32_t run_end_x = worker_x - 1;
                const auto previous_row_rectangle = std::find_if(
                    rectangles.rbegin(),
                    rectangles.rend(),
                    [&](const PendingRectangle& rectangle) {
                        return rectangle.start_x == *run_start_x && rectangle.end_x == run_end_x &&
                               rectangle.end_y + 1 == worker_y;
                    });
                if (previous_row_rectangle != rectangles.rend()) {
                    previous_row_rectangle->end_y = worker_y;
                } else {
                    rectangles.push_back(
                        {.start_x = *run_start_x,
                         .end_x = run_end_x,
                         .start_y = worker_y,
                         .end_y = worker_y});
                }
                run_start_x.reset();
            }
        }
    }

    line.segments.reserve(rectangles.size());
    for (const auto& rectangle : rectangles) {
        const tt::tt_metal::CoreRange workers(
            {rectangle.start_x, rectangle.start_y}, {rectangle.end_x, rectangle.end_y});
        const bool contains_sender = workers.contains(sender);
        const uint32_t destination_count = workers.size() - static_cast<uint32_t>(contains_sender);
        if (destination_count == 0) {
            continue;
        }
        line.segments.push_back(
            {.workers = workers,
             .destination_count = destination_count,
             .contains_sender = contains_sender});
    }

    uint32_t destination_count = 0;
    for (const auto& segment : line.segments) {
        destination_count += segment.destination_count;
    }
    if (destination_count != receiver_cores.num_cores()) {
        throw std::logic_error("multicast segments do not cover the exact receiver core set");
    }
    return line;
}

// Runtime format consumed by the segmented Conv multicast path:
//   [segment_count,
//    start_x, start_y, end_x, end_y, destination_count, contains_sender,
//    ...]
// Endpoint coordinates are converted to physical NoC coordinates by the
// caller-provided mapper. NOC1 reverses rectangle endpoints, matching the
// existing Conv setup_mcast_args convention.
template <typename WorkerToNoc>
std::vector<uint32_t> serialize_mcast_segments(
    const CorePlacementMcastLine& line, bool is_noc_0, WorkerToNoc&& worker_to_noc) {
    std::vector<uint32_t> args;
    args.reserve(1 + line.segments.size() * 6);
    args.push_back(static_cast<uint32_t>(line.segments.size()));
    for (const auto& segment : line.segments) {
        const auto start = worker_to_noc(segment.workers.start_coord);
        const auto end = worker_to_noc(segment.workers.end_coord);
        if (is_noc_0) {
            args.insert(args.end(), {start.x, start.y, end.x, end.y});
        } else {
            args.insert(args.end(), {end.x, end.y, start.x, start.y});
        }
        args.push_back(segment.destination_count);
        args.push_back(static_cast<uint32_t>(segment.contains_sender));
    }
    return args;
}

inline std::vector<CorePlacementAxisSegment> segment_worker_axis(const std::vector<uint32_t>& worker_coordinates) {
    if (worker_coordinates.empty()) {
        throw std::invalid_argument("worker axis must not be empty");
    }

    std::vector<CorePlacementAxisSegment> segments;
    segments.reserve(worker_coordinates.size());
    uint32_t segment_logical_start = 0;
    uint32_t segment_worker_start = worker_coordinates.front();

    for (uint32_t logical = 1; logical < worker_coordinates.size(); ++logical) {
        const uint32_t previous_worker = worker_coordinates[logical - 1];
        const uint32_t worker = worker_coordinates[logical];
        if (worker <= previous_worker) {
            throw std::invalid_argument("worker coordinates must be strictly increasing");
        }
        if (worker != previous_worker + 1) {
            segments.push_back(
                {.logical_start = segment_logical_start,
                 .worker_start = segment_worker_start,
                 .length = logical - segment_logical_start});
            segment_logical_start = logical;
            segment_worker_start = worker;
        }
    }
    segments.push_back(
        {.logical_start = segment_logical_start,
         .worker_start = segment_worker_start,
         .length = static_cast<uint32_t>(worker_coordinates.size()) - segment_logical_start});
    return segments;
}

// Maps a dense shard grid onto an ordered Cartesian subset of the TT-Metal
// worker grid. NoC conversion remains the responsibility of the program
// factory through IDevice::worker_core_from_logical_core().
struct CorePlacement {
    tt::tt_metal::CoreCoord shard_grid_size;
    tt::tt_metal::CoreCoord worker_grid_size;
    std::vector<uint32_t> worker_x;
    std::vector<uint32_t> worker_y;
    tt::tt_metal::CoreRangeSet worker_cores;

    tt::tt_metal::CoreCoord worker_core(ShardCoord shard) const {
        if (shard.x >= shard_grid_size.x || shard.y >= shard_grid_size.y) {
            throw std::out_of_range("shard coordinate is outside the shard grid");
        }
        return {worker_x.at(shard.x), worker_y.at(shard.y)};
    }

    std::optional<ShardCoord> shard_core(tt::tt_metal::CoreCoord worker) const {
        const auto x = std::find(worker_x.begin(), worker_x.end(), worker.x);
        const auto y = std::find(worker_y.begin(), worker_y.end(), worker.y);
        if (x == worker_x.end() || y == worker_y.end()) {
            return std::nullopt;
        }
        return ShardCoord{
            static_cast<uint32_t>(std::distance(worker_x.begin(), x)),
            static_cast<uint32_t>(std::distance(worker_y.begin(), y)),
        };
    }

    std::vector<CorePlacementAxisSegment> x_segments() const { return segment_worker_axis(worker_x); }
    std::vector<CorePlacementAxisSegment> y_segments() const { return segment_worker_axis(worker_y); }

    // CoreRangeSet preserves range order, and TT-Metal expands each range in
    // that order when assigning tensor shards. A segmented Cartesian grid
    // therefore uses ordered singleton ranges so logical shard order does not
    // depend on how the physical rectangles happen to be partitioned.
    tt::tt_metal::CoreRangeSet ordered_worker_cores(CorePlacementOrder order) const {
        if (!has_worker_gaps()) {
            return worker_cores;
        }

        std::vector<tt::tt_metal::CoreRange> ordered;
        ordered.reserve(shard_grid_size.x * shard_grid_size.y);
        if (order == CorePlacementOrder::RowMajor) {
            for (uint32_t logical_y = 0; logical_y < shard_grid_size.y; ++logical_y) {
                for (uint32_t logical_x = 0; logical_x < shard_grid_size.x; ++logical_x) {
                    ordered.emplace_back(worker_core({logical_x, logical_y}));
                }
            }
        } else {
            for (uint32_t logical_x = 0; logical_x < shard_grid_size.x; ++logical_x) {
                for (uint32_t logical_y = 0; logical_y < shard_grid_size.y; ++logical_y) {
                    ordered.emplace_back(worker_core({logical_x, logical_y}));
                }
            }
        }
        return tt::tt_metal::CoreRangeSet(std::move(ordered));
    }

    std::vector<CorePlacementRectangleSegment> rectangular_segments() const {
        const auto x_axis_segments = x_segments();
        const auto y_axis_segments = y_segments();
        std::vector<CorePlacementRectangleSegment> rectangles;
        rectangles.reserve(x_axis_segments.size() * y_axis_segments.size());
        for (const auto& y_segment : y_axis_segments) {
            for (const auto& x_segment : x_axis_segments) {
                rectangles.push_back({
                    .logical_start = {x_segment.logical_start, y_segment.logical_start},
                    .workers = tt::tt_metal::CoreRange(
                        {x_segment.worker_start, y_segment.worker_start},
                        {x_segment.worker_end(), y_segment.worker_end()}),
                });
            }
        }
        return rectangles;
    }

    CorePlacementMcastLine mcast_line_for_worker_sender(
        CorePlacementMcastAxis axis, uint32_t logical_fixed, tt::tt_metal::CoreCoord sender) const {
        const bool row_axis = axis == CorePlacementMcastAxis::Row;
        const auto varying_segments = row_axis ? x_segments() : y_segments();
        const auto& fixed_workers = row_axis ? worker_y : worker_x;
        if (logical_fixed >= fixed_workers.size()) {
            throw std::out_of_range("multicast logical line is outside the shard grid");
        }

        const uint32_t fixed_worker = fixed_workers[logical_fixed];
        if ((row_axis && sender.y != fixed_worker) || (!row_axis && sender.x != fixed_worker)) {
            throw std::invalid_argument("multicast sender must lie on the selected physical line");
        }
        const uint32_t sender_varying_worker = row_axis ? sender.x : sender.y;
        CorePlacementMcastLine line{
            .logical_line = row_axis ? ShardCoord{0, logical_fixed} : ShardCoord{logical_fixed, 0},
            .sender = sender,
            .segments = {},
        };
        for (const auto& segment : varying_segments) {
            const tt::tt_metal::CoreRange range = row_axis
                ? tt::tt_metal::CoreRange(
                      {segment.worker_start, fixed_worker}, {segment.worker_end(), fixed_worker})
                : tt::tt_metal::CoreRange(
                      {fixed_worker, segment.worker_start}, {fixed_worker, segment.worker_end()});
            const bool contains_sender = sender_varying_worker >= segment.worker_start &&
                                         sender_varying_worker <= segment.worker_end();
            const uint32_t destination_count = segment.length - static_cast<uint32_t>(contains_sender);
            // Preserve a singleton sender segment as protocol metadata. The
            // device kernel must perform a local copy instead of issuing a
            // zero-destination NoC multicast for it.
            line.segments.push_back(
                {.workers = range, .destination_count = destination_count, .contains_sender = contains_sender});
        }
        return line;
    }

    CorePlacementMcastLine mcast_line(
        CorePlacementMcastAxis axis, uint32_t logical_fixed, uint32_t logical_sender) const {
        const auto& varying_workers = axis == CorePlacementMcastAxis::Row ? worker_x : worker_y;
        if (logical_sender >= varying_workers.size()) {
            throw std::out_of_range("multicast logical sender is outside the shard grid");
        }
        const uint32_t fixed_worker =
            axis == CorePlacementMcastAxis::Row ? worker_y.at(logical_fixed) : worker_x.at(logical_fixed);
        const uint32_t varying_worker = varying_workers[logical_sender];
        return mcast_line_for_worker_sender(
            axis,
            logical_fixed,
            axis == CorePlacementMcastAxis::Row ? tt::tt_metal::CoreCoord{varying_worker, fixed_worker}
                                                : tt::tt_metal::CoreCoord{fixed_worker, varying_worker});
    }

    // Build the device multicast rectangles without the sender.  Keeping the
    // local copy separate makes every emitted rectangle use the same
    // remote-only protocol as the device-tested segmented one-to-all path.
    // A physical segment containing the sender can become zero, one, or two
    // remote rectangles depending on the sender position.
    CorePlacementMcastLine remote_mcast_line_for_worker_sender(
        CorePlacementMcastAxis axis, uint32_t logical_fixed, tt::tt_metal::CoreCoord sender) const {
        const auto line = mcast_line_for_worker_sender(axis, logical_fixed, sender);
        const bool row_axis = axis == CorePlacementMcastAxis::Row;
        const uint32_t sender_varying_worker = row_axis ? sender.x : sender.y;
        CorePlacementMcastLine remote_line{
            .logical_line = line.logical_line,
            .sender = sender,
            .segments = {},
        };

        for (const auto& segment : line.segments) {
            if (!segment.contains_sender) {
                remote_line.segments.push_back(segment);
                continue;
            }

            const uint32_t start = row_axis ? segment.workers.start_coord.x : segment.workers.start_coord.y;
            const uint32_t end = row_axis ? segment.workers.end_coord.x : segment.workers.end_coord.y;
            const uint32_t fixed = row_axis ? segment.workers.start_coord.y : segment.workers.start_coord.x;
            if (start < sender_varying_worker) {
                const auto workers = row_axis
                    ? tt::tt_metal::CoreRange({start, fixed}, {sender_varying_worker - 1, fixed})
                    : tt::tt_metal::CoreRange({fixed, start}, {fixed, sender_varying_worker - 1});
                remote_line.segments.push_back(
                    {.workers = workers, .destination_count = workers.size(), .contains_sender = false});
            }
            if (sender_varying_worker < end) {
                const auto workers = row_axis
                    ? tt::tt_metal::CoreRange({sender_varying_worker + 1, fixed}, {end, fixed})
                    : tt::tt_metal::CoreRange({fixed, sender_varying_worker + 1}, {fixed, end});
                remote_line.segments.push_back(
                    {.workers = workers, .destination_count = workers.size(), .contains_sender = false});
            }
        }
        return remote_line;
    }

    CorePlacementMcastPlan mcast_plan(CorePlacementMcastAxis axis) const {
        const bool row_axis = axis == CorePlacementMcastAxis::Row;
        const auto& fixed_workers = row_axis ? worker_y : worker_x;
        CorePlacementMcastPlan plan{.axis = axis, .lines = {}, .total_destination_count = 0};
        plan.lines.reserve(fixed_workers.size());

        for (uint32_t logical_fixed = 0; logical_fixed < fixed_workers.size(); ++logical_fixed) {
            auto line = mcast_line(axis, logical_fixed, /*logical_sender=*/0);
            for (const auto& segment : line.segments) {
                plan.total_destination_count += segment.destination_count;
            }
            plan.lines.push_back(std::move(line));
        }
        return plan;
    }

    bool has_worker_gaps() const { return x_segments().size() > 1 || y_segments().size() > 1; }
};

inline std::vector<uint32_t> select_worker_coordinates(
    uint32_t worker_dimension,
    uint32_t shard_dimension,
    const std::vector<uint32_t>& excluded_worker_coordinates) {
    if (worker_dimension == 0 || shard_dimension == 0 || shard_dimension > worker_dimension) {
        throw std::invalid_argument("shard dimension must fit in a non-empty worker dimension");
    }

    std::set<uint32_t> excluded;
    for (uint32_t coordinate : excluded_worker_coordinates) {
        if (coordinate >= worker_dimension) {
            throw std::invalid_argument("excluded worker coordinate is outside the worker grid");
        }
        if (!excluded.insert(coordinate).second) {
            throw std::invalid_argument("excluded worker coordinates must be unique");
        }
    }
    if (shard_dimension > worker_dimension - excluded.size()) {
        throw std::invalid_argument("worker grid has no replacement capacity after exclusion");
    }

    std::vector<uint32_t> selected;
    selected.reserve(shard_dimension);
    for (uint32_t worker = 0; worker < worker_dimension && selected.size() < shard_dimension; ++worker) {
        if (!excluded.contains(worker)) {
            selected.push_back(worker);
        }
    }
    if (selected.size() != shard_dimension) {
        throw std::invalid_argument("worker grid has no replacement capacity after exclusion");
    }
    return selected;
}

inline CorePlacement make_core_placement_from_worker_axes(
    tt::tt_metal::CoreCoord worker_grid_size,
    std::vector<uint32_t> worker_x,
    std::vector<uint32_t> worker_y) {
    // segment_worker_axis also validates that each axis is non-empty and
    // strictly increasing.
    segment_worker_axis(worker_x);
    segment_worker_axis(worker_y);
    if (worker_x.back() >= worker_grid_size.x || worker_y.back() >= worker_grid_size.y) {
        throw std::invalid_argument("worker axis is outside the worker grid");
    }

    CorePlacement placement{
        .shard_grid_size = {
            static_cast<uint32_t>(worker_x.size()), static_cast<uint32_t>(worker_y.size())},
        .worker_grid_size = worker_grid_size,
        .worker_x = std::move(worker_x),
        .worker_y = std::move(worker_y),
        .worker_cores = {},
    };
    std::set<tt::tt_metal::CoreRange> worker_ranges;
    for (const auto& rectangle : placement.rectangular_segments()) {
        worker_ranges.insert(rectangle.workers);
    }
    placement.worker_cores = tt::tt_metal::CoreRangeSet(std::move(worker_ranges));

    const uint32_t expected_core_count = placement.shard_grid_size.x * placement.shard_grid_size.y;
    if (placement.worker_cores.num_cores() != expected_core_count) {
        throw std::logic_error("worker placement is not a one-to-one Cartesian shard mapping");
    }
    return placement;
}

inline CorePlacement make_core_placement(
    tt::tt_metal::CoreCoord worker_grid_size,
    tt::tt_metal::CoreCoord shard_grid_size,
    const std::vector<uint32_t>& excluded_worker_rows,
    const std::vector<uint32_t>& excluded_worker_columns) {
    auto worker_x =
        select_worker_coordinates(worker_grid_size.x, shard_grid_size.x, excluded_worker_columns);
    auto worker_y = select_worker_coordinates(worker_grid_size.y, shard_grid_size.y, excluded_worker_rows);

    return make_core_placement_from_worker_axes(worker_grid_size, std::move(worker_x), std::move(worker_y));
}

// Normalize a user/planner-selected Cartesian CoreRangeSet into ordered axis
// coordinates. Non-Cartesian sets are rejected because a dense BLOCK_SHARDED
// logical grid cannot be inferred from them without explicit per-shard metadata.
inline CorePlacement make_core_placement_from_worker_grid(
    tt::tt_metal::CoreCoord worker_grid_size, const tt::tt_metal::CoreRangeSet& worker_cores) {
    if (worker_cores.empty()) {
        throw std::invalid_argument("worker core set must not be empty");
    }

    std::set<uint32_t> selected_x;
    std::set<uint32_t> selected_y;
    for (const auto& range : worker_cores.ranges()) {
        if (range.end_coord.x >= worker_grid_size.x || range.end_coord.y >= worker_grid_size.y) {
            throw std::invalid_argument("worker core set is outside the worker grid");
        }
        for (const auto& worker : range) {
            selected_x.insert(worker.x);
            selected_y.insert(worker.y);
        }
    }
    if (selected_x.size() * selected_y.size() != worker_cores.num_cores()) {
        throw std::invalid_argument("worker core set must be a Cartesian product of ordered x and y coordinates");
    }
    for (uint32_t y : selected_y) {
        for (uint32_t x : selected_x) {
            if (!worker_cores.contains({x, y})) {
                throw std::invalid_argument(
                    "worker core set must contain every coordinate in its Cartesian x/y product");
            }
        }
    }

    return make_core_placement_from_worker_axes(
        worker_grid_size,
        std::vector<uint32_t>(selected_x.begin(), selected_x.end()),
        std::vector<uint32_t>(selected_y.begin(), selected_y.end()));
}

inline CorePlacement make_core_placement(
    tt::tt_metal::CoreCoord worker_grid_size,
    tt::tt_metal::CoreCoord shard_grid_size,
    const std::optional<uint32_t>& excluded_worker_row = std::nullopt,
    const std::optional<uint32_t>& excluded_worker_column = std::nullopt) {
    const std::vector<uint32_t> excluded_worker_rows =
        excluded_worker_row.has_value() ? std::vector<uint32_t>{*excluded_worker_row} : std::vector<uint32_t>{};
    const std::vector<uint32_t> excluded_worker_columns =
        excluded_worker_column.has_value() ? std::vector<uint32_t>{*excluded_worker_column} : std::vector<uint32_t>{};
    return make_core_placement(
        worker_grid_size, shard_grid_size, excluded_worker_rows, excluded_worker_columns);
}

}  // namespace ttnn::operations::conv
