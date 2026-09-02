// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "core_placement.hpp"

#include <gtest/gtest.h>

namespace tt::tt_metal::unit_tests::conv_core_replacement {
namespace {

constexpr CoreCoord kWorkerGrid{5, 4};
constexpr CoreCoord kShardGrid{3, 3};

void expect_segment(
    const CorePlacementAxisSegment& segment, uint32_t logical_start, uint32_t worker_start, uint32_t length) {
    EXPECT_EQ(segment.logical_start, logical_start);
    EXPECT_EQ(segment.worker_start, worker_start);
    EXPECT_EQ(segment.length, length);
    EXPECT_EQ(segment.logical_end(), logical_start + length - 1);
    EXPECT_EQ(segment.worker_end(), worker_start + length - 1);
}

void expect_rectangle(
    const CorePlacementRectangleSegment& rectangle,
    ShardCoord logical_start,
    CoreCoord worker_start,
    CoreCoord worker_end) {
    EXPECT_EQ(rectangle.logical_start, logical_start);
    EXPECT_EQ(rectangle.workers.start_coord, worker_start);
    EXPECT_EQ(rectangle.workers.end_coord, worker_end);
    EXPECT_EQ(
        rectangle.logical_size(),
        CoreCoord(worker_end.x - worker_start.x + 1, worker_end.y - worker_start.y + 1));
}

void expect_round_trip(const CorePlacement& placement) {
    EXPECT_EQ(placement.worker_cores.num_cores(), 9u);
    for (uint32_t y = 0; y < kShardGrid.y; ++y) {
        for (uint32_t x = 0; x < kShardGrid.x; ++x) {
            const ShardCoord shard{x, y};
            const CoreCoord worker = placement.worker_core(shard);
            EXPECT_TRUE(placement.worker_cores.contains(worker));
            ASSERT_TRUE(placement.shard_core(worker).has_value());
            EXPECT_EQ(*placement.shard_core(worker), shard);
        }
    }
}

TEST(ConvCorePlacement, ContiguousBaselineIsOneToOne) {
    const auto placement = make_core_placement(kWorkerGrid, kShardGrid);
    EXPECT_EQ(placement.worker_x, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(placement.worker_y, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_FALSE(placement.has_worker_gaps());
    ASSERT_EQ(placement.x_segments().size(), 1u);
    ASSERT_EQ(placement.y_segments().size(), 1u);
    expect_segment(placement.x_segments().front(), 0, 0, 3);
    expect_segment(placement.y_segments().front(), 0, 0, 3);
    ASSERT_EQ(placement.rectangular_segments().size(), 1u);
    expect_rectangle(placement.rectangular_segments().front(), {0, 0}, {0, 0}, {2, 2});
    EXPECT_EQ(placement.worker_cores.ranges().size(), 1u);
    expect_round_trip(placement);
}

TEST(ConvCorePlacement, ExcludingWorkerBoundaryProducesContiguousShift) {
    const auto row_shift = make_core_placement(kWorkerGrid, kShardGrid, 0);
    EXPECT_EQ(row_shift.worker_y, (std::vector<uint32_t>{1, 2, 3}));
    EXPECT_FALSE(row_shift.has_worker_gaps());
    ASSERT_EQ(row_shift.y_segments().size(), 1u);
    expect_segment(row_shift.y_segments().front(), 0, 1, 3);
    expect_round_trip(row_shift);

    const auto column_shift = make_core_placement(kWorkerGrid, kShardGrid, std::nullopt, 0);
    EXPECT_EQ(column_shift.worker_x, (std::vector<uint32_t>{1, 2, 3}));
    EXPECT_FALSE(column_shift.has_worker_gaps());
    expect_round_trip(column_shift);

    const auto two_column_shift = make_core_placement(
        kWorkerGrid, kShardGrid, std::vector<uint32_t>{}, std::vector<uint32_t>{0, 1});
    EXPECT_EQ(two_column_shift.worker_x, (std::vector<uint32_t>{2, 3, 4}));
    EXPECT_FALSE(two_column_shift.has_worker_gaps());
    expect_round_trip(two_column_shift);
}

TEST(ConvCorePlacement, ExcludingMiddleWorkerAxesPreservesShardOrder) {
    const auto placement = make_core_placement(kWorkerGrid, kShardGrid, 2, 2);
    EXPECT_EQ(placement.worker_x, (std::vector<uint32_t>{0, 1, 3}));
    EXPECT_EQ(placement.worker_y, (std::vector<uint32_t>{0, 1, 3}));
    EXPECT_TRUE(placement.has_worker_gaps());
    EXPECT_EQ(placement.worker_core({2, 1}), CoreCoord(3, 1));
    EXPECT_EQ(placement.worker_core({1, 2}), CoreCoord(1, 3));
    EXPECT_FALSE(placement.worker_cores.contains({2, 1}));
    EXPECT_FALSE(placement.worker_cores.contains({1, 2}));
    const auto x_segments = placement.x_segments();
    const auto y_segments = placement.y_segments();
    ASSERT_EQ(x_segments.size(), 2u);
    ASSERT_EQ(y_segments.size(), 2u);
    expect_segment(x_segments[0], 0, 0, 2);
    expect_segment(x_segments[1], 2, 3, 1);
    expect_segment(y_segments[0], 0, 0, 2);
    expect_segment(y_segments[1], 2, 3, 1);
    const auto rectangles = placement.rectangular_segments();
    ASSERT_EQ(rectangles.size(), 4u);
    expect_rectangle(rectangles[0], {0, 0}, {0, 0}, {1, 1});
    expect_rectangle(rectangles[1], {2, 0}, {3, 0}, {3, 1});
    expect_rectangle(rectangles[2], {0, 2}, {0, 3}, {1, 3});
    expect_rectangle(rectangles[3], {2, 2}, {3, 3}, {3, 3});
    EXPECT_EQ(placement.worker_cores.ranges().size(), 4u);
    expect_round_trip(placement);
}

TEST(ConvCorePlacement, SingleAxisGapProducesTwoRectangles) {
    const auto column_gap = make_core_placement(kWorkerGrid, kShardGrid, std::nullopt, 1);
    const auto column_rectangles = column_gap.rectangular_segments();
    ASSERT_EQ(column_rectangles.size(), 2u);
    expect_rectangle(column_rectangles[0], {0, 0}, {0, 0}, {0, 2});
    expect_rectangle(column_rectangles[1], {1, 0}, {2, 0}, {3, 2});
    EXPECT_EQ(column_gap.worker_cores.ranges().size(), 2u);

    const auto row_gap = make_core_placement(kWorkerGrid, kShardGrid, 1);
    const auto row_rectangles = row_gap.rectangular_segments();
    ASSERT_EQ(row_rectangles.size(), 2u);
    expect_rectangle(row_rectangles[0], {0, 0}, {0, 0}, {2, 0});
    expect_rectangle(row_rectangles[1], {0, 1}, {0, 2}, {2, 3});
    EXPECT_EQ(row_gap.worker_cores.ranges().size(), 2u);
}

TEST(ConvCorePlacement, ColumnGapProducesSegmentedActivationRowsAndContiguousWeightColumns) {
    const auto placement = make_core_placement(kWorkerGrid, kShardGrid, std::nullopt, 2);
    const auto activation_plan = placement.mcast_plan(CorePlacementMcastAxis::Row);
    ASSERT_EQ(activation_plan.lines.size(), 3u);
    EXPECT_EQ(activation_plan.total_destination_count, 6u);
    for (uint32_t logical_y = 0; logical_y < activation_plan.lines.size(); ++logical_y) {
        const auto& line = activation_plan.lines[logical_y];
        EXPECT_EQ(line.logical_line, (ShardCoord{0, logical_y}));
        EXPECT_EQ(line.sender, CoreCoord(0, logical_y));
    }

    // Each activation row crosses the skipped x=2 worker, so its two physical
    // intervals are issued independently while retaining one logical line.
    for (const auto& line : activation_plan.lines) {
        ASSERT_EQ(line.segments.size(), 2u);
        EXPECT_EQ(line.segments[0].workers.start_coord.x, 0u);
        EXPECT_EQ(line.segments[0].workers.end_coord.x, 1u);
        EXPECT_EQ(line.segments[0].destination_count, 1u);
        EXPECT_EQ(line.segments[1].workers.start_coord.x, 3u);
        EXPECT_EQ(line.segments[1].workers.end_coord.x, 3u);
        EXPECT_EQ(line.segments[1].destination_count, 1u);
    }

    const auto weight_plan = placement.mcast_plan(CorePlacementMcastAxis::Column);
    ASSERT_EQ(weight_plan.lines.size(), 3u);
    EXPECT_EQ(weight_plan.total_destination_count, 6u);
    for (const auto& line : weight_plan.lines) {
        ASSERT_EQ(line.segments.size(), 1u);
        EXPECT_EQ(line.segments[0].workers.start_coord.y, 0u);
        EXPECT_EQ(line.segments[0].workers.end_coord.y, 2u);
        EXPECT_EQ(line.segments[0].destination_count, 2u);
    }
}

TEST(ConvCorePlacement, RowGapProducesContiguousActivationRowsAndSegmentedWeightColumns) {
    const auto placement = make_core_placement(kWorkerGrid, kShardGrid, 2);
    const auto activation_plan = placement.mcast_plan(CorePlacementMcastAxis::Row);
    ASSERT_EQ(activation_plan.lines.size(), 3u);
    EXPECT_EQ(activation_plan.total_destination_count, 6u);
    for (const auto& line : activation_plan.lines) {
        ASSERT_EQ(line.segments.size(), 1u);
        EXPECT_EQ(line.segments[0].workers.start_coord.x, 0u);
        EXPECT_EQ(line.segments[0].workers.end_coord.x, 2u);
        EXPECT_EQ(line.segments[0].destination_count, 2u);
    }

    const auto weight_plan = placement.mcast_plan(CorePlacementMcastAxis::Column);
    ASSERT_EQ(weight_plan.lines.size(), 3u);
    EXPECT_EQ(weight_plan.total_destination_count, 6u);
    for (const auto& line : weight_plan.lines) {
        ASSERT_EQ(line.segments.size(), 2u);
        EXPECT_EQ(line.segments[0].workers.start_coord.y, 0u);
        EXPECT_EQ(line.segments[0].workers.end_coord.y, 1u);
        EXPECT_EQ(line.segments[0].destination_count, 1u);
        EXPECT_EQ(line.segments[1].workers.start_coord.y, 3u);
        EXPECT_EQ(line.segments[1].workers.end_coord.y, 3u);
        EXPECT_EQ(line.segments[1].destination_count, 1u);
    }
}

TEST(ConvCorePlacement, RemoteOnlySegmentsExcludeEveryRotatingSender) {
    const auto placement = make_core_placement(kWorkerGrid, kShardGrid, std::nullopt, 2);
    for (uint32_t logical_y = 0; logical_y < kShardGrid.y; ++logical_y) {
        for (uint32_t logical_sender_x = 0; logical_sender_x < kShardGrid.x; ++logical_sender_x) {
            const auto sender = placement.worker_core({logical_sender_x, logical_y});
            const auto line = placement.remote_mcast_line_for_worker_sender(
                CorePlacementMcastAxis::Row, logical_y, sender);
            uint32_t destination_count = 0;
            for (const auto& segment : line.segments) {
                EXPECT_FALSE(segment.contains_sender);
                EXPECT_FALSE(segment.workers.contains(sender));
                EXPECT_EQ(segment.destination_count, segment.workers.size());
                destination_count += segment.destination_count;
            }
            EXPECT_EQ(destination_count, kShardGrid.x - 1);
        }
    }
}

TEST(ConvCorePlacement, RowGapWeightSegmentsExcludeTopRowSender) {
    const auto placement = make_core_placement(
        {5, 4}, {4, 3}, std::vector<uint32_t>{2}, std::vector<uint32_t>{});
    ASSERT_EQ(placement.worker_y, (std::vector<uint32_t>{0, 1, 3}));

    for (uint32_t logical_x = 0; logical_x < placement.shard_grid_size.x; ++logical_x) {
        const auto sender = placement.worker_core({logical_x, 0});
        const auto line = placement.remote_mcast_line_for_worker_sender(
            CorePlacementMcastAxis::Column, logical_x, sender);
        ASSERT_EQ(line.segments.size(), 2u);
        uint32_t destination_count = 0;
        for (const auto& segment : line.segments) {
            EXPECT_FALSE(segment.contains_sender);
            EXPECT_FALSE(segment.workers.contains(sender));
            EXPECT_EQ(segment.destination_count, segment.workers.size());
            destination_count += segment.destination_count;
        }
        EXPECT_EQ(destination_count, placement.shard_grid_size.y - 1);

        const auto runtime_args = serialize_mcast_segments(
            line, /*is_noc_0=*/true, [](CoreCoord worker) { return worker; });
        ASSERT_EQ(runtime_args.size(), 13u);
        EXPECT_EQ(runtime_args[0], 2u);
        EXPECT_EQ(runtime_args[1], logical_x);
        EXPECT_EQ(runtime_args[2], 1u);
        EXPECT_EQ(runtime_args[7], logical_x);
        EXPECT_EQ(runtime_args[8], 3u);
    }
}

TEST(ConvCorePlacement, SegmentedLinePlanTracksEveryRotatingSender) {
    const auto placement = make_core_placement(
        {6, 4}, {4, 3}, std::vector<uint32_t>{}, std::vector<uint32_t>{2});
    ASSERT_EQ(placement.worker_x, (std::vector<uint32_t>{0, 1, 3, 4}));

    for (uint32_t logical_sender = 0; logical_sender < placement.shard_grid_size.x; ++logical_sender) {
        const auto line =
            placement.mcast_line(CorePlacementMcastAxis::Row, /*logical_fixed=*/1, logical_sender);
        EXPECT_EQ(line.sender, placement.worker_core({logical_sender, 1}));
        ASSERT_EQ(line.segments.size(), 2u);

        uint32_t sender_segments = 0;
        uint32_t destinations = 0;
        for (const auto& segment : line.segments) {
            sender_segments += static_cast<uint32_t>(segment.contains_sender);
            destinations += segment.destination_count;
        }
        EXPECT_EQ(sender_segments, 1u);
        EXPECT_EQ(destinations, placement.shard_grid_size.x - 1);
    }
}

TEST(ConvCorePlacement, SingletonSenderSegmentIsPreservedWithoutZeroDestinationCommand) {
    const auto placement = make_core_placement(kWorkerGrid, kShardGrid, std::nullopt, 2);
    const auto line = placement.mcast_line(
        CorePlacementMcastAxis::Row, /*logical_fixed=*/0, /*logical_sender=*/2);

    ASSERT_EQ(line.segments.size(), 2u);
    EXPECT_FALSE(line.segments[0].contains_sender);
    EXPECT_EQ(line.segments[0].destination_count, 2u);
    EXPECT_TRUE(line.segments[1].contains_sender);
    EXPECT_EQ(line.segments[1].workers, CoreRange({3, 0}, {3, 0}));
    EXPECT_EQ(line.segments[1].destination_count, 0u);
}

TEST(ConvCorePlacement, SenderOutsideReceiverPlacementTargetsEveryReceiver) {
    const auto output_placement = make_core_placement(
        {6, 4}, {3, 3}, std::vector<uint32_t>{}, std::vector<uint32_t>{2});
    const auto line = output_placement.mcast_line_for_worker_sender(
        CorePlacementMcastAxis::Row, /*logical_fixed=*/1, /*sender=*/CoreCoord{4, 1});

    ASSERT_EQ(line.segments.size(), 2u);
    EXPECT_FALSE(line.segments[0].contains_sender);
    EXPECT_FALSE(line.segments[1].contains_sender);
    EXPECT_EQ(line.segments[0].destination_count, 2u);
    EXPECT_EQ(line.segments[1].destination_count, 1u);
}

TEST(ConvCorePlacement, OrderedWorkerGridMatchesTensorShardOrientation) {
    const auto placement = make_core_placement(kWorkerGrid, kShardGrid, 2, 2);

    const auto row_major_grid = placement.ordered_worker_cores(CorePlacementOrder::RowMajor);
    const auto row_major_cores = corerange_to_cores(row_major_grid, std::nullopt, /*row_wise=*/true);
    EXPECT_EQ(
        row_major_cores,
        (std::vector<CoreCoord>{
            {0, 0}, {1, 0}, {3, 0},
            {0, 1}, {1, 1}, {3, 1},
            {0, 3}, {1, 3}, {3, 3}}));

    const auto column_major_grid = placement.ordered_worker_cores(CorePlacementOrder::ColumnMajor);
    const auto column_major_cores = corerange_to_cores(column_major_grid, std::nullopt, /*row_wise=*/false);
    EXPECT_EQ(
        column_major_cores,
        (std::vector<CoreCoord>{
            {0, 0}, {0, 1}, {0, 3},
            {1, 0}, {1, 1}, {1, 3},
            {3, 0}, {3, 1}, {3, 3}}));
}

TEST(ConvCorePlacement, SerializesSegmentedRuntimeArgsForBothNocDirections) {
    const auto placement = make_core_placement(kWorkerGrid, kShardGrid, std::nullopt, 2);
    const auto line = placement.mcast_line(
        CorePlacementMcastAxis::Row, /*logical_fixed=*/1, /*logical_sender=*/2);
    const auto identity = [](CoreCoord worker) { return worker; };

    EXPECT_EQ(
        serialize_mcast_segments(line, /*is_noc_0=*/true, identity),
        (std::vector<uint32_t>{
            2,
            0, 1, 1, 1, 2, 0,
            3, 1, 3, 1, 0, 1}));
    EXPECT_EQ(
        serialize_mcast_segments(line, /*is_noc_0=*/false, identity),
        (std::vector<uint32_t>{
            2,
            1, 1, 0, 1, 2, 0,
            3, 1, 3, 1, 0, 1}));
}

TEST(ConvCorePlacement, AxisSegmentationPreservesLogicalOffsets) {
    const auto x_segments = segment_worker_axis({0, 1, 3, 4});
    ASSERT_EQ(x_segments.size(), 2u);
    expect_segment(x_segments[0], 0, 0, 2);
    expect_segment(x_segments[1], 2, 3, 2);

    const auto y_segments = segment_worker_axis({0, 2, 3});
    ASSERT_EQ(y_segments.size(), 2u);
    expect_segment(y_segments[0], 0, 0, 1);
    expect_segment(y_segments[1], 1, 2, 2);

    const auto singleton_segments = segment_worker_axis({4});
    ASSERT_EQ(singleton_segments.size(), 1u);
    expect_segment(singleton_segments[0], 0, 4, 1);
}

TEST(ConvCorePlacement, AxisSegmentationRejectsInvalidOrdering) {
    EXPECT_THROW(segment_worker_axis({}), std::invalid_argument);
    EXPECT_THROW(segment_worker_axis({0, 1, 1}), std::invalid_argument);
    EXPECT_THROW(segment_worker_axis({0, 3, 2}), std::invalid_argument);
}

TEST(ConvCorePlacement, NormalizesCartesianUserGridAndPreservesLogicalOrder) {
    const CoreRangeSet user_grid(std::set<CoreRange>{
        CoreRange({0, 0}, {1, 1}),
        CoreRange({3, 0}, {3, 1}),
        CoreRange({0, 3}, {1, 3}),
        CoreRange({3, 3}, {3, 3}),
    });
    const auto placement = make_core_placement_from_worker_grid(kWorkerGrid, user_grid);
    EXPECT_EQ(placement.shard_grid_size, kShardGrid);
    EXPECT_EQ(placement.worker_x, (std::vector<uint32_t>{0, 1, 3}));
    EXPECT_EQ(placement.worker_y, (std::vector<uint32_t>{0, 1, 3}));
    EXPECT_EQ(placement.worker_core({2, 2}), CoreCoord(3, 3));
    EXPECT_EQ(placement.worker_cores.ranges().size(), 4u);
    expect_round_trip(placement);
}

TEST(ConvCorePlacement, RejectsNonCartesianOrOutOfBoundsUserGrid) {
    const CoreRangeSet missing_cross_product_core(std::set<CoreRange>{
        CoreRange({0, 0}), CoreRange({1, 0}), CoreRange({0, 1})});
    EXPECT_THROW(
        make_core_placement_from_worker_grid(kWorkerGrid, missing_cross_product_core), std::invalid_argument);

    const CoreRangeSet out_of_bounds(std::set<CoreRange>{CoreRange({4, 3}, {5, 3})});
    EXPECT_THROW(make_core_placement_from_worker_grid(kWorkerGrid, out_of_bounds), std::invalid_argument);

    EXPECT_THROW(
        make_core_placement_from_worker_grid(kWorkerGrid, CoreRangeSet(std::set<CoreRange>{})),
        std::invalid_argument);
}

TEST(ConvCorePlacement, RejectsInvalidOrUnreplaceablePlacement) {
    EXPECT_THROW(make_core_placement({3, 3}, {3, 3}, 1), std::invalid_argument);
    EXPECT_THROW(make_core_placement(kWorkerGrid, kShardGrid, 4), std::invalid_argument);
    EXPECT_THROW(make_core_placement(kWorkerGrid, kShardGrid, std::nullopt, 5), std::invalid_argument);
    EXPECT_THROW(make_core_placement(kWorkerGrid, {0, 3}), std::invalid_argument);
    EXPECT_THROW(
        make_core_placement(kWorkerGrid, kShardGrid, std::vector<uint32_t>{}, std::vector<uint32_t>{0, 5}),
        std::invalid_argument);
    EXPECT_THROW(
        make_core_placement(kWorkerGrid, kShardGrid, std::vector<uint32_t>{}, std::vector<uint32_t>{0, 0}),
        std::invalid_argument);
    EXPECT_THROW(
        make_core_placement(kWorkerGrid, kShardGrid, std::vector<uint32_t>{0, 1}, std::vector<uint32_t>{}),
        std::invalid_argument);

    const auto placement = make_core_placement(kWorkerGrid, kShardGrid);
    EXPECT_THROW(placement.worker_core({3, 0}), std::out_of_range);
    EXPECT_FALSE(placement.shard_core({4, 3}).has_value());
}

TEST(ConvCorePlacement, MultipleExclusionsPreserveOrderAndStayInBounds) {
    const auto placement = make_core_placement(
        {7, 6}, {3, 3}, std::vector<uint32_t>{1, 4}, std::vector<uint32_t>{1, 3, 5});
    EXPECT_EQ(placement.worker_x, (std::vector<uint32_t>{0, 2, 4}));
    EXPECT_EQ(placement.worker_y, (std::vector<uint32_t>{0, 2, 3}));
    EXPECT_TRUE(placement.has_worker_gaps());
    EXPECT_EQ(placement.worker_cores.num_cores(), 9u);

    const auto x_segments = placement.x_segments();
    const auto y_segments = placement.y_segments();
    ASSERT_EQ(x_segments.size(), 3u);
    ASSERT_EQ(y_segments.size(), 2u);
    expect_segment(x_segments[0], 0, 0, 1);
    expect_segment(x_segments[1], 1, 2, 1);
    expect_segment(x_segments[2], 2, 4, 1);
    expect_segment(y_segments[0], 0, 0, 1);
    expect_segment(y_segments[1], 1, 2, 2);
    EXPECT_EQ(placement.rectangular_segments().size(), 6u);
    EXPECT_EQ(placement.worker_cores.ranges().size(), 6u);

    for (uint32_t y = 0; y < placement.shard_grid_size.y; ++y) {
        for (uint32_t x = 0; x < placement.shard_grid_size.x; ++x) {
            const ShardCoord shard{x, y};
            const CoreCoord worker = placement.worker_core(shard);
            EXPECT_LT(worker.x, placement.worker_grid_size.x);
            EXPECT_LT(worker.y, placement.worker_grid_size.y);
            ASSERT_TRUE(placement.shard_core(worker).has_value());
            EXPECT_EQ(*placement.shard_core(worker), shard);
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::unit_tests::conv_core_replacement
