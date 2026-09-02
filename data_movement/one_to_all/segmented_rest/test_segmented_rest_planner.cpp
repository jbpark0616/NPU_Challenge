// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "one_to_all/segmented_rest/segmented_rest_planner.hpp"

namespace tt::tt_metal::unit_tests::dm::core_to_all::segmented_rest {

namespace {

uint32_t axis_issue_count(const AxisPlan& axis_plan) {
    uint32_t total = 0;
    for (const auto& line : axis_plan.lines) {
        total += static_cast<uint32_t>(line.segments.size());
    }
    return total;
}

void expect_non_sender_coverage(const Plan& plan, CoreCoord grid_size) {
    for (uint32_t y = 0; y < grid_size.y; ++y) {
        for (uint32_t x = 0; x < grid_size.x; ++x) {
            const CoreCoord core{x, y};
            const uint32_t coverage = std::count_if(
                plan.segments.begin(), plan.segments.end(), [&core](const Segment& segment) {
                    return range_contains(segment.logical_range, core);
                });
            if (plan.active_cores.contains(core) && core == plan.sender) {
                EXPECT_LE(coverage, 1u) << "sender (" << core.x << "," << core.y << ")";
            } else if (plan.active_cores.contains(core)) {
                EXPECT_EQ(coverage, 1u) << "active receiver (" << core.x << "," << core.y << ")";
            } else {
                EXPECT_EQ(coverage, 0u) << "rest core (" << core.x << "," << core.y << ")";
            }
        }
    }
}

}  // namespace

TEST(SegmentedRestPlanner, FullGridUsesOneRectangle) {
    const auto plan = make_row_column_rest_plan({5, 4});
    EXPECT_EQ(plan.active_cores.num_cores(), 20u);
    EXPECT_EQ(plan.sender, CoreCoord(0, 0));
    EXPECT_EQ(plan.segments.size(), 1u);
    EXPECT_EQ(plan.destination_count, 19u);
    expect_non_sender_coverage(plan, {5, 4});
}

TEST(SegmentedRestPlanner, SingleRestLineUsesAtMostTwoRectangles) {
    for (uint32_t row = 0; row < 4; ++row) {
        const auto plan = make_row_column_rest_plan({5, 4}, row);
        EXPECT_EQ(plan.active_cores.num_cores(), 15u);
        EXPECT_LE(plan.segments.size(), 2u);
        EXPECT_EQ(plan.destination_count, 14u);
        expect_non_sender_coverage(plan, {5, 4});
    }
    for (uint32_t column = 0; column < 5; ++column) {
        const auto plan = make_row_column_rest_plan({5, 4}, std::nullopt, column);
        EXPECT_EQ(plan.active_cores.num_cores(), 16u);
        EXPECT_LE(plan.segments.size(), 2u);
        EXPECT_EQ(plan.destination_count, 15u);
        expect_non_sender_coverage(plan, {5, 4});
    }
}

TEST(SegmentedRestPlanner, CrossRestUsesAtMostFourRectangles) {
    for (uint32_t row = 0; row < 4; ++row) {
        for (uint32_t column = 0; column < 5; ++column) {
            const auto plan = make_row_column_rest_plan({5, 4}, row, column);
            EXPECT_TRUE(plan.active_cores.contains(plan.sender));
            EXPECT_EQ(plan.active_cores.num_cores(), 12u);
            EXPECT_LE(plan.segments.size(), 4u);
            EXPECT_EQ(plan.destination_count, 11u);
            expect_non_sender_coverage(plan, {5, 4});
        }
    }
}

TEST(SegmentedRestPlanner, ConvAxesSplitOnlyTheAffectedDirection) {
    const auto baseline_weight = make_conv_axis_plan({5, 4}, McastAxis::Column);
    const auto baseline_activation = make_conv_axis_plan({5, 4}, McastAxis::Row);
    EXPECT_EQ(baseline_weight.lines.size(), 5u);
    EXPECT_EQ(baseline_weight.total_destination_count, 15u);
    EXPECT_EQ(baseline_activation.lines.size(), 4u);
    EXPECT_EQ(baseline_activation.total_destination_count, 16u);

    const auto split_weight = make_conv_axis_plan({5, 4}, McastAxis::Column, 2);
    EXPECT_EQ(split_weight.lines.size(), 5u);
    EXPECT_EQ(split_weight.total_destination_count, 10u);
    for (const auto& line : split_weight.lines) {
        EXPECT_EQ(line.segments.size(), 2u);
    }

    const auto split_activation = make_conv_axis_plan({5, 4}, McastAxis::Row, std::nullopt, 2);
    EXPECT_EQ(split_activation.lines.size(), 4u);
    EXPECT_EQ(split_activation.total_destination_count, 12u);
    for (const auto& line : split_activation.lines) {
        EXPECT_EQ(line.segments.size(), 2u);
    }
}

TEST(SegmentedRestPlanner, CoreReplacementKeepsAThreeByThreeLogicalWorkloadConstant) {
    const auto baseline = make_core_replacement_plan({5, 4}, {3, 3});
    const auto row_gap = make_core_replacement_plan({5, 4}, {3, 3}, 2);
    const auto col_gap = make_core_replacement_plan({5, 4}, {3, 3}, std::nullopt, 2);
    const auto cross_gap = make_core_replacement_plan({5, 4}, {3, 3}, 2, 2);

    for (const auto* replacement : {&baseline, &row_gap, &col_gap, &cross_gap}) {
        EXPECT_EQ(replacement->active_cores.num_cores(), 9u);
        EXPECT_EQ(replacement->physical_x.size(), 3u);
        EXPECT_EQ(replacement->physical_y.size(), 3u);
    }
    EXPECT_EQ(row_gap.physical_y, (std::vector<uint32_t>{0, 1, 3}));
    EXPECT_EQ(col_gap.physical_x, (std::vector<uint32_t>{0, 1, 3}));

    const auto baseline_weight = make_conv_axis_plan(baseline, McastAxis::Column);
    const auto row_gap_weight = make_conv_axis_plan(row_gap, McastAxis::Column);
    const auto baseline_activation = make_conv_axis_plan(baseline, McastAxis::Row);
    const auto col_gap_activation = make_conv_axis_plan(col_gap, McastAxis::Row);
    const auto cross_weight = make_conv_axis_plan(cross_gap, McastAxis::Column);
    const auto cross_activation = make_conv_axis_plan(cross_gap, McastAxis::Row);

    EXPECT_EQ(baseline_weight.lines.size(), 3u);
    EXPECT_EQ(baseline_weight.total_destination_count, 6u);
    EXPECT_EQ(axis_issue_count(baseline_weight), 3u);
    EXPECT_EQ(row_gap_weight.total_destination_count, 6u);
    EXPECT_EQ(axis_issue_count(row_gap_weight), 6u);
    EXPECT_EQ(baseline_activation.lines.size(), 3u);
    EXPECT_EQ(baseline_activation.total_destination_count, 6u);
    EXPECT_EQ(axis_issue_count(baseline_activation), 3u);
    EXPECT_EQ(col_gap_activation.total_destination_count, 6u);
    EXPECT_EQ(axis_issue_count(col_gap_activation), 6u);
    EXPECT_EQ(cross_weight.total_destination_count, 6u);
    EXPECT_EQ(cross_activation.total_destination_count, 6u);
    EXPECT_EQ(axis_issue_count(cross_weight), 6u);
    EXPECT_EQ(axis_issue_count(cross_activation), 6u);
}

TEST(SegmentedRestPlanner, SingleSenderControlCoversNineCoresWithOneTwoOrFourSegments) {
    const auto baseline = make_single_sender_mcast_plan(make_core_replacement_plan({5, 4}, {3, 3}));
    const auto row_gap = make_single_sender_mcast_plan(make_core_replacement_plan({5, 4}, {3, 3}, 2));
    const auto col_gap = make_single_sender_mcast_plan(
        make_core_replacement_plan({5, 4}, {3, 3}, std::nullopt, 2));
    const auto cross_gap = make_single_sender_mcast_plan(make_core_replacement_plan({5, 4}, {3, 3}, 2, 2));

    for (const auto* plan : {&baseline, &row_gap, &col_gap, &cross_gap}) {
        EXPECT_EQ(plan->active_cores.num_cores(), 9u);
        EXPECT_EQ(plan->sender, CoreCoord(0, 0));
        EXPECT_EQ(plan->destination_count, 8u);
    }
    EXPECT_EQ(baseline.segments.size(), 1u);
    EXPECT_EQ(row_gap.segments.size(), 2u);
    EXPECT_EQ(col_gap.segments.size(), 2u);
    EXPECT_EQ(cross_gap.segments.size(), 4u);
}

TEST(SegmentedRestPlanner, CoreReplacementMovesTheSenderToTheReplacementBoundary) {
    const auto skip_top_row = make_single_sender_mcast_plan(
        make_core_replacement_plan({5, 4}, {3, 3}, 0));
    const auto skip_left_column = make_single_sender_mcast_plan(
        make_core_replacement_plan({5, 4}, {3, 3}, std::nullopt, 0));

    // The logical top-left sender follows the replacement mapping rather than
    // remaining on an excluded physical core.
    EXPECT_EQ(skip_top_row.sender, CoreCoord(0, 1));
    EXPECT_EQ(skip_left_column.sender, CoreCoord(1, 0));
    EXPECT_TRUE(skip_top_row.active_cores.contains(skip_top_row.sender));
    EXPECT_TRUE(skip_left_column.active_cores.contains(skip_left_column.sender));
    EXPECT_EQ(skip_top_row.segments.size(), 1u);
    EXPECT_EQ(skip_left_column.segments.size(), 1u);
}

}  // namespace tt::tt_metal::unit_tests::dm::core_to_all::segmented_rest
