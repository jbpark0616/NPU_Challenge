// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <tt_stl/small_vector.hpp>

#include "ttnn/device.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv_types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::conv::conv2d::core_replacement_test {

namespace {

constexpr uint32_t kBatch = 1;
constexpr uint32_t kInputChannels = 32;
constexpr uint32_t kOutputChannels = 96;
constexpr uint32_t kInputHeight = 24;
constexpr uint32_t kInputWidth = 24;
constexpr std::array<uint32_t, 2> kKernel{3, 3};
constexpr std::array<uint32_t, 2> kStride{1, 1};
constexpr std::array<uint32_t, 2> kPadding{1, 1};
constexpr std::array<uint32_t, 2> kDilation{1, 1};

double pearson_correlation(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.empty() || lhs.size() != rhs.size()) {
        return -1.0;
    }
    double lhs_sum = 0.0;
    double rhs_sum = 0.0;
    double lhs_squared_sum = 0.0;
    double rhs_squared_sum = 0.0;
    double product_sum = 0.0;
    for (size_t index = 0; index < lhs.size(); ++index) {
        const double lhs_value = lhs[index];
        const double rhs_value = rhs[index];
        lhs_sum += lhs_value;
        rhs_sum += rhs_value;
        lhs_squared_sum += lhs_value * lhs_value;
        rhs_squared_sum += rhs_value * rhs_value;
        product_sum += lhs_value * rhs_value;
    }
    const double count = static_cast<double>(lhs.size());
    const double lhs_variance = count * lhs_squared_sum - lhs_sum * lhs_sum;
    const double rhs_variance = count * rhs_squared_sum - rhs_sum * rhs_sum;
    const double denominator = std::sqrt(lhs_variance * rhs_variance);
    return denominator == 0.0 ? -1.0 : (count * product_sum - lhs_sum * rhs_sum) / denominator;
}

std::vector<float> reference_conv2d(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    uint32_t output_height,
    uint32_t output_width) {
    std::vector<float> output(kBatch * kOutputChannels * output_height * output_width, 0.0F);
    for (uint32_t n = 0; n < kBatch; ++n) {
        for (uint32_t co = 0; co < kOutputChannels; ++co) {
            for (uint32_t oh = 0; oh < output_height; ++oh) {
                for (uint32_t ow = 0; ow < output_width; ++ow) {
                    float sum = 0.0F;
                    for (uint32_t ci = 0; ci < kInputChannels; ++ci) {
                        for (uint32_t kh = 0; kh < kKernel[0]; ++kh) {
                            for (uint32_t kw = 0; kw < kKernel[1]; ++kw) {
                                const int32_t ih = static_cast<int32_t>(oh * kStride[0] + kh) - kPadding[0];
                                const int32_t iw = static_cast<int32_t>(ow * kStride[1] + kw) - kPadding[1];
                                if (ih < 0 || iw < 0 || ih >= static_cast<int32_t>(kInputHeight) ||
                                    iw >= static_cast<int32_t>(kInputWidth)) {
                                    continue;
                                }
                                const size_t input_index =
                                    ((n * kInputChannels + ci) * kInputHeight + static_cast<uint32_t>(ih)) *
                                        kInputWidth +
                                    static_cast<uint32_t>(iw);
                                const size_t weight_index =
                                    ((co * kInputChannels + ci) * kKernel[0] + kh) * kKernel[1] + kw;
                                sum += input[input_index] * weights[weight_index];
                            }
                        }
                    }
                    const size_t output_index =
                        ((n * kOutputChannels + co) * output_height + oh) * output_width + ow;
                    output[output_index] = sum;
                }
            }
        }
    }
    return output;
}

std::vector<float> nhwc_to_nchw(
    const std::vector<float>& input, uint32_t height, uint32_t width, uint32_t channels) {
    const uint32_t spatial_size = height * width;
    if (input.size() != kBatch * spatial_size * channels) {
        return {};
    }
    std::vector<float> output(input.size());
    for (uint32_t n = 0; n < kBatch; ++n) {
        for (uint32_t h = 0; h < height; ++h) {
            for (uint32_t w = 0; w < width; ++w) {
                for (uint32_t c = 0; c < channels; ++c) {
                    const size_t nhwc_index = ((n * height + h) * width + w) * channels + c;
                    const size_t nchw_index = ((n * channels + c) * height + h) * width + w;
                    output[nchw_index] = input[nhwc_index];
                }
            }
        }
    }
    return output;
}

ttnn::operations::conv::conv2d::Conv2dConfig automatic_conv_config() {
    ttnn::operations::conv::conv2d::Conv2dConfig config;
    config.weights_dtype = tt::tt_metal::DataType::BFLOAT16;
    config.output_layout = tt::tt_metal::Layout::TILE;
    return config;
}

}  // namespace

TEST(ConvCoreReplacementSafety, PreflightAllowsAtMostFourCartesianRectangles) {
    const tt::tt_metal::CoreRangeSet origin_rectangle(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({0, 0}, {2, 2})});
    EXPECT_NO_THROW(ttnn::operations::conv::validate_stock_block_sharded_worker_grid(
        origin_rectangle, {5, 4}, "test-origin"));

    const tt::tt_metal::CoreRangeSet translated_rectangle(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({1, 0}, {3, 2})});
    const auto translated_placement =
        ttnn::operations::conv::validate_stock_block_sharded_worker_grid(
            translated_rectangle, {5, 4}, "test-translated");
    EXPECT_EQ(translated_placement.worker_x, (std::vector<uint32_t>{1, 2, 3}));
    EXPECT_EQ(translated_placement.worker_y, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(translated_placement.worker_core({0, 0}), tt::tt_metal::CoreCoord(1, 0));
    EXPECT_FALSE(translated_placement.has_worker_gaps());

    // CoreRangeSet representation is not placement semantics: a dense rectangle
    // remains stock-compatible even when supplied as multiple adjacent ranges.
    const tt::tt_metal::CoreRangeSet split_but_dense_rectangle(std::set<tt::tt_metal::CoreRange>{
        tt::tt_metal::CoreRange({1, 0}, {2, 2}), tt::tt_metal::CoreRange({3, 0}, {3, 2})});
    const auto dense_placement = ttnn::operations::conv::validate_stock_block_sharded_worker_grid(
        split_but_dense_rectangle, {5, 4}, "test-split-dense");
    EXPECT_EQ(dense_placement.worker_x, translated_placement.worker_x);
    EXPECT_EQ(dense_placement.worker_y, translated_placement.worker_y);
    EXPECT_FALSE(dense_placement.has_worker_gaps());

    std::set<tt::tt_metal::CoreRange> gapped_ranges;
    for (uint32_t y : {0u, 1u, 2u}) {
        for (uint32_t x : {0u, 1u, 3u}) {
            gapped_ranges.emplace(tt::tt_metal::CoreCoord{x, y});
        }
    }
    const tt::tt_metal::CoreRangeSet gapped_grid(std::move(gapped_ranges));
    EXPECT_ANY_THROW(ttnn::operations::conv::validate_stock_block_sharded_worker_grid(
        gapped_grid, {5, 4}, "test-gapped"));
    const auto segmented_placement = ttnn::operations::conv::validate_stock_block_sharded_worker_grid(
        gapped_grid,
        {5, 4},
        "test-enabled-column-gap",
        ttnn::operations::conv::BlockShardedWorkerGridPolicy::RowMajorUpToFourRectangles);
    EXPECT_EQ(segmented_placement.worker_x, (std::vector<uint32_t>{0, 1, 3}));
    EXPECT_EQ(segmented_placement.worker_y, (std::vector<uint32_t>{0, 1, 2}));

    std::set<tt::tt_metal::CoreRange> row_gapped_ranges;
    for (uint32_t y : {0u, 1u, 3u}) {
        for (uint32_t x : {0u, 1u, 2u}) {
            row_gapped_ranges.emplace(tt::tt_metal::CoreCoord{x, y});
        }
    }
    const auto row_segmented_placement = ttnn::operations::conv::validate_stock_block_sharded_worker_grid(
        tt::tt_metal::CoreRangeSet(std::move(row_gapped_ranges)),
        {5, 4},
        "test-enabled-row-gap",
        ttnn::operations::conv::BlockShardedWorkerGridPolicy::RowMajorUpToFourRectangles);
    EXPECT_EQ(row_segmented_placement.worker_x, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(row_segmented_placement.worker_y, (std::vector<uint32_t>{0, 1, 3}));

    std::set<tt::tt_metal::CoreRange> two_column_gap_ranges;
    for (uint32_t y : {0u, 1u, 2u}) {
        for (uint32_t x : {0u, 2u, 4u}) {
            two_column_gap_ranges.emplace(tt::tt_metal::CoreCoord{x, y});
        }
    }
    EXPECT_ANY_THROW(ttnn::operations::conv::validate_stock_block_sharded_worker_grid(
        tt::tt_metal::CoreRangeSet(std::move(two_column_gap_ranges)),
        {5, 4},
        "test-two-column-gaps",
        ttnn::operations::conv::BlockShardedWorkerGridPolicy::RowMajorUpToFourRectangles));

    std::set<tt::tt_metal::CoreRange> cross_gapped_ranges;
    for (uint32_t y : {0u, 1u, 3u}) {
        for (uint32_t x : {0u, 1u, 3u}) {
            cross_gapped_ranges.emplace(tt::tt_metal::CoreCoord{x, y});
        }
    }
    const auto cross_segmented_placement = ttnn::operations::conv::validate_stock_block_sharded_worker_grid(
        tt::tt_metal::CoreRangeSet(std::move(cross_gapped_ranges)),
        {5, 4},
        "test-four-rectangles",
        ttnn::operations::conv::BlockShardedWorkerGridPolicy::RowMajorUpToFourRectangles);
    EXPECT_EQ(cross_segmented_placement.x_segments().size(), 2u);
    EXPECT_EQ(cross_segmented_placement.y_segments().size(), 2u);

    std::set<tt::tt_metal::CoreRange> translated_column_gap_ranges;
    for (uint32_t y : {0u, 1u, 2u}) {
        for (uint32_t x : {1u, 2u, 4u}) {
            translated_column_gap_ranges.emplace(tt::tt_metal::CoreCoord{x, y});
        }
    }
    const auto translated_segmented_placement = ttnn::operations::conv::validate_stock_block_sharded_worker_grid(
        tt::tt_metal::CoreRangeSet(std::move(translated_column_gap_ranges)),
        {5, 4},
        "test-translated-column-gap",
        ttnn::operations::conv::BlockShardedWorkerGridPolicy::RowMajorUpToFourRectangles);
    EXPECT_EQ(translated_segmented_placement.x_segments().size(), 2u);

    const tt::tt_metal::CoreRangeSet out_of_bounds(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({4, 3}, {5, 3})});
    EXPECT_ANY_THROW(ttnn::operations::conv::validate_stock_block_sharded_worker_grid(
        out_of_bounds, {5, 4}, "test-out-of-bounds"));
}

TEST(ConvCoreReplacementSafety, ExplicitExclusionsAllowBoundaryShiftAndFourRectangles) {
    constexpr tt::tt_metal::CoreCoord compute_grid{5, 5};
    const ttnn::operations::sliding_window::ParallelConfig stock_input{
        .grid = tt::tt_metal::CoreRangeSet(
            std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({0, 0}, {3, 3})}),
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };

    const auto selected = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{},
        /*excluded_worker_columns=*/{2});
    const auto selected_placement = ttnn::operations::conv::make_core_placement_from_worker_grid(
        compute_grid, selected.grid);
    EXPECT_EQ(selected_placement.worker_x, (std::vector<uint32_t>{0, 1, 3, 4}));
    EXPECT_EQ(selected_placement.worker_y, (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_TRUE(selected_placement.has_worker_gaps());

    const auto row_selected = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{2},
        /*excluded_worker_columns=*/{});
    const auto row_selected_placement = ttnn::operations::conv::make_core_placement_from_worker_grid(
        compute_grid, row_selected.grid);
    EXPECT_EQ(row_selected_placement.worker_x, (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(row_selected_placement.worker_y, (std::vector<uint32_t>{0, 1, 3, 4}));
    EXPECT_TRUE(row_selected_placement.has_worker_gaps());

    const ttnn::operations::sliding_window::ParallelConfig stock_three_by_three{
        .grid = tt::tt_metal::CoreRangeSet(
            std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({0, 0}, {2, 2})}),
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };
    const auto cross_selected = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_three_by_three,
        compute_grid,
        /*excluded_worker_rows=*/{1},
        /*excluded_worker_columns=*/{1});
    const auto cross_placement = ttnn::operations::conv::make_core_placement_from_worker_grid(
        compute_grid, cross_selected.grid);
    EXPECT_EQ(cross_placement.worker_x, (std::vector<uint32_t>{0, 2, 3}));
    EXPECT_EQ(cross_placement.worker_y, (std::vector<uint32_t>{0, 2, 3}));
    EXPECT_EQ(cross_placement.x_segments().size(), 2u);
    EXPECT_EQ(cross_placement.y_segments().size(), 2u);

    const auto boundary_shift = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_three_by_three,
        compute_grid,
        /*excluded_worker_rows=*/{0, 1},
        /*excluded_worker_columns=*/{});
    const auto boundary_shift_placement = ttnn::operations::conv::make_core_placement_from_worker_grid(
        compute_grid, boundary_shift.grid);
    EXPECT_EQ(boundary_shift_placement.worker_y, (std::vector<uint32_t>{2, 3, 4}));
    EXPECT_EQ(boundary_shift_placement.y_segments().size(), 1u);

    EXPECT_ANY_THROW(ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_three_by_three,
        compute_grid,
        /*excluded_worker_rows=*/{},
        /*excluded_worker_columns=*/{1, 3}));
    EXPECT_ANY_THROW(ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input,
        compute_grid,
        // A 4-high logical placement cannot retain four workers after two exclusions on a 5-high grid.
        /*excluded_worker_rows=*/{1, 2},
        /*excluded_worker_columns=*/{}));
    EXPECT_ANY_THROW(ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{},
        /*excluded_worker_columns=*/{1, 2}));
    const auto shifted_input = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input, compute_grid, /*excluded_worker_rows=*/{}, /*excluded_worker_columns=*/{0});
    const auto shifted_input_placement = ttnn::operations::conv::make_core_placement_from_worker_grid(
        compute_grid, shifted_input.grid);
    EXPECT_EQ(shifted_input_placement.worker_x, (std::vector<uint32_t>{1, 2, 3, 4}));
    EXPECT_EQ(shifted_input_placement.x_segments().size(), 1u);
}

TEST(ConvCoreReplacementSafety, HeightShardedExclusionsPreserveLogicalWorkerOrderAndLifetime) {
    constexpr tt::tt_metal::CoreCoord compute_grid{5, 4};
    const ttnn::operations::sliding_window::ParallelConfig stock_input{
        .grid = tt::tt_metal::num_cores_to_corerangeset(8, compute_grid, true),
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };

    const auto unchanged = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{},
        /*excluded_worker_columns=*/{});
    EXPECT_EQ(unchanged.grid, stock_input.grid);
    const auto inactive = ttnn::operations::conv::plan_conv_core_remapping(
        unchanged, stock_input, compute_grid, /*remapping_active=*/false);
    EXPECT_EQ(inactive.action, ttnn::operations::conv::ConvCoreRemappingAction::NotActive);
    EXPECT_EQ(inactive.input_parallel_config.grid, stock_input.grid);
    EXPECT_EQ(inactive.output_parallel_config.grid, stock_input.grid);

    const tt::tt_metal::CoreRangeSet production_fourteen_core_grid(
        std::vector<tt::tt_metal::CoreRange>{
            tt::tt_metal::CoreRange({0, 0}, {4, 1}),
            tt::tt_metal::CoreRange({0, 2}, {3, 2}),
        });
    EXPECT_FALSE(ttnn::operations::conv::is_remapped_height_sharded_worker_grid(
        production_fourteen_core_grid));
    const ttnn::operations::sliding_window::ParallelConfig production_stock_input{
        .grid = production_fourteen_core_grid,
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };
    const auto production_segmented = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        production_stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{},
        /*excluded_worker_columns=*/{2});
    EXPECT_EQ(
        ttnn::operations::conv::ordered_conv_worker_cores(production_segmented.grid),
        (std::vector<tt::tt_metal::CoreCoord>{
            {0, 0}, {1, 0}, {3, 0}, {4, 0},
            {0, 1}, {1, 1}, {3, 1}, {4, 1},
            {0, 2}, {1, 2}, {3, 2}, {4, 2},
            {0, 3}, {1, 3}}));
    EXPECT_TRUE(ttnn::operations::conv::is_remapped_height_sharded_worker_grid(
        production_segmented.grid));
    const tt::tt_metal::CoreRangeSet production_sender(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({0, 0})});
    const auto production_receivers = production_segmented.grid.subtract(production_sender);
    const auto production_mcast_line = ttnn::operations::conv::make_remote_rectangle_mcast_line_for_core_set(
        production_receivers, {0, 0}, compute_grid);
    ASSERT_EQ(production_mcast_line.segments.size(), 2u);
    EXPECT_EQ(
        production_mcast_line.segments[0].workers,
        tt::tt_metal::CoreRange({0, 0}, {1, 3}));
    EXPECT_EQ(
        production_mcast_line.segments[1].workers,
        tt::tt_metal::CoreRange({3, 0}, {4, 2}));
    EXPECT_TRUE(production_mcast_line.segments[0].contains_sender);
    EXPECT_EQ(production_mcast_line.segments[0].destination_count, 7u);
    EXPECT_FALSE(production_mcast_line.segments[1].contains_sender);
    EXPECT_EQ(production_mcast_line.segments[1].destination_count, 6u);
    uint32_t production_destination_count = 0;
    for (const auto& segment : production_mcast_line.segments) {
        production_destination_count += segment.destination_count;
    }
    EXPECT_EQ(production_destination_count, 13u);

    const auto production_boundary_translated =
        ttnn::operations::conv::select_conv_core_remapping_input_placement(
            production_stock_input,
            compute_grid,
            /*excluded_worker_rows=*/{},
            /*excluded_worker_columns=*/{0});
    EXPECT_EQ(
        ttnn::operations::conv::ordered_conv_worker_cores(production_boundary_translated.grid),
        (std::vector<tt::tt_metal::CoreCoord>{
            {1, 0}, {2, 0}, {3, 0}, {4, 0},
            {1, 1}, {2, 1}, {3, 1}, {4, 1},
            {1, 2}, {2, 2}, {3, 2}, {4, 2},
            {1, 3}, {2, 3}}));
    EXPECT_EQ(
        ttnn::operations::conv::make_stock_height_sharded_worker_grid(production_boundary_translated.grid),
        production_stock_input.grid);
    const tt::tt_metal::CoreCoord production_boundary_sender{1, 0};
    const tt::tt_metal::CoreRangeSet production_boundary_sender_set(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange(production_boundary_sender)});
    const auto production_boundary_receivers =
        production_boundary_translated.grid.subtract(production_boundary_sender_set);
    const auto production_boundary_mcast_line =
        ttnn::operations::conv::make_remote_rectangle_mcast_line_for_core_set(
            production_boundary_receivers,
            production_boundary_sender,
            compute_grid);
    ASSERT_EQ(production_boundary_mcast_line.segments.size(), 2u);
    EXPECT_EQ(
        production_boundary_mcast_line.segments[0].workers,
        tt::tt_metal::CoreRange({1, 0}, {4, 2}));
    EXPECT_EQ(
        production_boundary_mcast_line.segments[1].workers,
        tt::tt_metal::CoreRange({1, 3}, {2, 3}));
    EXPECT_TRUE(production_boundary_mcast_line.segments[0].contains_sender);
    EXPECT_EQ(production_boundary_mcast_line.segments[0].destination_count, 11u);
    EXPECT_FALSE(production_boundary_mcast_line.segments[1].contains_sender);
    EXPECT_EQ(production_boundary_mcast_line.segments[1].destination_count, 2u);
    const auto boundary_identity_worker_to_noc = [](tt::tt_metal::CoreCoord worker) { return worker; };
    EXPECT_EQ(
        ttnn::operations::conv::serialize_mcast_segments(
            production_boundary_mcast_line, true, boundary_identity_worker_to_noc),
        (std::vector<uint32_t>{2, 1, 0, 4, 2, 11, 1, 1, 3, 2, 3, 2, 0}));
    EXPECT_EQ(
        ttnn::operations::conv::serialize_mcast_segments(
            production_boundary_mcast_line, false, boundary_identity_worker_to_noc),
        (std::vector<uint32_t>{2, 4, 2, 1, 0, 11, 1, 2, 3, 1, 3, 2, 0}));
    const auto production_boundary_propagated = ttnn::operations::conv::plan_conv_core_remapping(
        production_boundary_translated,
        production_stock_input,
        compute_grid,
        /*remapping_active=*/true,
        /*current_placement_is_executable=*/true,
        ttnn::operations::conv::ConvCorePlacementProvenance{production_stock_input});
    EXPECT_EQ(
        production_boundary_propagated.action,
        ttnn::operations::conv::ConvCoreRemappingAction::Propagate);
    EXPECT_EQ(
        production_boundary_propagated.output_parallel_config.grid,
        production_boundary_translated.grid);
    auto invalid_boundary_provenance = production_stock_input;
    invalid_boundary_provenance.grid =
        tt::tt_metal::num_cores_to_corerangeset(13, compute_grid, true);
    EXPECT_THROW(
        ttnn::operations::conv::plan_conv_core_remapping(
            production_boundary_translated,
            production_stock_input,
            compute_grid,
            /*remapping_active=*/true,
            /*current_placement_is_executable=*/true,
            ttnn::operations::conv::ConvCorePlacementProvenance{invalid_boundary_provenance}),
        std::invalid_argument);

    const auto production_row_segmented = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        production_stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{2},
        /*excluded_worker_columns=*/{});
    EXPECT_EQ(
        ttnn::operations::conv::ordered_conv_worker_cores(production_row_segmented.grid),
        (std::vector<tt::tt_metal::CoreCoord>{
            {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0},
            {0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1},
            {0, 3}, {1, 3}, {2, 3}, {3, 3}}));
    EXPECT_TRUE(ttnn::operations::conv::is_remapped_height_sharded_worker_grid(
        production_row_segmented.grid));
    const auto production_row_receivers = production_row_segmented.grid.subtract(production_sender);
    const auto production_row_mcast_line = ttnn::operations::conv::make_remote_rectangle_mcast_line_for_core_set(
        production_row_receivers, {0, 0}, compute_grid);
    ASSERT_EQ(production_row_mcast_line.segments.size(), 2u);
    EXPECT_EQ(
        production_row_mcast_line.segments[0].workers,
        tt::tt_metal::CoreRange({0, 0}, {4, 1}));
    EXPECT_EQ(
        production_row_mcast_line.segments[1].workers,
        tt::tt_metal::CoreRange({0, 3}, {3, 3}));
    EXPECT_TRUE(production_row_mcast_line.segments[0].contains_sender);
    EXPECT_EQ(production_row_mcast_line.segments[0].destination_count, 9u);
    EXPECT_FALSE(production_row_mcast_line.segments[1].contains_sender);
    EXPECT_EQ(production_row_mcast_line.segments[1].destination_count, 4u);
    uint32_t production_row_destination_count = 0;
    for (const auto& segment : production_row_mcast_line.segments) {
        production_row_destination_count += segment.destination_count;
    }
    EXPECT_EQ(production_row_destination_count, 13u);
    const auto identity_worker_to_noc = [](tt::tt_metal::CoreCoord worker) { return worker; };
    const auto production_row_noc0_args = ttnn::operations::conv::serialize_mcast_segments(
        production_row_mcast_line, true, identity_worker_to_noc);
    const auto production_row_noc1_args = ttnn::operations::conv::serialize_mcast_segments(
        production_row_mcast_line, false, identity_worker_to_noc);
    EXPECT_EQ(
        production_row_noc0_args,
        (std::vector<uint32_t>{2, 0, 0, 4, 1, 9, 1, 0, 3, 3, 3, 4, 0}));
    EXPECT_EQ(
        production_row_noc1_args,
        (std::vector<uint32_t>{2, 4, 1, 0, 0, 9, 1, 3, 3, 0, 3, 4, 0}));
    const auto production_row_propagated = ttnn::operations::conv::plan_conv_core_remapping(
        production_row_segmented,
        production_stock_input,
        compute_grid,
        /*remapping_active=*/true);
    EXPECT_EQ(
        production_row_propagated.action,
        ttnn::operations::conv::ConvCoreRemappingAction::Propagate);
    EXPECT_EQ(production_row_propagated.output_parallel_config.grid, production_row_segmented.grid);

    const tt::tt_metal::CoreRangeSet twelve_core_grid(
        std::vector<tt::tt_metal::CoreRange>{
            tt::tt_metal::CoreRange({0, 0}, {4, 1}),
            tt::tt_metal::CoreRange({0, 2}, {1, 2}),
        });
    const ttnn::operations::sliding_window::ParallelConfig twelve_core_stock_input{
        .grid = twelve_core_grid,
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };
    const auto four_rectangle_segmented = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        twelve_core_stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{2},
        /*excluded_worker_columns=*/{2});
    EXPECT_EQ(
        ttnn::operations::conv::ordered_conv_worker_cores(four_rectangle_segmented.grid),
        (std::vector<tt::tt_metal::CoreCoord>{
            {0, 0}, {1, 0}, {3, 0}, {4, 0},
            {0, 1}, {1, 1}, {3, 1}, {4, 1},
            {0, 3}, {1, 3}, {3, 3}, {4, 3}}));
    const auto four_rectangle_receivers = four_rectangle_segmented.grid.subtract(production_sender);
    const auto four_rectangle_mcast_line = ttnn::operations::conv::make_remote_rectangle_mcast_line_for_core_set(
        four_rectangle_receivers, {0, 0}, compute_grid);
    ASSERT_EQ(four_rectangle_mcast_line.segments.size(), 4u);
    EXPECT_EQ(four_rectangle_mcast_line.segments[0].workers, tt::tt_metal::CoreRange({0, 0}, {1, 1}));
    EXPECT_EQ(four_rectangle_mcast_line.segments[1].workers, tt::tt_metal::CoreRange({3, 0}, {4, 1}));
    EXPECT_EQ(four_rectangle_mcast_line.segments[2].workers, tt::tt_metal::CoreRange({0, 3}, {1, 3}));
    EXPECT_EQ(four_rectangle_mcast_line.segments[3].workers, tt::tt_metal::CoreRange({3, 3}, {4, 3}));
    EXPECT_TRUE(four_rectangle_mcast_line.segments[0].contains_sender);
    EXPECT_EQ(four_rectangle_mcast_line.segments[0].destination_count, 3u);
    EXPECT_EQ(four_rectangle_mcast_line.segments[1].destination_count, 4u);
    EXPECT_EQ(four_rectangle_mcast_line.segments[2].destination_count, 2u);
    EXPECT_EQ(four_rectangle_mcast_line.segments[3].destination_count, 2u);
    const auto four_rectangle_noc0_args = ttnn::operations::conv::serialize_mcast_segments(
        four_rectangle_mcast_line, true, identity_worker_to_noc);
    const auto four_rectangle_noc1_args = ttnn::operations::conv::serialize_mcast_segments(
        four_rectangle_mcast_line, false, identity_worker_to_noc);
    EXPECT_EQ(
        four_rectangle_noc0_args,
        (std::vector<uint32_t>{
            4,
            0, 0, 1, 1, 3, 1,
            3, 0, 4, 1, 4, 0,
            0, 3, 1, 3, 2, 0,
            3, 3, 4, 3, 2, 0}));
    EXPECT_EQ(
        four_rectangle_noc1_args,
        (std::vector<uint32_t>{
            4,
            1, 1, 0, 0, 3, 1,
            4, 1, 3, 0, 4, 0,
            1, 3, 0, 3, 2, 0,
            4, 3, 3, 3, 2, 0}));
    const auto four_rectangle_propagated = ttnn::operations::conv::plan_conv_core_remapping(
        four_rectangle_segmented,
        twelve_core_stock_input,
        compute_grid,
        /*remapping_active=*/true);
    EXPECT_EQ(
        four_rectangle_propagated.action,
        ttnn::operations::conv::ConvCoreRemappingAction::Propagate);
    EXPECT_EQ(four_rectangle_propagated.output_parallel_config.grid, four_rectangle_segmented.grid);
    const ttnn::operations::sliding_window::ParallelConfig eleven_core_stock_output{
        .grid = tt::tt_metal::CoreRangeSet(
            std::vector<tt::tt_metal::CoreRange>{
                tt::tt_metal::CoreRange({0, 0}, {4, 1}),
                tt::tt_metal::CoreRange({0, 2}, {0, 2}),
            }),
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };
    const auto four_rectangle_exit = ttnn::operations::conv::plan_conv_core_remapping(
        four_rectangle_segmented,
        eleven_core_stock_output,
        compute_grid,
        /*remapping_active=*/true,
        /*current_placement_is_executable=*/false,
        ttnn::operations::conv::ConvCorePlacementProvenance{twelve_core_stock_input});
    EXPECT_EQ(
        four_rectangle_exit.action,
        ttnn::operations::conv::ConvCoreRemappingAction::ExitToStock);
    EXPECT_EQ(four_rectangle_exit.input_parallel_config.grid, twelve_core_stock_input.grid);
    EXPECT_EQ(four_rectangle_exit.output_parallel_config.grid, eleven_core_stock_output.grid);

    // The stock input is provenance, not a shape inferred from the remapped bounding box or the next output grid.
    // This is the boundary-translation case that previously confused x1..4 with a four-wide stock topology.
    const auto boundary_exit = ttnn::operations::conv::plan_conv_core_remapping(
        production_boundary_translated,
        eleven_core_stock_output,
        compute_grid,
        /*remapping_active=*/true,
        /*current_placement_is_executable=*/false,
        ttnn::operations::conv::ConvCorePlacementProvenance{production_stock_input});
    EXPECT_EQ(boundary_exit.action, ttnn::operations::conv::ConvCoreRemappingAction::ExitToStock);
    EXPECT_EQ(boundary_exit.input_parallel_config.grid, production_stock_input.grid);
    EXPECT_EQ(boundary_exit.output_parallel_config.grid, eleven_core_stock_output.grid);

    const auto production_exit = ttnn::operations::conv::plan_conv_core_remapping(
        production_segmented,
        production_stock_input,
        compute_grid,
        /*remapping_active=*/true,
        /*current_placement_is_executable=*/false);
    EXPECT_EQ(production_exit.action, ttnn::operations::conv::ConvCoreRemappingAction::ExitToStock);
    EXPECT_EQ(production_exit.input_parallel_config.grid, production_stock_input.grid);
    EXPECT_EQ(production_exit.output_parallel_config.grid, production_stock_input.grid);

    const auto selected = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{},
        /*excluded_worker_columns=*/{2});
    EXPECT_TRUE(ttnn::operations::conv::is_remapped_height_sharded_worker_grid(selected.grid));
    EXPECT_EQ(
        ttnn::operations::conv::ordered_conv_worker_cores(selected.grid),
        (std::vector<tt::tt_metal::CoreCoord>{{0, 0}, {1, 0}, {3, 0}, {4, 0}, {0, 1}, {1, 1}, {3, 1}, {4, 1}}));

    const auto row_shifted = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{0},
        /*excluded_worker_columns=*/{});
    const auto row_shifted_workers = ttnn::operations::conv::ordered_conv_worker_cores(row_shifted.grid);
    ASSERT_EQ(row_shifted_workers.size(), 8u);
    EXPECT_EQ(row_shifted_workers.front(), tt::tt_metal::CoreCoord(0, 1));
    EXPECT_EQ(row_shifted_workers.back(), tt::tt_metal::CoreCoord(2, 2));

    const auto propagated = ttnn::operations::conv::plan_conv_core_remapping(
        selected, stock_input, compute_grid, /*remapping_active=*/true);
    EXPECT_EQ(propagated.action, ttnn::operations::conv::ConvCoreRemappingAction::Propagate);
    EXPECT_EQ(propagated.output_parallel_config.grid, selected.grid);
    EXPECT_FALSE(propagated.requires_input_reshard());

    const auto exited = ttnn::operations::conv::plan_conv_core_remapping(
        selected,
        stock_input,
        compute_grid,
        /*remapping_active=*/true,
        /*current_placement_is_executable=*/false);
    EXPECT_EQ(exited.action, ttnn::operations::conv::ConvCoreRemappingAction::ExitToStock);
    EXPECT_TRUE(exited.requires_input_reshard());
    EXPECT_EQ(exited.input_parallel_config.grid, stock_input.grid);
    EXPECT_EQ(exited.output_parallel_config.grid, stock_input.grid);

    EXPECT_ANY_THROW(ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{0, 1, 2, 3},
        /*excluded_worker_columns=*/{}));
    EXPECT_ANY_THROW(ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{},
        /*excluded_worker_columns=*/{2, 2}));
    EXPECT_ANY_THROW(ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{},
        /*excluded_worker_columns=*/{compute_grid.x}));
}

TEST(ConvCoreReplacementSafety, HeightShardedMulticastSegmentsExactlyCoverRemoteWorkers) {
    constexpr tt::tt_metal::CoreCoord compute_grid{13, 10};
    constexpr uint32_t num_cores = 14;
    const ttnn::operations::sliding_window::ParallelConfig stock_input{
        .grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_grid, true),
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };
    const auto selected = ttnn::operations::conv::select_conv_core_remapping_input_placement(
        stock_input,
        compute_grid,
        /*excluded_worker_rows=*/{},
        /*excluded_worker_columns=*/{2});
    const auto workers = ttnn::operations::conv::ordered_conv_worker_cores(selected.grid);
    ASSERT_EQ(workers.size(), num_cores);
    EXPECT_EQ(workers.front(), tt::tt_metal::CoreCoord(0, 0));
    EXPECT_EQ(workers.back(), tt::tt_metal::CoreCoord(1, 1));

    const tt::tt_metal::CoreRangeSet sender(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange(workers.front())});
    const auto receivers = selected.grid.subtract(sender);
    const auto line = ttnn::operations::conv::make_remote_rectangle_mcast_line_for_core_set(
        receivers, workers.front(), compute_grid);
    ASSERT_EQ(line.segments.size(), 2u);
    EXPECT_EQ(line.segments[0].workers, tt::tt_metal::CoreRange({0, 0}, {1, 1}));
    EXPECT_EQ(line.segments[1].workers, tt::tt_metal::CoreRange({3, 0}, {12, 0}));
    EXPECT_TRUE(line.segments[0].contains_sender);
    EXPECT_EQ(line.segments[0].destination_count, 3u);
    EXPECT_FALSE(line.segments[1].contains_sender);
    EXPECT_EQ(line.segments[1].destination_count, 10u);
    uint32_t destination_count = 0;
    for (const auto& segment : line.segments) {
        destination_count += segment.destination_count;
    }
    EXPECT_EQ(destination_count, num_cores - 1);

    const auto identity = [](tt::tt_metal::CoreCoord worker) { return worker; };
    const auto noc0_args = ttnn::operations::conv::serialize_mcast_segments(line, true, identity);
    const auto noc1_args = ttnn::operations::conv::serialize_mcast_segments(line, false, identity);
    ASSERT_EQ(noc0_args.size(), 1u + line.segments.size() * 6u);
    ASSERT_EQ(noc1_args.size(), noc0_args.size());
    EXPECT_EQ(noc0_args[0], 2u);
    EXPECT_EQ((std::vector<uint32_t>(noc0_args.begin() + 1, noc0_args.begin() + 7)),
              (std::vector<uint32_t>{0, 0, 1, 1, 3, 1}));
    EXPECT_EQ((std::vector<uint32_t>(noc1_args.begin() + 7, noc1_args.begin() + 13)),
              (std::vector<uint32_t>{12, 0, 3, 0, 10, 0}));

    EXPECT_ANY_THROW(ttnn::operations::conv::make_remote_rectangle_mcast_line_for_core_set(
        selected.grid, workers.front(), compute_grid));
}

TEST(ConvCoreReplacementSafety, WidthShardedExclusionsRemainHostRejected) {
    constexpr tt::tt_metal::CoreCoord compute_grid{5, 4};
    const ttnn::operations::sliding_window::ParallelConfig width_sharded{
        .grid = tt::tt_metal::num_cores_to_corerangeset(8, compute_grid, true),
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };
    EXPECT_ANY_THROW(ttnn::operations::conv::select_conv_core_remapping_input_placement(
        width_sharded,
        compute_grid,
        /*excluded_worker_rows=*/{},
        /*excluded_worker_columns=*/{2}));
}

TEST(ConvCoreReplacementSafety, PlannerPropagatesCompatibleTranslationAndExitsOnExpansion) {
    const tt::tt_metal::CoreRangeSet translated_grid(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({1, 0}, {4, 3})});
    const ttnn::operations::sliding_window::ParallelConfig input_config{
        .grid = translated_grid,
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };
    auto stock_input_config = input_config;
    stock_input_config.grid = tt::tt_metal::CoreRangeSet(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({0, 0}, {3, 3})});
    const auto stock_output_config = ttnn::operations::conv::determine_output_parallel_config(
        stock_input_config,
        tt::tt_metal::CoreCoord{8, 8},
        /*out_channels=*/128,
        tt::tt_metal::ShardOrientation::ROW_MAJOR,
        /*is_mm_conv=*/false);
    EXPECT_EQ(stock_output_config.grid.bounding_box().start_coord, tt::tt_metal::CoreCoord(0, 0));

    const auto propagation_plan = ttnn::operations::conv::plan_conv_core_remapping(
        input_config, stock_output_config, tt::tt_metal::CoreCoord{8, 8}, /*remapping_active=*/true);

    EXPECT_EQ(propagation_plan.action, ttnn::operations::conv::ConvCoreRemappingAction::Propagate);
    EXPECT_FALSE(propagation_plan.requires_input_reshard());
    EXPECT_EQ(
        propagation_plan.output_parallel_config.grid.bounding_box().start_coord,
        tt::tt_metal::CoreCoord(1, 0));
    EXPECT_EQ(
        propagation_plan.output_parallel_config.grid.bounding_box().end_coord,
        tt::tt_metal::CoreCoord(4, 3));
    const auto integrated_propagated_output = ttnn::operations::conv::determine_output_parallel_config(
        input_config,
        tt::tt_metal::CoreCoord{8, 8},
        /*out_channels=*/128,
        tt::tt_metal::ShardOrientation::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*enable_core_remapping=*/true);
    EXPECT_EQ(integrated_propagated_output.grid, propagation_plan.output_parallel_config.grid);
    const auto explicitly_disabled_output = ttnn::operations::conv::determine_output_parallel_config(
        input_config,
        tt::tt_metal::CoreCoord{8, 8},
        /*out_channels=*/128,
        tt::tt_metal::ShardOrientation::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*enable_core_remapping=*/false);
    EXPECT_EQ(
        explicitly_disabled_output.grid.bounding_box().start_coord,
        tt::tt_metal::CoreCoord(0, 0));
    EXPECT_EQ(
        explicitly_disabled_output.grid.bounding_box().end_coord,
        tt::tt_metal::CoreCoord(3, 3));

    const auto stock_smaller_output_config = ttnn::operations::conv::determine_output_parallel_config(
        stock_input_config,
        tt::tt_metal::CoreCoord{5, 4},
        /*out_channels=*/64,
        tt::tt_metal::ShardOrientation::ROW_MAJOR,
        /*is_mm_conv=*/false);
    const auto smaller_output_plan = ttnn::operations::conv::plan_conv_core_remapping(
        input_config,
        stock_smaller_output_config,
        tt::tt_metal::CoreCoord{5, 4},
        /*remapping_active=*/true);

    EXPECT_EQ(smaller_output_plan.action, ttnn::operations::conv::ConvCoreRemappingAction::Propagate);
    EXPECT_EQ(
        smaller_output_plan.output_parallel_config.grid.bounding_box().start_coord,
        tt::tt_metal::CoreCoord(1, 0));
    EXPECT_EQ(
        smaller_output_plan.output_parallel_config.grid.bounding_box().end_coord,
        tt::tt_metal::CoreCoord(2, 3));

    // Layer4's downsample produces 2048 channels on a 5x4 device. Its stock 5x4 output topology cannot inherit
    // x=1, so output placement falls back to the stock origin without reducing the output core count. The Conv
    // preparation path uses this origin change to end remapping propagation and reanchor its input before dispatch.
    const auto stock_layer4_downsample_output_config = ttnn::operations::conv::determine_output_parallel_config(
        stock_input_config,
        tt::tt_metal::CoreCoord{5, 4},
        /*out_channels=*/2048,
        tt::tt_metal::ShardOrientation::ROW_MAJOR,
        /*is_mm_conv=*/false);
    const auto layer4_downsample_plan = ttnn::operations::conv::plan_conv_core_remapping(
        input_config,
        stock_layer4_downsample_output_config,
        tt::tt_metal::CoreCoord{5, 4},
        /*remapping_active=*/true);

    EXPECT_EQ(layer4_downsample_plan.action, ttnn::operations::conv::ConvCoreRemappingAction::ExitToStock);
    EXPECT_TRUE(layer4_downsample_plan.requires_input_reshard());
    EXPECT_EQ(
        layer4_downsample_plan.input_parallel_config.grid.bounding_box().start_coord,
        tt::tt_metal::CoreCoord(0, 0));
    EXPECT_EQ(
        layer4_downsample_plan.output_parallel_config.grid.bounding_box().start_coord,
        tt::tt_metal::CoreCoord(0, 0));
    EXPECT_EQ(
        layer4_downsample_plan.output_parallel_config.grid.bounding_box().end_coord,
        tt::tt_metal::CoreCoord(4, 3));
    const auto integrated_layer4_downsample_output = ttnn::operations::conv::determine_output_parallel_config(
        input_config,
        tt::tt_metal::CoreCoord{5, 4},
        /*out_channels=*/2048,
        tt::tt_metal::ShardOrientation::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*enable_core_remapping=*/true);
    EXPECT_EQ(
        integrated_layer4_downsample_output.grid,
        layer4_downsample_plan.output_parallel_config.grid);

    // Growth ends remapping even when the larger rectangle would still fit at the translated origin.
    const auto stock_expanded_but_fitting_output_config =
        ttnn::operations::conv::determine_output_parallel_config(
            stock_input_config,
            tt::tt_metal::CoreCoord{8, 8},
            /*out_channels=*/160,
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            /*is_mm_conv=*/false);
    const auto expanded_but_fitting_plan = ttnn::operations::conv::plan_conv_core_remapping(
        input_config,
        stock_expanded_but_fitting_output_config,
        tt::tt_metal::CoreCoord{8, 8},
        /*remapping_active=*/true);
    EXPECT_EQ(expanded_but_fitting_plan.action, ttnn::operations::conv::ConvCoreRemappingAction::ExitToStock);
    EXPECT_EQ(
        expanded_but_fitting_plan.output_parallel_config.grid.bounding_box().start_coord,
        tt::tt_metal::CoreCoord(0, 0));
    EXPECT_EQ(
        expanded_but_fitting_plan.output_parallel_config.grid.bounding_box().end_coord,
        tt::tt_metal::CoreCoord(4, 3));

    const ttnn::operations::sliding_window::ParallelConfig row_translated_input_config{
        .grid = tt::tt_metal::CoreRangeSet(
            std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({0, 1}, {3, 3})}),
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::COL_MAJOR,
    };
    auto stock_row_input_config = row_translated_input_config;
    stock_row_input_config.grid = tt::tt_metal::CoreRangeSet(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({0, 0}, {3, 2})});
    const auto stock_row_output_config = ttnn::operations::conv::determine_output_parallel_config(
        stock_row_input_config,
        tt::tt_metal::CoreCoord{5, 4},
        /*out_channels=*/96,
        tt::tt_metal::ShardOrientation::COL_MAJOR,
        /*is_mm_conv=*/false);
    const auto row_translated_plan = ttnn::operations::conv::plan_conv_core_remapping(
        row_translated_input_config,
        stock_row_output_config,
        tt::tt_metal::CoreCoord{5, 4},
        /*remapping_active=*/true);
    EXPECT_EQ(row_translated_plan.action, ttnn::operations::conv::ConvCoreRemappingAction::Propagate);
    EXPECT_EQ(
        row_translated_plan.output_parallel_config.grid.bounding_box().start_coord,
        tt::tt_metal::CoreCoord(0, 1));
    EXPECT_EQ(
        row_translated_plan.output_parallel_config.grid.bounding_box().end_coord,
        tt::tt_metal::CoreCoord(3, 3));

    const auto stock_row_expanded_output_config = ttnn::operations::conv::determine_output_parallel_config(
        stock_row_input_config,
        tt::tt_metal::CoreCoord{5, 4},
        /*out_channels=*/2048,
        tt::tt_metal::ShardOrientation::COL_MAJOR,
        /*is_mm_conv=*/false);
    const auto row_expanded_plan = ttnn::operations::conv::plan_conv_core_remapping(
        row_translated_input_config,
        stock_row_expanded_output_config,
        tt::tt_metal::CoreCoord{5, 4},
        /*remapping_active=*/true);
    EXPECT_EQ(row_expanded_plan.action, ttnn::operations::conv::ConvCoreRemappingAction::ExitToStock);
    EXPECT_EQ(
        row_expanded_plan.output_parallel_config.grid.bounding_box().start_coord,
        tt::tt_metal::CoreCoord(0, 0));
    EXPECT_EQ(
        row_expanded_plan.output_parallel_config.grid.bounding_box().end_coord,
        tt::tt_metal::CoreCoord(3, 3));

    const ttnn::operations::sliding_window::ParallelConfig out_of_bounds_input{
        .grid = tt::tt_metal::CoreRangeSet(
            std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({5, 5}, {7, 8})}),
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };
    EXPECT_ANY_THROW(ttnn::operations::conv::determine_output_parallel_config(
        out_of_bounds_input,
        tt::tt_metal::CoreCoord{8, 8},
        /*out_channels=*/128,
        tt::tt_metal::ShardOrientation::ROW_MAJOR,
        /*is_mm_conv=*/false));
}

TEST(ConvCoreReplacementSafety, PlannerPropagatesSegmentedAxesByLogicalPrefixAndExitsOnExpansion) {
    constexpr tt::tt_metal::CoreCoord compute_grid{6, 5};
    const auto input_placement = ttnn::operations::conv::make_core_placement_from_worker_axes(
        compute_grid, std::vector<uint32_t>{0, 1, 3, 4}, std::vector<uint32_t>{0, 2, 3});
    const ttnn::operations::sliding_window::ParallelConfig input_config{
        .grid = input_placement.ordered_worker_cores(ttnn::operations::conv::CorePlacementOrder::RowMajor),
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };

    EXPECT_EQ(ttnn::operations::conv::get_num_cores_channels_from_parallel_config(input_config), 4u);
    EXPECT_EQ(ttnn::operations::conv::get_num_cores_nhw_from_parallel_config(input_config), 3u);

    const tt::tt_metal::MemoryConfig segmented_output_memory_config{
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec{input_config.grid, {64, 32}, tt::tt_metal::ShardOrientation::ROW_MAJOR}};
    const auto segmented_conv_parallel_config =
        ttnn::operations::conv::determine_conv_op_parallel_config_from_conv_output_mem_config(
            segmented_output_memory_config,
            /*num_cores_nhw=*/3,
            /*num_cores_c_in=*/4,
            /*num_cores_c_out=*/4);
    EXPECT_EQ(segmented_conv_parallel_config.grid_size, tt::tt_metal::CoreCoord(4, 3));

    const auto stock_smaller_placement =
        ttnn::operations::conv::make_core_placement(compute_grid, tt::tt_metal::CoreCoord{3, 2});
    auto stock_smaller_output = input_config;
    stock_smaller_output.grid = stock_smaller_placement.worker_cores;
    const auto smaller_plan = ttnn::operations::conv::plan_conv_core_remapping(
        input_config, stock_smaller_output, compute_grid, /*remapping_active=*/true);
    ASSERT_EQ(smaller_plan.action, ttnn::operations::conv::ConvCoreRemappingAction::Propagate);
    const auto smaller_placement = ttnn::operations::conv::make_core_placement_from_worker_grid(
        compute_grid, smaller_plan.output_parallel_config.grid);
    EXPECT_EQ(smaller_placement.worker_x, (std::vector<uint32_t>{0, 1, 3}));
    EXPECT_EQ(smaller_placement.worker_y, (std::vector<uint32_t>{0, 2}));
    EXPECT_EQ(smaller_placement.worker_cores.num_cores(), 6u);

    // A same-or-smaller topology normally propagates, but an operation that cannot execute a segmented placement
    // must end its lifetime and request a stock input reshard before dispatch.
    const auto unsupported_next_op_plan = ttnn::operations::conv::plan_conv_core_remapping(
        input_config,
        stock_smaller_output,
        compute_grid,
        /*remapping_active=*/true,
        /*current_placement_is_executable=*/false);
    ASSERT_EQ(
        unsupported_next_op_plan.action,
        ttnn::operations::conv::ConvCoreRemappingAction::ExitToStock);
    EXPECT_TRUE(unsupported_next_op_plan.requires_input_reshard());
    const auto stock_input_at_unsupported_boundary =
        ttnn::operations::conv::make_core_placement_from_worker_grid(
            compute_grid, unsupported_next_op_plan.input_parallel_config.grid);
    EXPECT_FALSE(stock_input_at_unsupported_boundary.has_worker_gaps());
    EXPECT_EQ(
        stock_input_at_unsupported_boundary.worker_x,
        (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(
        stock_input_at_unsupported_boundary.worker_y,
        (std::vector<uint32_t>{0, 1, 2}));
    const auto stock_output_at_unsupported_boundary =
        ttnn::operations::conv::make_core_placement_from_worker_grid(
            compute_grid, unsupported_next_op_plan.output_parallel_config.grid);
    EXPECT_FALSE(stock_output_at_unsupported_boundary.has_worker_gaps());
    EXPECT_EQ(stock_output_at_unsupported_boundary.worker_x, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(stock_output_at_unsupported_boundary.worker_y, (std::vector<uint32_t>{0, 1}));

    // The integrated output selector must count selected axes, not the
    // segmented placement's larger physical bounding box.
    const auto integrated_output = ttnn::operations::conv::determine_output_parallel_config(
        input_config,
        compute_grid,
        /*out_channels=*/64,
        tt::tt_metal::ShardOrientation::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*enable_core_remapping=*/true);
    const auto integrated_placement =
        ttnn::operations::conv::make_core_placement_from_worker_grid(compute_grid, integrated_output.grid);
    EXPECT_EQ(integrated_placement.worker_x, (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(integrated_placement.worker_y, (std::vector<uint32_t>{0, 2, 3}));

    // A matmul-lowered 1x1 Conv preserves the complete input topology. Once the exact ROW_MAJOR 2D reuse-mcast
    // backend is selected, the semantic Conv planner must keep the segmented placement alive across this boundary.
    const auto integrated_mm_output = ttnn::operations::conv::determine_output_parallel_config(
        input_config,
        compute_grid,
        /*out_channels=*/128,
        tt::tt_metal::ShardOrientation::ROW_MAJOR,
        /*is_mm_conv=*/true,
        /*enable_core_remapping=*/true);
    EXPECT_EQ(integrated_mm_output.grid, input_config.grid);

    auto stock_expanded_output = input_config;
    stock_expanded_output.grid =
        ttnn::operations::conv::make_core_placement(compute_grid, tt::tt_metal::CoreCoord{5, 3}).worker_cores;
    const auto expanded_plan = ttnn::operations::conv::plan_conv_core_remapping(
        input_config, stock_expanded_output, compute_grid, /*remapping_active=*/true);
    ASSERT_EQ(expanded_plan.action, ttnn::operations::conv::ConvCoreRemappingAction::ExitToStock);
    const auto stock_input_after_exit = ttnn::operations::conv::make_core_placement_from_worker_grid(
        compute_grid, expanded_plan.input_parallel_config.grid);
    EXPECT_EQ(stock_input_after_exit.worker_x, (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(stock_input_after_exit.worker_y, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_FALSE(stock_input_after_exit.has_worker_gaps());
}

TEST(ConvCoreReplacementSafety, MatmulLoweredConvMulticastLinesExactlyCoverSegmentedRowsAndColumns) {
    const auto placement = ttnn::operations::conv::make_core_placement_from_worker_axes(
        /*worker_grid_size=*/{6, 5},
        /*worker_x=*/std::vector<uint32_t>{0, 1, 3, 4},
        /*worker_y=*/std::vector<uint32_t>{0, 2, 3, 4});

    const auto validate_axis = [&](ttnn::operations::conv::CorePlacementMcastAxis axis) {
        const bool row_axis = axis == ttnn::operations::conv::CorePlacementMcastAxis::Row;
        const uint32_t fixed_count = row_axis ? placement.shard_grid_size.y : placement.shard_grid_size.x;
        const uint32_t varying_count = row_axis ? placement.shard_grid_size.x : placement.shard_grid_size.y;
        for (uint32_t fixed = 0; fixed < fixed_count; ++fixed) {
            for (uint32_t sender_index = 0; sender_index < varying_count; ++sender_index) {
                const auto sender = row_axis ? placement.worker_core({sender_index, fixed})
                                             : placement.worker_core({fixed, sender_index});
                const auto line = placement.remote_mcast_line_for_worker_sender(axis, fixed, sender);
                std::set<tt::tt_metal::CoreCoord> expected;
                std::set<tt::tt_metal::CoreCoord> actual;
                for (uint32_t receiver_index = 0; receiver_index < varying_count; ++receiver_index) {
                    if (receiver_index != sender_index) {
                        expected.insert(row_axis ? placement.worker_core({receiver_index, fixed})
                                                 : placement.worker_core({fixed, receiver_index}));
                    }
                }

                uint32_t destination_count = 0;
                for (const auto& segment : line.segments) {
                    destination_count += segment.destination_count;
                    EXPECT_FALSE(segment.contains_sender);
                    EXPECT_FALSE(segment.workers.contains(sender));
                    for (const auto& worker : segment.workers) {
                        EXPECT_TRUE(actual.insert(worker).second);
                    }
                }
                EXPECT_EQ(destination_count, varying_count - 1);
                EXPECT_EQ(actual, expected);
            }
        }
    };
    validate_axis(ttnn::operations::conv::CorePlacementMcastAxis::Row);
    validate_axis(ttnn::operations::conv::CorePlacementMcastAxis::Column);

    const auto row_sender = placement.worker_core({1, 2});
    const auto row = placement.remote_mcast_line_for_worker_sender(
        ttnn::operations::conv::CorePlacementMcastAxis::Row, /*logical_fixed=*/2, row_sender);
    const auto column_sender = placement.worker_core({3, 0});
    const auto column = placement.remote_mcast_line_for_worker_sender(
        ttnn::operations::conv::CorePlacementMcastAxis::Column, /*logical_fixed=*/3, column_sender);

    const auto identity_worker_to_noc = [](tt::tt_metal::CoreCoord worker) { return worker; };
    const auto row_args =
        ttnn::operations::conv::serialize_mcast_segments(row, /*is_noc_0=*/true, identity_worker_to_noc);
    const auto column_args =
        ttnn::operations::conv::serialize_mcast_segments(column, /*is_noc_0=*/false, identity_worker_to_noc);
    ASSERT_FALSE(row_args.empty());
    ASSERT_FALSE(column_args.empty());
    EXPECT_EQ(row_args.size(), 1u + row_args.front() * 6u);
    EXPECT_EQ(column_args.size(), 1u + column_args.front() * 6u);
}

TEST(ConvCoreReplacementSafety, EquivalentStockGridRetainsRemappedBlockExecutionPlacement) {
    const tt::tt_metal::CoreCoord compute_grid{5, 4};
    const tt::tt_metal::CoreRangeSet stock_grid(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({0, 0}, {3, 3})});
    const auto translated_grid = ttnn::operations::conv::make_core_placement_from_worker_axes(
                                     compute_grid,
                                     std::vector<uint32_t>{1, 2, 3, 4},
                                     std::vector<uint32_t>{0, 1, 2, 3})
                                     .ordered_worker_cores(ttnn::operations::conv::CorePlacementOrder::RowMajor);
    const auto column_gap_grid = ttnn::operations::conv::make_core_placement_from_worker_axes(
                                     compute_grid,
                                     std::vector<uint32_t>{0, 1, 3, 4},
                                     std::vector<uint32_t>{0, 1, 2, 3})
                                     .ordered_worker_cores(ttnn::operations::conv::CorePlacementOrder::RowMajor);
    const tt::tt_metal::CoreRangeSet smaller_grid(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({0, 0}, {3, 2})});

    EXPECT_TRUE(ttnn::operations::conv::block_sharded_override_matches_remapped_logical_grid(
        compute_grid, translated_grid, stock_grid));
    EXPECT_TRUE(ttnn::operations::conv::block_sharded_override_matches_remapped_logical_grid(
        compute_grid, column_gap_grid, stock_grid));
    EXPECT_FALSE(ttnn::operations::conv::block_sharded_override_matches_remapped_logical_grid(
        compute_grid, smaller_grid, stock_grid));

    // Both an explicit stock override and a stock grid selected by
    // reshard_if_not_optimal use this execution-grid decision.
    EXPECT_EQ(
        ttnn::operations::conv::select_equivalent_remapped_block_execution_grid(
            compute_grid, translated_grid, stock_grid, /*allow_equivalent_remapped_input=*/true),
        translated_grid);
    EXPECT_EQ(
        ttnn::operations::conv::select_equivalent_remapped_block_execution_grid(
            compute_grid, column_gap_grid, stock_grid, /*allow_equivalent_remapped_input=*/true),
        column_gap_grid);
    EXPECT_EQ(
        ttnn::operations::conv::select_equivalent_remapped_block_execution_grid(
            compute_grid, smaller_grid, stock_grid, /*allow_equivalent_remapped_input=*/true),
        stock_grid);
    EXPECT_EQ(
        ttnn::operations::conv::select_equivalent_remapped_block_execution_grid(
            compute_grid, translated_grid, stock_grid, /*allow_equivalent_remapped_input=*/false),
        stock_grid);
}

TEST(ConvCoreReplacementSafety, BlockShardedSlidingWindowReplicationUsesRectangleSizeNotAbsoluteEnd) {
    const tt::tt_metal::CoreRangeSet translated_grid(
        std::set<tt::tt_metal::CoreRange>{tt::tt_metal::CoreRange({1, 2}, {4, 4})});

    auto config = ttnn::operations::sliding_window::ParallelConfig{
        .grid = translated_grid,
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };
    EXPECT_EQ(
        ttnn::operations::sliding_window::get_repeat_factor_for_replicating_nhw_config_across_grid(config), 4u);

    config.shard_orientation = tt::tt_metal::ShardOrientation::COL_MAJOR;
    EXPECT_EQ(
        ttnn::operations::sliding_window::get_repeat_factor_for_replicating_nhw_config_across_grid(config), 3u);

    const auto segmented = ttnn::operations::conv::make_core_placement_from_worker_axes(
        /*worker_grid_size=*/{6, 5},
        /*worker_x=*/std::vector<uint32_t>{0, 1, 3, 4},
        /*worker_y=*/std::vector<uint32_t>{0, 2, 3});
    config.grid = segmented.ordered_worker_cores(ttnn::operations::conv::CorePlacementOrder::RowMajor);
    config.shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR;
    EXPECT_EQ(
        ttnn::operations::sliding_window::get_repeat_factor_for_replicating_nhw_config_across_grid(config), 4u);

    config.grid = segmented.ordered_worker_cores(ttnn::operations::conv::CorePlacementOrder::ColumnMajor);
    config.shard_orientation = tt::tt_metal::ShardOrientation::COL_MAJOR;
    EXPECT_EQ(
        ttnn::operations::sliding_window::get_repeat_factor_for_replicating_nhw_config_across_grid(config), 3u);
}

TEST(ConvCoreReplacementSafety, BlockShardedHaloMapsLogicalNhwAcrossSelectedWorkerAxis) {
    const auto row_gap = ttnn::operations::conv::make_core_placement_from_worker_axes(
        /*worker_grid_size=*/{5, 4},
        /*worker_x=*/std::vector<uint32_t>{0, 1, 2, 3},
        /*worker_y=*/std::vector<uint32_t>{0, 1, 3});
    ttnn::operations::sliding_window::ParallelConfig config{
        .grid = row_gap.ordered_worker_cores(ttnn::operations::conv::CorePlacementOrder::RowMajor),
        .shard_scheme = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR,
    };
    EXPECT_EQ(
        ttnn::operations::sliding_window::get_block_sharded_worker_core_for_nhw_core(config, 0),
        tt::tt_metal::CoreCoord(0, 0));
    EXPECT_EQ(
        ttnn::operations::sliding_window::get_block_sharded_worker_core_for_nhw_core(config, 2),
        tt::tt_metal::CoreCoord(0, 3));

    const auto column_gap = ttnn::operations::conv::make_core_placement_from_worker_axes(
        /*worker_grid_size=*/{5, 4},
        /*worker_x=*/std::vector<uint32_t>{0, 1, 3},
        /*worker_y=*/std::vector<uint32_t>{0, 1, 2, 3});
    config.grid = column_gap.ordered_worker_cores(ttnn::operations::conv::CorePlacementOrder::ColumnMajor);
    config.shard_orientation = tt::tt_metal::ShardOrientation::COL_MAJOR;
    EXPECT_EQ(
        ttnn::operations::sliding_window::get_block_sharded_worker_core_for_nhw_core(config, 2),
        tt::tt_metal::CoreCoord(3, 0));
}

// Phase 1 baseline: exercise the unmodified TTNN Conv path with its automatic
// sharding policy. A forced 3x3 grid is a separate experiment because an
// unsupported override can enqueue a device program that never completes.
// Disabled until the unmodified BOS ResNet50 path passes after a clean device
// reset.  A host timeout cannot safely recover a device program that has
// already deadlocked, so this integration test must be enabled deliberately.
TEST(ConvCoreReplacement, DISABLED_AutomaticShardingBaseline) {
    constexpr size_t l1_small_size = 16384;
    log_info(tt::LogTest, "CONV_CORE_REPLACEMENT STEP 1: creating mesh device");
    auto device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(/*device_id=*/0, l1_small_size);
    log_info(tt::LogTest, "CONV_CORE_REPLACEMENT STEP 2: mesh device ready");
    ASSERT_GE(device->get_device(0)->compute_with_storage_grid_size().x, 5u);
    ASSERT_GE(device->get_device(0)->compute_with_storage_grid_size().y, 4u);

    const tt::tt_metal::MemoryConfig dram_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    const tt::tt_metal::Shape input_shape{kBatch, kInputChannels, kInputHeight, kInputWidth};
    const tt::tt_metal::Shape weight_shape{
        kOutputChannels, kInputChannels, kKernel[0], kKernel[1]};

    ttnn::random::seed(42);
    log_info(tt::LogTest, "CONV_CORE_REPLACEMENT STEP 3: creating input and weight tensors");
    auto host_input_tensor = ttnn::random::random(input_shape, tt::tt_metal::DataType::BFLOAT16);
    auto weights = ttnn::random::random(weight_shape, tt::tt_metal::DataType::BFLOAT16);
    const auto host_input = host_input_tensor.to_vector<float>();
    const auto host_weights = weights.to_vector<float>();
    log_info(tt::LogTest, "CONV_CORE_REPLACEMENT STEP 4: golden inputs saved; uploading input to DRAM");
    auto input = host_input_tensor.to_device(device.get(), dram_config);

    log_info(tt::LogTest, "CONV_CORE_REPLACEMENT STEP 5: input uploaded; permuting NCHW to NHWC");
    input = ttnn::permute(input, tt::stl::SmallVector<int64_t>{0, 2, 3, 1});  // NCHW -> NHWC
    const auto config = automatic_conv_config();
    log_info(tt::LogTest, "CONV_CORE_REPLACEMENT STEP 6: launching ttnn::conv2d");
    auto [output, output_dimensions] =
        std::get<static_cast<int>(ttnn::operations::conv::ResultType::OUTPUT_DIM)>(ttnn::conv2d(
            input,
            weights,
            device.get(),
            kInputChannels,
            kOutputChannels,
            kBatch,
            kInputHeight,
            kInputWidth,
            kKernel,
            kStride,
            kPadding,
            kDilation,
            1,
            std::nullopt,
            std::nullopt,
            config,
            std::nullopt,
            std::nullopt,
            ttnn::operations::conv::conv2d::Conv2dSliceConfig{
                .slice_type =
                    ttnn::operations::conv::conv2d::Conv2dSliceConfig::SliceType::L1_FULL},
            true));
    log_info(tt::LogTest, "CONV_CORE_REPLACEMENT STEP 7: ttnn::conv2d returned");

    const bool output_is_sharded = output.shard_spec().has_value();
    const uint32_t output_height = std::get<0>(output_dimensions);
    const uint32_t output_width = std::get<1>(output_dimensions);
    log_info(
        tt::LogTest,
        "CONV_CORE_REPLACEMENT STEP 8: L1_FULL output returned (sharded={}); copying raw output to host",
        output_is_sharded);
    const auto host_output = ttnn::from_device(output, /*blocking=*/true);
    log_info(tt::LogTest, "CONV_CORE_REPLACEMENT STEP 9: reading flattened NHWC host output");
    const auto raw_result = host_output.to_vector<float>();
    const auto result = nhwc_to_nchw(raw_result, output_height, output_width, kOutputChannels);
    log_info(tt::LogTest, "CONV_CORE_REPLACEMENT STEP 10: calculating CPU golden");
    const auto golden = reference_conv2d(host_input, host_weights, output_height, output_width);
    log_info(tt::LogTest, "CONV_CORE_REPLACEMENT STEP 11: calculating PCC");
    const double pcc = pearson_correlation(result, golden);
    log_info(
        tt::LogTest,
        "CONV_CORE_REPLACEMENT config=automatic_sharding_baseline "
        "output={}x{} pcc_threshold=0.99 pcc={}",
        output_height,
        output_width,
        pcc);
    EXPECT_GT(pcc, 0.99);
}

}  // namespace ttnn::operations::conv::conv2d::core_replacement_test
