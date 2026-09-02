// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/buffer_distribution_spec.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::unit_tests::conv_core_replacement {
namespace {

CoreRangeSet make_segmented_grid() {
    std::set<CoreRange> cores;
    for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x : {0u, 1u, 3u, 4u}) {
            cores.emplace(CoreCoord{x, y});
        }
    }
    return CoreRangeSet(cores);
}

BufferDistributionSpec make_grid_2d_spec(const CoreRangeSet& grid, ShardOrientation orientation) {
    return BufferDistributionSpec(
        Shape{128, 128},
        Shape{32, 32},
        grid,
        orientation,
        ShardDistributionStrategy::GRID_2D);
}

TensorSpec make_legacy_block_sharded_tensor_spec(
    Shape shape, const CoreRangeSet& grid, ShardOrientation orientation = ShardOrientation::ROW_MAJOR) {
    MemoryConfig memory_config{
        TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, ShardSpec{grid, {32, 32}, orientation}};
    return TensorSpec(
        std::move(shape), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), std::move(memory_config)));
}

TEST(ConvCoreReplacementBufferDistribution, SegmentedCartesianGridPreservesRowMajorLogicalOrder) {
    const auto spec = make_grid_2d_spec(make_segmented_grid(), ShardOrientation::ROW_MAJOR);
    EXPECT_EQ(
        spec.cores(),
        (std::vector<CoreCoord>{
            {0, 0}, {1, 0}, {3, 0}, {4, 0},
            {0, 1}, {1, 1}, {3, 1}, {4, 1},
            {0, 2}, {1, 2}, {3, 2}, {4, 2},
            {0, 3}, {1, 3}, {3, 3}, {4, 3}}));
}

TEST(ConvCoreReplacementBufferDistribution, SegmentedCartesianGridPreservesColumnMajorLogicalOrder) {
    const auto spec = make_grid_2d_spec(make_segmented_grid(), ShardOrientation::COL_MAJOR);
    EXPECT_EQ(
        spec.cores(),
        (std::vector<CoreCoord>{
            {0, 0}, {0, 1}, {0, 2}, {0, 3},
            {1, 0}, {1, 1}, {1, 2}, {1, 3},
            {3, 0}, {3, 1}, {3, 2}, {3, 3},
            {4, 0}, {4, 1}, {4, 2}, {4, 3}}));
}

TEST(ConvCoreReplacementBufferDistribution, SegmentedCartesianGridTrimsUnusedLogicalAxes) {
    const auto spec = BufferDistributionSpec(
        Shape{64, 96},
        Shape{32, 32},
        make_segmented_grid(),
        ShardOrientation::ROW_MAJOR,
        ShardDistributionStrategy::GRID_2D);
    EXPECT_EQ(spec.cores(), (std::vector<CoreCoord>{{0, 0}, {1, 0}, {3, 0}, {0, 1}, {1, 1}, {3, 1}}));
}

TEST(ConvCoreReplacementBufferDistribution, Grid2DRejectsNonCartesianSegmentedGrid) {
    const CoreRangeSet non_cartesian(std::set<CoreRange>{
        CoreRange({0, 0}), CoreRange({1, 0}), CoreRange({0, 1})});
    EXPECT_ANY_THROW(make_grid_2d_spec(non_cartesian, ShardOrientation::ROW_MAJOR));
}

TEST(ConvCoreReplacementTensorSpec, LegacyBlockShardingAcceptsSegmentedCartesianGrid) {
    const auto grid = make_segmented_grid();
    const auto spec = make_legacy_block_sharded_tensor_spec(Shape{128, 128}, grid);
    ASSERT_TRUE(spec.memory_config().shard_spec().has_value());
    EXPECT_EQ(spec.memory_config().shard_spec()->grid, grid);
}

TEST(ConvCoreReplacementTensorSpec, BlockShardingRejectsNonCartesianGrid) {
    const CoreRangeSet non_cartesian(std::set<CoreRange>{
        CoreRange({0, 0}), CoreRange({1, 0}), CoreRange({0, 1})});
    EXPECT_ANY_THROW(make_legacy_block_sharded_tensor_spec(Shape{64, 64}, non_cartesian));
}

TEST(ConvCoreReplacementTensorSpec, ValidationUsesSelectedAxesInsteadOfBoundingBox) {
    const CoreRangeSet two_by_two_with_column_gap(std::set<CoreRange>{
        CoreRange({0, 0}), CoreRange({2, 0}), CoreRange({0, 1}), CoreRange({2, 1})});
    // Three width shards fit in the physical 3-column bounding box but not in the two selected logical columns.
    EXPECT_ANY_THROW(make_legacy_block_sharded_tensor_spec(Shape{64, 96}, two_by_two_with_column_gap));
}

TEST(ConvCoreReplacementTensorSpec, NdGrid2DPopulatesLegacySpecForSegmentedCartesianGrid) {
    const auto grid = make_segmented_grid();
    MemoryConfig memory_config{
        BufferType::L1,
        NdShardSpec{Shape{32, 32}, grid, ShardOrientation::ROW_MAJOR, ShardDistributionStrategy::GRID_2D}};
    const TensorSpec spec(
        Shape{128, 128}, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), std::move(memory_config)));
    ASSERT_TRUE(spec.memory_config().shard_spec().has_value());
    EXPECT_EQ(spec.memory_config().shard_spec()->grid, grid);
}

}  // namespace
}  // namespace tt::tt_metal::unit_tests::conv_core_replacement
