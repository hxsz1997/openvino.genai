// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "scheduler.hpp"

using TestBlockAllocatorWithNumLayers = ::testing::TestWithParam<size_t>;

TEST_P(TestBlockAllocatorWithNumLayers, AllocatesBlocksAccordingToNumLayers) {
    size_t num_layers = GetParam();
    size_t initial_num_free_blocks = 10;
    auto allocator = ov::genai::BlockAllocator(initial_num_free_blocks, false, num_layers);
    for (size_t i = 0; i < num_layers; i++) {
        EXPECT_EQ(allocator.num_free_blocks(i), initial_num_free_blocks);
    }

    auto blocks = allocator.allocate_block();
    ASSERT_EQ(blocks.size(), num_layers);

    for (size_t i = 0; i < num_layers; i++) {
        EXPECT_EQ(allocator.num_free_blocks(i), initial_num_free_blocks - 1);
    }
}

INSTANTIATE_TEST_SUITE_P(VariousNumLayers, TestBlockAllocatorWithNumLayers, ::testing::Values(1, 2, 15, 23, 42));

TEST(TestBlockAllocator, AllocatesBlocksIndependentlyToLayers) {
    size_t num_layers = 3;
    size_t initial_num_free_blocks = 10;
    auto allocator = ov::genai::BlockAllocator(initial_num_free_blocks, false, num_layers);

    allocator.allocate_block(0);
    allocator.allocate_block(0);
    EXPECT_EQ(allocator.num_free_blocks(0), 8);
    EXPECT_EQ(allocator.num_free_blocks(1), 10);
    EXPECT_EQ(allocator.num_free_blocks(2), 10);

    allocator.allocate_block(2);

    EXPECT_EQ(allocator.num_free_blocks(0), 8);
    EXPECT_EQ(allocator.num_free_blocks(1), 10);
    EXPECT_EQ(allocator.num_free_blocks(2), 9);

    allocator.allocate_block(1);
    allocator.allocate_block(1);
    allocator.allocate_block(1);

    EXPECT_EQ(allocator.num_free_blocks(0), 8);
    EXPECT_EQ(allocator.num_free_blocks(1), 7);
    EXPECT_EQ(allocator.num_free_blocks(2), 9);
}

TEST(TestBlockAllocator, FreesBlocksIndependentlyFromLayers) {
    size_t num_layers = 3;
    size_t initial_num_free_blocks = 10;
    auto allocator = ov::genai::BlockAllocator(initial_num_free_blocks, false, num_layers);

    auto block_01 = allocator.allocate_block(0);
    auto block_02 = allocator.allocate_block(0);
    auto block_10 = allocator.allocate_block(1);
    auto block_11 = allocator.allocate_block(1);
    auto block_12 = allocator.allocate_block(1);
    auto block_20 = allocator.allocate_block(2);
    ASSERT_EQ(allocator.num_free_blocks(0), 8);
    ASSERT_EQ(allocator.num_free_blocks(1), 7);
    ASSERT_EQ(allocator.num_free_blocks(2), 9);

    allocator.free(block_02, 0);
    EXPECT_EQ(allocator.num_free_blocks(0), 9);
    EXPECT_EQ(allocator.num_free_blocks(1), 7);
    EXPECT_EQ(allocator.num_free_blocks(2), 9);

    allocator.free(block_20, 2);
    EXPECT_EQ(allocator.num_free_blocks(0), 9);
    EXPECT_EQ(allocator.num_free_blocks(1), 7);
    EXPECT_EQ(allocator.num_free_blocks(2), 10);

    allocator.free(block_12, 1);
    allocator.free(block_10, 1);
    EXPECT_EQ(allocator.num_free_blocks(0), 9);
    EXPECT_EQ(allocator.num_free_blocks(1), 9);
    EXPECT_EQ(allocator.num_free_blocks(2), 10);
}

class PrefixCachingBlockAllocatorTest : public testing::Test {
protected:
    PrefixCachingBlockAllocatorTest(): allocator(initial_num_free_blocks, true, num_layers) {}
    size_t num_layers = 3;
    size_t initial_num_free_blocks = 10;
    ov::genai::BlockAllocator allocator;
    std::map<uint64_t, ov::genai::BlocksPerLayer> cached_blocks_map;
};

TEST_F(PrefixCachingBlockAllocatorTest, OnlyAllocatesAndFreesBlocksFromAllLayers) {
    auto allocator = ov::genai::BlockAllocator(initial_num_free_blocks, true, num_layers);
    EXPECT_THROW(allocator.allocate_block(0), ov::Exception);

    // allocate one block so that there is something to free
    auto blocks_per_layer = allocator.allocate_block(0, cached_blocks_map);

    EXPECT_THROW(allocator.free(blocks_per_layer[0], 0), ov::Exception);
    EXPECT_NO_THROW(allocator.free(blocks_per_layer));

    // with prefix caching freed blocks should go into the overwritable store first, not in the actual free pool
    EXPECT_EQ(allocator.num_overwriteable_blocks(), 1);
}


TEST_F(PrefixCachingBlockAllocatorTest, HandlesFreesCorrectlyWithMixedHashFrees) {
    // allocate one block so that there is something to free
    allocator.allocate_block(0, cached_blocks_map);
    allocator.allocate_block(1, cached_blocks_map);
    allocator.allocate_block(2, cached_blocks_map);
    ASSERT_EQ(allocator.num_free_blocks(0), 7);

    ov::genai::BlocksPerLayer mixed_hash_blocks;
    mixed_hash_blocks.reserve(num_layers);
    auto hash_0_blocks = cached_blocks_map[0];
    auto hash_1_blocks = cached_blocks_map[1];
    std::copy(hash_0_blocks.begin(), hash_0_blocks.begin() + num_layers / 2, std::back_inserter(mixed_hash_blocks));
    std::copy(hash_1_blocks.begin() + num_layers / 2, hash_1_blocks.end(), std::back_inserter(mixed_hash_blocks));

    EXPECT_NO_THROW(allocator.free(mixed_hash_blocks));
    EXPECT_EQ(allocator.num_free_blocks(0), 8);
    EXPECT_EQ(allocator.num_free_blocks(num_layers - 1), 8);
    EXPECT_EQ(allocator.num_overwriteable_blocks(), 0);  // mixed hash, can't store under blocks across layers under same hash
}

TEST_F(PrefixCachingBlockAllocatorTest, AllocatesFromOverwriteableBlocksWhenFreePoolIsExhausted) {
    allocator.allocate_block(0, cached_blocks_map);
    allocator.allocate_block(1, cached_blocks_map);
    allocator.allocate_block(2, cached_blocks_map);

    allocator.free(cached_blocks_map[0]);
    allocator.free(cached_blocks_map[1]);
    allocator.free(cached_blocks_map[2]);

    ASSERT_EQ(allocator.num_overwriteable_blocks(), 3);

    for (size_t i = 0; i < initial_num_free_blocks - 3; i++) {
        allocator.allocate_block(1337 + i, cached_blocks_map);
        EXPECT_EQ(allocator.num_overwriteable_blocks(), 3);
    }

    EXPECT_EQ(allocator.num_overwriteable_blocks(), 3);
    allocator.allocate_block(31337, cached_blocks_map);
    EXPECT_EQ(allocator.num_overwriteable_blocks(), 2);
}

TEST_F(PrefixCachingBlockAllocatorTest, ThrowsAtAllocationWhenFull) {
    for (size_t i = 0; i < initial_num_free_blocks; i++) {
        allocator.allocate_block(1337 + i, cached_blocks_map);
    }

    ASSERT_EQ(allocator.num_overwriteable_blocks(), 0);
    ASSERT_EQ(allocator.num_free_blocks(0), 0);

    EXPECT_THROW(allocator.allocate_block(31337, cached_blocks_map), ov::Exception);
}

TEST_F(PrefixCachingBlockAllocatorTest, HandlesHashCollisionsAtFreeCorrectly) {
    // TODO (vshampor): also handle collisions during allocations (multimap instead of map?)
    auto cached_blocks_map = std::map<uint64_t, ov::genai::BlocksPerLayer>{};
    auto first_hash_0_block = allocator.allocate_block(0, cached_blocks_map);
    allocator.free(first_hash_0_block);
    ASSERT_EQ(allocator.num_overwriteable_blocks(), 1);

    // double free
    ASSERT_THROW(allocator.free(first_hash_0_block), ov::Exception);

    allocator.allocate_block(1, cached_blocks_map);
    auto second_hash_0_block = allocator.allocate_block(0, cached_blocks_map);
    EXPECT_EQ(allocator.num_overwriteable_blocks(), 1);

    // this "free" should replace the old block with the same hash in the overwritable store
    allocator.free(second_hash_0_block);
    EXPECT_EQ(allocator.num_overwriteable_blocks(), 1);
    std::map<uint64_t, ov::genai::BlocksPerLayer> empty_map{};  // to force allocator to take the block from overwritable store
    auto internal_overwriteable_block = allocator.get_cached_block(0, empty_map);
    for (size_t layer_idx = 0; layer_idx < internal_overwriteable_block.size(); layer_idx++) {
        EXPECT_EQ(internal_overwriteable_block[layer_idx], second_hash_0_block[layer_idx]);
    }
}

TEST(TestBlockAllocator, CalculatesUsagePercentageCorrectly) {
    size_t num_layers = 10;
    size_t initial_num_free_blocks = 10;
    auto allocator = ov::genai::BlockAllocator(initial_num_free_blocks, false, num_layers);
    EXPECT_NEAR(allocator.get_used_percentage(), 0.0, 1e-5);

    auto one_block_from_each_layer = allocator.allocate_block();
    EXPECT_NEAR(allocator.get_used_percentage(), 10.0, 1e-5);

    auto one_block_from_some_layer = allocator.allocate_block(7);
    EXPECT_NEAR(allocator.get_used_percentage(), 11.0, 1e-5);

    allocator.free(one_block_from_each_layer);
    EXPECT_NEAR(allocator.get_used_percentage(), 1.0, 1e-5);
}


TEST(TestBlockAllocator, CalculatesUsagePercentageCorrectlyWithPrefixCaching) {
    size_t num_layers = 10;
    size_t initial_num_free_blocks = 10;
    auto allocator = ov::genai::BlockAllocator(initial_num_free_blocks, true, num_layers);
    ASSERT_NEAR(allocator.get_used_percentage(), 0.0, 1e-5);

    std::map<uint64_t, ov::genai::BlocksPerLayer> prefix_hash_map;

    for (uint64_t mock_hash: {13, 42, 1337}) {
        auto one_block_from_each_layer = allocator.allocate_block(mock_hash, prefix_hash_map);
    }
    ASSERT_NEAR(allocator.get_used_percentage(), 30.0, 1e-5);

    allocator.free(prefix_hash_map[13]);
    ASSERT_NEAR(allocator.get_used_percentage(), 20.0, 1e-5);

    allocator.allocate_block(13, prefix_hash_map);
    ASSERT_NEAR(allocator.get_used_percentage(), 30.0, 1e-5);
}
