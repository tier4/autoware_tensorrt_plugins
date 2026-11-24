// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/scatter_ops/segment_csr.h"
#include "cuda_gtest_utils.hpp"
#include "test_utils.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

using autoware::tensorrt_plugins::test::copy_to_device;
using autoware::tensorrt_plugins::test::copy_to_host;
using autoware::tensorrt_plugins::test::CudaStreamGuard;
using autoware::tensorrt_plugins::test::DeviceBuffer;

struct SegmentCsrCase
{
  std::string name;
  std::size_t rows;
  std::size_t cols;
  std::vector<float> src;
  std::vector<std::int64_t> indptr;
};

std::vector<float> make_segment_reference_mean(
  const std::vector<float> & src, const std::size_t rows, const std::size_t cols,
  const std::vector<std::int64_t> & indptr)
{
  std::vector<float> out((indptr.size() - 1U) * cols, 0.0F);
  for (std::size_t segment = 0; segment + 1U < indptr.size(); ++segment) {
    const auto start = static_cast<std::size_t>(indptr[segment]);
    const auto end = static_cast<std::size_t>(indptr[segment + 1U]);
    const auto count = std::max<std::size_t>(end - start, 1U);
    for (std::size_t col = 0; col < cols; ++col) {
      float accum = 0.0F;
      for (std::size_t row = start; row < end; ++row) {
        accum += src[row * cols + col];
      }
      out[segment * cols + col] = accum / static_cast<float>(count);
    }
  }
  (void)rows;
  return out;
}

std::vector<float> make_segment_reference_max(
  const std::vector<float> & src, const std::size_t rows, const std::size_t cols,
  const std::vector<std::int64_t> & indptr)
{
  std::vector<float> out((indptr.size() - 1U) * cols, 0.0F);
  for (std::size_t segment = 0; segment + 1U < indptr.size(); ++segment) {
    const auto start = static_cast<std::size_t>(indptr[segment]);
    const auto end = static_cast<std::size_t>(indptr[segment + 1U]);
    for (std::size_t col = 0; col < cols; ++col) {
      float best = 0.0F;
      if (start < end) {
        best = -std::numeric_limits<float>::infinity();
        for (std::size_t row = start; row < end; ++row) {
          best = std::max(best, src[row * cols + col]);
        }
      }
      out[segment * cols + col] = best;
    }
  }
  (void)rows;
  return out;
}

void expect_float_vectors_equal(
  const std::vector<float> & actual, const std::vector<float> & expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_FLOAT_EQ(actual[index], expected[index]) << "index=" << index;
  }
}

template <ReductionType REDUCE>
void run_segment_csr_case(const SegmentCsrCase & test_case, const std::vector<float> & reference)
{
  SKIP_TEST_IF_CUDA_UNAVAILABLE();

  CudaStreamGuard stream;
  DeviceBuffer<float> src_d(std::max<std::size_t>(test_case.src.size(), 1U));
  DeviceBuffer<std::int64_t> indptr_d(test_case.indptr.size());
  DeviceBuffer<float> out_d(std::max<std::size_t>(reference.size(), 1U));

  if (!test_case.src.empty()) {
    copy_to_device(src_d.get(), test_case.src);
  }
  copy_to_device(indptr_d.get(), test_case.indptr);

  ASSERT_EQ(
    (segment_csr_launch<float, REDUCE>(
      src_d.get(), static_cast<std::int32_t>(test_case.rows),
      static_cast<std::int32_t>(test_case.cols), indptr_d.get(),
      static_cast<std::int32_t>(test_case.indptr.size()), out_d.get(),
      static_cast<std::int64_t *>(nullptr), stream.get())),
    0);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  if (!reference.empty()) {
    expect_float_vectors_equal(copy_to_host(out_d.get(), reference.size()), reference);
  }
}

class SegmentCsrTest : public ::testing::TestWithParam<SegmentCsrCase>
{
};

TEST_P(SegmentCsrTest, MeanMatchesCpuReference)
{
  const auto & test_case = GetParam();
  const auto reference =
    make_segment_reference_mean(test_case.src, test_case.rows, test_case.cols, test_case.indptr);
  run_segment_csr_case<MEAN>(test_case, reference);
}

TEST_P(SegmentCsrTest, MaxMatchesCpuReference)
{
  const auto & test_case = GetParam();
  const auto reference =
    make_segment_reference_max(test_case.src, test_case.rows, test_case.cols, test_case.indptr);
  run_segment_csr_case<MAX>(test_case, reference);
}

TEST(SegmentCsrZeroOutputTest, MaxWithArgIndicesAcceptsZeroOutputShape)
{
  SKIP_TEST_IF_CUDA_UNAVAILABLE();

  const std::vector<SegmentCsrCase> test_cases{
    SegmentCsrCase{
      "NoSegmentsWithRows",
      6U,
      2U,
      {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F},
      {0}},
    SegmentCsrCase{"ZeroColumns", 3U, 0U, {}, {0, 1, 3}},
    SegmentCsrCase{"EmptyIndptr", 0U, 2U, {}, {}}};

  for (const auto & test_case : test_cases) {
    SCOPED_TRACE(test_case.name);

    CudaStreamGuard stream;
    DeviceBuffer<float> src_d(std::max<std::size_t>(test_case.src.size(), 1U));
    DeviceBuffer<std::int64_t> indptr_d(std::max<std::size_t>(test_case.indptr.size(), 1U));
    DeviceBuffer<float> out_d(1U);
    DeviceBuffer<std::int64_t> arg_indices_d(1U);

    if (!test_case.src.empty()) {
      copy_to_device(src_d.get(), test_case.src);
    }
    if (!test_case.indptr.empty()) {
      copy_to_device(indptr_d.get(), test_case.indptr);
    }

    ASSERT_EQ(
      (segment_csr_launch<float, MAX>(
        src_d.get(), static_cast<std::int32_t>(test_case.rows),
        static_cast<std::int32_t>(test_case.cols), indptr_d.get(),
        static_cast<std::int32_t>(test_case.indptr.size()), out_d.get(), arg_indices_d.get(),
        stream.get())),
      0);
    ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  }
}

std::string segment_csr_case_name(const testing::TestParamInfo<SegmentCsrCase> & info)
{
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
  ReferenceKernelsTest, SegmentCsrTest,
  testing::Values(
    SegmentCsrCase{
      "NonEmptySegments",
      6U,
      2U,
      {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F},
      {0, 2, 5, 6}},
    SegmentCsrCase{
      "EmptyMiddleSegment",
      6U,
      2U,
      {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F},
      {0, 2, 2, 5, 6}},
    SegmentCsrCase{
      "EmptyBoundarySegments",
      6U,
      2U,
      {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F},
      {0, 0, 3, 6, 6}},
    SegmentCsrCase{
      "NoSegmentsWithRows",
      6U,
      2U,
      {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F},
      {0}},
    SegmentCsrCase{"EmptyInputNoSegments", 0U, 2U, {}, {0}},
    SegmentCsrCase{"EmptyInputEmptySegments", 0U, 2U, {}, {0, 0, 0}},
    SegmentCsrCase{"ZeroColumns", 3U, 0U, {}, {0, 1, 3}}),
  segment_csr_case_name);

// Runs every launch path (the vectorized kernel and its fallbacks) for half and float and compares
// bit by bit against a CPU reference.

struct SegmentCsrPathCase
{
  std::string name;
  std::size_t cols;
  std::size_t src_offset;  // 1 misaligns src
  std::size_t out_offset;  // 1 misaligns out
  bool with_arg;           // MAX only: request argmax indices
};

constexpr std::size_t kNumSegments = 300U;

// Includes empty segments and lengths around the kernel's 4-row unroll.
std::vector<std::int64_t> make_indptr()
{
  const std::vector<std::int64_t> lengths{0, 1, 2, 3, 4, 5, 0, 9, 3, 0, 8};
  std::vector<std::int64_t> indptr{0};
  for (std::size_t segment = 0; segment < kNumSegments; ++segment) {
    indptr.push_back(indptr.back() + lengths[segment % lengths.size()]);
  }
  return indptr;
}

// Small integers, so sums are exact in half. `with_specials` mixes in NaN, -0 and +0, with
// column 1 all NaN.
float make_value(const std::size_t row, const std::size_t col, const bool with_specials)
{
  const auto pattern = static_cast<float>(static_cast<int>((row * 7U + col * 13U) % 17U) - 8);
  if (!with_specials) {
    return pattern;
  }
  if (col == 1U) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (col % 3U == 0U) {
    return row % 4U == 1U ? -0.0F : (row % 4U == 2U ? 0.0F : -std::abs(pattern) - 1.0F);
  }
  switch ((row * 5U + col) % 11U) {
    case 0U:
      return std::numeric_limits<float>::quiet_NaN();
    case 1U:
      return -0.0F;
    case 2U:
      return 0.0F;
    default:
      return pattern;
  }
}

template <typename T>
T from_float(const float value)
{
  if constexpr (std::is_same_v<T, half>) {
    return __float2half(value);
  } else {
    return value;
  }
}

template <typename T>
float to_float(const T value)
{
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(value);
  } else {
    return value;
  }
}

// Mirrors Reducer: MAX replaces only on `new > current`, so NaN never wins and -0 stays over a
// later +0.
template <typename T, ReductionType REDUCE>
void make_reference(
  const std::vector<T> & src, const std::size_t num_rows, const std::size_t cols,
  const std::vector<std::int64_t> & indptr, std::vector<T> & out, std::vector<std::int64_t> & arg)
{
  constexpr float max_init = std::is_same_v<T, half> ? -std::numeric_limits<float>::infinity()
                                                     : std::numeric_limits<float>::lowest();
  out.assign((indptr.size() - 1U) * cols, from_float<T>(0.0F));
  arg.assign(out.size(), static_cast<std::int64_t>(num_rows));
  for (std::size_t segment = 0; segment + 1U < indptr.size(); ++segment) {
    const auto start = static_cast<std::size_t>(indptr[segment]);
    const auto end = static_cast<std::size_t>(indptr[segment + 1U]);
    for (std::size_t col = 0; col < cols; ++col) {
      float value = REDUCE == MAX ? max_init : 0.0F;
      std::int64_t argmax = 0;
      for (std::size_t row = start; row < end; ++row) {
        const float element = to_float(src[row * cols + col]);
        if (REDUCE == MAX) {
          if (element > value) {
            value = element;
            argmax = static_cast<std::int64_t>(row);
          }
        } else {
          value += element;
        }
      }
      if (REDUCE == MEAN && end > start) {
        value /= static_cast<float>(end - start);
      }
      if (end > start) {
        out[segment * cols + col] = from_float<T>(value);
        arg[segment * cols + col] = argmax;
      }
    }
  }
}

template <typename T, ReductionType REDUCE>
void run_segment_csr_path_case(const SegmentCsrPathCase & test_case)
{
  SKIP_TEST_IF_CUDA_UNAVAILABLE();

  const bool with_arg = REDUCE == MAX && test_case.with_arg;
  const auto indptr = make_indptr();
  const auto num_rows = static_cast<std::size_t>(indptr.back());
  const auto cols = test_case.cols;

  std::vector<T> src(num_rows * cols);
  for (std::size_t row = 0; row < num_rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      src[row * cols + col] = from_float<T>(make_value(row, col, REDUCE == MAX));
    }
  }
  std::vector<T> expected_out;
  std::vector<std::int64_t> expected_arg;
  make_reference<T, REDUCE>(src, num_rows, cols, indptr, expected_out, expected_arg);

  CudaStreamGuard stream;
  DeviceBuffer<T> src_d(src.size() + test_case.src_offset);
  DeviceBuffer<std::int64_t> indptr_d(indptr.size());
  DeviceBuffer<T> out_d(expected_out.size() + test_case.out_offset);
  DeviceBuffer<std::int64_t> arg_d(expected_arg.size());
  copy_to_device(src_d.get() + test_case.src_offset, src);
  copy_to_device(indptr_d.get(), indptr);
  // Poison the output: the kernel must write every cell.
  ASSERT_EQ(
    cudaMemset(out_d.get(), 0x7F, sizeof(T) * (expected_out.size() + test_case.out_offset)),
    cudaSuccess);

  ASSERT_EQ(
    (segment_csr_launch<T, REDUCE>(
      src_d.get() + test_case.src_offset, static_cast<std::int32_t>(num_rows),
      static_cast<std::int32_t>(cols), indptr_d.get(), static_cast<std::int32_t>(indptr.size()),
      out_d.get() + test_case.out_offset, with_arg ? arg_d.get() : nullptr, stream.get())),
    0);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  const auto actual_out = copy_to_host(out_d.get() + test_case.out_offset, expected_out.size());
  for (std::size_t index = 0; index < expected_out.size(); ++index) {
    ASSERT_EQ(std::memcmp(&actual_out[index], &expected_out[index], sizeof(T)), 0)
      << "index=" << index << " actual=" << to_float(actual_out[index])
      << " expected=" << to_float(expected_out[index]);
  }
  if (with_arg) {
    EXPECT_EQ(copy_to_host(arg_d.get(), expected_arg.size()), expected_arg);
  }
}

class SegmentCsrPathTest : public ::testing::TestWithParam<SegmentCsrPathCase>
{
};

TEST_P(SegmentCsrPathTest, HalfMaxIsBitExact)
{
  run_segment_csr_path_case<half, MAX>(GetParam());
}

TEST_P(SegmentCsrPathTest, FloatMaxIsBitExact)
{
  run_segment_csr_path_case<float, MAX>(GetParam());
}

TEST_P(SegmentCsrPathTest, HalfSumIsBitExact)
{
  run_segment_csr_path_case<half, SUM>(GetParam());
}

TEST_P(SegmentCsrPathTest, FloatMeanIsBitExact)
{
  run_segment_csr_path_case<float, MEAN>(GetParam());
}

std::string segment_csr_path_case_name(const testing::TestParamInfo<SegmentCsrPathCase> & info)
{
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
  ReferenceKernelsTest, SegmentCsrPathTest,
  testing::Values(
    SegmentCsrPathCase{"Vectorized", 16U, 0U, 0U, false},
    SegmentCsrPathCase{"VectorizedManyVectors", 2080U, 0U, 0U, false},
    SegmentCsrPathCase{"VectorizedOddVectorCount", 24U, 0U, 0U, false},
    SegmentCsrPathCase{"MisalignedSrcFallback", 16U, 1U, 0U, false},
    SegmentCsrPathCase{"MisalignedOutFallback", 16U, 0U, 1U, false},
    SegmentCsrPathCase{"OddColumnsFallback", 7U, 0U, 0U, false},
    SegmentCsrPathCase{"SingleColumn", 1U, 0U, 0U, false},
    SegmentCsrPathCase{"ArgOutputFallback", 16U, 0U, 0U, true}),
  segment_csr_path_case_name);

}  // namespace
