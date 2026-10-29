// Copyright 2022 TIER IV, Inc.
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

// Vendored from autoware_cuda_utils/cuda_gtest_utils.hpp so the tests do not depend on a
// package that is not released as a binary.

#ifndef CUDA_GTEST_UTILS_HPP_
#define CUDA_GTEST_UTILS_HPP_

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#define SKIP_TEST_IF_CUDA_UNAVAILABLE()                                 \
  if (!autoware::tensorrt_plugins::test::is_cuda_runtime_available()) { \
    GTEST_SKIP() << "CUDA runtime is not available. Skipping test.";    \
  }

namespace autoware::tensorrt_plugins::test
{

/**
 * @brief Check if CUDA is available at runtime.
 * @return true if CUDA is available, false otherwise.
 */
inline bool is_cuda_runtime_available()
{
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return (error == cudaSuccess && device_count > 0);
}

}  // namespace autoware::tensorrt_plugins::test

#endif  // CUDA_GTEST_UTILS_HPP_
