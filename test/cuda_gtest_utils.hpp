// This code is licensed under CC0 1.0 Universal (Public Domain).
// You can use this without any limitation.
// https://creativecommons.org/publicdomain/zero/1.0/deed.en
// borrowed from https://proc-cpuinfo.fixstars.com/2019/02/cuda_smart_pointer/
// (via autoware_cuda_utils/cuda_gtest_utils.hpp, vendored here so the tests do not depend
// on a package that is not released as a binary)

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
