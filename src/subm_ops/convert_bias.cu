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
#include "autoware/subm_ops/convert_bias.hpp"

namespace
{
constexpr int kBlockSize = 256;

__global__ void float_to_half_kernel(
  const float * __restrict__ input, __half * __restrict__ output, std::int64_t num_elements)
{
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2half_rn(input[idx]);
  }
}

__global__ void half_to_float_kernel(
  const __half * __restrict__ input, float * __restrict__ output, std::int64_t num_elements)
{
  std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __half2float(input[idx]);
  }
}

int grid_for(std::int64_t num_elements)
{
  return static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
}
}  // namespace

void launch_float_to_half(
  const float * input, __half * output, std::int64_t num_elements, cudaStream_t stream)
{
  float_to_half_kernel<<<grid_for(num_elements), kBlockSize, 0, stream>>>(
    input, output, num_elements);
}

void launch_half_to_float(
  const __half * input, float * output, std::int64_t num_elements, cudaStream_t stream)
{
  half_to_float_kernel<<<grid_for(num_elements), kBlockSize, 0, stream>>>(
    input, output, num_elements);
}
