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
#ifndef AUTOWARE__SUBM_OPS__CONVERT_BIAS_HPP_
#define AUTOWARE__SUBM_OPS__CONVERT_BIAS_HPP_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// SubMConv takes its bias in float32 or float16, whatever the features are; spconv's kernels
// add a bias of one fixed type per kernel family. These convert the C_out values.

void launch_float_to_half(
  const float * input, __half * output, std::int64_t num_elements, cudaStream_t stream);

void launch_half_to_float(
  const __half * input, float * output, std::int64_t num_elements, cudaStream_t stream);

#endif  // AUTOWARE__SUBM_OPS__CONVERT_BIAS_HPP_
