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

// The SubMConv node contract, shared with t4sparse (tier4/t4sparse, src/trt/subm_contract.hpp at
// 19f1167). The table, the checks and their messages are t4sparse's, so one exported graph
// builds against either library; keep the two copies in step.
//
// What a SubMConv node carries. The scale attributes select the mode, the
// element types have to agree with it, and every violation is refused at
// build. This header is the one copy of the table.
//
//                     float         8-bit in          8-bit in and out
//   attributes
//   input_scale       -             floats [1]        floats [1]
//   weight_scale      -             floats [C_out]    floats [C_out]
//   output_scale      -             -                 floats [1]
//   inputs
//   0 features        T: f32 | f16  T: s8 | e4m3      T: s8 | e4m3
//                        | bf16
//   1 weights         T [C_out, k0, k1, k2, C_in]
//   2 pairs           int32 [k0, k1, k2, P], P = N rounded up to kPadRows
//   3 point_mask      int32 [N, 1]
//   4 mask_argsort    int32 [N]
//   5 tile_mask       int32 [P / kMaskRows]
//   6 bias            f32 | f16 | bf16 [C_out]
//   output            T             f16               T
//
// Which T has a kernel is the library's answer, not this table's: spconv_refusal() below says
// what the spconv-backed plugin runs.
//
// Epilogue, in fp32:  out = act(input_scale * weight_scale[n] * acc + bias[n]) / output_scale
//
// Every scale is positive and finite. "-" is an absent attribute or an empty
// list; the ONNX parser warns about each absent one, so a writer sets it empty.
#ifndef AUTOWARE__TENSORRT_PLUGINS__SUBM_CONTRACT_HPP_
#define AUTOWARE__TENSORRT_PLUGINS__SUBM_CONTRACT_HPP_

#include <NvInferRuntime.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace autoware::tensorrt_plugins::subm
{


enum class Mode { kFloat, k8BitIn, k8BitInOut };
const char* to_string(Mode mode) noexcept;

/// Input positions.
enum class Slot : std::int32_t {
  kFeatures,
  kWeights,
  kPairs,
  kPointMask,
  kMaskArgsort,
  kTileMask,
  kBias,
};

constexpr std::int32_t kInputs = 7;

/// The rulebook tensors' granularities, t4sparse's: `tile_mask` has one word per kMaskRows rows
/// and `pairs` is padded to a multiple of kPadRows rows.
constexpr std::int32_t kMaskRows = 32;
constexpr std::int32_t kPadRows = 256;

/// The rows `pairs` has for `rows` points.
constexpr std::int32_t padded_rows(std::int32_t rows) noexcept
{
  return (rows + kPadRows - 1) / kPadRows * kPadRows;
}

/// "[3, 3, 3, 256]"
std::string dims_string(const nvinfer1::Dims & dims);

/// The position of a slot among the inputs.
constexpr std::int32_t at(Slot slot) noexcept { return static_cast<std::int32_t>(slot); }

/// The scale attributes a node carries.
struct Scales {
  std::optional<float> input;
  std::vector<float> weight;  ///< empty when absent
  std::optional<float> output;
};

/// Why something is refused, or nothing when it is accepted.
using Refusal = std::optional<std::string>;

/// kFloat when no scale is set, k8BitIn with input_scale and weight_scale,
/// k8BitInOut with all three; nothing for any other combination.
std::optional<Mode> mode_for(const Scales& scales) noexcept;

/// Refuses scales set without their partner, and values that are not positive and finite.
Refusal scale_refusal(const Scales& scales);

bool is_8bit(nvinfer1::DataType type) noexcept;
bool is_float(nvinfer1::DataType type) noexcept;
const char* type_name(nvinfer1::DataType type) noexcept;

/// The output element type for these features.
nvinfer1::DataType output_type(Mode mode, nvinfer1::DataType features_type) noexcept;

/// Whether the tensor at `position` (`kInputs` is the output) may have `type`
/// when the features have `features_type`. Says nothing about which features
/// have a kernel.
bool accepts(std::int32_t position, nvinfer1::DataType type, nvinfer1::DataType features_type,
             Mode mode) noexcept;

/// Refuses the scales, the input count or an input's type. The weights' type is
/// left to `refusal`.
Refusal type_refusal(const nvinfer1::DataType* input_types, std::int32_t n_inputs,
                     const Scales& scales);

/// Refuses everything the table says, plus shapes that disagree with each
/// other; nothing about the kernel table.
Refusal refusal(const nvinfer1::DynamicPluginTensorDesc* input_descs, std::int32_t n_inputs,
                nvinfer1::DataType output_type_given, const Scales& scales);

/// What the contract accepts and spconv does not run: bfloat16 and float8e4m3 features, and
/// an 8-bit output.
Refusal spconv_refusal(nvinfer1::DataType features_type, const Scales & scales);

}  // namespace autoware::tensorrt_plugins::subm

#endif  // AUTOWARE__TENSORRT_PLUGINS__SUBM_CONTRACT_HPP_
