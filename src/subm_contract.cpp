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

#include "autoware/tensorrt_plugins/subm_contract.hpp"

#include <cmath>
#include <cstdio>

#include <string>

namespace autoware::tensorrt_plugins::subm
{

using nvinfer1::DataType;
using nvinfer1::Dims;
using nvinfer1::DynamicPluginTensorDesc;

namespace {

/// The kernel taps are bits of one 32-bit mask word.
constexpr std::int32_t kMaxKernelVolume = 32;

constexpr const char* kSlotNames[kInputs] = {
    "features",
    "weights",
    "pairs",
    "point_mask",
    "mask_argsort",
    "tile_mask",
    "bias",
};

template <class... Args>
std::string format(const char* pattern, Args... args) {
  const int length = std::snprintf(nullptr, 0, pattern, args...);
  if (length <= 0) return {};
  std::string text(static_cast<std::size_t>(length), '\0');
  std::snprintf(text.data(), text.size() + 1, pattern, args...);
  return text;
}

long long as_long(std::int64_t value) { return static_cast<long long>(value); }

bool is_positive_and_finite(float value) { return value > 0.f && std::isfinite(value); }

/// A node's input tensors, addressed by slot.
struct NodeInputs {
  const DynamicPluginTensorDesc* descs;

  DataType type(Slot slot) const { return descs[at(slot)].desc.type; }
  /// The shape as the graph declares it; a dynamic axis is -1.
  const Dims& dims(Slot slot) const { return descs[at(slot)].desc.dims; }
  /// The shape at the largest size the engine may see.
  const Dims& max_dims(Slot slot) const { return descs[at(slot)].max; }
  std::string shape(Slot slot) const { return dims_string(max_dims(slot)); }
};

/// The layer's dimensions as the weights give them: [C_out, k0, k1, k2, C_in].
struct LayerDims {
  std::int32_t c_out;
  std::int32_t c_in;
  std::int64_t kernel[3];
};

LayerDims layer_dims_of(const Dims& weights) {
  return {static_cast<std::int32_t>(weights.d[0]), static_cast<std::int32_t>(weights.d[4]),
          {weights.d[1], weights.d[2], weights.d[3]}};
}

/// Features [N, C_in] and weights [C_out, k0, k1, k2, C_in] with an odd kernel.
Refusal layer_dims_refusal(const NodeInputs& inputs) {
  const Dims& features = inputs.dims(Slot::kFeatures);
  if (features.nbDims != 2) {
    return format("features are %s; want [N, C_in]", inputs.shape(Slot::kFeatures).c_str());
  }

  const Dims& weights = inputs.dims(Slot::kWeights);
  if (weights.nbDims != 5) {
    return format("weights are %s; want [C_out, k0, k1, k2, C_in]",
                  inputs.shape(Slot::kWeights).c_str());
  }

  const LayerDims layer = layer_dims_of(weights);
  const std::int64_t feature_channels = features.d[1];
  if (feature_channels > 0 && feature_channels != layer.c_in) {
    return format("features have %lld channels, the weights %d", as_long(feature_channels),
                  layer.c_in);
  }

  std::int64_t kernel_volume = 1;
  for (const std::int64_t extent : layer.kernel) {
    const bool is_odd = extent >= 1 && extent % 2 == 1;
    if (!is_odd) {
      return format("weights are %s; want an odd kernel on every axis",
                    inputs.shape(Slot::kWeights).c_str());
    }
    kernel_volume *= extent;
  }
  if (kernel_volume > kMaxKernelVolume) {
    return format("kernel volume %lld; at most %d", as_long(kernel_volume), kMaxKernelVolume);
  }
  return std::nullopt;
}

/// The rulebook's four tensors are functions of the point count N and the
/// kernel. TensorRT gives the largest N the engine may see, and each tensor's
/// largest shape has to be that function of it.
Refusal rulebook_refusal(const NodeInputs& inputs, const LayerDims& layer) {
  const auto max_points = static_cast<std::int32_t>(inputs.max_dims(Slot::kFeatures).d[0]);
  const std::int32_t max_rows = padded_rows(max_points);
  const std::int32_t max_tiles = max_rows / kMaskRows;
  const bool points_known = max_points > 0;

  /// Whether a tensor's largest extent on `axis` is `expected`.
  const auto max_extent_is = [&](Slot slot, int axis, std::int32_t expected) {
    return !points_known || inputs.max_dims(slot).d[axis] == expected;
  };

  // pairs: [k0, k1, k2, P], P = N padded
  const Dims& pairs = inputs.dims(Slot::kPairs);
  const bool pairs_match_kernel = pairs.nbDims == 4 && pairs.d[0] == layer.kernel[0] &&
                                  pairs.d[1] == layer.kernel[1] && pairs.d[2] == layer.kernel[2];
  if (!pairs_match_kernel || !max_extent_is(Slot::kPairs, 3, max_rows)) {
    return format("pairs is %s for N=%d; the weights' kernel wants [%lld, %lld, %lld, %d]",
                  inputs.shape(Slot::kPairs).c_str(), max_points, as_long(layer.kernel[0]),
                  as_long(layer.kernel[1]), as_long(layer.kernel[2]), max_rows);
  }

  // point_mask: [N, 1]
  const Dims& point_mask = inputs.dims(Slot::kPointMask);
  const bool point_mask_is_column = point_mask.nbDims == 2 && point_mask.d[1] == 1;
  if (!point_mask_is_column || !max_extent_is(Slot::kPointMask, 0, max_points)) {
    return format("point_mask is %s; want [%d, 1]", inputs.shape(Slot::kPointMask).c_str(),
                  max_points);
  }

  // mask_argsort: [N]
  const bool argsort_is_vector = inputs.dims(Slot::kMaskArgsort).nbDims == 1;
  if (!argsort_is_vector || !max_extent_is(Slot::kMaskArgsort, 0, max_points)) {
    return format("mask_argsort is %s; want [%d]", inputs.shape(Slot::kMaskArgsort).c_str(),
                  max_points);
  }

  // tile_mask: [P / kMaskRows]
  const bool tile_mask_is_vector = inputs.dims(Slot::kTileMask).nbDims == 1;
  if (!tile_mask_is_vector || !max_extent_is(Slot::kTileMask, 0, max_tiles)) {
    return format("tile_mask is %s; want [%d] for N=%d", inputs.shape(Slot::kTileMask).c_str(),
                  max_tiles, max_points);
  }
  return std::nullopt;
}

/// Which scale is set without the one it needs; for scales mode_for() names no mode.
std::string unpaired_scale(const Scales& scales) {
  const bool has_input = scales.input.has_value();
  const bool has_weight = !scales.weight.empty();
  const char* present = has_input ? "input_scale" : has_weight ? "weight_scale" : "output_scale";
  const char* missing = has_input ? "weight_scale" : "input_scale";
  return format("%s without %s; the 8-bit modes carry input_scale and weight_scale", present,
                missing);
}

/// Refuses a set scale that is not positive and finite.
Refusal scale_value_refusal(const Scales& scales) {
  if (scales.input && !is_positive_and_finite(*scales.input)) {
    return format("input_scale is %g; want positive and finite", *scales.input);
  }
  for (std::size_t channel = 0; channel < scales.weight.size(); ++channel) {
    if (!is_positive_and_finite(scales.weight[channel])) {
      return format("weight_scale[%zu] is %g; want positive and finite", channel,
                    scales.weight[channel]);
    }
  }
  if (scales.output && !is_positive_and_finite(*scales.output)) {
    return format("output_scale is %g; want positive and finite", *scales.output);
  }
  return std::nullopt;
}

}  // namespace

const char* to_string(Mode mode) noexcept {
  switch (mode) {
    case Mode::kFloat: return "float";
    case Mode::k8BitIn: return "8-bit in";
    case Mode::k8BitInOut: return "8-bit in and out";
  }
  return "?";
}

std::optional<Mode> mode_for(const Scales& scales) noexcept {
  const bool has_input = scales.input.has_value();
  const bool has_weight = !scales.weight.empty();
  const bool has_output = scales.output.has_value();

  // input_scale and weight_scale come as a pair; output_scale needs the pair.
  if (has_input != has_weight) return std::nullopt;
  if (!has_input) return has_output ? std::nullopt : std::optional(Mode::kFloat);
  return has_output ? Mode::k8BitInOut : Mode::k8BitIn;
}

Refusal scale_refusal(const Scales& scales) {
  if (!mode_for(scales)) return unpaired_scale(scales);
  return scale_value_refusal(scales);
}

bool is_8bit(DataType type) noexcept { return type == DataType::kINT8 || type == DataType::kFP8; }

bool is_float(DataType type) noexcept {
  return type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kBF16;
}

const char* type_name(DataType type) noexcept {
  switch (type) {
    case DataType::kFLOAT: return "float32";
    case DataType::kHALF: return "float16";
    case DataType::kBF16: return "bfloat16";
    case DataType::kINT8: return "int8";
    case DataType::kFP8: return "float8e4m3";
    case DataType::kINT32: return "int32";
    case DataType::kINT64: return "int64";
    default: return "other";
  }
}

DataType output_type(Mode mode, DataType features_type) noexcept {
  return mode == Mode::k8BitIn ? DataType::kHALF : features_type;
}

bool accepts(std::int32_t position, DataType type, DataType features_type, Mode mode) noexcept {
  if (position == kInputs) return type == output_type(mode, features_type);
  if (position < 0 || position > kInputs) return false;

  switch (static_cast<Slot>(position)) {
    case Slot::kFeatures: return mode == Mode::kFloat ? is_float(type) : is_8bit(type);
    case Slot::kWeights: return type == features_type;
    case Slot::kPairs:
    case Slot::kPointMask:
    case Slot::kMaskArgsort:
    case Slot::kTileMask: return type == DataType::kINT32;
    case Slot::kBias: return is_float(type);
  }
  return false;
}

Refusal type_refusal(const DataType* input_types, std::int32_t n_inputs, const Scales& scales) {
  if (n_inputs != kInputs) return format("%d inputs; want %d", n_inputs, kInputs);

  const std::optional<Mode> named_mode = mode_for(scales);
  if (!named_mode) return unpaired_scale(scales);
  const Mode mode = *named_mode;
  if (auto why = scale_value_refusal(scales)) return why;

  const DataType features_type = input_types[at(Slot::kFeatures)];
  if (!accepts(at(Slot::kFeatures), features_type, features_type, mode)) {
    // mode_for() names the float mode exactly when no scale attribute is set.
    if (mode == Mode::kFloat) {
      return format("%s features without scale attributes; want float32, float16 or bfloat16",
                    type_name(features_type));
    }
    return format("%s features with scale attributes; want int8 or float8e4m3",
                  type_name(features_type));
  }

  // The rulebook tensors and the bias; the weights' type is `refusal`'s.
  for (std::int32_t position = at(Slot::kPairs); position < kInputs; ++position) {
    if (!accepts(position, input_types[position], features_type, mode)) {
      const char* expected =
          position == at(Slot::kBias) ? "float32, float16 or bfloat16" : "int32";
      return format("%s is %s; want %s", kSlotNames[position], type_name(input_types[position]),
                    expected);
    }
  }
  return std::nullopt;
}

Refusal refusal(const DynamicPluginTensorDesc* input_descs, std::int32_t n_inputs,
                DataType output_type_given, const Scales& scales) {
  // Types first: the same checks getOutputDataTypes made while the graph was built.
  DataType input_types[kInputs];
  for (std::int32_t position = 0; position < n_inputs && position < kInputs; ++position) {
    input_types[position] = input_descs[position].desc.type;
  }
  if (auto why = type_refusal(input_types, n_inputs, scales)) return why;

  const std::optional<Mode> named_mode = mode_for(scales);
  if (!named_mode) return unpaired_scale(scales);
  const Mode mode = *named_mode;

  const NodeInputs inputs{input_descs};
  const DataType features_type = inputs.type(Slot::kFeatures);
  const DataType weights_type = inputs.type(Slot::kWeights);
  if (weights_type != features_type) {
    return format("weights are %s, the features %s; they share one element type",
                  type_name(weights_type), type_name(features_type));
  }

  // Shapes, each against the layer the weights describe.
  if (auto why = layer_dims_refusal(inputs)) return why;
  const LayerDims layer = layer_dims_of(inputs.dims(Slot::kWeights));
  if (auto why = rulebook_refusal(inputs, layer)) return why;

  const Dims& bias = inputs.dims(Slot::kBias);
  const bool bias_is_per_channel = bias.nbDims == 1 && bias.d[0] == layer.c_out;
  if (!bias_is_per_channel) {
    return format("bias is %s; want [%d]", inputs.shape(Slot::kBias).c_str(), layer.c_out);
  }

  const bool is_quantized = mode != Mode::kFloat;
  if (is_quantized && scales.weight.size() != static_cast<std::size_t>(layer.c_out)) {
    return format("weight_scale has %zu entries, the weights %d output channels",
                  scales.weight.size(), layer.c_out);
  }

  const DataType output_type_expected = output_type(mode, features_type);
  if (output_type_given != output_type_expected) {
    return format("output is %s; the %s mode writes %s", type_name(output_type_given),
                  to_string(mode), type_name(output_type_expected));
  }
  return std::nullopt;
}

std::string dims_string(const Dims & dims)
{
  std::string text = "[";
  for (std::int32_t i = 0; i < dims.nbDims; ++i) {
    if (i > 0) {
      text += ", ";
    }
    text += std::to_string(dims.d[i]);
  }
  return text + "]";
}

Refusal spconv_refusal(DataType features_type, const Scales & scales)
{
  if (features_type == DataType::kBF16 || features_type == DataType::kFP8) {
    return format(
      "%s features; the spconv library runs float32, float16 and int8",
      type_name(features_type));
  }
  if (mode_for(scales) == Mode::k8BitInOut) {
    return std::string("output_scale is set; the spconv library writes no 8-bit output");
  }
  return std::nullopt;
}

}  // namespace autoware::tensorrt_plugins::subm
