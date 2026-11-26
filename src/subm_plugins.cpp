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

// SubMRulebook and SubMConv on spconv. The node pair, its attributes and its tensor shapes are
// t4sparse's (subm_contract.hpp), so a graph exported for one library builds against the other.
// What the four rulebook tensors hold is each library's own: a pair is always one library's.

#include "autoware/subm_ops/convert_bias.hpp"
#include "autoware/tensorrt_plugins/plugin_utils.hpp"
#include "autoware/tensorrt_plugins/subm_contract.hpp"
#include "autoware/tensorrt_plugins/subm_plugin_creators.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_runtime.h>
#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>  // cSpell:ignore spconvlib
#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/gemmops/GemmTunerSimple.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::tensorrt_plugins
{
namespace
{

using nvinfer1::DataType;
using nvinfer1::Dims;
using nvinfer1::DimsExprs;
using nvinfer1::DynamicPluginTensorDesc;
using nvinfer1::IPluginV3;
using nvinfer1::PluginField;
using nvinfer1::PluginFieldCollection;
using nvinfer1::PluginFieldType;
using nvinfer1::PluginTensorDesc;
using nvinfer1::TensorRTPhase;
using subm::at;
using subm::Slot;

void log_error(const std::string & message)
{
  getLogger()->log(nvinfer1::ILogger::Severity::kERROR, message.c_str());
}

/// Logs "<plugin> <node>: <why>" and returns -1, the build-refusing status.
std::int32_t refuse(const char * plugin, const std::string & node, const std::string & why)
{
  log_error(std::string(plugin) + " " + node + ": " + why);
  return -1;
}

// --- attribute readers ---------------------------------------------------------------------
// The ONNX parser passes INTS as int32 or int64 and counts a string's terminator in its length;
// a hand-built field may do neither.

constexpr std::int32_t kMaxIntsLength = 8;

void read_ints(const PluginField & field, std::vector<std::int32_t> & out)
{
  if (field.data == nullptr || field.length < 1 || field.length > kMaxIntsLength) {
    return;
  }
  if (field.type == PluginFieldType::kDIMS) {
    const auto * dims = static_cast<const Dims *>(field.data);
    out.assign(dims->d, dims->d + dims->nbDims);
  } else if (field.type == PluginFieldType::kINT64) {
    const auto * values = static_cast<const std::int64_t *>(field.data);
    out.assign(values, values + field.length);
  } else {
    const auto * values = static_cast<const std::int32_t *>(field.data);
    out.assign(values, values + field.length);
  }
}

std::int32_t read_int(const PluginField & field, std::int32_t fallback)
{
  if (field.data == nullptr) {
    return fallback;
  }
  if (field.type == PluginFieldType::kINT64) {
    return static_cast<std::int32_t>(*static_cast<const std::int64_t *>(field.data));
  }
  return *static_cast<const std::int32_t *>(field.data);
}

std::string read_string(const PluginField & field)
{
  if (field.data == nullptr || field.type != PluginFieldType::kCHAR || field.length < 0) {
    return {};
  }
  const auto * chars = static_cast<const char *>(field.data);
  return std::string(chars, strnlen(chars, static_cast<std::size_t>(field.length)));
}

float read_float(const PluginField & field, float fallback)
{
  if (field.data == nullptr || field.type != PluginFieldType::kFLOAT32) {
    return fallback;
  }
  return *static_cast<const float *>(field.data);
}

/// One float32, or a problem naming the attribute; an empty list is no scale.
void read_scale(const PluginField & field, std::optional<float> & out, std::string & problem)
{
  if (field.length == 0) {
    return;
  }
  if (field.type != PluginFieldType::kFLOAT32 || field.length != 1) {
    problem = std::string(field.name) + " is not one float32";
  } else {
    out = *static_cast<const float *>(field.data);
  }
}

/// A float32 list, or a problem naming the attribute; an empty list is no scale.
void read_scales(const PluginField & field, std::vector<float> & out, std::string & problem)
{
  if (field.length == 0) {
    return;
  }
  if (field.type != PluginFieldType::kFLOAT32 || field.length < 1) {
    problem = std::string(field.name) + " is not a float32 list";
    return;
  }
  const auto * values = static_cast<const float *>(field.data);
  out.assign(values, values + field.length);
}

std::string extent(const std::vector<std::int32_t> & values)
{
  std::string text;
  for (const auto value : values) {
    text += (text.empty() ? "" : "x") + std::to_string(value);
  }
  return text;
}

// --- SubMRulebook --------------------------------------------------------------------------
//
// attributes: kernel_size ints[3], spatial_shape ints[3]
// in : 0 coords       int32 [N, 4]
// out: 0 pairs        int32 [k0, k1, k2, P], P = N rounded up to kPadRows
//      1 point_mask   int32 [N, 1]
//      2 mask_argsort int32 [N]
//      3 tile_mask    int32 [P / kMaskRows]
//
// The shapes are the contract's. The contents are spconv's implicit-GEMM rulebook: `pairs`
// holds spconv's [kernel_volume, N] pair table at its start, `point_mask` and `mask_argsort`
// are spconv's, and `tile_mask`, which spconv has no use for, is zero.

struct RulebookAttrs
{
  std::vector<std::int32_t> kernel_size;
  std::vector<std::int32_t> spatial_shape;
  /// spconv's pair-mask argsort. Not a node attribute: on unless the engine was built with
  /// AUTOWARE_SUBM_RULEBOOK_SORT=0, and carried in the engine.
  std::int32_t do_sort{1};
  std::string node_name;

  std::int32_t kernel_volume() const
  {
    return std::accumulate(kernel_size.begin(), kernel_size.end(), 1, std::multiplies<>());
  }

  /// Why a node has no rulebook, or empty. t4sparse's limits, so that a graph one library
  /// builds is not refused by the other: odd kernels of at most 32 taps.
  std::string problem() const
  {
    if (kernel_size.empty()) {
      return "kernel_size is missing";
    }
    if (spatial_shape.empty()) {
      return "spatial_shape is missing";
    }
    if (kernel_size.size() != 3 || spatial_shape.size() != 3) {
      return "kernel_size " + extent(kernel_size) + ", spatial_shape " + extent(spatial_shape) +
             "; three ints each";
    }
    std::int64_t taps = 1;
    for (const auto k : kernel_size) {
      if (k < 1 || k % 2 == 0) {
        return "kernel_size " + extent(kernel_size) + "; odd on every axis";
      }
      taps *= k;
    }
    if (taps > 32) {
      return "kernel_size " + extent(kernel_size) + " has " + std::to_string(taps) +
             " taps; at most 32";
    }
    for (const auto s : spatial_shape) {
      if (s < 1) {
        return "spatial_shape " + extent(spatial_shape) + "; positive on every axis";
      }
    }
    return {};
  }
};

class SubMRulebookPlugin : public IPluginV3,
                           public nvinfer1::IPluginV3OneCore,
                           public nvinfer1::IPluginV3OneBuild,
                           public nvinfer1::IPluginV3OneRuntime
{
public:
  SubMRulebookPlugin(RulebookAttrs attrs, TensorRTPhase phase)
  : attrs_(std::move(attrs)), build_phase_(phase == TensorRTPhase::kBUILD)
  {
    fields_.emplace_back(
      "kernel_size", attrs_.kernel_size.data(), PluginFieldType::kINT32,
      static_cast<std::int32_t>(attrs_.kernel_size.size()));
    fields_.emplace_back(
      "spatial_shape", attrs_.spatial_shape.data(), PluginFieldType::kINT32,
      static_cast<std::int32_t>(attrs_.spatial_shape.size()));
    fields_.emplace_back("do_sort", &attrs_.do_sort, PluginFieldType::kINT32, 1);
    fc_.nbFields = static_cast<std::int32_t>(fields_.size());
    fc_.fields = fields_.data();
  }

  nvinfer1::IPluginCapability * getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept override
  {
    switch (type) {
      case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore *>(this);
      case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild *>(this);
      case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime *>(this);
    }
    return nullptr;
  }

  IPluginV3 * clone() noexcept override
  {
    try {
      return new SubMRulebookPlugin(
        attrs_, build_phase_ ? TensorRTPhase::kBUILD : TensorRTPhase::kRUNTIME);
    } catch (const std::exception & e) {
      caughtError(e);
    }
    return nullptr;
  }

  const char * getPluginName() const noexcept override { return kSUBM_RULEBOOK_PLUGIN_NAME; }
  const char * getPluginVersion() const noexcept override { return kSUBM_PLUGIN_VERSION; }
  const char * getPluginNamespace() const noexcept override { return kSUBM_PLUGIN_NAMESPACE; }

  std::int32_t getNbOutputs() const noexcept override { return 4; }

  std::int32_t configurePlugin(
    const DynamicPluginTensorDesc * in, std::int32_t num_inputs, const DynamicPluginTensorDesc *,
    std::int32_t num_outputs) noexcept override
  {
    if (num_inputs != 1 || num_outputs != 4) {
      return fail(
        std::to_string(num_inputs) + " inputs, " + std::to_string(num_outputs) +
        " outputs; want 1 and 4");
    }
    if (in[0].desc.dims.nbDims != 2 || in[0].desc.dims.d[1] != 4) {
      return fail("coords are " + subm::dims_string(in[0].desc.dims) + "; want [N, 4]");
    }
    if (const auto why = attrs_.problem(); !why.empty()) {
      return fail(why);
    }
    if (in[0].max.d[0] > kMaxPoints) {
      return fail(
        "up to " + std::to_string(in[0].max.d[0]) + " points; this plugin indexes at most " +
        std::to_string(kMaxPoints));
    }
    return 0;
  }

  std::int32_t getOutputDataTypes(
    DataType * out, std::int32_t num_outputs, const DataType *, std::int32_t) const noexcept override
  {
    for (std::int32_t i = 0; i < num_outputs; ++i) {
      out[i] = DataType::kINT32;
    }
    return 0;
  }

  std::int32_t getOutputShapes(
    const DimsExprs * inputs, std::int32_t num_inputs, const DimsExprs *, std::int32_t,
    DimsExprs * out, std::int32_t num_outputs, nvinfer1::IExprBuilder & builder) noexcept override
  {
    if (num_inputs != 1 || num_outputs != 4 || inputs[0].nbDims != 2) {
      return -1;
    }
    if (const auto why = attrs_.problem(); !why.empty()) {
      return fail(why);
    }
    using nvinfer1::DimensionOperation;
    const nvinfer1::IDimensionExpr * n = inputs[0].d[0];
    const auto * padded = builder.operation(
      DimensionOperation::kPROD,
      *builder.operation(
        DimensionOperation::kFLOOR_DIV,
        *builder.operation(DimensionOperation::kSUM, *n, *builder.constant(subm::kPadRows - 1)),
        *builder.constant(subm::kPadRows)),
      *builder.constant(subm::kPadRows));

    out[0].nbDims = 4;
    for (int axis = 0; axis < 3; ++axis) {
      out[0].d[axis] = builder.constant(attrs_.kernel_size[axis]);
    }
    out[0].d[3] = padded;

    out[1].nbDims = 2;
    out[1].d[0] = n;
    out[1].d[1] = builder.constant(1);

    out[2].nbDims = 1;
    out[2].d[0] = n;

    out[3].nbDims = 1;
    out[3].d[0] = builder.operation(
      DimensionOperation::kFLOOR_DIV, *padded, *builder.constant(subm::kMaskRows));
    return 0;
  }

  bool supportsFormatCombination(
    std::int32_t pos, const DynamicPluginTensorDesc * in_out, std::int32_t,
    std::int32_t) noexcept override
  {
    return in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
           in_out[pos].desc.type == DataType::kINT32;
  }

  std::size_t getWorkspaceSize(
    const DynamicPluginTensorDesc *, std::int32_t, const DynamicPluginTensorDesc *,
    std::int32_t) const noexcept override
  {
    if (!attrs_.problem().empty()) {
      return 0;
    }
    const Layout layout = layout_of(attrs_);
    return layout.thrust_tmp_offset + kThrustTempBytes;
  }

  std::int32_t onShapeChange(
    const PluginTensorDesc *, std::int32_t, const PluginTensorDesc *, std::int32_t) noexcept override
  {
    return 0;
  }

  IPluginV3 * attachToContext(nvinfer1::IPluginResourceContext *) noexcept override
  {
    return clone();
  }

  const PluginFieldCollection * getFieldsToSerialize() noexcept override { return &fc_; }

  std::int32_t enqueue(
    const PluginTensorDesc * input_desc, const PluginTensorDesc *, const void * const * inputs,
    void * const * outputs, void * workspace, cudaStream_t stream) noexcept override
  {
    using SpconvOps = spconvlib::spconv::csrc::sparse::all::SpconvOps;
    using StaticAllocator = spconvlib::spconv::csrc::sparse::alloc::StaticAllocator;

    const std::int64_t n = input_desc[0].dims.d[0];
    // TensorRT times a plugin during the build against buffers it never filled; nothing reads
    // this output then, so a build-phase instance does nothing.
    if (n <= 0 || build_phase_) {
      return 0;
    }
    if (n > kMaxPoints) {
      return fail(std::to_string(n) + " points; at most " + std::to_string(kMaxPoints));
    }

    try {
      const Layout layout = layout_of(attrs_);
      const int kernel_volume = attrs_.kernel_volume();
      const std::int32_t padded = subm::padded_rows(static_cast<std::int32_t>(n));
      auto * const ws = static_cast<std::uint8_t *>(workspace);

      const std::vector<int> ksize(attrs_.kernel_size.begin(), attrs_.kernel_size.end());
      const std::vector<int> spatial(attrs_.spatial_shape.begin(), attrs_.spatial_shape.end());
      const std::vector<int> ones{1, 1, 1};
      std::vector<int> padding;
      for (const auto k : ksize) {
        padding.push_back(k / 2);
      }

      const auto max_act_out_theory = SpconvOps::get_handcrafted_max_act_out(
        static_cast<std::size_t>(n), ksize, ones, padding, ones);
      auto ws_tensors = SpconvOps::get_indice_gen_tensors_from_workspace(
        ws, kernel_volume, kMaxPoints, kMaxPoints, max_act_out_theory, true,
        layout.use_int64_hash_k, false);

      tv::Tensor input_indices =
        tv::from_blob(const_cast<void *>(inputs[0]), {n, 4}, tv::int32, 0);
      tv::Tensor pair_fwd = tv::from_blob(outputs[0], {kernel_volume, n}, tv::int32, 0);
      tv::Tensor pair_mask = tv::from_blob(outputs[1], {1, n}, tv::int32, 0);
      tv::Tensor mask_argsort = tv::from_blob(outputs[2], {1, n}, tv::int32, 0);
      tv::Tensor out_indices =
        tv::from_blob(ws + layout.out_indices_offset, {n, 4}, tv::int32, 0);
      tv::Tensor indices_kernel_num =
        tv::from_blob(ws + layout.kernel_num_offset, {kernel_volume}, tv::int32, 0);
      tv::Tensor thrust_tmp = tv::from_blob(
        ws + layout.thrust_tmp_offset, {static_cast<std::int64_t>(kThrustTempBytes)}, tv::uint8,
        0);

      if (
        PLUGIN_CUDA_CHECK(cudaMemsetAsync(
          ws + layout.kernel_num_offset, 0, static_cast<std::size_t>(kernel_volume) * 4, stream)) !=
          cudaSuccess ||
        PLUGIN_CUDA_CHECK(cudaMemsetAsync(
          outputs[3], 0, static_cast<std::size_t>(padded / subm::kMaskRows) * 4, stream)) !=
          cudaSuccess) {
        return -1;
      }

      tv::Context ctx;
      ctx.set_cuda_stream_int(reinterpret_cast<std::uintptr_t>(stream));
      out_indices.copy_(input_indices, ctx);

      ws_tensors.emplace(SPCONV_ALLOC_PAIR_FWD, pair_fwd);
      ws_tensors.emplace(SPCONV_ALLOC_PAIR_MASK, pair_mask);
      ws_tensors.emplace(SPCONV_ALLOC_MASK_ARG_SORT, mask_argsort);
      ws_tensors.emplace(SPCONV_ALLOC_OUT_INDICES, out_indices);
      ws_tensors.emplace(
        SPCONV_ALLOC_INDICE_NUM_PER_LOC, indices_kernel_num);  // cSpell:ignore INDICE
      StaticAllocator alloc(ws_tensors);
      alloc.thrust_tmp_tensor_ = thrust_tmp;

      // cSpell:ignore indice subm
      SpconvOps::get_indice_pairs_implicit_gemm(
        alloc, input_indices, 1 /*batch_size*/, spatial,
        static_cast<int>(tv::gemm::SparseConvAlgo::kMaskImplicitGemm), ksize, ones, padding, ones,
        {0, 0, 0}, true /*subm*/, false /*transposed*/, false /*is_train*/,
        reinterpret_cast<std::uintptr_t>(stream), kMaxPoints, tv::CUDAKernelTimer(false),
        false /*direct_table*/, attrs_.do_sort != 0);
    } catch (const std::exception & e) {
      caughtError(e);
      return -1;
    }
    return 0;
  }

private:
  /// The most points a rulebook is built for; bounds the workspace, as in
  /// GetIndicePairsImplicitGemm.
  static constexpr int kMaxPoints{512000};
  /// Scratch for spconv's sorts.
  static constexpr std::size_t kThrustTempBytes{8U * 1024U * 1024U};

  /// The workspace: spconv's index-generation buffers, then the per-tap counts, spconv's copy
  /// of the coordinates and the sort scratch.
  struct Layout
  {
    bool use_int64_hash_k;
    std::size_t kernel_num_offset;
    std::size_t out_indices_offset;
    std::size_t thrust_tmp_offset;
  };

  static Layout layout_of(const RulebookAttrs & attrs)
  {
    using SpconvOps = spconvlib::spconv::csrc::sparse::all::SpconvOps;
    std::int64_t spatial_volume = 1;
    for (const auto s : attrs.spatial_shape) {
      spatial_volume *= s;
    }
    Layout layout{};
    layout.use_int64_hash_k =
      spatial_volume >= static_cast<std::int64_t>(std::numeric_limits<int>::max());
    const int kernel_volume = attrs.kernel_volume();
    layout.kernel_num_offset = static_cast<std::size_t>(SpconvOps::get_indice_gen_workspace_size(
      kernel_volume, kMaxPoints, kMaxPoints, kMaxPoints, true, layout.use_int64_hash_k, false));
    layout.out_indices_offset =
      layout.kernel_num_offset + static_cast<std::size_t>(kernel_volume) * sizeof(std::int32_t);
    layout.thrust_tmp_offset =
      layout.out_indices_offset + static_cast<std::size_t>(kMaxPoints) * 4 * sizeof(std::int32_t);
    return layout;
  }

  std::int32_t fail(const std::string & why) const
  {
    return refuse(kSUBM_RULEBOOK_PLUGIN_NAME, attrs_.node_name, why);
  }

  RulebookAttrs attrs_;
  bool build_phase_{false};
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

class SubMRulebookPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
  SubMRulebookPluginCreator()
  {
    fields_.emplace_back("kernel_size", nullptr, PluginFieldType::kINT32, 3);
    fields_.emplace_back("spatial_shape", nullptr, PluginFieldType::kINT32, 3);
    fc_.nbFields = static_cast<std::int32_t>(fields_.size());
    fc_.fields = fields_.data();
  }

  IPluginV3 * createPlugin(
    const char * name, const PluginFieldCollection * fc, TensorRTPhase phase) noexcept override
  {
    try {
      RulebookAttrs attrs;
      if (name != nullptr) {
        attrs.node_name = name;
      }
      if (phase == TensorRTPhase::kBUILD) {
        const char * sort = std::getenv("AUTOWARE_SUBM_RULEBOOK_SORT");
        attrs.do_sort = (sort != nullptr && std::strcmp(sort, "0") == 0) ? 0 : 1;
      }
      for (std::int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i) {
        const PluginField & field = fc->fields[i];
        if (field.data == nullptr) {
          continue;
        }
        const std::string field_name = field.name;
        if (field_name == "kernel_size") {
          read_ints(field, attrs.kernel_size);
        } else if (field_name == "spatial_shape") {
          read_ints(field, attrs.spatial_shape);
        } else if (field_name == "do_sort") {
          attrs.do_sort = read_int(field, attrs.do_sort);
        }
      }
      return new SubMRulebookPlugin(std::move(attrs), phase);
    } catch (const std::exception & e) {
      caughtError(e);
    }
    return nullptr;
  }

  const PluginFieldCollection * getFieldNames() noexcept override { return &fc_; }
  const char * getPluginName() const noexcept override { return kSUBM_RULEBOOK_PLUGIN_NAME; }
  const char * getPluginVersion() const noexcept override { return kSUBM_PLUGIN_VERSION; }
  const char * getPluginNamespace() const noexcept override { return kSUBM_PLUGIN_NAMESPACE; }

private:
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

// --- SubMConv ------------------------------------------------------------------------------
//
// attributes: activation string, act_alpha float, and the scales
// scales, inputs and output: subm_contract.hpp

struct ConvAttrs
{
  std::string activation{"none"};
  float act_alpha{0.0F};
  subm::Scales scales;
  std::string problem;  ///< an attribute of the wrong type, or empty
  std::string node_name;
  /// False for SubMConv: one output row per input point. True for SparseConv: the output rows
  /// are point_mask's, which SparseRulebook computed.
  bool computed_rows{false};

  const char * plugin_name() const
  {
    return computed_rows ? kSPARSE_CONV_PLUGIN_NAME : kSUBM_CONV_PLUGIN_NAME;
  }
  subm::Rows rows() const
  {
    return computed_rows ? subm::Rows::kPointMask : subm::Rows::kFeatures;
  }
};

/// spconv's activation for the node's, or nothing for a name the node does not define.
std::optional<tv::gemm::Activation> activation_of(const std::string & name)
{
  if (name == "none") {
    return tv::gemm::Activation::kNone;
  }
  if (name == "relu") {
    return tv::gemm::Activation::kReLU;
  }
  if (name == "leaky_relu") {
    return tv::gemm::Activation::kLeakyReLU;
  }
  return std::nullopt;
}

/// A device copy of a float list.
struct DeviceFloats
{
  float * data{nullptr};
  DeviceFloats() = default;
  DeviceFloats(const DeviceFloats &) = delete;
  DeviceFloats & operator=(const DeviceFloats &) = delete;
  ~DeviceFloats() { cudaFree(data); }
};

class SubMConvPlugin : public IPluginV3,
                       public nvinfer1::IPluginV3OneCore,
                       public nvinfer1::IPluginV3OneBuild,
                       public nvinfer1::IPluginV3OneRuntime
{
public:
  SubMConvPlugin(ConvAttrs attrs, TensorRTPhase phase)
  : attrs_(std::move(attrs)), build_phase_(phase == TensorRTPhase::kBUILD)
  {
    using ConvGemmOps = spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;
    using ConvMain = spconvlib::cumm::conv::main::ConvMainUnitTest;

    fields_.emplace_back(
      "activation", attrs_.activation.c_str(), PluginFieldType::kCHAR,
      static_cast<std::int32_t>(attrs_.activation.size()));
    fields_.emplace_back("act_alpha", &attrs_.act_alpha, PluginFieldType::kFLOAT32, 1);
    const subm::Scales & scales = attrs_.scales;
    if (scales.input) {
      fields_.emplace_back("input_scale", &*scales.input, PluginFieldType::kFLOAT32, 1);
    }
    if (!scales.weight.empty()) {
      fields_.emplace_back(
        "weight_scale", scales.weight.data(), PluginFieldType::kFLOAT32,
        static_cast<std::int32_t>(scales.weight.size()));
    }
    if (scales.output) {
      fields_.emplace_back("output_scale", &*scales.output, PluginFieldType::kFLOAT32, 1);
    }
    fc_.nbFields = static_cast<std::int32_t>(fields_.size());
    fc_.fields = fields_.data();

    // spconv picks its kernel per layer by timing the first calls; one tuner per element type.
    if (!build_phase_) {
      arch_ = ConvGemmOps::get_compute_capability();
      tuner_ = std::make_unique<ConvTuner>(ConvMain::get_all_conv_algo_desp());  // cSpell:ignore desp
      // Read on the host by spconv; allocated here, outside CUDA graph capture.
      mask_tensor_ = tv::zeros({1}, tv::uint32, -1);
      mask_tensor_.data_ptr<std::uint32_t>()[0] = 0xffffffffU;
    }
  }

  nvinfer1::IPluginCapability * getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept override
  {
    switch (type) {
      case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore *>(this);
      case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild *>(this);
      case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime *>(this);
    }
    return nullptr;
  }

  IPluginV3 * clone() noexcept override
  {
    try {
      auto * plugin =
        new SubMConvPlugin(attrs_, build_phase_ ? TensorRTPhase::kBUILD : TensorRTPhase::kRUNTIME);
      plugin->gemm_scale_ = gemm_scale_;
      return plugin;
    } catch (const std::exception & e) {
      caughtError(e);
    }
    return nullptr;
  }

  const char * getPluginName() const noexcept override { return attrs_.plugin_name(); }
  const char * getPluginVersion() const noexcept override { return kSUBM_PLUGIN_VERSION; }
  const char * getPluginNamespace() const noexcept override { return kSUBM_PLUGIN_NAMESPACE; }

  std::int32_t getNbOutputs() const noexcept override { return 1; }

  /// The contract first, then what spconv runs of it.
  std::int32_t configurePlugin(
    const DynamicPluginTensorDesc * in, std::int32_t num_inputs,
    const DynamicPluginTensorDesc * out, std::int32_t) noexcept override
  {
    if (!attrs_.problem.empty()) {
      return fail(attrs_.problem);
    }
    if (const auto why =
          subm::refusal(in, num_inputs, out[0].desc.type, attrs_.scales, attrs_.rows())) {
      return fail(*why);
    }
    if (!activation_of(attrs_.activation)) {
      return fail("activation \"" + attrs_.activation + "\" is not none, relu or leaky_relu");
    }
    return library_refusal(in[at(Slot::kFeatures)].desc.type, in[at(Slot::kBias)].desc.type);
  }

  std::int32_t getOutputDataTypes(
    DataType * out, std::int32_t, const DataType * in,
    std::int32_t num_inputs) const noexcept override
  {
    // A type supportsFormatCombination refuses fails the build with only "could not find any
    // supported formats"; the graph's types reach here first, so the refusal can say why.
    if (!attrs_.problem.empty()) {
      return fail(attrs_.problem);
    }
    if (const auto why = subm::type_refusal(in, num_inputs, attrs_.scales)) {
      return fail(*why);
    }
    if (library_refusal(in[at(Slot::kFeatures)], in[at(Slot::kBias)]) != 0) {
      return -1;
    }
    const std::optional<subm::Mode> mode = subm::mode_for(attrs_.scales);
    if (!mode) {
      return fail("the scale attributes name no mode");
    }
    out[0] = subm::output_type(*mode, in[at(Slot::kFeatures)]);
    return 0;
  }

  std::int32_t getOutputShapes(
    const DimsExprs * inputs, std::int32_t num_inputs, const DimsExprs *, std::int32_t,
    DimsExprs * out, std::int32_t, nvinfer1::IExprBuilder &) noexcept override
  {
    if (num_inputs != subm::kInputs) {
      return fail(
        std::to_string(num_inputs) + " inputs; want " + std::to_string(subm::kInputs));
    }
    if (inputs[0].nbDims != 2 || inputs[1].nbDims != 5) {
      return fail(
        "features of rank " + std::to_string(inputs[0].nbDims) + ", weights of rank " +
        std::to_string(inputs[1].nbDims) + "; want 2 and 5");
    }
    if (attrs_.computed_rows && inputs[at(Slot::kPointMask)].nbDims != 2) {
      return fail(
        "point_mask of rank " + std::to_string(inputs[at(Slot::kPointMask)].nbDims) + "; want 2");
    }
    out[0].nbDims = 2;
    // SubMConv: one row per input point. SparseConv: the rows the rulebook computed.
    out[0].d[0] = attrs_.computed_rows ? inputs[at(Slot::kPointMask)].d[0] : inputs[0].d[0];
    out[0].d[1] = inputs[1].d[0];  // C_out
    return 0;
  }

  bool supportsFormatCombination(
    std::int32_t pos, const DynamicPluginTensorDesc * in_out, std::int32_t num_inputs,
    std::int32_t) noexcept override
  {
    if (in_out[pos].desc.format != nvinfer1::TensorFormat::kLINEAR) {
      return false;
    }
    const auto mode = subm::mode_for(attrs_.scales);
    if (!mode || num_inputs != subm::kInputs) {
      return false;
    }
    const DataType features = in_out[0].desc.type;
    if (!subm::accepts(pos, in_out[pos].desc.type, features, *mode)) {
      return false;
    }
    if (pos == at(Slot::kBias) && in_out[pos].desc.type == DataType::kBF16) {
      return false;
    }
    return pos != 0 || !subm::spconv_refusal(features, attrs_.scales);
  }

  /// Room for the bias in the type the kernels add, when the node carries it in another.
  std::size_t getWorkspaceSize(
    const DynamicPluginTensorDesc * inputs, std::int32_t num_inputs,
    const DynamicPluginTensorDesc *, std::int32_t) const noexcept override
  {
    if (num_inputs != subm::kInputs || inputs[at(Slot::kWeights)].desc.dims.nbDims != 5) {
      return 0;
    }
    const std::int64_t c_out = inputs[at(Slot::kWeights)].desc.dims.d[0];
    return c_out > 0 ? static_cast<std::size_t>(c_out) * sizeof(float) : 0;
  }

  std::int32_t onShapeChange(
    const PluginTensorDesc *, std::int32_t, const PluginTensorDesc *, std::int32_t) noexcept override
  {
    return 0;
  }

  /// The 8-bit GEMM multiplies its accumulator by one scale per output channel,
  /// input_scale * weight_scale[n]; it is uploaded here, once per context, so enqueue copies
  /// nothing.
  IPluginV3 * attachToContext(nvinfer1::IPluginResourceContext *) noexcept override
  {
    auto * plugin = static_cast<SubMConvPlugin *>(clone());
    const subm::Scales & scales = attrs_.scales;
    if (plugin == nullptr || build_phase_ || !scales.input || plugin->gemm_scale_) {
      return plugin;
    }
    std::vector<float> gemm_scale(scales.weight);
    for (auto & scale : gemm_scale) {
      scale *= *scales.input;
    }
    auto device = std::make_shared<DeviceFloats>();
    const std::size_t bytes = gemm_scale.size() * sizeof(float);
    if (
      PLUGIN_CUDA_CHECK(cudaMalloc(&device->data, bytes)) != cudaSuccess ||
      PLUGIN_CUDA_CHECK(cudaMemcpy(
        device->data, gemm_scale.data(), bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
      delete plugin;
      fail("the scales did not reach the device");
      return nullptr;
    }
    plugin->gemm_scale_ = std::move(device);
    return plugin;
  }

  const PluginFieldCollection * getFieldsToSerialize() noexcept override { return &fc_; }

  std::int32_t enqueue(
    const PluginTensorDesc * input_desc, const PluginTensorDesc * output_desc,
    const void * const * inputs, void * const * outputs, void * workspace,
    cudaStream_t stream) noexcept override
  {
    using ConvGemmOps = spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;
    using StaticAllocator = spconvlib::spconv::csrc::sparse::alloc::StaticAllocator;

    // n: output rows; n_in: input rows, the same unless the rulebook computed the output's.
    const std::int64_t n = output_desc[0].dims.d[0];
    const std::int64_t n_in = input_desc[at(Slot::kFeatures)].dims.d[0];
    // TensorRT times a plugin during the build against buffers it never filled. Here those are
    // indices, so a build-phase instance does nothing.
    if (n <= 0 || n_in <= 0 || build_phase_) {
      return 0;
    }

    const Dims & weight_dims = input_desc[at(Slot::kWeights)].dims;
    const std::int64_t c_out = weight_dims.d[0];
    const std::int64_t c_in = weight_dims.d[4];
    const std::int64_t kernel_volume = weight_dims.d[1] * weight_dims.d[2] * weight_dims.d[3];
    const Dims & pair_dims = input_desc[at(Slot::kPairs)].dims;
    const std::int32_t padded =
      attrs_.computed_rows ? static_cast<std::int32_t>(n)
                           : subm::padded_rows(static_cast<std::int32_t>(n));
    if (
      pair_dims.nbDims != 4 || pair_dims.d[0] != weight_dims.d[1] ||
      pair_dims.d[1] != weight_dims.d[2] || pair_dims.d[2] != weight_dims.d[3] ||
      pair_dims.d[3] != padded) {
      return fail(
        "pairs is " + subm::dims_string(pair_dims) + "; the weights and N=" + std::to_string(n) +
        " want " + std::to_string(padded) + " rows per tap");
    }

    const DataType features_type = input_desc[at(Slot::kFeatures)].type;
    const bool is_8bit = attrs_.scales.input.has_value();
    if (is_8bit && !gemm_scale_) {
      return fail("enqueue on a plugin no context attached");
    }
    const auto activation = activation_of(attrs_.activation);
    if (!activation) {
      return fail("activation \"" + attrs_.activation + "\" is not none, relu or leaky_relu");
    }

    try {
      const tv::DType dtype =
        is_8bit ? tv::int8 : (features_type == DataType::kFLOAT ? tv::float32 : tv::float16);
      // The type the kernels write, and the type of the bias they add: the 8-bit kernels
      // dequantize in their epilogue, in float32, and write float16.
      const tv::DType out_dtype = is_8bit ? tv::float16 : dtype;
      const tv::DType bias_dtype = is_8bit ? tv::float32 : dtype;

      tv::Tensor features =
        tv::from_blob(const_cast<void *>(inputs[at(Slot::kFeatures)]), {n_in, c_in}, dtype, 0);
      tv::Tensor weights = tv::from_blob(
        const_cast<void *>(inputs[at(Slot::kWeights)]),
        {c_out, weight_dims.d[1], weight_dims.d[2], weight_dims.d[3], c_in}, dtype, 0);
      // spconv's rulebook as the rulebook node wrote it: [kernel_volume, rows] at the start of
      // pairs.
      tv::Tensor pair_fwd = tv::from_blob(
        const_cast<void *>(inputs[at(Slot::kPairs)]), {kernel_volume, n}, tv::int32, 0);
      tv::Tensor pair_mask =
        tv::from_blob(const_cast<void *>(inputs[at(Slot::kPointMask)]), {1, n}, tv::int32, 0);
      tv::Tensor mask_argsort =
        tv::from_blob(const_cast<void *>(inputs[at(Slot::kMaskArgsort)]), {1, n}, tv::int32, 0);
      tv::Tensor out_features = tv::from_blob(outputs[0], {n, c_out}, out_dtype, 0);

      tv::Tensor bias;
      const DataType bias_type = input_desc[at(Slot::kBias)].type;
      const void * const bias_in = inputs[at(Slot::kBias)];
      if ((bias_type == DataType::kFLOAT) == (bias_dtype == tv::float32)) {
        bias = tv::from_blob(const_cast<void *>(bias_in), {c_out}, bias_dtype, 0);
      } else {
        if (bias_dtype == tv::float32) {
          launch_half_to_float(
            static_cast<const __half *>(bias_in), static_cast<float *>(workspace), c_out, stream);
        } else {
          launch_float_to_half(
            static_cast<const float *>(bias_in), static_cast<__half *>(workspace), c_out, stream);
        }
        if (PLUGIN_CUDA_CHECK(cudaGetLastError()) != cudaSuccess) {
          return -1;
        }
        bias = tv::from_blob(workspace, {c_out}, bias_dtype, 0);
      }

      tv::Tensor scale;
      if (is_8bit) {
        scale = tv::from_blob(gemm_scale_->data, {c_out}, tv::float32, 0);
      }

      std::vector<tv::Tensor> pair_mask_splits{pair_mask};
      std::vector<tv::Tensor> mask_argsort_splits{mask_argsort};
      std::unordered_map<std::string, tv::Tensor> tensors{
        {SPCONV_ALLOC_FEATURES, features},
        {SPCONV_ALLOC_FILTERS, weights},
        {SPCONV_ALLOC_OUT_FEATURES, out_features}};
      StaticAllocator alloc(tensors);

      ConvGemmOps::implicit_gemm(
        alloc, *tuner_, features, weights, pair_fwd, pair_mask_splits, mask_argsort_splits,
        static_cast<int>(n), mask_tensor_, arch_, false /*is_train*/,
        !attrs_.computed_rows /*is_subm*/,
        reinterpret_cast<std::uintptr_t>(stream), tv::CUDAKernelTimer(false),
        true /*auto_fp32_accum*/, false /*fp32_accum*/, bias, attrs_.act_alpha, 0.0F /*act_beta*/,
        *activation, false /*use_tf32*/, 1.0F /*output_scale*/, scale, tv::Tensor() /*output_add*/,
        0.0F /*output_add_scale*/, is_8bit ? static_cast<int>(tv::float16) : -1);
    } catch (const std::exception & e) {
      caughtError(e);
      return -1;
    }
    return 0;
  }

private:
  using ConvTuner = spconvlib::spconv::csrc::sparse::convops::spops::ConvTuner;

  /// Refuses what the contract accepts and spconv does not run.
  std::int32_t library_refusal(DataType features_type, DataType bias_type) const
  {
    if (const auto why = subm::spconv_refusal(features_type, attrs_.scales)) {
      return fail(*why);
    }
    if (bias_type == DataType::kBF16) {
      return fail("bias is bfloat16; the spconv library takes float32 or float16");
    }
    return 0;
  }

  std::int32_t fail(const std::string & why) const
  {
    return refuse(attrs_.plugin_name(), attrs_.node_name, why);
  }

  ConvAttrs attrs_;
  /// True for the instance TensorRT builds with, false for the one it runs.
  bool build_phase_{false};
  std::shared_ptr<DeviceFloats> gemm_scale_;  ///< shared by a context's clones
  std::unique_ptr<ConvTuner> tuner_;
  tv::Tensor mask_tensor_;
  std::tuple<int, int> arch_{};
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

class SubMConvPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
  explicit SubMConvPluginCreator(bool computed_rows = false) : computed_rows_(computed_rows)
  {
    fields_.emplace_back("activation", nullptr, PluginFieldType::kCHAR, 1);
    fields_.emplace_back("act_alpha", nullptr, PluginFieldType::kFLOAT32, 1);
    fields_.emplace_back("input_scale", nullptr, PluginFieldType::kFLOAT32, 1);
    fields_.emplace_back("weight_scale", nullptr, PluginFieldType::kFLOAT32, 1);
    fields_.emplace_back("output_scale", nullptr, PluginFieldType::kFLOAT32, 1);
    fc_.nbFields = static_cast<std::int32_t>(fields_.size());
    fc_.fields = fields_.data();
  }

  IPluginV3 * createPlugin(
    const char * name, const PluginFieldCollection * fc, TensorRTPhase phase) noexcept override
  {
    try {
      ConvAttrs attrs;
      attrs.computed_rows = computed_rows_;
      if (name != nullptr) {
        attrs.node_name = name;
      }
      for (std::int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i) {
        const PluginField & field = fc->fields[i];
        if (field.data == nullptr) {
          continue;
        }
        const std::string field_name = field.name;
        if (field_name == "activation") {
          attrs.activation = read_string(field);
        } else if (field_name == "act_alpha") {
          attrs.act_alpha = read_float(field, attrs.act_alpha);
        } else if (field_name == "input_scale") {
          read_scale(field, attrs.scales.input, attrs.problem);
        } else if (field_name == "weight_scale") {
          read_scales(field, attrs.scales.weight, attrs.problem);
        } else if (field_name == "output_scale") {
          read_scale(field, attrs.scales.output, attrs.problem);
        }
      }
      return new SubMConvPlugin(std::move(attrs), phase);
    } catch (const std::exception & e) {
      caughtError(e);
    }
    return nullptr;
  }

  const PluginFieldCollection * getFieldNames() noexcept override { return &fc_; }
  const char * getPluginName() const noexcept override
  {
    return computed_rows_ ? kSPARSE_CONV_PLUGIN_NAME : kSUBM_CONV_PLUGIN_NAME;
  }
  const char * getPluginVersion() const noexcept override { return kSUBM_PLUGIN_VERSION; }
  const char * getPluginNamespace() const noexcept override { return kSUBM_PLUGIN_NAMESPACE; }

private:
  bool computed_rows_;
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

class SparseConvPluginCreator : public SubMConvPluginCreator
{
public:
  SparseConvPluginCreator() : SubMConvPluginCreator(true) {}
};

// --- SparseRulebook ------------------------------------------------------------------------
//
// The rulebook of a convolution whose output points are computed: a stride, or a padding other
// than kernel / 2. A proposal; t4sparse has no such node yet.
//
// attributes: kernel_size, stride, padding, dilation, spatial_shape (the input grid): ints[3]
// in : 0 coords       int32 [N, 4]
// out: 0 pairs        int32 [k0, k1, k2, M]
//      1 point_mask   int32 [M, 1]
//      2 mask_argsort int32 [M]
//      3 tile_mask    int32 [M]
//      4 out_coords   int32 [M, 4]
//      5 out_count    int32 scalar, M: the size tensor every other shape has as an extent
//
// Outputs 0 to 3 are SubMRulebook's with M, the output point count, in place of N, and without
// its padding: TensorRT allows no arithmetic on a size tensor. They feed a SparseConv;
// out_coords feeds the next rulebook. The contents are spconv's, as in SubMRulebook.

struct SparseRulebookAttrs
{
  std::vector<std::int32_t> kernel_size;
  std::vector<std::int32_t> stride;
  std::vector<std::int32_t> padding;
  std::vector<std::int32_t> dilation{1, 1, 1};
  std::vector<std::int32_t> spatial_shape;
  std::int32_t do_sort{1};  ///< as in RulebookAttrs
  std::string node_name;

  std::int32_t kernel_volume() const
  {
    return std::accumulate(kernel_size.begin(), kernel_size.end(), 1, std::multiplies<>());
  }

  /// Why a node has no rulebook, or empty. The kernel limits are SubMRulebook's.
  std::string problem() const
  {
    const std::pair<const char *, const std::vector<std::int32_t> *> lists[] = {
      {"kernel_size", &kernel_size},
      {"stride", &stride},
      {"padding", &padding},
      {"dilation", &dilation},
      {"spatial_shape", &spatial_shape}};
    for (const auto & [name, values] : lists) {
      if (values->empty()) {
        return std::string(name) + " is missing";
      }
      if (values->size() != 3) {
        return std::string(name) + " " + extent(*values) + "; three ints";
      }
    }
    std::int64_t taps = 1;
    for (const auto k : kernel_size) {
      if (k < 1 || k % 2 == 0) {
        return "kernel_size " + extent(kernel_size) + "; odd on every axis";
      }
      taps *= k;
    }
    if (taps > 32) {
      return "kernel_size " + extent(kernel_size) + " has " + std::to_string(taps) +
             " taps; at most 32";
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (stride[axis] < 1 || dilation[axis] < 1 || spatial_shape[axis] < 1) {
        return "stride " + extent(stride) + ", dilation " + extent(dilation) +
               ", spatial_shape " + extent(spatial_shape) + "; positive on every axis";
      }
      if (padding[axis] < 0) {
        return "padding " + extent(padding) + "; not negative";
      }
    }
    return {};
  }
};

class SparseRulebookPlugin : public IPluginV3,
                             public nvinfer1::IPluginV3OneCore,
                             public nvinfer1::IPluginV3OneBuild,
                             public nvinfer1::IPluginV3OneRuntime
{
public:
  SparseRulebookPlugin(SparseRulebookAttrs attrs, TensorRTPhase phase)
  : attrs_(std::move(attrs)), build_phase_(phase == TensorRTPhase::kBUILD)
  {
    const auto add = [this](const char * name, const std::vector<std::int32_t> & values) {
      fields_.emplace_back(
        name, values.data(), PluginFieldType::kINT32, static_cast<std::int32_t>(values.size()));
    };
    add("kernel_size", attrs_.kernel_size);
    add("stride", attrs_.stride);
    add("padding", attrs_.padding);
    add("dilation", attrs_.dilation);
    add("spatial_shape", attrs_.spatial_shape);
    fields_.emplace_back("do_sort", &attrs_.do_sort, PluginFieldType::kINT32, 1);
    fc_.nbFields = static_cast<std::int32_t>(fields_.size());
    fc_.fields = fields_.data();
  }

  nvinfer1::IPluginCapability * getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept override
  {
    switch (type) {
      case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore *>(this);
      case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild *>(this);
      case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime *>(this);
    }
    return nullptr;
  }

  IPluginV3 * clone() noexcept override
  {
    try {
      return new SparseRulebookPlugin(
        attrs_, build_phase_ ? TensorRTPhase::kBUILD : TensorRTPhase::kRUNTIME);
    } catch (const std::exception & e) {
      caughtError(e);
    }
    return nullptr;
  }

  const char * getPluginName() const noexcept override { return kSPARSE_RULEBOOK_PLUGIN_NAME; }
  const char * getPluginVersion() const noexcept override { return kSUBM_PLUGIN_VERSION; }
  const char * getPluginNamespace() const noexcept override { return kSUBM_PLUGIN_NAMESPACE; }

  std::int32_t getNbOutputs() const noexcept override { return kOutputs; }

  std::int32_t configurePlugin(
    const DynamicPluginTensorDesc * in, std::int32_t num_inputs, const DynamicPluginTensorDesc *,
    std::int32_t num_outputs) noexcept override
  {
    if (num_inputs != 1 || num_outputs != kOutputs) {
      return fail(
        std::to_string(num_inputs) + " inputs, " + std::to_string(num_outputs) +
        " outputs; want 1 and " + std::to_string(kOutputs));
    }
    if (in[0].desc.dims.nbDims != 2 || in[0].desc.dims.d[1] != 4) {
      return fail("coords are " + subm::dims_string(in[0].desc.dims) + "; want [N, 4]");
    }
    if (const auto why = attrs_.problem(); !why.empty()) {
      return fail(why);
    }
    if (in[0].max.d[0] > kMaxPoints) {
      return fail(
        "up to " + std::to_string(in[0].max.d[0]) + " points; this plugin indexes at most " +
        std::to_string(kMaxPoints));
    }
    return 0;
  }

  std::int32_t getOutputDataTypes(
    DataType * out, std::int32_t num_outputs, const DataType *, std::int32_t) const noexcept override
  {
    for (std::int32_t i = 0; i < num_outputs; ++i) {
      out[i] = DataType::kINT32;
    }
    return 0;
  }

  std::int32_t getOutputShapes(
    const DimsExprs * inputs, std::int32_t num_inputs, const DimsExprs *, std::int32_t,
    DimsExprs * out, std::int32_t num_outputs, nvinfer1::IExprBuilder & builder) noexcept override
  {
    if (num_inputs != 1 || num_outputs != kOutputs || inputs[0].nbDims != 2) {
      return -1;
    }
    if (const auto why = attrs_.problem(); !why.empty()) {
      return fail(why);
    }
    using nvinfer1::DimensionOperation;
    // M is known only once the rulebook ran: output 5 is the size tensor that carries it. The
    // builder tunes for half the input points, and the plugin's index limit bounds it.
    const nvinfer1::IDimensionExpr * m = builder.declareSizeTensor(
      kCountOutput,
      *builder.operation(DimensionOperation::kCEIL_DIV, *inputs[0].d[0], *builder.constant(2)),
      *builder.constant(kMaxPoints));
    // TensorRT refuses arithmetic on a size tensor, so no shape here is M rounded or divided:
    // pairs and tile_mask have M rows where SubMRulebook's are padded.
    out[0].nbDims = 4;
    for (int axis = 0; axis < 3; ++axis) {
      out[0].d[axis] = builder.constant(attrs_.kernel_size[axis]);
    }
    out[0].d[3] = m;

    out[1].nbDims = 2;
    out[1].d[0] = m;
    out[1].d[1] = builder.constant(1);

    out[2].nbDims = 1;
    out[2].d[0] = m;

    out[3].nbDims = 1;
    out[3].d[0] = m;

    out[4].nbDims = 2;
    out[4].d[0] = m;
    out[4].d[1] = builder.constant(4);

    out[kCountOutput].nbDims = 0;
    return 0;
  }

  bool supportsFormatCombination(
    std::int32_t pos, const DynamicPluginTensorDesc * in_out, std::int32_t,
    std::int32_t) noexcept override
  {
    return in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
           in_out[pos].desc.type == DataType::kINT32;
  }

  std::size_t getWorkspaceSize(
    const DynamicPluginTensorDesc *, std::int32_t, const DynamicPluginTensorDesc *,
    std::int32_t) const noexcept override
  {
    if (!attrs_.problem().empty()) {
      return 0;
    }
    return layout_of(attrs_).thrust_tmp_offset + kThrustTempBytes;
  }

  std::int32_t onShapeChange(
    const PluginTensorDesc *, std::int32_t, const PluginTensorDesc *, std::int32_t) noexcept override
  {
    return 0;
  }

  IPluginV3 * attachToContext(nvinfer1::IPluginResourceContext *) noexcept override
  {
    return clone();
  }

  const PluginFieldCollection * getFieldsToSerialize() noexcept override { return &fc_; }

  std::int32_t enqueue(
    const PluginTensorDesc * input_desc, const PluginTensorDesc *, const void * const * inputs,
    void * const * outputs, void * workspace, cudaStream_t stream) noexcept override
  {
    using SpconvOps = spconvlib::spconv::csrc::sparse::all::SpconvOps;
    using StaticAllocator = spconvlib::spconv::csrc::sparse::alloc::StaticAllocator;

    const std::int64_t n = input_desc[0].dims.d[0];
    if (n > kMaxPoints) {
      return fail(std::to_string(n) + " points; at most " + std::to_string(kMaxPoints));
    }
    // TensorRT times a plugin during the build against buffers it never filled, and still reads
    // the count: a build-phase instance reports the count the builder tunes for and writes
    // nothing else.
    if (n <= 0 || build_phase_) {
      return write_count(static_cast<std::int32_t>(n <= 0 ? 0 : (n + 1) / 2), outputs, stream);
    }

    std::int32_t count = 0;
    try {
      const Layout layout = layout_of(attrs_);
      const int kernel_volume = attrs_.kernel_volume();
      auto * const ws = static_cast<std::uint8_t *>(workspace);

      const std::vector<int> ksize(attrs_.kernel_size.begin(), attrs_.kernel_size.end());
      const std::vector<int> stride(attrs_.stride.begin(), attrs_.stride.end());
      const std::vector<int> padding(attrs_.padding.begin(), attrs_.padding.end());
      const std::vector<int> dilation(attrs_.dilation.begin(), attrs_.dilation.end());
      const std::vector<int> spatial(attrs_.spatial_shape.begin(), attrs_.spatial_shape.end());

      const auto max_act_out_theory = SpconvOps::get_handcrafted_max_act_out(
        static_cast<std::size_t>(n), ksize, stride, padding, dilation);
      auto ws_tensors = SpconvOps::get_indice_gen_tensors_from_workspace(
        ws, kernel_volume, kMaxPoints, kMaxPoints, max_act_out_theory, false /*subm*/,
        layout.use_int64_hash_k, true /*direct_table*/);

      // TensorRT sizes every output for kMaxPoints rows, the size tensor's bound; spconv takes
      // what it needs of each.
      tv::Tensor input_indices =
        tv::from_blob(const_cast<void *>(inputs[0]), {n, 4}, tv::int32, 0);
      tv::Tensor pair_fwd = tv::from_blob(outputs[0], {kernel_volume, kMaxPoints}, tv::int32, 0);
      tv::Tensor pair_mask = tv::from_blob(outputs[1], {1, kMaxPoints}, tv::int32, 0);
      tv::Tensor mask_argsort = tv::from_blob(outputs[2], {1, kMaxPoints}, tv::int32, 0);
      tv::Tensor out_indices = tv::from_blob(outputs[4], {kMaxPoints, 4}, tv::int32, 0);

      std::uint8_t * extra = ws + layout.kernel_num_offset;
      tv::Tensor indices_kernel_num = tv::from_blob(extra, {kernel_volume}, tv::int32, 0);
      extra += static_cast<std::size_t>(kernel_volume) * sizeof(std::int32_t);
      // spconv also writes the backward tables, which only training reads.
      tv::Tensor pair_bwd = tv::from_blob(extra, {kernel_volume, kMaxPoints}, tv::int32, 0);
      extra += static_cast<std::size_t>(kernel_volume) * kMaxPoints * sizeof(std::int32_t);
      tv::Tensor pair_mask_bwd = tv::from_blob(extra, {1, kMaxPoints}, tv::int32, 0);
      extra += static_cast<std::size_t>(kMaxPoints) * sizeof(std::int32_t);
      tv::Tensor mask_argsort_bwd = tv::from_blob(extra, {1, kMaxPoints}, tv::int32, 0);
      tv::Tensor thrust_tmp = tv::from_blob(
        ws + layout.thrust_tmp_offset, {static_cast<std::int64_t>(kThrustTempBytes)}, tv::uint8,
        0);

      if (
        PLUGIN_CUDA_CHECK(cudaMemsetAsync(
          ws + layout.kernel_num_offset, 0, static_cast<std::size_t>(kernel_volume) * 4, stream)) !=
        cudaSuccess) {
        return -1;
      }

      ws_tensors.emplace(SPCONV_ALLOC_PAIR_FWD, pair_fwd);
      ws_tensors.emplace(SPCONV_ALLOC_PAIR_BWD, pair_bwd);
      ws_tensors.emplace(SPCONV_ALLOC_PAIR_MASK, pair_mask);
      ws_tensors.emplace(SPCONV_ALLOC_PAIR_MASK_BWD, pair_mask_bwd);
      ws_tensors.emplace(SPCONV_ALLOC_MASK_ARG_SORT, mask_argsort);
      ws_tensors.emplace(SPCONV_ALLOC_MASK_ARG_SORT_BWD, mask_argsort_bwd);
      ws_tensors.emplace(SPCONV_ALLOC_OUT_INDICES, out_indices);
      ws_tensors.emplace(
        SPCONV_ALLOC_INDICE_NUM_PER_LOC, indices_kernel_num);  // cSpell:ignore INDICE
      StaticAllocator alloc(ws_tensors);
      alloc.thrust_tmp_tensor_ = thrust_tmp;

      // cSpell:ignore indice subm
      const auto result = SpconvOps::get_indice_pairs_implicit_gemm(
        alloc, input_indices, 1 /*batch_size*/, spatial,
        static_cast<int>(tv::gemm::SparseConvAlgo::kMaskImplicitGemm), ksize, stride, padding,
        dilation, {0, 0, 0}, false /*subm*/, false /*transposed*/, false /*is_train*/,
        reinterpret_cast<std::uintptr_t>(stream), kMaxPoints, tv::CUDAKernelTimer(false),
        true /*direct_table*/, attrs_.do_sort != 0);
      count = std::get<1>(result);
      // tile_mask, which spconv has no use for, is zero.
      if (
        PLUGIN_CUDA_CHECK(cudaMemsetAsync(
          outputs[3], 0, static_cast<std::size_t>(count) * sizeof(std::int32_t), stream)) !=
        cudaSuccess) {
        return -1;
      }
    } catch (const std::exception & e) {
      caughtError(e);
      return -1;
    }
    return write_count(count, outputs, stream);
  }

private:
  static constexpr std::int32_t kOutputs{6};
  static constexpr std::int32_t kCountOutput{5};
  static constexpr int kMaxPoints{512000};
  static constexpr std::size_t kThrustTempBytes{8U * 1024U * 1024U};

  /// The workspace: spconv's index-generation buffers, then the per-tap counts, the backward
  /// tables and the sort scratch.
  struct Layout
  {
    bool use_int64_hash_k;
    std::size_t kernel_num_offset;
    std::size_t thrust_tmp_offset;
  };

  static Layout layout_of(const SparseRulebookAttrs & attrs)
  {
    using SpconvOps = spconvlib::spconv::csrc::sparse::all::SpconvOps;
    const std::vector<int> ksize(attrs.kernel_size.begin(), attrs.kernel_size.end());
    const std::vector<int> stride(attrs.stride.begin(), attrs.stride.end());
    const std::vector<int> padding(attrs.padding.begin(), attrs.padding.end());
    const std::vector<int> dilation(attrs.dilation.begin(), attrs.dilation.end());
    const std::vector<int> spatial(attrs.spatial_shape.begin(), attrs.spatial_shape.end());
    std::int64_t out_volume = 1;
    const auto out_shape =
      SpconvOps::get_conv_output_size(spatial, ksize, stride, padding, dilation);
    for (const auto extent : out_shape) {
      out_volume *= extent;
    }
    Layout layout{};
    layout.use_int64_hash_k =
      out_volume >= static_cast<std::int64_t>(std::numeric_limits<int>::max());
    const int kernel_volume = attrs.kernel_volume();
    layout.kernel_num_offset = static_cast<std::size_t>(SpconvOps::get_indice_gen_workspace_size(
      kernel_volume, kMaxPoints, kMaxPoints, kMaxPoints, false, layout.use_int64_hash_k, true));
    layout.thrust_tmp_offset =
      layout.kernel_num_offset +
      (static_cast<std::size_t>(kernel_volume) +
       (static_cast<std::size_t>(kernel_volume) + 2) * kMaxPoints) *
        sizeof(std::int32_t);
    return layout;
  }

  std::int32_t write_count(std::int32_t count, void * const * outputs, cudaStream_t stream)
  {
    count_ = count;
    return PLUGIN_CUDA_CHECK(cudaMemcpyAsync(
             outputs[kCountOutput], &count_, sizeof(count_), cudaMemcpyHostToDevice, stream)) ==
               cudaSuccess
             ? 0
             : -1;
  }

  std::int32_t fail(const std::string & why) const
  {
    return refuse(kSPARSE_RULEBOOK_PLUGIN_NAME, attrs_.node_name, why);
  }

  SparseRulebookAttrs attrs_;
  bool build_phase_{false};
  std::int32_t count_{0};  ///< outlives the copy enqueue starts
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

class SparseRulebookPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
  SparseRulebookPluginCreator()
  {
    for (const char * name : {"kernel_size", "stride", "padding", "dilation", "spatial_shape"}) {
      fields_.emplace_back(name, nullptr, PluginFieldType::kINT32, 3);
    }
    fc_.nbFields = static_cast<std::int32_t>(fields_.size());
    fc_.fields = fields_.data();
  }

  IPluginV3 * createPlugin(
    const char * name, const PluginFieldCollection * fc, TensorRTPhase phase) noexcept override
  {
    try {
      SparseRulebookAttrs attrs;
      if (name != nullptr) {
        attrs.node_name = name;
      }
      if (phase == TensorRTPhase::kBUILD) {
        const char * sort = std::getenv("AUTOWARE_SUBM_RULEBOOK_SORT");
        attrs.do_sort = (sort != nullptr && std::strcmp(sort, "0") == 0) ? 0 : 1;
      }
      for (std::int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i) {
        const PluginField & field = fc->fields[i];
        if (field.data == nullptr) {
          continue;
        }
        const std::string field_name = field.name;
        if (field_name == "kernel_size") {
          read_ints(field, attrs.kernel_size);
        } else if (field_name == "stride") {
          read_ints(field, attrs.stride);
        } else if (field_name == "padding") {
          read_ints(field, attrs.padding);
        } else if (field_name == "dilation") {
          read_ints(field, attrs.dilation);
        } else if (field_name == "spatial_shape") {
          read_ints(field, attrs.spatial_shape);
        } else if (field_name == "do_sort") {
          attrs.do_sort = read_int(field, attrs.do_sort);
        }
      }
      return new SparseRulebookPlugin(std::move(attrs), phase);
    } catch (const std::exception & e) {
      caughtError(e);
    }
    return nullptr;
  }

  const PluginFieldCollection * getFieldNames() noexcept override { return &fc_; }
  const char * getPluginName() const noexcept override { return kSPARSE_RULEBOOK_PLUGIN_NAME; }
  const char * getPluginVersion() const noexcept override { return kSUBM_PLUGIN_VERSION; }
  const char * getPluginNamespace() const noexcept override { return kSUBM_PLUGIN_NAMESPACE; }

private:
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

REGISTER_TENSORRT_PLUGIN(SubMRulebookPluginCreator);
REGISTER_TENSORRT_PLUGIN(SubMConvPluginCreator);
REGISTER_TENSORRT_PLUGIN(SparseRulebookPluginCreator);
REGISTER_TENSORRT_PLUGIN(SparseConvPluginCreator);

}  // namespace

nvinfer1::IPluginCreatorV3One * subm_rulebook_creator() noexcept
{
  static SubMRulebookPluginCreator creator;
  return &creator;
}

nvinfer1::IPluginCreatorV3One * subm_conv_creator() noexcept
{
  static SubMConvPluginCreator creator;
  return &creator;
}

nvinfer1::IPluginCreatorV3One * sparse_rulebook_creator() noexcept
{
  static SparseRulebookPluginCreator creator;
  return &creator;
}

nvinfer1::IPluginCreatorV3One * sparse_conv_creator() noexcept
{
  static SparseConvPluginCreator creator;
  return &creator;
}

}  // namespace autoware::tensorrt_plugins
