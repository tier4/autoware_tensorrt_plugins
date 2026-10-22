// Copyright 2025 TIER IV, Inc.
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

// Registration for the spconv-independent plugin library
// (libautoware_tensorrt_plugins.so).

#include "autoware/tensorrt_plugins/plugin_registration.hpp"

#include "autoware/tensorrt_plugins/argsort_plugin_creator.hpp"
#include "autoware/tensorrt_plugins/logger_finder.hpp"
#include "autoware/tensorrt_plugins/multi_scale_deformable_attention_plugin_creator.hpp"
#include "autoware/tensorrt_plugins/quick_cumsum_cuda_plugin_creator.hpp"
#include "autoware/tensorrt_plugins/rotate_plugin_creator.hpp"
#include "autoware/tensorrt_plugins/segment_csr_plugin_creator.hpp"
#include "autoware/tensorrt_plugins/select_and_pad_plugin_creator.hpp"
#include "autoware/tensorrt_plugins/unique_plugin_creator.hpp"

#include <NvInferRuntime.h>

#include <cstdint>

namespace
{
autoware::tensorrt_plugins::ThreadSafeLoggerFinder logger_finder;
}  // namespace

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder * finder)
{
  logger_finder.setLoggerFinder(finder);
}

extern "C" nvinfer1::IPluginCreatorInterface * const * getCreators(std::int32_t & num_creators)
{
  static nvinfer1::plugin::ArgsortPluginCreator argsort_plugin_creator{};
  static nvinfer1::plugin::QuickCumsumCudaPluginCreator quick_cumsum_cuda_plugin_creator{};
  static autoware::tensorrt_plugins::MultiScaleDeformableAttentionPluginCreator
    multi_scale_deformable_attention_plugin_creator{};
  static autoware::tensorrt_plugins::RotatePluginCreator rotate_plugin_creator{};
  static nvinfer1::plugin::SegmentCSRPluginCreator segment_csr_plugin_creator{};
  static autoware::tensorrt_plugins::SelectAndPadPluginCreator select_and_pad_plugin_creator{};
  static nvinfer1::plugin::UniquePluginCreator unique_plugin_creator{};

  static nvinfer1::IPluginCreatorInterface * const plugin_creator_list[] = {
    &argsort_plugin_creator,
    &quick_cumsum_cuda_plugin_creator,
    &multi_scale_deformable_attention_plugin_creator,
    &rotate_plugin_creator,
    &segment_csr_plugin_creator,
    &select_and_pad_plugin_creator,
    &unique_plugin_creator};
  num_creators =
    static_cast<std::int32_t>(sizeof(plugin_creator_list) / sizeof(plugin_creator_list[0]));
  return plugin_creator_list;
}
