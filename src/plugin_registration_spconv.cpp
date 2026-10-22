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

// Registration for the spconv-dependent plugin library
// (libautoware_tensorrt_plugins_spconv.so).

#include "autoware/tensorrt_plugins/get_indices_pairs_implicit_gemm_plugin_creator.hpp"
#include "autoware/tensorrt_plugins/get_indices_pairs_plugin_creator.hpp"
#include "autoware/tensorrt_plugins/implicit_gemm_plugin_creator.hpp"
#include "autoware/tensorrt_plugins/indice_conv_plugin_creator.hpp"
#include "autoware/tensorrt_plugins/logger_finder.hpp"
#include "autoware/tensorrt_plugins/plugin_registration.hpp"

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
  static nvinfer1::plugin::GetIndicesPairsImplicitGemmPluginCreator
    get_indices_pairs_implicit_gemm_plugin_creator{};
  static nvinfer1::plugin::GetIndicesPairsPluginCreator get_indices_pairs_plugin_creator{};
  static nvinfer1::plugin::ImplicitGemmPluginCreator implicit_gemm_plugin_creator{};
  static nvinfer1::plugin::IndiceConvPluginCreator
    indice_conv_plugin_creator{};  // cSpell:ignore Indice

  static nvinfer1::IPluginCreatorInterface * const plugin_creator_list[] = {
    &get_indices_pairs_implicit_gemm_plugin_creator, &get_indices_pairs_plugin_creator,
    &implicit_gemm_plugin_creator, &indice_conv_plugin_creator};
  num_creators =
    static_cast<std::int32_t>(sizeof(plugin_creator_list) / sizeof(plugin_creator_list[0]));
  return plugin_creator_list;
}
