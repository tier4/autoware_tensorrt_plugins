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
#ifndef AUTOWARE__TENSORRT_PLUGINS__SUBM_PLUGIN_CREATORS_HPP_
#define AUTOWARE__TENSORRT_PLUGINS__SUBM_PLUGIN_CREATORS_HPP_

#include <NvInferRuntime.h>

// SubMRulebook and SubMConv: the submanifold sparse convolution as the node pair t4sparse
// defines (tier4/t4sparse), run by spconv. One exported graph builds against either library;
// the node contract is in subm_contract.hpp.

namespace autoware::tensorrt_plugins
{

constexpr char const * const kSUBM_RULEBOOK_PLUGIN_NAME{"SubMRulebook"};
constexpr char const * const kSUBM_CONV_PLUGIN_NAME{"SubMConv"};
constexpr char const * const kSUBM_PLUGIN_VERSION{"1"};
constexpr char const * const kSUBM_PLUGIN_NAMESPACE{""};

nvinfer1::IPluginCreatorV3One * subm_rulebook_creator() noexcept;
nvinfer1::IPluginCreatorV3One * subm_conv_creator() noexcept;

// SparseRulebook and SparseConv: the same node contract for convolutions whose output points are
// computed (strided, or any padding). A proposal: t4sparse does not have this pair yet.
constexpr char const * const kSPARSE_RULEBOOK_PLUGIN_NAME{"SparseRulebook"};
constexpr char const * const kSPARSE_CONV_PLUGIN_NAME{"SparseConv"};

nvinfer1::IPluginCreatorV3One * sparse_rulebook_creator() noexcept;
nvinfer1::IPluginCreatorV3One * sparse_conv_creator() noexcept;

}  // namespace autoware::tensorrt_plugins

#endif  // AUTOWARE__TENSORRT_PLUGINS__SUBM_PLUGIN_CREATORS_HPP_
