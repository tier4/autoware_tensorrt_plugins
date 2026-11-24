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

// Checks which plugins each built library provides (the sets must match the documented
// split and be disjoint) and that TensorRT's own loader, IPluginRegistry::loadLibrary,
// registers them.

#include <NvInferRuntime.h>
#include <dlfcn.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <set>
#include <string>

namespace
{

using GetCreatorsFn = nvinfer1::IPluginCreatorInterface * const * (*)(std::int32_t &);

// Whether the global TensorRT registry has a plugin of this name.
bool is_registered(const std::string & plugin_name)
{
  return getPluginRegistry()->getCreator(plugin_name.c_str(), "1", "") != nullptr;
}

// Reads the plugin names from the library's getCreators. Opening a library registers
// nothing, so it is left open.
std::set<std::string> plugin_names_of(const char * library_path)
{
  void * handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    ADD_FAILURE() << "dlopen(" << library_path << "): " << dlerror();
    return {};
  }
  EXPECT_NE(dlsym(handle, "setLoggerFinder"), nullptr) << dlerror();
  auto get_creators = reinterpret_cast<GetCreatorsFn>(dlsym(handle, "getCreators"));
  if (get_creators == nullptr) {
    ADD_FAILURE() << "dlsym(getCreators): " << dlerror();
    return {};
  }

  std::int32_t num_creators = 0;
  nvinfer1::IPluginCreatorInterface * const * creators = get_creators(num_creators);
  EXPECT_GT(num_creators, 0);

  std::set<std::string> names;
  for (std::int32_t i = 0; i < num_creators; ++i) {
    const nvinfer1::InterfaceInfo info = creators[i]->getInterfaceInfo();
    EXPECT_STREQ(info.kind, "PLUGIN CREATOR_V3ONE");
    auto * creator = static_cast<nvinfer1::IPluginCreatorV3One *>(creators[i]);
    EXPECT_STREQ(creator->getPluginNamespace(), "");
    EXPECT_STREQ(creator->getPluginVersion(), "1");
    EXPECT_FALSE(is_registered(creator->getPluginName()))
      << creator->getPluginName() << " is registered although its library was only opened";
    EXPECT_TRUE(names.insert(creator->getPluginName()).second)
      << "duplicate plugin name " << creator->getPluginName();
  }
  return names;
}

const std::set<std::string> kCORE_PLUGINS{
  "Argsort",
  "QuickCumsumCuda",
  "SegmentCSR",
  "CustomUnique",
  "RotatePlugin",
  "SelectAndPadPlugin",
  "MultiScaleDeformableAttentionPlugin"};

const std::set<std::string> kSPCONV_PLUGINS{
  "GetIndicePairs", "GetIndicePairsImplicitGemm", "ImplicitGemm", "IndiceConv"};

// Loads a library the way consumers do and checks that exactly then its plugins are
// registered.
void expect_registered_while_loaded(
  const char * library_path, const std::set<std::string> & plugins)
{
  nvinfer1::IPluginRegistry * registry = getPluginRegistry();
  const nvinfer1::IPluginRegistry::PluginLibraryHandle handle =
    registry->loadLibrary(library_path);
  ASSERT_NE(handle, nullptr) << "loadLibrary(" << library_path << ")";
  for (const std::string & plugin : plugins) {
    EXPECT_TRUE(is_registered(plugin)) << plugin;
  }

  registry->deregisterLibrary(handle);
  for (const std::string & plugin : plugins) {
    EXPECT_FALSE(is_registered(plugin)) << plugin;
  }
}

}  // namespace

TEST(PluginLibrariesTest, CoreLibraryLoadsThroughTheRegistry)
{
  expect_registered_while_loaded(CORE_PLUGIN_LIBRARY, kCORE_PLUGINS);
}

TEST(PluginLibrariesTest, CoreLibraryProvidesTheCudaOnlyPlugins)
{
  EXPECT_EQ(plugin_names_of(CORE_PLUGIN_LIBRARY), kCORE_PLUGINS);
}

#ifdef SPCONV_PLUGIN_LIBRARY
TEST(PluginLibrariesTest, SpconvLibraryLoadsThroughTheRegistry)
{
  expect_registered_while_loaded(SPCONV_PLUGIN_LIBRARY, kSPCONV_PLUGINS);
}

TEST(PluginLibrariesTest, SpconvLibraryProvidesTheSparseConvolutionPlugins)
{
  EXPECT_EQ(plugin_names_of(SPCONV_PLUGIN_LIBRARY), kSPCONV_PLUGINS);
}

TEST(PluginLibrariesTest, LibrariesProvideDisjointPluginSets)
{
  const std::set<std::string> core = plugin_names_of(CORE_PLUGIN_LIBRARY);
  const std::set<std::string> spconv = plugin_names_of(SPCONV_PLUGIN_LIBRARY);
  std::set<std::string> both;
  std::set_intersection(
    core.begin(), core.end(), spconv.begin(), spconv.end(), std::inserter(both, both.begin()));
  EXPECT_TRUE(both.empty()) << "plugin provided by both libraries: " << *both.begin();
}
#endif
