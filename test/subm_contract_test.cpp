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


// From t4sparse (tests/test_subm_contract.cpp at 19f1167), with the spconv library's own
// refusals added at the end.
//
// SubMConv's contract with an exported graph, row by row of the table in
// subm_contract.hpp. No device: the table is about tensors, not kernels.
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "autoware/tensorrt_plugins/subm_contract.hpp"

namespace {

using namespace nvinfer1;
namespace subm = autoware::tensorrt_plugins::subm;

constexpr std::int32_t kC = 64;
constexpr std::int32_t kN = 256;  // a multiple of the pad, so P = N

Dims dims2(std::int64_t a, std::int64_t b) {
  Dims d{};
  d.nbDims = 2;
  d.d[0] = a;
  d.d[1] = b;
  return d;
}

Dims vec(std::int64_t n) {
  Dims d{};
  d.nbDims = 1;
  d.d[0] = n;
  return d;
}

DynamicPluginTensorDesc dyn(Dims d, DataType t) {
  DynamicPluginTensorDesc p{};
  p.desc.dims = d;
  p.desc.type = t;
  p.desc.format = TensorFormat::kLINEAR;
  p.desc.scale = 1.f;
  p.min = p.opt = p.max = d;
  return p;
}

/// The scale attributes of each mode.
subm::Scales scales(subm::Mode m) {
  subm::Scales s;
  if (m == subm::Mode::kFloat) return s;
  s.input = 0.02f;
  s.weight.assign(kC, 0.01f);
  if (m == subm::Mode::k8BitInOut) s.output = 0.05f;
  return s;
}
const subm::Scales kNone = scales(subm::Mode::kFloat);
const subm::Scales kIn = scales(subm::Mode::k8BitIn);
const subm::Scales kInOut = scales(subm::Mode::k8BitInOut);

/// The inputs of a 3x3x3 layer and its output.
std::vector<DynamicPluginTensorDesc> node(DataType in, DataType out) {
  return {
      dyn(dims2(kN, kC), in),
      dyn(Dims{5, {kC, 3, 3, 3, kC}}, in),
      dyn(Dims{4, {3, 3, 3, kN}}, DataType::kINT32),
      dyn(dims2(kN, 1), DataType::kINT32),
      dyn(vec(kN), DataType::kINT32),
      dyn(vec(kN / 32), DataType::kINT32),
      dyn(vec(kC), DataType::kFLOAT),
      dyn(dims2(kN, kC), out),
  };
}

std::string why(const std::vector<DynamicPluginTensorDesc>& io, const subm::Scales& s) {
  const auto nb_in = static_cast<std::int32_t>(io.size()) - 1;
  return subm::refusal(io.data(), nb_in, io.back().desc.type, s).value_or("");
}
#define EXPECT_ACCEPTED(io, s) EXPECT_EQ(why(io, s), "")
#define EXPECT_REFUSED(io, s) EXPECT_NE(why(io, s), "")

}  // namespace

TEST(SubMContract, TheScaleAttributesNameTheMode) {
  EXPECT_EQ(subm::mode_for(kNone), subm::Mode::kFloat);
  EXPECT_EQ(subm::mode_for(kIn), subm::Mode::k8BitIn);
  EXPECT_EQ(subm::mode_for(kInOut), subm::Mode::k8BitInOut);
  const auto refused = [](subm::Scales s) { return subm::scale_refusal(s).value_or(""); };
  subm::Scales s = kIn;
  s.weight.clear();
  EXPECT_EQ(refused(s),
            "input_scale without weight_scale; the 8-bit modes carry input_scale and weight_scale");
  s = kIn;
  s.input.reset();
  EXPECT_EQ(refused(s),
            "weight_scale without input_scale; the 8-bit modes carry input_scale and weight_scale");
  s = kNone;
  s.output = 1.f;
  EXPECT_EQ(refused(s),
            "output_scale without input_scale; the 8-bit modes carry input_scale and weight_scale");
}

TEST(SubMContract, ScalesArePositiveAndFinite) {
  const auto refused = [](subm::Scales s) { return subm::scale_refusal(s).value_or(""); };
  EXPECT_EQ(refused(kInOut), "");
  subm::Scales s = kInOut;
  s.input = 0.f;
  EXPECT_EQ(refused(s), "input_scale is 0; want positive and finite");
  s = kInOut;
  s.weight[3] = -0.5f;
  EXPECT_EQ(refused(s), "weight_scale[3] is -0.5; want positive and finite");
  s = kInOut;
  s.output = std::numeric_limits<float>::infinity();
  EXPECT_EQ(refused(s), "output_scale is inf; want positive and finite");
  s = kInOut;
  s.output = std::numeric_limits<float>::quiet_NaN();
  EXPECT_NE(refused(s), "");
}

TEST(SubMContract, Float) {
  EXPECT_ACCEPTED(node(DataType::kHALF, DataType::kHALF), kNone);
  EXPECT_ACCEPTED(node(DataType::kFLOAT, DataType::kFLOAT), kNone)
      << "a kernel is the table's question";
  EXPECT_ACCEPTED(node(DataType::kBF16, DataType::kBF16), kNone);
  EXPECT_REFUSED(node(DataType::kINT8, DataType::kHALF), kNone)
      << "8-bit features want their scales";
  EXPECT_REFUSED(node(DataType::kHALF, DataType::kBF16), kNone)
      << "the output is the features' type";
  EXPECT_REFUSED(node(DataType::kHALF, DataType::kHALF), kIn) << "scales on a float node";
  EXPECT_REFUSED(node(DataType::kHALF, DataType::kHALF), kInOut) << "scales on a float node";
}

TEST(SubMContract, EightBitIn) {
  EXPECT_ACCEPTED(node(DataType::kINT8, DataType::kHALF), kIn);
  EXPECT_ACCEPTED(node(DataType::kFP8, DataType::kHALF), kIn);
  EXPECT_REFUSED(node(DataType::kINT8, DataType::kINT8), kIn) << "no output_scale: float16";
  EXPECT_REFUSED(node(DataType::kINT8, DataType::kBF16), kIn) << "no output_scale: float16";
  subm::Scales s = kIn;
  s.weight.pop_back();
  EXPECT_EQ(why(node(DataType::kINT8, DataType::kHALF), s),
            "weight_scale has 63 entries, the weights 64 output channels");
}

TEST(SubMContract, EightBitInAndOut) {
  EXPECT_ACCEPTED(node(DataType::kINT8, DataType::kINT8), kInOut);
  EXPECT_ACCEPTED(node(DataType::kFP8, DataType::kFP8), kInOut);
  EXPECT_REFUSED(node(DataType::kINT8, DataType::kFP8), kInOut)
      << "the output is the features' type";
  EXPECT_REFUSED(node(DataType::kINT8, DataType::kHALF), kInOut) << "output_scale: 8-bit";
}

TEST(SubMContract, EveryTensorHasOneType) {
  auto io = node(DataType::kINT8, DataType::kINT8);
  io[1] = dyn(Dims{5, {kC, 3, 3, 3, kC}}, DataType::kFP8);
  EXPECT_REFUSED(io, kInOut) << "weights of another 8-bit type";
  io = node(DataType::kHALF, DataType::kHALF);
  io[2] = dyn(Dims{4, {3, 3, 3, kN}}, DataType::kINT64);
  EXPECT_REFUSED(io, kNone) << "int64 pairs";
  io = node(DataType::kHALF, DataType::kHALF);
  io[5] = dyn(vec(kN / 32), DataType::kFLOAT);
  EXPECT_REFUSED(io, kNone) << "a float tile mask";
  io = node(DataType::kHALF, DataType::kHALF);
  io[6] = dyn(vec(kC), DataType::kINT8);
  EXPECT_REFUSED(io, kNone) << "an 8-bit bias";
}

// getOutputDataTypes has only the types, and in a weakly typed build the
// graph's: float32 features and weights there are a float16 engine's.
TEST(SubMContract, TypesAlone) {
  const DataType ok[7] = {DataType::kHALF,  DataType::kHALF,  DataType::kINT32, DataType::kINT32,
                          DataType::kINT32, DataType::kINT32, DataType::kFLOAT};
  const auto refused = [&](std::int32_t pos, DataType t, const subm::Scales& s,
                           std::int32_t nb = 7) {
    DataType in[7];
    std::copy(ok, ok + 7, in);
    in[pos] = t;
    return subm::type_refusal(in, nb, s).value_or("");
  };
  EXPECT_EQ(refused(0, DataType::kHALF, kNone), "");
  EXPECT_EQ(refused(0, DataType::kFLOAT, kNone), "");
  EXPECT_EQ(refused(1, DataType::kFLOAT, kNone), "") << "the weights are refusal's";
  EXPECT_EQ(refused(0, DataType::kINT8, kIn), "");
  EXPECT_EQ(refused(0, DataType::kHALF, kNone, 9), "9 inputs; want 7");
  EXPECT_EQ(refused(0, DataType::kINT8, kNone),
            "int8 features without scale attributes; want float32, float16 or bfloat16");
  EXPECT_EQ(refused(0, DataType::kHALF, kIn),
            "float16 features with scale attributes; want int8 or float8e4m3");
  EXPECT_EQ(refused(2, DataType::kINT64, kNone), "pairs is int64; want int32");
  EXPECT_EQ(refused(5, DataType::kFLOAT, kNone), "tile_mask is float32; want int32");
  for (const subm::Scales& s : {kNone, kIn, kInOut}) {
    const DataType features = s.input ? DataType::kINT8 : DataType::kHALF;
    for (const DataType bias : {DataType::kFLOAT, DataType::kHALF, DataType::kBF16}) {
      DataType in[7];
      std::copy(ok, ok + 7, in);
      in[0] = features;
      in[6] = bias;
      EXPECT_EQ(subm::type_refusal(in, 7, s).value_or(""), "") << subm::type_name(bias);
    }
  }
  EXPECT_EQ(refused(6, DataType::kINT8, kNone),
            "bias is int8; want float32, float16 or bfloat16");
  subm::Scales s = kIn;
  s.input = -1.f;
  EXPECT_EQ(refused(0, DataType::kINT8, s), "input_scale is -1; want positive and finite");
}

TEST(SubMContract, ShapesAgreeWithEachOther) {
  auto io = node(DataType::kINT8, DataType::kINT8);
  io[6] = dyn(vec(kC + 1), DataType::kFLOAT);
  EXPECT_REFUSED(io, kInOut) << "bias not [C_out]";
  io = node(DataType::kHALF, DataType::kHALF);
  io[0] = dyn(dims2(kN, kC / 2), DataType::kHALF);
  EXPECT_REFUSED(io, kNone) << "C_in differs between features and weights";
  io = node(DataType::kHALF, DataType::kHALF);
  io[2] = dyn(Dims{4, {3, 3, 1, kN}}, DataType::kINT32);
  EXPECT_REFUSED(io, kNone) << "pairs for another kernel volume";
  io = node(DataType::kHALF, DataType::kHALF);
  io[1] = dyn(Dims{5, {kC, 1, 3, 3, kC}}, DataType::kHALF);
  io[2] = dyn(Dims{4, {3, 3, 1, kN}}, DataType::kINT32);
  EXPECT_REFUSED(io, kNone) << "pairs for another kernel with as many taps";
  io = node(DataType::kHALF, DataType::kHALF);
  io[2] = dyn(Dims{4, {3, 3, 3, 200}}, DataType::kINT32);
  EXPECT_REFUSED(io, kNone) << "pairs not padded to the M tile";
  io = node(DataType::kHALF, DataType::kHALF);
  io[3] = dyn(vec(kN), DataType::kINT32);
  EXPECT_REFUSED(io, kNone) << "point_mask is [N, 1]";
  io = node(DataType::kHALF, DataType::kHALF);
  io[4] = dyn(vec(kN - 1), DataType::kINT32);
  EXPECT_REFUSED(io, kNone) << "mask_argsort is [N]";
  io = node(DataType::kHALF, DataType::kHALF);
  io[5] = dyn(vec(kN / 32 + 1), DataType::kINT32);
  EXPECT_REFUSED(io, kNone) << "tile_mask is [P / 32]";
  io = node(DataType::kHALF, DataType::kHALF);
  io[1] = dyn(Dims{5, {kC, 3, 2, 3, kC}}, DataType::kHALF);
  EXPECT_REFUSED(io, kNone) << "an even kernel";
  io = node(DataType::kHALF, DataType::kHALF);
  io[1] = dyn(Dims{5, {kC, 5, 5, 3, kC}}, DataType::kHALF);
  EXPECT_REFUSED(io, kNone) << "a kernel volume over 32";
}

// supportsFormatCombination asks per slot, before the shapes are known.
TEST(SubMContract, AcceptsPerSlot) {
  using subm::Mode;
  EXPECT_TRUE(subm::accepts(0, DataType::kHALF, DataType::kHALF, Mode::kFloat));
  EXPECT_TRUE(subm::accepts(0, DataType::kFLOAT, DataType::kFLOAT, Mode::kFloat));
  EXPECT_FALSE(subm::accepts(0, DataType::kINT8, DataType::kINT8, Mode::kFloat));
  EXPECT_TRUE(subm::accepts(0, DataType::kINT8, DataType::kINT8, Mode::k8BitIn));
  EXPECT_FALSE(subm::accepts(0, DataType::kHALF, DataType::kHALF, Mode::k8BitIn));
  EXPECT_TRUE(subm::accepts(1, DataType::kFP8, DataType::kFP8, Mode::k8BitInOut));
  EXPECT_FALSE(subm::accepts(1, DataType::kINT8, DataType::kFP8, Mode::k8BitInOut));
  for (std::int32_t pos : {2, 3, 4, 5}) {
    EXPECT_TRUE(subm::accepts(pos, DataType::kINT32, DataType::kHALF, Mode::kFloat)) << pos;
    EXPECT_FALSE(subm::accepts(pos, DataType::kINT64, DataType::kHALF, Mode::kFloat)) << pos;
  }
  EXPECT_TRUE(subm::accepts(6, DataType::kFLOAT, DataType::kINT8, Mode::k8BitIn));
  EXPECT_TRUE(subm::accepts(7, DataType::kHALF, DataType::kINT8, Mode::k8BitIn)) << "the output";
  EXPECT_TRUE(subm::accepts(7, DataType::kINT8, DataType::kINT8, Mode::k8BitInOut));
  EXPECT_FALSE(subm::accepts(7, DataType::kHALF, DataType::kINT8, Mode::k8BitInOut));
  EXPECT_FALSE(subm::accepts(8, DataType::kFLOAT, DataType::kINT8, Mode::k8BitIn))
      << "past the output";
}

// What the contract accepts and the spconv library does not run.
TEST(SubMContract, SpconvRunsFloat32Float16AndInt8In) {
  EXPECT_FALSE(subm::spconv_refusal(DataType::kFLOAT, kNone));
  EXPECT_FALSE(subm::spconv_refusal(DataType::kHALF, kNone));
  EXPECT_FALSE(subm::spconv_refusal(DataType::kINT8, kIn));

  const auto bf16 = subm::spconv_refusal(DataType::kBF16, kNone);
  ASSERT_TRUE(bf16);
  EXPECT_NE(bf16->find("bfloat16 features"), std::string::npos) << *bf16;
  const auto fp8 = subm::spconv_refusal(DataType::kFP8, kIn);
  ASSERT_TRUE(fp8);
  EXPECT_NE(fp8->find("float8e4m3 features"), std::string::npos) << *fp8;
  const auto out8 = subm::spconv_refusal(DataType::kINT8, kInOut);
  ASSERT_TRUE(out8);
  EXPECT_NE(out8->find("output_scale is set"), std::string::npos) << *out8;
}

// The rulebook tensors' shapes are functions of N at t4sparse's granularities.
TEST(SubMContract, PairsArePaddedTo256Rows) {
  EXPECT_EQ(subm::padded_rows(1), 256);
  EXPECT_EQ(subm::padded_rows(256), 256);
  EXPECT_EQ(subm::padded_rows(257), 512);
  EXPECT_EQ(subm::kPadRows % subm::kMaskRows, 0);
}

// SparseConv: the rows are point_mask's, and pairs and tile_mask are not padded.
TEST(SubMContract, ComputedRowsTakeUnpaddedRulebookTensors)
{
  using namespace autoware::tensorrt_plugins::subm;  // NOLINT
  const auto desc = [](nvinfer1::DataType type, std::initializer_list<std::int64_t> dims) {
    nvinfer1::DynamicPluginTensorDesc d{};
    d.desc.type = type;
    d.desc.dims.nbDims = static_cast<std::int32_t>(dims.size());
    std::int32_t axis = 0;
    for (const auto extent : dims) {
      d.desc.dims.d[axis++] = extent;
    }
    d.min = d.opt = d.max = d.desc.dims;
    return d;
  };
  using nvinfer1::DataType;
  const std::int64_t n = 1000, m = 700;
  nvinfer1::DynamicPluginTensorDesc inputs[kInputs] = {
    desc(DataType::kHALF, {n, 32}),          desc(DataType::kHALF, {64, 3, 3, 3, 32}),
    desc(DataType::kINT32, {3, 3, 3, m}),    desc(DataType::kINT32, {m, 1}),
    desc(DataType::kINT32, {m}),             desc(DataType::kINT32, {m}),
    desc(DataType::kFLOAT, {64})};
  EXPECT_FALSE(refusal(inputs, kInputs, DataType::kHALF, Scales{}, Rows::kPointMask));
  // the same tensors are not a SubMConv's: its rows are the features', padded
  const Refusal as_subm = refusal(inputs, kInputs, DataType::kHALF, Scales{}, Rows::kFeatures);
  ASSERT_TRUE(as_subm);
  EXPECT_NE(as_subm->find("pairs is"), std::string::npos) << *as_subm;
  // a padded pairs tensor is not a SparseConv's
  inputs[at(Slot::kPairs)] = desc(DataType::kINT32, {3, 3, 3, padded_rows(700)});
  const Refusal padded = refusal(inputs, kInputs, DataType::kHALF, Scales{}, Rows::kPointMask);
  ASSERT_TRUE(padded);
  EXPECT_NE(padded->find("pairs is"), std::string::npos) << *padded;
}
