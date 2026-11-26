// Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef AUTOWARE__SCATTER_OPS_REDUCTION_CUH
#define AUTOWARE__SCATTER_OPS_REDUCTION_CUH

#include "autoware/scatter_ops/atomics.cuh"
#include "autoware/scatter_ops/reduction.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <limits>
#include <type_traits>

// cSpell:ignore bfloat
// The single-column kernel reduces inside a warp.
static __device__ inline Float8E4M3 __shfl_down_sync(
  const unsigned int mask, const Float8E4M3 value, const unsigned int delta)
{
  return Float8E4M3{
    static_cast<std::uint8_t>(__shfl_down_sync(mask, static_cast<int>(value.bits), delta))};
}

// The branches on REDUCE are resolved at compile time, so an element type only has to provide
// what its reductions use: ordering for MIN and MAX, arithmetic for the others. The 8-bit types
// (int8, Float8E4M3) are MIN and MAX only.
template <typename scalar_t, ReductionType REDUCE>
struct Reducer
{
  static inline __host__ __device__ scalar_t zero()
  {
    if constexpr (std::is_same<scalar_t, Float8E4M3>::value) {
      return Float8E4M3{0x00U};
    } else {
      return (scalar_t)0;
    }
  }

  static inline __host__ __device__ scalar_t init()
  {
    if constexpr (REDUCE == MUL || REDUCE == DIV) {
      return (scalar_t)1;
    } else if constexpr (REDUCE == MIN) {
      if constexpr (
        std::is_same<scalar_t, __half>::value || std::is_same<scalar_t, __nv_bfloat16>::value) {
        scalar_t max_val;
        *(reinterpret_cast<unsigned short *>(&(max_val))) =
          std::is_same<scalar_t, __half>::value ? 0x7C00 : 0x7F80;
        return max_val;
      } else if constexpr (std::is_same<scalar_t, Float8E4M3>::value) {
        return Float8E4M3{0x7EU};  // 448, the largest value: E4M3 has no infinity
      } else {
        return std::numeric_limits<scalar_t>::max();
      }
    } else if constexpr (REDUCE == MAX) {
      if constexpr (
        std::is_same<scalar_t, __half>::value || std::is_same<scalar_t, __nv_bfloat16>::value) {
        scalar_t min_val;
        *(reinterpret_cast<unsigned short *>(&(min_val))) =
          std::is_same<scalar_t, __half>::value ? 0xFC00 : 0xFF80;
        return min_val;
      } else if constexpr (std::is_same<scalar_t, Float8E4M3>::value) {
        return Float8E4M3{0xFEU};  // -448, the smallest value
      } else {
        return std::numeric_limits<scalar_t>::lowest();
      }
    } else {
      return (scalar_t)0;
    }
  }

  static inline __host__ __device__ void update(scalar_t * val, scalar_t new_val)
  {
    if constexpr (REDUCE == SUM || REDUCE == MEAN) {
      *val = *val + new_val;
    } else if constexpr (REDUCE == MUL) {
      *val = *val * new_val;
    } else if constexpr (REDUCE == DIV) {
      *val = *val / new_val;
    } else if constexpr (REDUCE == MIN) {
      if (new_val < *val) *val = new_val;
    } else {
      if (new_val > *val) *val = new_val;
    }
  }

  static inline __host__ __device__ void update(
    scalar_t * val, scalar_t new_val, int64_t * arg, int64_t new_arg)
  {
    if constexpr (REDUCE == SUM || REDUCE == MEAN) {
      *val = *val + new_val;
    } else if constexpr (REDUCE == MUL) {
      *val = *val * new_val;
    } else if constexpr (REDUCE == DIV) {
      *val = *val / new_val;
    } else if constexpr (REDUCE == MIN) {
      if (new_val < *val) {
        *val = new_val;
        *arg = new_arg;
      }
    } else {
      if (new_val > *val) {
        *val = new_val;
        *arg = new_arg;
      }
    }
  }

  static inline __host__ __device__ void write(scalar_t * address, scalar_t val, int count)
  {
    if constexpr (REDUCE == SUM || REDUCE == MUL || REDUCE == DIV) {
      *address = val;
    } else if constexpr (REDUCE == MEAN) {
      *address = val / (scalar_t)(count > 0 ? count : 1);
    } else {
      *address = count > 0 ? val : zero();
    }
  }

  static inline __host__ __device__ void write(
    scalar_t * address, scalar_t val, int64_t * arg_address, int64_t arg, int count)
  {
    if constexpr (REDUCE == SUM || REDUCE == MUL || REDUCE == DIV) {
      *address = val;
    } else if constexpr (REDUCE == MEAN) {
      *address = val / (scalar_t)(count > 0 ? count : 1);
    } else {
      if (count > 0) {
        *address = val;
        *arg_address = arg;
      } else {
        *address = zero();
      }
    }
  }

  static inline __device__ void atomic_write(scalar_t * address, scalar_t val)
  {
    if (REDUCE == SUM || REDUCE == MEAN)
      atomAdd(address, val);
    else if (REDUCE == MUL)
      atomMul(address, val);
    else if (REDUCE == DIV)
      atomDiv(address, val);
    else if (REDUCE == MIN)
      atomMin(address, val);
    else if (REDUCE == MAX)
      atomMax(address, val);
  }
};

#endif  // AUTOWARE__SCATTER_OPS_REDUCTION_CUH
