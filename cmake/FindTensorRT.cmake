# Copyright 2025 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FindTensorRT
# ------------
# Locates the TensorRT headers and the nvinfer library. Set TENSORRT_ROOT (variable or
# environment) to look in a non-standard prefix.
#
# Result variables:  TensorRT_FOUND, TensorRT_VERSION, TensorRT_INCLUDE_DIRS, TensorRT_LIBRARIES
# Imported target:   TensorRT::nvinfer

include(FindPackageHandleStandardArgs)

if(NOT DEFINED TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
endif()

find_path(TensorRT_INCLUDE_DIR NvInferRuntime.h
  HINTS ${TENSORRT_ROOT} ${CUDAToolkit_INCLUDE_DIRS}
  PATH_SUFFIXES include include/${CMAKE_LIBRARY_ARCHITECTURE})
find_library(TensorRT_nvinfer_LIBRARY nvinfer
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/${CMAKE_LIBRARY_ARCHITECTURE})

if(TensorRT_INCLUDE_DIR)
  if(EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_version_header)
  else()
    file(READ "${TensorRT_INCLUDE_DIR}/NvInfer.h" _trt_version_header)
  endif()
  foreach(_part MAJOR MINOR PATCH)
    string(REGEX MATCH "define[ \t]+NV_TENSORRT_${_part}[ \t]+([0-9]+)" _ "${_trt_version_header}")
    set(_trt_${_part} "${CMAKE_MATCH_1}")
    if("${_trt_${_part}}" STREQUAL "")
      # TensorRT 10.11+ defines NV_TENSORRT_* in terms of TRT_*_ENTERPRISE
      string(REGEX MATCH "define[ \t]+TRT_${_part}_ENTERPRISE[ \t]+([0-9]+)" _ "${_trt_version_header}")
      set(_trt_${_part} "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  set(TensorRT_VERSION "${_trt_MAJOR}.${_trt_MINOR}.${_trt_PATCH}")
  unset(_trt_version_header)
endif()

find_package_handle_standard_args(TensorRT
  REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY
  VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND)
  set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
  set(TensorRT_LIBRARIES "${TensorRT_nvinfer_LIBRARY}")
  if(NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer SHARED IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
      IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY)
