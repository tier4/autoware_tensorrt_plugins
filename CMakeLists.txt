cmake_minimum_required(VERSION 3.22)

# Plain CMake project: builds in a bare CUDA + TensorRT environment without ROS, ament or
# colcon. When ament_cmake is available (a ROS 2 environment), ament_package() additionally
# registers the package in the ament index so ROS 2 nodes locate it via the usual lookup.
project(autoware_tensorrt_plugins LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_SPCONV_PLUGINS "Build the sparse convolution plugin library (needs spconv/cumm)" ON)
option(CUDA_VERBOSE "Verbose output of CUDA modules" OFF)
include(CTest)  # defines BUILD_TESTING (default ON)

# --- Dependencies ---------------------------------------------------------------------------
find_package(CUDAToolkit 12 REQUIRED)
find_package(TensorRT 10 REQUIRED)
find_package(ament_cmake QUIET)

if(BUILD_SPCONV_PLUGINS)
  find_package(cumm QUIET)
  find_package(spconv QUIET)
  if(cumm_FOUND AND spconv_FOUND)
    message(STATUS "spconv is available: building the sparse convolution plugins")
  else()
    message(WARNING "spconv is NOT available: the sparse convolution plugins will not be built")
    set(BUILD_SPCONV_PLUGINS OFF)
  endif()
endif()

# --- CUDA -----------------------------------------------------------------------------------
# The target architectures must be set before the CUDA language is enabled. The kernels need
# sm_70 or newer (16-bit atomics).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real 87-real 89-real)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
    if(CUDAToolkit_VERSION VERSION_LESS 13.0)
      list(APPEND CMAKE_CUDA_ARCHITECTURES 101-real)
    else()  # CUDA 13.0 renamed SM101 to SM110
      list(APPEND CMAKE_CUDA_ARCHITECTURES 110-real)
    endif()
    list(APPEND CMAKE_CUDA_ARCHITECTURES 120)
  else()
    list(APPEND CMAKE_CUDA_ARCHITECTURES 89-virtual)
  endif()
endif()
enable_language(CUDA)
if(CUDA_VERBOSE)
  message(STATUS "CUDA ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_DIR}), architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "TensorRT ${TensorRT_VERSION}: ${TensorRT_LIBRARIES}")
endif()

# --- Common compile options -----------------------------------------------------------------
add_library(plugin_build_options INTERFACE)
target_compile_options(plugin_build_options INTERFACE
  # Objects compiled by nvcc must not take part in link-time optimization.
  -fno-lto
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wno-deprecated-declarations>
  # cSpell:ignore expt
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-diag-suppress=1675;-Xcompiler=-Wno-deprecated-declarations>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)
target_compile_definitions(plugin_build_options INTERFACE _GLIBCXX_USE_CXX11_ABI=1)

# --- Kernels (no TensorRT dependency) -------------------------------------------------------
# Built as a static library so every plugin library is self-contained: an installed plugin
# .so has no companion library to locate at load time.
add_library(cuda_ops STATIC
  src/argsort_ops/argsort.cu
  src/bev_ops/bev_pool_cuda.cu
  src/scatter_ops/segment_csr.cu
  src/unique_ops/unique.cu
  src/multi_scale_deform_attn_ops/ms_deform_attn_kernel.cu
  src/rotate_ops/rotate_kernel.cu
  src/select_and_pad_ops/select_and_pad_kernel.cu
)
target_include_directories(cuda_ops PUBLIC include)
target_link_libraries(cuda_ops PUBLIC CUDA::cudart PRIVATE plugin_build_options)
# Kernel symbols stay internal to the plugin libraries that link this archive.
set_target_properties(cuda_ops PROPERTIES CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)

# --- Plugin libraries -----------------------------------------------------------------------
# The plugins are split into two shared libraries by external dependency:
#   libautoware_tensorrt_plugins.so        pure CUDA plugins (cudart + nvinfer only)
#   libautoware_tensorrt_plugins_spconv.so sparse-convolution plugins (+ spconv/cumm)
# Consumers load the libraries they need.
set(PLUGIN_LIBRARIES ${PROJECT_NAME})

add_library(${PROJECT_NAME} SHARED
  src/argsort_plugin.cpp
  src/argsort_plugin_creator.cpp
  src/quick_cumsum_cuda_plugin.cpp
  src/quick_cumsum_cuda_plugin_creator.cpp
  src/multi_scale_deformable_attention_plugin.cpp
  src/multi_scale_deformable_attention_plugin_creator.cpp
  src/rotate_plugin.cpp
  src/rotate_plugin_creator.cpp
  src/segment_csr_plugin_creator.cpp
  src/segment_csr_plugin.cpp
  src/select_and_pad_plugin.cpp
  src/select_and_pad_plugin_creator.cpp
  src/unique_plugin.cpp
  src/unique_plugin_creator.cpp
  src/plugin_registration.cpp
  src/plugin_utils.cpp
)
target_link_libraries(${PROJECT_NAME} PRIVATE TensorRT::nvinfer CUDA::cudart cuda_ops)

if(BUILD_SPCONV_PLUGINS)
  add_library(${PROJECT_NAME}_spconv SHARED
    src/get_indices_pairs_implicit_gemm_plugin_creator.cpp
    src/get_indices_pairs_implicit_gemm_plugin.cpp
    src/get_indices_pairs_plugin_creator.cpp
    src/get_indices_pairs_plugin.cpp
    src/implicit_gemm_plugin_creator.cpp
    src/implicit_gemm_plugin.cpp
    src/indice_conv_plugin_creator.cpp # cSpell:ignore indice
    src/indice_conv_plugin.cpp
    src/plugin_registration_spconv.cpp
    src/plugin_utils.cpp
  )
  target_compile_definitions(${PROJECT_NAME}_spconv PRIVATE TV_CUDA)
  target_link_libraries(${PROJECT_NAME}_spconv PRIVATE TensorRT::nvinfer CUDA::cudart spconv::spconv)
  list(APPEND PLUGIN_LIBRARIES ${PROJECT_NAME}_spconv)
endif()

foreach(plugin_library IN LISTS PLUGIN_LIBRARIES)
  target_include_directories(${plugin_library} PRIVATE include)
  target_link_libraries(${plugin_library} PRIVATE plugin_build_options)
  # Only the TensorRT entry points (getCreators, setLoggerFinder) are exported, so that
  # nothing leaks between plugin libraries loaded into the same process.
  set_target_properties(${plugin_library} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
endforeach()

# --- Tests ----------------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(GTest)
  if(GTest_FOUND)
    add_executable(reference_kernels_test
      test/argsort_kernel_test.cpp
      test/segment_csr_kernel_test.cpp
      test/unique_kernel_test.cpp
    )
    target_include_directories(reference_kernels_test PRIVATE test)
    target_link_libraries(reference_kernels_test PRIVATE cuda_ops GTest::gtest_main plugin_build_options)
    add_test(NAME reference_kernels_test COMMAND reference_kernels_test)

    # Loads the built libraries through dlopen and checks which plugins each provides.
    add_executable(plugin_libraries_test test/plugin_libraries_test.cpp)
    add_dependencies(plugin_libraries_test ${PLUGIN_LIBRARIES})
    target_link_libraries(plugin_libraries_test PRIVATE
      TensorRT::nvinfer CUDA::cudart ${CMAKE_DL_LIBS} GTest::gtest_main plugin_build_options)
    target_compile_definitions(plugin_libraries_test PRIVATE
      CORE_PLUGIN_LIBRARY="$<TARGET_FILE:${PROJECT_NAME}>")
    if(BUILD_SPCONV_PLUGINS)
      target_compile_definitions(plugin_libraries_test PRIVATE
        SPCONV_PLUGIN_LIBRARY="$<TARGET_FILE:${PROJECT_NAME}_spconv>")
    endif()
    add_test(NAME plugin_libraries_test COMMAND plugin_libraries_test)
  else()
    message(WARNING "GTest not found: tests will not be built")
  endif()
endif()

# --- Install --------------------------------------------------------------------------------
# Same layout with or without ament: consumers keep resolving
# $(find-pkg-share autoware_tensorrt_plugins)/plugins/<library>.so.
set(PACKAGE_SHARE_DIR share/${PROJECT_NAME})
install(TARGETS ${PLUGIN_LIBRARIES} LIBRARY DESTINATION ${PACKAGE_SHARE_DIR}/plugins)

if(ament_cmake_FOUND)
  # Registers the package in the ament index and installs package.xml and the environment
  # hooks colcon sources, so ros2 pkg / ament_index / launch's find-pkg-share find this prefix.
  install(FILES package.xml DESTINATION ${PACKAGE_SHARE_DIR})
  ament_package()
else()
  message(STATUS "ament_cmake not found: the package will not be registered in the ament index")
endif()
