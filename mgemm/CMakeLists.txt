cmake_minimum_required(VERSION 3.18)
project(MixedGemm LANGUAGES CXX CUDA)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# 执行 Python 脚本来获取 Torch 的 CMAKE 前缀路径
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _torch_found
)

if(NOT _torch_found EQUAL 0)
  message(FATAL_ERROR "PyTorch not found. Please make sure 'import torch' works in your Python environment.")
else()
  # 将找到的路径添加到 CMAKE_PREFIX_PATH 的最前面
  list(INSERT CMAKE_PREFIX_PATH 0 "${TORCH_CMAKE_PREFIX_PATH}")
  message(STATUS "Found PyTorch CMAKE_PREFIX_PATH: ${TORCH_CMAKE_PREFIX_PATH}")
endif()
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH}/../../)
find_package(Torch REQUIRED)
set(TORCH_CUDA_ARCHITECTURES "")
# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
find_package(pybind11 CONFIG REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_CMAKE_PREFIX_PATH}/../../lib/ NO_DEFAULT_PATH)

# 设置 CUTLASS 根目录
set(CUTLASS_ROOT ../cutlass)

# Remove sm_120 compile option
#message(STATUS "CMAKE_CUDA_FLAGS before cleanup: ${CMAKE_CUDA_FLAGS}")
string(REPLACE "-gencode arch=compute_120,code=sm_120" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
#message(STATUS "CMAKE_CUDA_FLAGS after cleanup: ${CMAKE_CUDA_FLAGS}")

# 包含 CUTLASS 头文件
include_directories(./include)
include_directories(/usr/local/cuda/include)
include_directories(${PYTHON_ROOT})
link_directories(${PYTHON_ROOT}/site-packages/torch/lib)
include_directories(${PYTHON_ROOT}/site-packages/torch/include)
include_directories(${PYTHON_ROOT}/site-packages/torch/include/torch/csrc/api/include)
include_directories(${CUTLASS_ROOT}/include)
include_directories(${CUTLASS_ROOT}/include)
include_directories(${CUTLASS_ROOT}/tools/util/include)
include_directories(${CUTLASS_ROOT}/examples/common)

# 设置 CMake CUDA 架构目标
set(CMAKE_CUDA_ARCHITECTURES 120a)
add_compile_options(-g)
add_compile_options(-lineinfo)
add_compile_options(-w)
add_compile_options(-fpermissive)

file(GLOB GEMM_SOURCES CONFIGURE_DEPENDS "src/*.cu")
add_library(gemm_objs OBJECT ${GEMM_SOURCES})

# 创建目标可执行文件

# pybind11_add_module(mixedgemm src/bindings.cpp
# $<TARGET_OBJECTS:gemm_objs>)

add_library(mixedgemm MODULE src/bindings.cpp $<TARGET_OBJECTS:gemm_objs>)
set_source_files_properties(src/bindings.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(mixedgemm PRIVATE pybind11::module "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" CUDA::cudart)

# 设置 CUDA 编译选项（允许宽松 constexpr）
target_compile_options(gemm_objs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)
target_compile_options(mixedgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# 添加 CUTLASS 的架构宏定义（确保针对 SM120 架构）
target_compile_definitions(mixedgemm PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)

set_property(TARGET mixedgemm PROPERTY CXX_STANDARD 17)
set_target_properties(mixedgemm PROPERTIES PREFIX "")
add_definitions(-DTORCH_EXTENSION_NAME=mixedgemm)


add_executable(bench_mxf4f6f8 benchmark/mxf4f6f8_bench.cu)
target_compile_options(bench_mxf4f6f8 PRIVATE -Xptxas=-v -lineinfo -w -O3)
target_link_libraries(bench_mxf4f6f8 gemm_objs)

add_executable(bench_mxf4f8 benchmark/mxf4f8_bench.cu)
target_compile_options(bench_mxf4f8 PRIVATE -Xptxas=-v -lineinfo -w -O3)
target_link_libraries(bench_mxf4f8 gemm_objs)

add_executable(bench_cutlass_mxf4f6f8 benchmark/cutlass_mxf4f6f8_bench.cu)
target_compile_options(bench_cutlass_mxf4f6f8 PRIVATE -Xptxas=-v -lineinfo -w -O3)
target_link_libraries(bench_cutlass_mxf4f6f8 gemm_objs)