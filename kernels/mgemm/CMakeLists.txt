cmake_minimum_required(VERSION 3.18)
project(MixedGemm LANGUAGES CXX CUDA)

set(CMAKE_PREFIX_PATH "/home/actypedef/miniconda3/envs/mhq/lib/python3.12/site-packages/torch")
find_package(Torch REQUIRED)

# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
include_directories(${Python3_INCLUDE_DIRS})
find_package(pybind11 CONFIG REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS /home/actypedef/miniconda3/envs/mhq/lib/python3.12/site-packages/torch/lib NO_DEFAULT_PATH)

# 设置 CUTLASS 根目录
set(CUTLASS_ROOT ../../cutlass)

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

add_compile_options(-w)
add_compile_options(-fpermissive)

add_library(gemm_objs OBJECT src/fp4.cu src/fp6.cu src/fp8.cu src/gemm.cu src/reorder.cu src/rmsnorm.cu src/activate.cu)


# 创建目标可执行文件

add_executable(bench_fp4 benchmark/bench_fp4.cu
$<TARGET_OBJECTS:gemm_objs>)
add_executable(bench_fp6 benchmark/bench_fp6.cu
$<TARGET_OBJECTS:gemm_objs>)
add_executable(bench_fp8 benchmark/bench_fp8.cu
$<TARGET_OBJECTS:gemm_objs>)
add_executable(bench_gemm benchmark/bench_gemm.cu
$<TARGET_OBJECTS:gemm_objs>)
add_executable(bench_reorder benchmark/bench_reorder.cu
$<TARGET_OBJECTS:gemm_objs>)
add_executable(bench_reorder_gemm benchmark/bench_reorder_gemm.cu
$<TARGET_OBJECTS:gemm_objs>)
# pybind11_add_module(mixedgemm src/bindings.cpp
# $<TARGET_OBJECTS:gemm_objs>)

add_library(mixedgemm MODULE src/bindings.cpp $<TARGET_OBJECTS:gemm_objs>)
set_source_files_properties(src/bindings.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(mixedgemm PRIVATE pybind11::module "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" CUDA::cudart)

# 设置 CUDA 编译选项（允许宽松 constexpr）
target_compile_options(bench_fp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(bench_fp6 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(bench_fp8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(bench_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(bench_reorder PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(bench_reorder_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(gemm_objs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)
target_compile_options(mixedgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# 添加 CUTLASS 的架构宏定义（确保针对 SM120 架构）
target_compile_definitions(bench_fp4 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(bench_fp6 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(bench_fp8 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(bench_gemm PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(bench_reorder PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(bench_reorder_gemm PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(mixedgemm PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)

set_property(TARGET mixedgemm PROPERTY CXX_STANDARD 17)
set_target_properties(mixedgemm PROPERTIES PREFIX "")
add_definitions(-DTORCH_EXTENSION_NAME=mixedgemm)
