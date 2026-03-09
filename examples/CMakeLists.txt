# =============================================================================
# CUDA 学习示例 - 根 CMakeLists.txt
# =============================================================================
#
# 本文件用于构建 CUDA 学习教程的所有示例程序
#
# 使用方法:
#   mkdir build && cd build
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # 根据你的 GPU 架构设置
#   make -j
#
# =============================================================================

# CMake 最低版本要求 (3.18 支持现代 CUDA 特性)
cmake_minimum_required(VERSION 3.18)

# 项目名称和语言配置
project(cuda_examples LANGUAGES CUDA CXX)

# 查找 CUDA 工具包 (现代 CMake 方式)
find_package(CUDAToolkit REQUIRED)

# =============================================================================
# CUDA 架构配置
# =============================================================================
# 默认使用 sm_80 (A100), 可通过命令行覆盖
# 示例: cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 3090
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA 架构版本" FORCE)
endif()

# 常见 GPU 架构参考:
# - sm_75: Turing (RTX 2080, T4)
# - sm_80: Ampere (A100)
# - sm_86: Ampere (RTX 3090, A40)
# - sm_89: Ada Lovelace (RTX 4090)
# - sm_90: Hopper (H100)

# =============================================================================
# 编译选项配置
# =============================================================================
# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 添加编译选项
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>           # CUDA 优化等级
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>  # 允许在 constexpr 中使用 __device__ 函数
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>    # 支持 lambda 中的 __device__ 标注
)

# =============================================================================
# 示例子目录
# =============================================================================
# 按章节顺序添加所有示例
add_subdirectory(01_gpu_basics)      # 第 1 章: GPU 基础知识
add_subdirectory(02_cuda_intro)      # 第 2 章: CUDA 编程入门
add_subdirectory(03_gpu_hardware)    # 第 3 章: GPU 硬件架构
add_subdirectory(04_thread_hierarchy)# 第 4 章: 线程层次结构
add_subdirectory(05_hello_cuda)      # 第 5 章: Hello CUDA
add_subdirectory(06_memory_basics)   # 第 6 章: 内存模型基础
add_subdirectory(07_kernel_deep)     # 第 7 章: 核函数深入
add_subdirectory(08_profiling)       # 第 8 章: 性能分析
add_subdirectory(09_memory_opt)      # 第 9 章: 内存优化
add_subdirectory(10_precision)       # 第 10 章: 精度与数值
add_subdirectory(11_roofline)        # 第 11 章: Roofline 模型
add_subdirectory(12_atomic_race)     # 第 12 章: 原子操作与竞争条件
add_subdirectory(13_shared_memory)   # 第 13 章: 共享内存深入
add_subdirectory(14_reduce)          # 第 14 章: 规约算法优化
add_subdirectory(15_bank_conflict)   # 第 15 章: Bank Conflict优化
add_subdirectory(16_cooperative_groups) # 第 16 章: Cooperative Groups
add_subdirectory(17_gemm_basics)       # 第 17 章: GEMM优化入门
add_subdirectory(18_gemm_tiling)       # 第 18 章: GEMM分块优化
add_subdirectory(19_tensor_core)       # 第 19 章: Tensor Core编程
add_subdirectory(20_convolution)       # 第 20 章: 卷积算子实现
add_subdirectory(21_async_execution)   # 第 21 章: 异步执行与延迟隐藏
add_subdirectory(22_streams)           # 第 22 章: CUDA流与并发
add_subdirectory(23_data_transfer)     # 第 23 章: 数据传输优化
add_subdirectory(24_cuda_graph)        # 第 24 章: CUDA Graph
add_subdirectory(25_multi_gpu)         # 第 25 章: 多GPU编程
add_subdirectory(26_quantization)      # 第 26 章: 低精度与量化
add_subdirectory(27_ptx_optimization)# 第 27 章: PTX与底层优化
add_subdirectory(28_micro_optimization) # 第 28 章: 微指令级调优
add_subdirectory(29_ilp_divergence)  # 第 29 章: ILP与Warp Divergence
add_subdirectory(30_cuda_libraries)  # 第 30 章: CUDA官方库实战

# =============================================================================
# 构建信息输出
# =============================================================================
message(STATUS "")
message(STATUS "=== CUDA 学习示例构建配置 ===")
message(STATUS "CMake 版本:        ${CMAKE_VERSION}")
message(STATUS "CUDA 编译器:       ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA 架构:         ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ 标准:          ${CMAKE_CXX_STANDARD}")
message(STATUS "构建类型:          ${CMAKE_BUILD_TYPE}")
message(STATUS "=============================")
message(STATUS "")