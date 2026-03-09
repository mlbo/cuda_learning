# =============================================================================
# CMakeLists.txt for Chapter 30: CUDA Libraries
# =============================================================================

# cuBLAS GEMM示例
add_executable(30_01_cublas_gemm 01_cublas_gemm.cu)
target_link_libraries(30_01_cublas_gemm CUDA::cudart CUDA::cublas)

# cuDNN Conv示例 (需要cuDNN)
find_path(CUDNN_INCLUDE_DIR cudnn.h)
find_library(CUDNN_LIBRARY cudnn)

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    add_executable(30_02_cudnn_conv 02_cudnn_conv.cu)
    target_link_libraries(30_02_cudnn_conv CUDA::cudart ${CUDNN_LIBRARY})
    target_include_directories(30_02_cudnn_conv PRIVATE ${CUDNN_INCLUDE_DIR})
else()
    message(STATUS "cuDNN not found, skipping cudnn_conv example")
endif()

# CUB Reduce示例 (CUB随CUDA Toolkit提供)
add_executable(30_03_cub_reduce 03_cub_reduce.cu)
target_link_libraries(30_03_cub_reduce CUDA::cudart)

# CUTLASS GEMM示例 (需要CUTLASS)
find_path(CUTLASS_INCLUDE_DIR cutlass/gemm/device/gemm.h)

if(CUTLASS_INCLUDE_DIR)
    add_executable(30_04_cutlass_gemm 04_cutlass_gemm.cu)
    target_link_libraries(30_04_cutlass_gemm CUDA::cudart)
    target_include_directories(30_04_cutlass_gemm PRIVATE ${CUTLASS_INCLUDE_DIR})
else()
    message(STATUS "CUTLASS not found, skipping cutlass_gemm example")
    # 创建一个简化版本
    add_executable(30_04_cutlass_gemm 04_cutlass_gemm.cu)
    target_link_libraries(30_04_cutlass_gemm CUDA::cudart)
endif()