# =============================================================================
# 第26章: 低精度与量化 - CMakeLists.txt
# =============================================================================

# 查找cuBLAS
find_package(CUDAToolkit REQUIRED)

# 01 - FP16基础
add_executable(26_01_fp16_basics 01_fp16_basics.cu)
target_link_libraries(26_01_fp16_basics CUDA::cudart)

# 02 - BF16和TF32
add_executable(26_02_bf16_tf32 02_bf16_tf32.cu)
target_link_libraries(26_02_bf16_tf32 CUDA::cudart CUDA::cublas)

# 03 - INT8量化
add_executable(26_03_int8_quantize 03_int8_quantize.cu)
target_link_libraries(26_03_int8_quantize CUDA::cudart)

# 04 - 量化GEMM
add_executable(26_04_quantized_gemm 04_quantized_gemm.cu)
target_link_libraries(26_04_quantized_gemm CUDA::cudart CUDA::cublas)