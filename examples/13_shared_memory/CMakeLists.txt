# =============================================================================
# 第 13 章: 共享内存深入 - CMakeLists.txt
# =============================================================================

# 01 - 共享内存基础
add_executable(13_01_smem_basics 01_smem_basics.cu)
target_link_libraries(13_01_smem_basics CUDA::cudart)

# 02 - 静态vs动态共享内存
add_executable(13_02_static_vs_dynamic 02_static_vs_dynamic.cu)
target_link_libraries(13_02_static_vs_dynamic CUDA::cudart)

# 03 - __syncthreads() 同步详解
add_executable(13_03_syncthreads 03_syncthreads.cu)
target_link_libraries(13_03_syncthreads CUDA::cudart)

# 04 - 矩阵乘法优化
add_executable(13_04_matrix_mul_smem 04_matrix_mul_smem.cu)
target_link_libraries(13_04_matrix_mul_smem CUDA::cudart)

# 05 - 共享内存配置
add_executable(13_05_smem_config 05_smem_config.cu)
target_link_libraries(13_05_smem_config CUDA::cudart)