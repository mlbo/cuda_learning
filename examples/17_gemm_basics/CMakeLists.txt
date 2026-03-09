# =============================================================================
# 第 17 章: GEMM优化入门 - CMakeLists.txt
# =============================================================================

# 01 - 朴素GEMM实现
add_executable(17_01_naive_gemm 01_naive_gemm.cu)
target_link_libraries(17_01_naive_gemm CUDA::cudart)

# 02 - 内存合并优化的GEMM
add_executable(17_02_coalesced_gemm 02_coalesced_gemm.cu)
target_link_libraries(17_02_coalesced_gemm CUDA::cudart)

# 03 - 分块GEMM实现
add_executable(17_03_tiled_gemm 03_tiled_gemm.cu)
target_link_libraries(17_03_tiled_gemm CUDA::cudart)