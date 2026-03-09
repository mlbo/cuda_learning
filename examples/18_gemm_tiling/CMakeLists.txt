# =============================================================================
# 第 18 章: GEMM分块优化 - CMakeLists.txt
# =============================================================================

# 01 - 共享内存缓存优化
add_executable(18_01_smem_caching 01_smem_caching.cu)
target_link_libraries(18_01_smem_caching CUDA::cudart)

# 02 - 1D Block Tiling优化
add_executable(18_02_1d_blocktiling 02_1d_blocktiling.cu)
target_link_libraries(18_02_1d_blocktiling CUDA::cudart)

# 03 - 2D Block Tiling优化
add_executable(18_03_2d_blocktiling 03_2d_blocktiling.cu)
target_link_libraries(18_03_2d_blocktiling CUDA::cudart)

# 04 - Warp Tiling优化
add_executable(18_04_warptiling 04_warptiling.cu)
target_link_libraries(18_04_warptiling CUDA::cudart)