# =============================================================================
# 第 16 章: Cooperative Groups - CMakeLists.txt
# =============================================================================

# 01 - thread_block 基础
add_executable(16_01_thread_block 01_thread_block.cu)
target_link_libraries(16_01_thread_block CUDA::cudart)

# 02 - warp_group 操作
add_executable(16_02_warp_group 02_warp_group.cu)
target_link_libraries(16_02_warp_group CUDA::cudart)

# 03 - grid.sync() 跨块同步
add_executable(16_03_grid_sync 03_grid_sync.cu)
target_link_libraries(16_03_grid_sync CUDA::cudart)

# 04 - 使用 CG 的规约实现
add_executable(16_04_reduce_with_cg 04_reduce_with_cg.cu)
target_link_libraries(16_04_reduce_with_cg CUDA::cudart)

# 05 - cudaLaunchCooperativeKernel 使用
add_executable(16_05_launch_cooperative 05_launch_cooperative.cu)
target_link_libraries(16_05_launch_cooperative CUDA::cudart)

# 06 - Thread Block Cluster (需要 CC 9.0+)
add_executable(16_06_thread_block_cluster 06_thread_block_cluster.cu)
target_link_libraries(16_06_thread_block_cluster CUDA::cudart)
# Cluster 示例需要为 sm_90 编译才能完全支持所有特性
# 如果只有 sm_80 设备，可以注释掉上面两行