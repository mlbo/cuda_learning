# =============================================================================
# 第 14 章: 规约算法优化 - CMakeLists.txt
# =============================================================================

# 01 - 朴素规约实现
add_executable(14_01_naive_reduce 01_naive_reduce.cu)
target_link_libraries(14_01_naive_reduce CUDA::cudart)

# 02 - 树状规约（共享内存）
add_executable(14_02_tree_reduce 02_tree_reduce.cu)
target_link_libraries(14_02_tree_reduce CUDA::cudart)

# 03 - Warp Shuffle 规约
add_executable(14_03_warp_shuffle_reduce 03_warp_shuffle_reduce.cu)
target_link_libraries(14_03_warp_shuffle_reduce CUDA::cudart)

# 04 - Two-Pass 规约
add_executable(14_04_two_pass_reduce 04_two_pass_reduce.cu)
target_link_libraries(14_04_two_pass_reduce CUDA::cudart)

# 05 - Cooperative Groups 规约
add_executable(14_05_cooperative_groups_reduce 05_cooperative_groups_reduce.cu)
target_link_libraries(14_05_cooperative_groups_reduce CUDA::cudart)

# 06 - 性能对比测试
add_executable(14_06_performance_comparison 06_performance_comparison.cu)
target_link_libraries(14_06_performance_comparison CUDA::cudart)