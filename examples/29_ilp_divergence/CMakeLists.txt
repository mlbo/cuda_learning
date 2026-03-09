# =============================================================================
# CMakeLists.txt for Chapter 29: ILP and Warp Divergence
# =============================================================================

# ILP基础示例
add_executable(29_01_ilp_basics 01_ilp_basics.cu)
target_link_libraries(29_01_ilp_basics CUDA::cudart)

# Dual Issue示例
add_executable(29_02_dual_issue 02_dual_issue.cu)
target_link_libraries(29_02_dual_issue CUDA::cudart)

# Warp Divergence示例
add_executable(29_03_warp_divergence 03_warp_divergence.cu)
target_link_libraries(29_03_warp_divergence CUDA::cudart)

# 分支优化示例
add_executable(29_04_branch_optimization 04_branch_optimization.cu)
target_link_libraries(29_04_branch_optimization CUDA::cudart)