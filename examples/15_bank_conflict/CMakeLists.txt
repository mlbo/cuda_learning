# =============================================================================
# 第 15 章: Bank Conflict 优化 - CMakeLists.txt
# =============================================================================

# 01 - Bank 结构演示
add_executable(15_01_bank_structure 01_bank_structure.cu)
target_link_libraries(15_01_bank_structure CUDA::cudart)

# 02 - Bank Conflict 演示
add_executable(15_02_bank_conflict_demo 02_bank_conflict_demo.cu)
target_link_libraries(15_02_bank_conflict_demo CUDA::cudart)

# 03 - Padding 解决方案
add_executable(15_03_padding_solution 03_padding_solution.cu)
target_link_libraries(15_03_padding_solution CUDA::cudart)

# 04 - XOR Swizzling 解决方案
add_executable(15_04_xor_swizzling 04_xor_swizzling.cu)
target_link_libraries(15_04_xor_swizzling CUDA::cudart)

# 05 - 矩阵转置优化案例
add_executable(15_05_matrix_transpose 05_matrix_transpose.cu)
target_link_libraries(15_05_matrix_transpose CUDA::cudart)
