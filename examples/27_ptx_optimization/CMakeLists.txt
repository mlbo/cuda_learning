# =============================================================================
# 第 27 章: PTX与底层优化 - CMakeLists.txt
# =============================================================================

# 01 - PTX基础语法
add_executable(27_01_ptx_basics 01_ptx_basics.cu)
target_link_libraries(27_01_ptx_basics CUDA::cudart)

# 02 - 内联PTX汇编
add_executable(27_02_inline_ptx 02_inline_ptx.cu)
target_link_libraries(27_02_inline_ptx CUDA::cudart)

# 03 - PTX性能优化
add_executable(27_03_ptx_optimization 03_ptx_optimization.cu)
target_link_libraries(27_03_ptx_optimization CUDA::cudart)

# 04 - 查看PTX代码
add_executable(27_04_inspect_ptx 04_inspect_ptx.cu)
target_link_libraries(27_04_inspect_ptx CUDA::cudart)