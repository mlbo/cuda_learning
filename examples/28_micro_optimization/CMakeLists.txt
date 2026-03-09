# =============================================================================
# 第 28 章: 微指令级调优 - CMakeLists.txt
# =============================================================================

# 01 - 循环展开优化
add_executable(28_01_pragma_unroll 01_pragma_unroll.cu)
target_link_libraries(28_01_pragma_unroll CUDA::cudart)

# 02 - __restrict__限定符
add_executable(28_02_restrict_qualifier 02_restrict_qualifier.cu)
target_link_libraries(28_02_restrict_qualifier CUDA::cudart)

# 03 - 编译器选项调优
add_executable(28_03_compiler_options 03_compiler_options.cu)
target_link_libraries(28_03_compiler_options CUDA::cudart)

# 04 - 内存对齐优化
add_executable(28_04_alignment 04_alignment.cu)
target_link_libraries(28_04_alignment CUDA::cudart)

# 05 - __ldg与常量内存
add_executable(28_05_ldg_const 05_ldg_const.cu)
target_link_libraries(28_05_ldg_const CUDA::cudart)