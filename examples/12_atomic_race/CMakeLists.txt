# =============================================================================
# 第 12 章: 原子操作与竞争条件 - CMakeLists.txt
# =============================================================================

# 01 - 竞争条件演示
add_executable(12_01_race_condition 01_race_condition.cu)
target_link_libraries(12_01_race_condition CUDA::cudart)

# 02 - 原子操作修复
add_executable(12_02_atomic_add 02_atomic_add.cu)
target_link_libraries(12_02_atomic_add CUDA::cudart)

# 03 - 性能对比
add_executable(12_03_atomic_performance 03_atomic_performance.cu)
target_link_libraries(12_03_atomic_performance CUDA::cudart)

# 04 - Warp规约
add_executable(12_04_warp_reduce 04_warp_reduce.cu)
target_link_libraries(12_04_warp_reduce CUDA::cudart)

# 05 - Block规约
add_executable(12_05_block_reduce 05_block_reduce.cu)
target_link_libraries(12_05_block_reduce CUDA::cudart)

# 06 - atomicCAS
add_executable(12_06_atomic_cas 06_atomic_cas.cu)
target_link_libraries(12_06_atomic_cas CUDA::cudart)

# 07 - 并行直方图
add_executable(12_07_histogram 07_histogram.cu)
target_link_libraries(12_07_histogram CUDA::cudart)