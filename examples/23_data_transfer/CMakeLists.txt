# =============================================================================
# 第 23 章: 数据传输优化 - CMakeLists.txt
# =============================================================================

# 01 - 锁页内存
add_executable(23_01_pinned_memory 01_pinned_memory.cu)
target_link_libraries(23_01_pinned_memory CUDA::cudart)

# 02 - 异步传输
add_executable(23_02_async_transfer 02_async_transfer.cu)
target_link_libraries(23_02_async_transfer CUDA::cudart)

# 03 - 零拷贝
add_executable(23_03_zero_copy 03_zero_copy.cu)
target_link_libraries(23_03_zero_copy CUDA::cudart)

# 04 - 统一内存
add_executable(23_04_unified_memory 04_unified_memory.cu)
target_link_libraries(23_04_unified_memory CUDA::cudart)

# 05 - 预取与建议
add_executable(23_05_prefetch_advise 05_prefetch_advise.cu)
target_link_libraries(23_05_prefetch_advise CUDA::cudart)