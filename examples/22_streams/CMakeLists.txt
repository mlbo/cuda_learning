# =============================================================================
# 第 22 章: CUDA流与并发 - CMakeLists.txt
# =============================================================================

# 01 - CUDA流基础
add_executable(22_01_stream_basics 01_stream_basics.cu)
target_link_libraries(22_01_stream_basics CUDA::cudart)

# 02 - 多流并发
add_executable(22_02_multi_stream 02_multi_stream.cu)
target_link_libraries(22_02_multi_stream CUDA::cudart)

# 03 - 流同步机制
add_executable(22_03_stream_sync 03_stream_sync.cu)
target_link_libraries(22_03_stream_sync CUDA::cudart)

# 04 - CUDA事件
add_executable(22_04_cuda_events 04_cuda_events.cu)
target_link_libraries(22_04_cuda_events CUDA::cudart)

# 05 - 并发核函数
add_executable(22_05_concurrent_kernels 05_concurrent_kernels.cu)
target_link_libraries(22_05_concurrent_kernels CUDA::cudart)