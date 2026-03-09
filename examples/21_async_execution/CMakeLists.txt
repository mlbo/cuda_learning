# =============================================================================
# 第 21 章: 异步执行与延迟隐藏 - CMakeLists.txt
# =============================================================================

# 01 - 双缓冲技术
add_executable(21_01_double_buffer 01_double_buffer.cu)
target_link_libraries(21_01_double_buffer CUDA::cudart)

# 02 - 软件流水线
add_executable(21_02_software_pipeline 02_software_pipeline.cu)
target_link_libraries(21_02_software_pipeline CUDA::cudart)

# 03 - 异步拷贝API
add_executable(21_03_memcpy_async 03_memcpy_async.cu)
target_link_libraries(21_03_memcpy_async CUDA::cudart)

# 04 - Pipeline Primitives
add_executable(21_04_pipeline_primitives 04_pipeline_primitives.cu)
target_link_libraries(21_04_pipeline_primitives CUDA::cudart)