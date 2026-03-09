# =============================================================================
# 第 24 章: CUDA Graph - CMakeLists.txt
# =============================================================================

# 01 - CUDA Graph基础
add_executable(24_01_graph_basics 01_graph_basics.cu)
target_link_libraries(24_01_graph_basics CUDA::cudart)

# 02 - 流捕获
add_executable(24_02_stream_capture 02_stream_capture.cu)
target_link_libraries(24_02_stream_capture CUDA::cudart)

# 03 - 图实例化
add_executable(24_03_graph_instantiate 03_graph_instantiate.cu)
target_link_libraries(24_03_graph_instantiate CUDA::cudart)

# 04 - 图优化
add_executable(24_04_graph_optimize 04_graph_optimize.cu)
target_link_libraries(24_04_graph_optimize CUDA::cudart)