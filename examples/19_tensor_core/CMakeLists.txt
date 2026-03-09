# Chapter 19: Tensor Core
add_executable(19_01_tensor_core_basics 01_tensor_core_basics.cu)
target_link_libraries(19_01_tensor_core_basics CUDA::cudart)

add_executable(19_02_wmma_gemm 02_wmma_gemm.cu)
target_link_libraries(19_02_wmma_gemm CUDA::cudart)

add_executable(19_03_mixed_precision 03_mixed_precision.cu)
target_link_libraries(19_03_mixed_precision CUDA::cudart)