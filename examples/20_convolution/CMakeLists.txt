# Chapter 20: Convolution
add_executable(20_01_direct_conv 01_direct_conv.cu)
target_link_libraries(20_01_direct_conv CUDA::cudart)

add_executable(20_02_im2col_conv 02_im2col_conv.cu)
target_link_libraries(20_02_im2col_conv CUDA::cudart)

add_executable(20_03_smem_conv 03_smem_conv.cu)
target_link_libraries(20_03_smem_conv CUDA::cudart)