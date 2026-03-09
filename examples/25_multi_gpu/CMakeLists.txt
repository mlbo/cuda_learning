# =============================================================================
# 第25章: 多GPU编程 - CMakeLists.txt
# =============================================================================

# 01 - 设备枚举与选择（无需NCCL）
add_executable(25_01_device_selection 01_device_selection.cu)
target_link_libraries(25_01_device_selection CUDA::cudart)

# 02 - P2P传输（无需NCCL）
add_executable(25_02_p2p_transfer 02_p2p_transfer.cu)
target_link_libraries(25_02_p2p_transfer CUDA::cudart)

# 03/04 - NCCL示例（需要NCCL库，可选）
find_package(NCCL QUIET)
if(NCCL_FOUND)
    message(STATUS "NCCL found, enabling NCCL examples")

    add_executable(25_03_nccl_basics 03_nccl_basics.cu)
    target_link_libraries(25_03_nccl_basics CUDA::cudart NCCL::nccl)

    add_executable(25_04_allreduce 04_allreduce.cu)
    target_link_libraries(25_04_allreduce CUDA::cudart NCCL::nccl)
else()
    message(STATUS "NCCL not found, skipping NCCL examples (03, 04)")
    message(STATUS "Install NCCL to enable these examples")
endif()