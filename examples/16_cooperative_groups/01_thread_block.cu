/**
 * 01_thread_block.cu - thread_block 基础用法
 *
 * 本示例演示：
 * 1. 获取 thread_block 组对象
 * 2. 使用 block.sync() 替代 __syncthreads()
 * 3. 查询组属性（rank, size, index 等）
 *
 * 编译要求：CUDA 9.0+, CC 5.0+
 */

#include <stdio.h>
#include <cmath>
#include <utility>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// ============================================================================
// 核函数：演示 thread_block 的基本用法
// ============================================================================
__global__ void thread_block_basics(float* data, float* output) {
    // 1. 获取当前线程块组
    //    这是集合操作，所有线程必须参与
    //    建议在函数开头创建，避免在条件分支中创建
    cg::thread_block block = cg::this_thread_block();

    // 2. 获取组属性
    // ------------------------------------------------------------------------

    // 线程在块内的 rank（0 到 num_threads-1）
    unsigned int rank = block.thread_rank();

    // 块内的总线程数
    unsigned int size = block.num_threads();

    // 块在网格中的 3D 索引
    dim3 block_idx = block.group_index();

    // 线程在块内的 3D 索引
    dim3 thread_idx = block.thread_index();

    // 块的维度
    dim3 block_dim = block.dim_threads();

    // 3. 使用共享内存演示同步
    // ------------------------------------------------------------------------
    __shared__ float shared_data[256];

    // 每个线程加载数据到共享内存
    shared_data[rank] = data[rank];

    // 同步：等待所有线程完成加载
    // 这等价于 __syncthreads()
    block.sync();

    // 现在可以安全读取其他线程写入的数据
    // 演示：每个线程读取相邻线程的数据（循环）
    int neighbor = (rank + 1) % size;
    output[rank] = shared_data[neighbor];
}

// ============================================================================
// 核函数：演示使用组参数的函数设计
// ============================================================================

// 使用组参数的设备函数
// 明确表示需要块级协作
__device__ float block_wide_sum(const cg::thread_block& block, float* shared_data) {
    unsigned int rank = block.thread_rank();
    unsigned int size = block.num_threads();

    // 树状规约
    for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
        if (rank < stride) {
            shared_data[rank] += shared_data[rank + stride];
        }
        // 使用组同步
        block.sync();
    }

    return shared_data[0];
}

__global__ void sum_with_cg(float* input, float* output, int N) {
    // 在函数开头创建组
    cg::thread_block block = cg::this_thread_block();

    __shared__ float s_data[256];

    // 加载数据
    if (block.thread_rank() < N) {
        s_data[block.thread_rank()] = input[block.thread_rank()];
    } else {
        s_data[block.thread_rank()] = 0.0f;
    }
    block.sync();

    // 调用需要组参数的函数
    float sum = block_wide_sum(block, s_data);

    // 只有 rank 0 写入结果
    if (block.thread_rank() == 0) {
        output[0] = sum;
    }
}

// ============================================================================
// 核函数：演示 barrier_arrive 和 barrier_wait（CUDA 12.2+）
// ============================================================================
__global__ void barrier_arrive_wait_demo(float* data, int N) {
    cg::thread_block block = cg::this_thread_block();
    __shared__ float shared_buffer[256];

    // 加载数据
    if (block.thread_rank() < N) {
        shared_buffer[block.thread_rank()] = data[block.thread_rank()];
    }

    // 到达屏障并获取 token
    // 这允许线程在等待时做独立工作
    auto token = block.barrier_arrive();

    // 可以在这里做一些不依赖其他线程的工作
    // 例如：预计算某些局部值
    float local_value = shared_buffer[block.thread_rank()] * 2.0f;

    // 等待所有线程到达屏障
    block.barrier_wait(std::move(token));

    // 现在可以安全访问其他线程写入的数据
    data[block.thread_rank()] = local_value;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== Cooperative Groups: thread_block 基础示例 ===\n\n");

    // 检查 CUDA 版本
    int cuda_version = 0;
    cudaRuntimeGetVersion(&cuda_version);
    printf("CUDA Runtime 版本: %d\n", cuda_version);

    // 设置
    const int N = 256;
    const int threads_per_block = 256;
    const int blocks = 1;

    // 分配内存
    float *d_data, *d_output;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // 初始化数据
    float* h_data = new float[N];
    float* h_output = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // 运行基础示例
    printf("\n--- 运行 thread_block_basics ---\n");
    thread_block_basics<<<blocks, threads_per_block>>>(d_data, d_output);
    cudaDeviceSynchronize();

    // 检查结果
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("原始数据[0..4]: ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_data[i]);
    printf("\n");
    printf("输出数据[0..4] (应为相邻值): ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_output[i]);
    printf("\n");

    // 运行规约示例
    printf("\n--- 运行 sum_with_cg ---\n");
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));

    sum_with_cg<<<blocks, threads_per_block>>>(d_data, d_sum, N);
    cudaDeviceSynchronize();

    float h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    float expected_sum = 0.0f;
    for (int i = 0; i < N; i++) expected_sum += h_data[i];
    printf("计算的和: %.0f\n", h_sum);
    printf("期望的和: %.0f\n", expected_sum);
    printf("结果: %s\n", std::fabs(h_sum - expected_sum) < 1e-5 ? "正确" : "错误");

    // 清理
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_sum);
    delete[] h_data;
    delete[] h_output;

    printf("\n=== 示例完成 ===\n");
    return 0;
}
