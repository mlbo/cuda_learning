/**
 * 06_thread_block_cluster.cu - Thread Block Cluster 示例
 *
 * 本示例演示：
 * 1. Thread Block Cluster 的基本概念
 * 2. cluster_group 的获取和使用
 * 3. 跨 block 的分布式共享内存访问
 * 4. Cluster 级同步
 *
 * 计算能力要求：CC 9.0+ (Hopper 架构: H100, H200 等)
 *
 * 注意：如果没有 H100+ GPU，此代码将无法运行
 *       但可以作为学习参考
 */

#include <stdio.h>
#include <utility>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// ============================================================================
// 检查计算能力
// ============================================================================
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define CLUSTER_SUPPORTED 1
#else
#define CLUSTER_SUPPORTED 0
#endif

// ============================================================================
// 核函数：Cluster 基本操作
// 使用 __cluster_dims__ 属性指定 cluster 大小
// ============================================================================

// 2x1x1 = 2 blocks per cluster
__cluster_dims__(2, 1, 1)
__global__ void cluster_basics(int* output) {
#if CLUSTER_SUPPORTED
    // 获取 cluster 组
    cg::cluster_group cluster = cg::this_cluster();

    // 获取当前线程块在 cluster 内的 rank
    unsigned int block_rank = cluster.block_rank();

    // 获取 cluster 内的 block 数量
    unsigned int num_blocks = cluster.num_blocks();

    // 获取 cluster 内的总线程数
    unsigned long long total_threads = cluster.num_threads();

    // 获取当前线程在 cluster 内的全局 rank
    unsigned long long thread_rank = cluster.thread_rank();

    // Cluster 级同步
    cluster.sync();

    // 存储信息
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid * 4] = block_rank;
    output[tid * 4 + 1] = num_blocks;
    output[tid * 4 + 2] = thread_rank;
    output[tid * 4 + 3] = total_threads;

#else
    printf("Cluster 需要 Hopper 架构 (CC 9.0+)\n");
#endif
}

// ============================================================================
// 核函数：分布式共享内存访问
// Cluster 内的 block 可以直接访问彼此的共享内存
// ============================================================================

// 2x2x1 = 4 blocks per cluster
__cluster_dims__(2, 2, 1)
__global__ void distributed_shared_memory(int* input, int* output, int N) {
#if CLUSTER_SUPPORTED
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();

    // 声明共享内存
    extern __shared__ int shared_data[];

    int block_rank = cluster.block_rank();
    int num_blocks = cluster.num_blocks();

    // 每个 block 加载一部分数据到自己的共享内存
    int tid = block.thread_rank();
    int block_size = block.num_threads();

    // 每个线程加载数据
    int load_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (load_idx < N) {
        shared_data[tid] = input[load_idx];
    } else {
        shared_data[tid] = 0;
    }

    // Block 内同步
    block.sync();

    // 访问其他 block 的共享内存
    // map_shared_rank 返回其他 block 的共享内存指针
    int neighbor_rank = (block_rank + 1) % num_blocks;
    int* neighbor_smem = cluster.map_shared_rank(shared_data, neighbor_rank);

    // 读取邻居 block 的数据并计算
    int neighbor_tid = tid;  // 对应邻居 block 的相同位置
    int result = shared_data[tid] + neighbor_smem[neighbor_tid];

    // Cluster 同步确保所有 block 完成读取
    cluster.sync();

    // 写入结果
    if (load_idx < N) {
        output[load_idx] = result;
    }

#else
    printf("Distributed Shared Memory 需要 Hopper 架构 (CC 9.0+)\n");
#endif
}

// ============================================================================
// 核函数：使用 barrier_arrive/wait 的 Cluster 同步
// ============================================================================

__cluster_dims__(2, 1, 1)
__global__ void cluster_barrier_demo(float* data, int N) {
#if CLUSTER_SUPPORTED
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();

    extern __shared__ float smem[];

    // 加载数据
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        smem[threadIdx.x] = data[tid];
    }

    // 到达屏障
    auto token = cluster.barrier_arrive();

    // 可以在这里做一些独立工作
    float local_calc = smem[threadIdx.x] * 2.0f;

    // 等待所有 block 到达
    cluster.barrier_wait(std::move(token));

    // 现在可以安全访问其他 block 的数据
    if (block.thread_rank() == 0) {
        float* other_smem = cluster.map_shared_rank(smem,
                                                    (cluster.block_rank() + 1) % cluster.num_blocks());
        // 使用其他 block 的共享内存...
        (void)other_smem;
    }

    cluster.sync();

    if (tid < N) {
        data[tid] = local_calc;
    }

#else
    printf("Cluster barrier 需要 Hopper 架构 (CC 9.0+)\n");
#endif
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== Thread Block Cluster 示例 ===\n\n");

    // 检查设备属性
    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);

    if (prop.major < 9) {
        printf("\n警告：Thread Block Cluster 需要 Hopper 架构 (CC 9.0+)\n");
        printf("当前设备不支持，示例将跳过实际执行\n\n");
        printf("支持的 GPU:\n");
        printf("  - NVIDIA H100\n");
        printf("  - NVIDIA H200\n");
        printf("  - 未来 Hopper 架构产品\n");

        printf("\n--- Cluster API 概念说明 ---\n\n");

        printf("Thread Block Cluster 是 Hopper 架构引入的新特性:\n\n");

        printf("1. 概念:\n");
        printf("   - 多个线程块组成一个 Cluster\n");
        printf("   - Cluster 内的 block 可以同步和协作\n");
        printf("   - 可以访问彼此的共享内存（Distributed Shared Memory）\n\n");

        printf("2. Cluster 大小:\n");
        printf("   - 使用 __cluster_dims__(X, Y, Z) 属性指定\n");
        printf("   - 或在运行时通过 cudaLaunchKernelEx 指定\n\n");

        printf("3. 分布式共享内存:\n");
        printf("   - cluster.map_shared_rank(smem, rank) 获取其他 block 的共享内存指针\n");
        printf("   - 无需通过全局内存即可共享数据\n\n");

        printf("4. Cluster 同步:\n");
        printf("   - cluster.sync(): 同步所有 block\n");
        printf("   - cluster.barrier_arrive/wait(): 分步同步\n\n");

        printf("示例代码已编写，可参考 06_thread_block_cluster.cu 源码\n");

        return 0;
    }

    // Hopper 架构：运行实际示例
    printf("检测到 Hopper 架构，运行实际示例\n\n");

    // 示例 1：基本操作
    printf("--- Cluster 基本操作 ---\n");
    {
        const int blocks = 4;
        const int threads = 128;
        const int total_threads = blocks * threads;

        int *d_output;
        cudaMalloc(&d_output, total_threads * 4 * sizeof(int));

        cluster_basics<<<blocks, threads, 0, 0>>>(d_output);
        cudaDeviceSynchronize();

        int* h_output = new int[total_threads * 4];
        cudaMemcpy(h_output, d_output, total_threads * 4 * sizeof(int), cudaMemcpyDeviceToHost);

        printf("Block 0, Thread 0: block_rank=%d, num_blocks=%d\n",
               h_output[0], h_output[1]);

        cudaFree(d_output);
        delete[] h_output;
    }

    // 示例 2：分布式共享内存
    printf("\n--- 分布式共享内存 ---\n");
    {
        const int N = 1024;
        const int blocks = 4;
        const int threads = 256;

        int *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(int));
        cudaMalloc(&d_output, N * sizeof(int));

        int* h_input = new int[N];
        for (int i = 0; i < N; i++) h_input[i] = i;
        cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

        // 需要 cluster 启动 API (CUDA 13+)
        cudaLaunchConfig_t config = {
            .gridDim = dim3(blocks),
            .blockDim = dim3(threads),
            .dynamicSmemBytes = threads * sizeof(int),
        };

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {2, 2, 1};

        config.attrs = attrs;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, distributed_shared_memory,
                           d_input, d_output, N);
        cudaDeviceSynchronize();

        int* h_output = new int[N];
        cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

        printf("输入[0] = %d, 输出[0] = %d (期望 %d + 邻居值)\n",
               h_input[0], h_output[0], h_input[0]);

        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
        delete[] h_output;
    }

    printf("\n=== 示例完成 ===\n");
    return 0;
}
