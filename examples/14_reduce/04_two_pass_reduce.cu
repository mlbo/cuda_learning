/**
 * 04_two_pass_reduce.cu
 * 两阶段规约实现
 *
 * Two-Pass 规约使用两个 Kernel 完成规约：
 * 1. Kernel 1: 每个 Block 产生一个部分和
 * 2. Kernel 2: 对部分和进行最终规约
 *
 * 优点：完全避免原子操作
 * 缺点：需要额外的内存和 Kernel 启动开销
 *
 * 编译: nvcc -o 04_two_pass_reduce 04_two_pass_reduce.cu
 * 运行: ./04_two_pass_reduce
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 错误检查宏
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            printf("CUDA错误 %s:%d: %s\n", __FILE__, __LINE__,                 \
                   cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// =============================================================================
// Warp Shuffle 规约函数
// =============================================================================
__device__ float warp_reduce(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// =============================================================================
// Kernel 1: Block 内规约，产生部分和
// =============================================================================
__global__ void reduce_block_partial(float* data, float* partial_sums, int N) {
    // 共享内存存储每个 Warp 的部分和
    __shared__ float warp_sums[32];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int idx = blockIdx.x * blockDim.x + tid;

    // Grid-stride 循环累加
    float sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        sum += data[i];
    }

    // Warp 内规约
    sum = warp_reduce(sum);

    // 将 Warp 部分和写入共享内存
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // 第一个 Warp 对 Warp 部分和进行规约
    if (warp_id == 0) {
        sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce(sum);

        // lane 0 写入部分和到全局内存（无原子操作！）
        if (lane == 0) {
            partial_sums[blockIdx.x] = sum;
        }
    }
}

// =============================================================================
// Kernel 2: 对部分和进行最终规约
// =============================================================================
__global__ void reduce_partial_final(float* partial_sums, float* result, int numPartials) {
    __shared__ float warp_sums[32];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    // Block-stride 循环累加部分和
    float sum = 0.0f;
    for (int i = tid; i < numPartials; i += blockDim.x) {
        sum += partial_sums[i];
    }

    // Warp 内规约
    sum = warp_reduce(sum);

    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // 第一个 Warp 进行最终规约
    if (warp_id == 0) {
        sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce(sum);

        // 直接写入最终结果（无原子操作！）
        if (lane == 0) {
            *result = sum;
        }
    }
}

// =============================================================================
// Two-Pass 规约封装函数
// =============================================================================
void two_pass_reduce(float* d_data, float* d_result, int N) {
    // 计算 Block 和 Grid 配置
    int blockSize = 256;
    int gridSize = 256;  // 固定 Block 数量

    // 分配部分和数组
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    // Kernel 1: Block 内规约
    reduce_block_partial<<<gridSize, blockSize>>>(d_data, d_partial_sums, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel 2: 对部分和进行最终规约
    // GridSize = 1，因为只需要规约部分和
    reduce_partial_final<<<1, blockSize>>>(d_partial_sums, d_result, gridSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 释放临时内存
    CUDA_CHECK(cudaFree(d_partial_sums));
}

// =============================================================================
// 单 Kernel + 原子操作的对比版本
// =============================================================================
__global__ void reduce_single_kernel(float* data, float* result, int N) {
    __shared__ float warp_sums[32];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int idx = blockIdx.x * blockDim.x + tid;

    float sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        sum += data[i];
    }

    sum = warp_reduce(sum);

    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce(sum);

        if (lane == 0) {
            atomicAdd(result, sum);  // 使用原子操作
        }
    }
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    int N = 16 * 1024 * 1024;  // 16M 元素，更适合 Two-Pass
    size_t bytes = N * sizeof(float);

    printf("========== Two-Pass 规约详解 ==========\n");
    printf("数据量: %d 个 float (%.2f MB)\n\n", N, (float)bytes / 1024 / 1024);

    // 分配内存
    float* h_data = (float*)malloc(bytes);
    float *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    // 初始化：全为 1
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // 创建计时事件
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blockSize = 256;
    int gridSize = 256;
    float zero = 0.0f;
    float result;
    float ms;

    // -------------------- Two-Pass 版本测试 --------------------
    printf(">>> Two-Pass 规约\n");

    CUDA_CHECK(cudaEventRecord(start));
    two_pass_reduce(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("结果: %.0f (期望 %d)\n", result, N);
    printf("时间: %.3f ms\n", ms);
    printf("原子操作次数: 0\n\n");

    // -------------------- 单 Kernel + 原子操作版本测试 --------------------
    printf(">>> 单 Kernel + 原子操作\n");
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    reduce_single_kernel<<<gridSize, blockSize>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("结果: %.0f (期望 %d)\n", result, N);
    printf("时间: %.3f ms\n", ms);
    printf("原子操作次数: %d (GridSize)\n\n", gridSize);

    // -------------------- 不同数据规模对比 --------------------
    printf("========== 不同数据规模对比 ==========\n");
    printf("%-12s %-15s %-15s\n", "数据量", "Two-Pass(ms)", "单Kernel(ms)");
    printf("---------------------------------------------\n");

    int sizes[] = {1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < numSizes; i++) {
        int testN = sizes[i];

        // 重新分配和初始化
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaMalloc(&d_data, testN * sizeof(float)));

        // 初始化数据
        float* h_test = (float*)malloc(testN * sizeof(float));
        for (int j = 0; j < testN; j++) {
            h_test[j] = 1.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_data, h_test, testN * sizeof(float), cudaMemcpyHostToDevice));
        free(h_test);

        // Two-Pass
        CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start));
        two_pass_reduce(d_data, d_result, testN);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float twoPassMs;
        CUDA_CHECK(cudaEventElapsedTime(&twoPassMs, start, stop));

        // 单 Kernel
        CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start));
        reduce_single_kernel<<<gridSize, blockSize>>>(d_data, d_result, testN);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float singleMs;
        CUDA_CHECK(cudaEventElapsedTime(&singleMs, start, stop));

        printf("%-12d %-15.3f %-15.3f\n", testN, twoPassMs, singleMs);
    }

    printf("\n");

    // -------------------- 分析总结 --------------------
    printf("========== Two-Pass 分析 ==========\n");
    printf("优点:\n");
    printf("  1. 完全避免原子操作\n");
    printf("  2. 避免原子操作导致的串行化\n");
    printf("  3. 数据量大时性能更好\n\n");

    printf("缺点:\n");
    printf("  1. 需要两次 Kernel 启动开销\n");
    printf("  2. 需要额外的临时内存\n");
    printf("  3. Kernel 之间有隐式同步\n\n");

    printf("适用场景:\n");
    printf("  - 数据量大（> 1M）\n");
    printf("  - 需要避免原子操作竞争\n");
    printf("  - 有足够的显存存储中间结果\n");

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    free(h_data);

    return 0;
}