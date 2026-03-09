/**
 * 01_naive_reduce.cu
 * 朴素规约实现 - 展示原子操作的性能问题
 *
 * 这个示例展示了最简单的并行规约方法及其性能问题：
 * - 每个线程直接使用原子操作累加到全局结果
 * - N 个线程 = N 次原子操作 = 严重的串行化
 *
 * 编译: nvcc -o 01_naive_reduce 01_naive_reduce.cu
 * 运行: ./01_naive_reduce
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 朴素原子规约 - 正确但性能很差
// 每个线程都直接对全局结果做原子加法
__global__ void naive_atomic_reduce(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程对全局结果做一次原子操作
    // N 个线程 -> N 次串行化的原子操作
    if (idx < N) {
        atomicAdd(result, data[idx]);
    }
}

// 错误的并行规约 - 有竞争条件
// 这个版本会得到错误的结果
__global__ void wrong_reduce(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // 竞争条件：多个线程同时读写 result
        // 结果不可预测
        *result += data[idx];
    }
}

// CPU 参考实现
float cpu_reduce(float* data, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }
    return sum;
}

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

int main() {
    // 数据规模
    int N = 1024 * 1024;  // 1M 元素
    size_t bytes = N * sizeof(float);

    printf("========== 朴素规约演示 ==========\n");
    printf("数据量: %d 个 float (%.2f MB)\n\n", N, (float)bytes / 1024 / 1024);

    // 分配主机内存
    float* h_data = (float*)malloc(bytes);

    // 初始化数据：全为 1，期望结果 = N
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    // 分配设备内存
    float *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // 创建计时事件
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // -------------------- CPU 参考实现 --------------------
    printf(">>> CPU 参考实现\n");
    float cpu_result = cpu_reduce(h_data, N);
    printf("CPU 结果: %.0f\n", cpu_result);
    printf("期望结果: %d\n\n", N);

    // -------------------- 错误版本演示 --------------------
    printf(">>> 错误版本 (竞争条件)\n");
    float zero = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));

    // 启动错误版本
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    wrong_reduce<<<gridSize, blockSize>>>(d_data, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float wrong_result;
    CUDA_CHECK(cudaMemcpy(&wrong_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("错误版本结果: %.0f (期望 %d)\n", wrong_result, N);
    printf("结果正确: %s\n\n", (wrong_result == (float)N) ? "是" : "否");

    // -------------------- 朴素原子版本 --------------------
    printf(">>> 朴素原子规约\n");
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));

    // 计时开始
    CUDA_CHECK(cudaEventRecord(start));
    naive_atomic_reduce<<<gridSize, blockSize>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float naive_ms;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));

    float naive_result;
    CUDA_CHECK(cudaMemcpy(&naive_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("原子规约结果: %.0f\n", naive_result);
    printf("结果正确: %s\n", (naive_result == (float)N) ? "是" : "否");
    printf("执行时间: %.3f ms\n", naive_ms);

    // -------------------- 性能分析 --------------------
    printf("\n========== 性能分析 ==========\n");
    printf("朴素原子规约的问题:\n");
    printf("  - 每个线程执行 1 次原子操作\n");
    printf("  - 总共 %d 次原子操作\n", N);
    printf("  - 原子操作导致完全串行化\n");
    printf("  - 并行变成串行，性能极差\n\n");

    printf("优化思路:\n");
    printf("  1. 先在更小范围内合并（Warp/Block）\n");
    printf("  2. 减少原子操作次数\n");
    printf("  3. 使用树状规约分治\n");

    // 清理资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    free(h_data);

    return 0;
}