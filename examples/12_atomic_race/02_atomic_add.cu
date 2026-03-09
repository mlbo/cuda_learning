/**
 * 02_atomic_add.cu
 * 使用原子操作修复竞争条件
 *
 * 编译: nvcc -o 02_atomic_add 02_atomic_add.cu
 * 运行: ./02_atomic_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 使用原子操作的并行累加核函数
__global__ void sum_atomic(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // atomicAdd: 原子地执行 *result += data[idx]
        // 返回旧值（这里没有使用返回值）
        atomicAdd(result, data[idx]);
    }
}

// 错误版本用于对比
__global__ void sum_naive(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        *result += data[idx];
    }
}

int main() {
    int N = 1024 * 1024;  // 1M 元素
    size_t bytes = N * sizeof(float);

    // 分配主机内存
    float* h_data = (float*)malloc(bytes);
    float* h_result = (float*)malloc(sizeof(float));

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    // 分配设备内存
    float *d_data, *d_result;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_result, sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    printf("========== 原子操作演示 ==========\n");
    printf("数据量: %d\n", N);
    printf("期望结果: %d\n\n", N);

    // 测试普通版本（有竞争条件）
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    sum_naive<<<numBlocks, blockSize>>>(d_data, d_result, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("普通版本 (有竞争条件):\n");
    printf("  结果: %.0f\n", *h_result);
    printf("  正确: %s\n\n", (*h_result == (float)N) ? "是" : "否 (竞争条件!)");

    // 测试原子操作版本
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    sum_atomic<<<numBlocks, blockSize>>>(d_data, d_result, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("原子操作版本:\n");
    printf("  结果: %.0f\n", *h_result);
    printf("  正确: %s\n", (*h_result == (float)N) ? "是" : "否");

    printf("==================================\n");

    // 清理
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    free(h_result);

    return 0;
}