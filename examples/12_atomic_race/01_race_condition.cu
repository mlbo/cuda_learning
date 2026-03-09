/**
 * 01_race_condition.cu
 * 演示竞争条件：并行累加的错误示例
 *
 * 编译: nvcc -o 01_race_condition 01_race_condition.cu
 * 运行: ./01_race_condition
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 错误的并行累加核函数（存在竞争条件）
__global__ void sum_naive(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 多个线程同时读写 result，存在竞争条件！
        *result += data[idx];
    }
}

// CPU版本用于对比
float sum_cpu(float* data, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    int N = 1024;  // 数据量
    size_t bytes = N * sizeof(float);

    // 分配主机内存
    float* h_data = (float*)malloc(bytes);
    float* h_result = (float*)malloc(sizeof(float));

    // 初始化数据：全部为1.0，正确结果应该是 N
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    // 分配设备内存
    float *d_data, *d_result;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_result, sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // 初始化结果为0
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    printf("Launching kernel with %d blocks x %d threads\n", numBlocks, blockSize);

    sum_naive<<<numBlocks, blockSize>>>(d_data, d_result, N);
    cudaDeviceSynchronize();

    // 获取结果
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // 计算CPU结果对比
    float expected = sum_cpu(h_data, N);

    printf("\n========== 竞争条件演示 ==========\n");
    printf("数据量: %d\n", N);
    printf("期望结果: %.0f (每个元素为1.0)\n", expected);
    printf("实际结果: %.0f\n", *h_result);
    printf("结果正确: %s\n", (*h_result == expected) ? "是" : "否");
    printf("==================================\n");

    // 多次运行演示结果不稳定
    printf("\n多次运行结果演示（结果会变化）:\n");
    for (int run = 0; run < 5; run++) {
        cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
        sum_naive<<<numBlocks, blockSize>>>(d_data, d_result, N);
        cudaDeviceSynchronize();
        cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        printf("  Run %d: %.0f %s\n", run + 1, *h_result,
               (*h_result == expected) ? "✓" : "✗ (竞争条件!)");
    }

    // 清理
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    free(h_result);

    return 0;
}