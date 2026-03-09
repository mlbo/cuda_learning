/**
 * 03_atomic_performance.cu
 * 原子操作性能对比分析
 *
 * 编译: nvcc -o 03_atomic_performance 03_atomic_performance.cu
 * 运行: ./03_atomic_performance
 * 分析: ncu ./03_atomic_performance
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 直接原子操作版本
__global__ void sum_atomic_direct(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(result, data[idx]);
    }
}

// Warp级别规约版本
__global__ void sum_atomic_warp(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = idx & 31;  // threadIdx.x % 32
    int warpIdx = idx >> 5;  // idx / 32

    // 每个线程读取数据
    float val = (idx < N) ? data[idx] : 0.0f;

    // Warp内规约：使用shuffle指令
    // __shfl_down_sync: 将warp内后面的线程的值传递给前面的线程
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    // 只有每个warp的第一个线程写入结果
    if (lane == 0) {
        atomicAdd(result, val);
    }
}

// Block级别规约版本（使用共享内存）
__global__ void sum_atomic_block(float* data, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    sdata[tid] = (idx < N) ? data[idx] : 0.0f;
    __syncthreads();

    // 树状规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 只有线程0写入全局结果
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// 计时辅助函数
template <void (*Kernel)(float*, float*, int)>
float time_kernel(float* d_data, float* d_result, int N,
                  int warmup = 5, int runs = 10) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    for (int i = 0; i < warmup; i++) {
        Kernel<<<256, 256>>>(d_data, d_result, N);
    }
    cudaDeviceSynchronize();

    // 重置结果
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    // 正式计时
    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) {
        cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
        Kernel<<<256, 256>>>(d_data, d_result, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= runs;  // 平均时间

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    int N = 1024 * 1024;  // 1M 元素
    size_t bytes = N * sizeof(float);

    // 分配内存
    float* h_data = (float*)malloc(bytes);
    float* d_data, *d_result;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_result, sizeof(float));

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    printf("========== 原子操作性能对比 ==========\n");
    printf("数据量: %d (%.2f MB)\n", N, (float)bytes / 1024 / 1024);
    printf("期望结果: %d\n\n", N);

    float h_result;
    float time_ms;

    // 测试直接原子操作版本
    time_ms = time_kernel<sum_atomic_direct>(d_data, d_result, N);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("1. 直接原子操作:\n");
    printf("   时间: %.4f ms\n", time_ms);
    printf("   结果: %.0f (%s)\n", h_result,
           h_result == (float)N ? "正确" : "错误");
    printf("   原子操作次数: %d\n\n", N);

    // 测试Warp规约版本
    time_ms = time_kernel<sum_atomic_warp>(d_data, d_result, N);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("2. Warp规约 + 原子操作:\n");
    printf("   时间: %.4f ms\n", time_ms);
    printf("   结果: %.0f (%s)\n", h_result,
           h_result == (float)N ? "正确" : "错误");
    printf("   原子操作次数: ~%d (减少32倍)\n\n", N / 32);

    // 测试Block规约版本
    time_ms = time_kernel<sum_atomic_block>(d_data, d_result, N);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("3. Block规约 + 原子操作:\n");
    printf("   时间: %.4f ms\n", time_ms);
    printf("   结果: %.0f (%s)\n", h_result,
           h_result == (float)N ? "正确" : "错误");
    printf("   原子操作次数: ~%d (减少256倍)\n", N / 256);

    printf("\n========================================\n");
    printf("结论: 通过规约减少原子操作次数可以显著提升性能!\n");

    // 清理
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);

    return 0;
}
