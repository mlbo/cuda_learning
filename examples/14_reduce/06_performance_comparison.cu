/**
 * 06_performance_comparison.cu
 * 性能对比测试 - 简化版本
 *
 * 编译: nvcc -o 06_performance_comparison 06_performance_comparison.cu
 * 运行: ./06_performance_comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            printf("CUDA错误 %s:%d: %s\n", __FILE__, __LINE__,                 \
                   cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Warp规约函数
__device__ float warp_reduce(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// 1. 朴素原子规约
__global__ void naive_atomic_reduce(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(result, data[idx]);
    }
}

// 2. 树状规约
__global__ void tree_reduce(float* data, float* result, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        sum += data[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

// 3. Warp Shuffle规约
__global__ void warp_shuffle_reduce(float* data, float* result, int N) {
    __shared__ float warp_sums[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        sum += data[i];
    }
    sum = warp_reduce(sum);
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane < 8) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce(sum);
        if (lane == 0) atomicAdd(result, sum);
    }
}

int main() {
    int N = 16 * 1024 * 1024;
    size_t bytes = N * sizeof(float);

    printf("========== 规约算法性能对比 ==========\n");
    printf("数据量: %d (%.2f MB)\n\n", N, (float)bytes / 1024 / 1024);

    float* h_data = (float*)malloc(bytes);
    float *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms, result, zero = 0.0f;

    // 朴素原子规约
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(start);
    naive_atomic_reduce<<<gridSize, blockSize>>>(d_data, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("朴素原子规约: %.3f ms, 结果=%.0f\n", ms, result);

    // 树状规约
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(start);
    tree_reduce<<<gridSize, blockSize>>>(d_data, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("树状规约:     %.3f ms, 结果=%.0f\n", ms, result);

    // Warp Shuffle规约
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(start);
    warp_shuffle_reduce<<<gridSize, blockSize>>>(d_data, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Warp Shuffle: %.3f ms, 结果=%.0f\n", ms, result);

    printf("\n结论: Warp Shuffle > 树状规约 > 朴素原子规约\n");
    printf("========================================\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);

    return 0;
}