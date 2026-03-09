/**
 * 05_block_reduce.cu
 * Block级别规约详解
 *
 * 使用共享内存进行Block内规约，进一步减少原子操作次数
 *
 * 编译: nvcc -o 05_block_reduce 05_block_reduce.cu
 * 运行: ./05_block_reduce
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Block规约 - 基础版本
__global__ void block_reduce_basic(float* data, float* result, int N) {
    // 共享内存：每个block共享
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 步骤1: 加载数据到共享内存
    sdata[tid] = (idx < N) ? data[idx] : 0.0f;
    __syncthreads();

    // 步骤2: 树状规约
    // 每轮将活跃线程数减半
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // 每轮规约后同步
    }

    // 步骤3: 只有线程0拥有完整的block规约结果
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// Block规约 - 优化版本（减少同步次数）
__global__ void block_reduce_optimized(float* data, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据（每个线程可以加载多个元素以提高效率）
    sdata[tid] = (idx < N) ? data[idx] : 0.0f;
    __syncthreads();

    // 展开循环以减少同步次数
    // 当s <= 32时，属于同一个warp，可以省略同步

    // s = 128
    if (tid < 128) {
        sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();

    // s = 64
    if (tid < 64) {
        sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();

    // s = 32 及以下：在同一个warp内，无需同步
    if (tid < 32) {
        // 使用volatile确保读取最新值
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// 结合Warp Shuffle的Block规约（最高效）
__device__ float warp_reduce_shfl(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__global__ void block_reduce_shfl(float* data, float* result, int N) {
    __shared__ float sdata[32];  // 只需要存储warp级别的部分和

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    // 加载数据
    float val = (idx < N) ? data[idx] : 0.0f;

    // Warp内规约
    val = warp_reduce_shfl(val);

    // 每个warp的第一个线程将结果写入共享内存
    if (lane == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    // 第一个warp对共享内存中的结果再做规约
    if (warp_id == 0) {
        val = (lane < blockDim.x / 32) ? sdata[lane] : 0.0f;
        val = warp_reduce_shfl(val);
        if (lane == 0) {
            atomicAdd(result, val);
        }
    }
}

// Grid-Stride循环 + Block规约
__global__ void block_reduce_grid_stride(float* data, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;

    // Grid-stride循环：每个线程累加多个元素
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        sum += data[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Block内规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

int main() {
    int N = 1024 * 1024;  // 1M 元素
    size_t bytes = N * sizeof(float);

    float* h_data = (float*)malloc(bytes);
    float* d_data, *d_result;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_result, sizeof(float));

    // 初始化
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    printf("========== Block规约详解 ==========\n");
    printf("数据量: %d\n", N);
    printf("期望结果: %d\n\n", N);

    float zero = 0.0f;
    float h_result;

    // 测试基础版本
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
    block_reduce_basic<<<4096, 256>>>(d_data, d_result, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("基础版本: 结果=%.0f, 正确=%s\n", h_result,
           h_result == (float)N ? "是" : "否");

    // 测试优化版本
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
    block_reduce_optimized<<<4096, 256>>>(d_data, d_result, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("优化版本: 结果=%.0f, 正确=%s\n", h_result,
           h_result == (float)N ? "是" : "否");

    // 测试Shuffle版本
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
    block_reduce_shfl<<<4096, 256>>>(d_data, d_result, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Shuffle版本: 结果=%.0f, 正确=%s\n", h_result,
           h_result == (float)N ? "是" : "否");

    // 测试Grid-Stride版本
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
    block_reduce_grid_stride<<<256, 256>>>(d_data, d_result, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Grid-Stride版本: 结果=%.0f, 正确=%s\n", h_result,
           h_result == (float)N ? "是" : "否");

    printf("\n========== 优化要点 ==========\n");
    printf("1. 减少同步次数：warp内无需同步\n");
    printf("2. 使用Warp Shuffle：避免共享内存bank冲突\n");
    printf("3. Grid-Stride循环：提高数据局部性\n");
    printf("4. 减少原子操作：从N次降到GridSize次\n");

    // 清理
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);

    return 0;
}