/**
 * 02_tree_reduce.cu
 * 树状规约实现 - 使用共享内存
 *
 * 树状规约采用分治策略：
 * 1. 每个线程加载数据到共享内存
 * 2. 迭代地将相邻元素相加
 * 3. 最终得到一个结果
 *
 * 编译: nvcc -o 02_tree_reduce 02_tree_reduce.cu
 * 运行: ./02_tree_reduce
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
// 基础树状规约
// =============================================================================
__global__ void tree_reduce_basic(float* data, float* result, int N) {
    // 共享内存：每个 Block 共享
    // 大小需要是 blockDim.x
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 步骤 1：加载数据到共享内存
    // 超出边界的线程加载 0
    sdata[tid] = (idx < N) ? data[idx] : 0.0f;

    // 同步：确保所有线程都完成了加载
    __syncthreads();

    // 步骤 2：树状规约
    // 每轮将活跃线程数减半
    // s = 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // 每轮规约后同步
        __syncthreads();
    }

    // 步骤 3：只有线程 0 拥有完整的 Block 规约结果
    // 使用原子操作跨 Block 合并
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// =============================================================================
// 优化版本：减少 Bank Conflict
// =============================================================================
// 注意：这里的顺序规约方式可以避免 Bank Conflict
// tid 和 tid+s 访问的地址差为 s
// 当 s 是 2 的幂次时，这种访问模式是 Bank Conflict Free 的
__global__ void tree_reduce_bank_optimized(float* data, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? data[idx] : 0.0f;
    __syncthreads();

    // 顺序规约：避免 Bank Conflict
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // tid 访问 Bank[tid % 32]
            // tid + s 访问 Bank[(tid + s) % 32]
            // 当 s 是 2 的幂次时，这两个 Bank 不同
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// =============================================================================
// 展开版本：减少同步次数
// =============================================================================
// 当活跃线程数 <= 32 时，属于同一个 Warp，可以省略同步
// 同时展开最后几轮循环以提高性能
__global__ void tree_reduce_unrolled(float* data, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? data[idx] : 0.0f;
    __syncthreads();

    // 展开循环：当 s > 32 时需要同步
    if (blockDim.x >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }

    if (blockDim.x >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    // s <= 32 时在同一个 Warp 内，无需同步
    // 使用 volatile 确保读取最新值
    if (tid < 32) {
        volatile float* smem = sdata;

        // 展开最后 5 轮
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8)   smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4)   smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2)   smem[tid] += smem[tid + 1];
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// =============================================================================
// Grid-Stride 循环版本
// =============================================================================
// 每个 Block 处理更多数据，减少 Block 数量和原子操作次数
__global__ void tree_reduce_grid_stride(float* data, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;

    // Grid-stride 循环：每个线程累加多个元素
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        sum += data[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // 树状规约
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

// =============================================================================
// 详细步骤演示（小数据量）
// =============================================================================
__global__ void tree_reduce_demo(float* data, float* result, int N) {
    __shared__ float sdata[8];  // 演示用：8 个元素

    int tid = threadIdx.x;

    // 加载数据
    sdata[tid] = (tid < N) ? data[tid] : 0.0f;

    printf("初始: tid=%d, sdata[%d]=%.0f\n", tid, tid, sdata[tid]);
    __syncthreads();

    // 树状规约演示
    // 第 1 轮：s = 4
    if (tid < 4) {
        float old = sdata[tid];
        sdata[tid] += sdata[tid + 4];
        printf("第1轮: tid=%d, sdata[%d]=%.0f + %.0f = %.0f\n",
               tid, tid, old, sdata[tid + 4], sdata[tid]);
    }
    __syncthreads();

    // 第 2 轮：s = 2
    if (tid < 2) {
        float old = sdata[tid];
        sdata[tid] += sdata[tid + 2];
        printf("第2轮: tid=%d, sdata[%d]=%.0f + %.0f = %.0f\n",
               tid, tid, old, sdata[tid + 2], sdata[tid]);
    }
    __syncthreads();

    // 第 3 轮：s = 1
    if (tid < 1) {
        float old = sdata[tid];
        sdata[tid] += sdata[tid + 1];
        printf("第3轮: tid=%d, sdata[%d]=%.0f + %.0f = %.0f\n",
               tid, tid, old, sdata[tid + 1], sdata[tid]);
    }
    __syncthreads();

    if (tid == 0) {
        printf("最终: sdata[0] = %.0f\n", sdata[0]);
        atomicAdd(result, sdata[0]);
    }
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    int N = 1024 * 1024;  // 1M 元素
    size_t bytes = N * sizeof(float);

    printf("========== 树状规约详解 ==========\n");
    printf("数据量: %d 个 float\n\n", N);

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
    int gridSize = (N + blockSize - 1) / blockSize;
    float zero = 0.0f;
    float result;
    float ms;

    // -------------------- 小规模演示 --------------------
    printf(">>> 小规模演示 (8 个元素)\n");
    float h_demo[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float* d_demo;
    CUDA_CHECK(cudaMalloc(&d_demo, 8 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_demo, h_demo, 8 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));

    printf("输入数据: 1, 2, 3, 4, 5, 6, 7, 8\n");
    printf("期望结果: %d\n\n", 1+2+3+4+5+6+7+8);

    tree_reduce_demo<<<1, 8>>>(d_demo, d_result, 8);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("\n演示结果: %.0f\n\n", result);
    CUDA_CHECK(cudaFree(d_demo));

    // -------------------- 基础版本测试 --------------------
    printf(">>> 基础树状规约\n");
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    tree_reduce_basic<<<gridSize, blockSize>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("结果: %.0f (期望 %d)\n", result, N);
    printf("时间: %.3f ms\n", ms);
    printf("原子操作次数: %d (GridSize)\n\n", gridSize);

    // -------------------- 展开版本测试 --------------------
    printf(">>> 展开优化版本\n");
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    tree_reduce_unrolled<<<gridSize, blockSize>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("结果: %.0f (期望 %d)\n", result, N);
    printf("时间: %.3f ms\n\n", ms);

    // -------------------- Grid-Stride 版本测试 --------------------
    printf(">>> Grid-Stride 循环版本\n");
    gridSize = 256;  // 固定 Block 数量
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    tree_reduce_grid_stride<<<gridSize, blockSize>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("结果: %.0f (期望 %d)\n", result, N);
    printf("时间: %.3f ms\n", ms);
    printf("原子操作次数: %d (固定 GridSize)\n\n", gridSize);

    // -------------------- 优化要点总结 --------------------
    printf("========== 优化要点 ==========\n");
    printf("1. 树状规约将 O(N) 串行操作变为 O(log N) 并行操作\n");
    printf("2. 展开循环减少同步次数\n");
    printf("3. Warp 内无需同步（锁步执行）\n");
    printf("4. Grid-Stride 循环减少原子操作次数\n");
    printf("5. 原子操作次数: N -> GridSize\n");

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    free(h_data);

    return 0;
}