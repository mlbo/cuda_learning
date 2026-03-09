/**
 * 03_warp_shuffle_reduce.cu
 * Warp Shuffle 规约实现
 *
 * Warp Shuffle 允许同一 Warp 内的线程直接交换寄存器数据，
 * 无需使用共享内存，效率更高。
 *
 * 主要函数：
 * - __shfl_down_sync: 从 lane + delta 获取值
 * - __shfl_xor_sync: 从 lane ^ laneMask 获取值
 *
 * 编译: nvcc -o 03_warp_shuffle_reduce 03_warp_shuffle_reduce.cu
 * 运行: ./03_warp_shuffle_reduce
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
// Warp Shuffle 规约函数 - 使用 __shfl_down_sync
// =============================================================================
__device__ float warp_reduce_shfl_down(float val) {
    // __shfl_down_sync(mask, var, delta)
    // mask: 参与的线程掩码，0xffffffff 表示全部 32 个线程
    // var: 要交换的变量
    // delta: 向下偏移量（从高 lane 向低 lane 传递数据）

    // 5 轮规约：32 -> 16 -> 8 -> 4 -> 2 -> 1
    // offset = 16: lane 0 从 lane 16 获取值，相加后 lane 0 有 32 个元素的和
    // offset = 8:  lane 0 从 lane 8 获取值
    // 以此类推...

    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    // 现在 lane 0 拥有完整的 warp 规约结果
    return val;
}

// =============================================================================
// Warp Shuffle 规约函数 - 使用 __shfl_xor_sync
// =============================================================================
// XOR 方式的特点是：所有 lane 最终都会得到完整的规约结果
// 这在某些场景下很有用（比如需要在每个线程都保留结果）
__device__ float warp_reduce_shfl_xor(float val) {
    // __shfl_xor_sync(mask, var, laneMask)
    // laneMask 与 lane ID 进行 XOR 运算得到目标 lane

    // 例如：offset = 16
    // lane 0 从 lane 0^16=16 获取值
    // lane 1 从 lane 1^16=17 获取值
    // ...

    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }

    // 所有 lane 都拥有完整的 warp 规约结果
    return val;
}

// =============================================================================
// 使用 Warp Shuffle 的完整规约 Kernel
// =============================================================================
__global__ void reduce_warp_shuffle(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;  // lane ID (0-31)

    // 加载数据
    float val = (idx < N) ? data[idx] : 0.0f;

    // Warp 内规约（无共享内存，无同步）
    val = warp_reduce_shfl_down(val);

    // 只有 lane 0 写入结果
    if (lane == 0) {
        atomicAdd(result, val);
    }
}

// =============================================================================
// 结合 Grid-Stride 循环的 Warp Shuffle 规约
// =============================================================================
__global__ void reduce_warp_shuffle_grid_stride(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    // Grid-stride 循环：每个线程累加多个元素
    float sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        sum += data[i];
    }

    // Warp 内规约
    sum = warp_reduce_shfl_down(sum);

    // 只有 lane 0 写入结果
    if (lane == 0) {
        atomicAdd(result, sum);
    }
}

// =============================================================================
// 结合共享内存的多级规约：Warp + Block
// =============================================================================
__global__ void reduce_warp_shuffle_block(float* data, float* result, int N) {
    // 共享内存存储每个 Warp 的部分和
    __shared__ float warp_sums[32];  // 最多 32 个 Warp

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;  // tid / 32
    int idx = blockIdx.x * blockDim.x + tid;

    // Grid-stride 循环累加
    float sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        sum += data[i];
    }

    // Level 1: Warp 内规约
    sum = warp_reduce_shfl_down(sum);

    // Level 2: 将 Warp 部分和写入共享内存
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Level 3: 第一个 Warp 对 Warp 部分和进行规约
    if (warp_id == 0) {
        // 只有线程数 / 32 个有效元素
        sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;

        // 再次 Warp 规约
        sum = warp_reduce_shfl_down(sum);

        // 只有 lane 0 写入全局结果
        if (lane == 0) {
            atomicAdd(result, sum);
        }
    }
}

// =============================================================================
// Shuffle 详细演示
// =============================================================================
__global__ void shuffle_demo() {
    int lane = threadIdx.x & 31;

    // 演示 __shfl_down_sync
    float val = (float)(lane + 1);  // 每个线程有不同的值：1, 2, 3, ..., 32

    if (lane < 8) {
        printf("初始: lane %d, val = %.0f\n", lane, val);
    }

    // 第 1 步：offset = 16
    float tmp = __shfl_down_sync(0xffffffff, val, 16);
    if (lane < 8) {
        printf("offset=16: lane %d, 从 lane %d 获取 %.0f, 原 val=%.0f\n",
               lane, lane + 16, tmp, val);
    }
    val += tmp;

    // 第 2 步：offset = 8
    tmp = __shfl_down_sync(0xffffffff, val, 8);
    if (lane < 8) {
        printf("offset=8: lane %d, 从 lane %d 获取 %.0f, 原 val=%.0f\n",
               lane, lane + 8, tmp, val);
    }
    val += tmp;

    // 继续剩余步骤...
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (lane == 0) {
        printf("\n最终: lane 0 的 val = %.0f (应该等于 1+2+...+32 = %d)\n",
               val, 32 * 33 / 2);
    }
}

// =============================================================================
// XOR 演示
// =============================================================================
__global__ void xor_demo() {
    int lane = threadIdx.x & 31;

    // 演示 __shfl_xor_sync
    float val = (float)(lane + 1);

    // XOR 方式：所有 lane 最终都会得到相同的值
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }

    // 所有 lane 都有完整结果
    if (lane < 4 || lane == 31) {
        printf("lane %d: 最终 val = %.0f\n", lane, val);
    }
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    int N = 1024 * 1024;  // 1M 元素
    size_t bytes = N * sizeof(float);

    printf("========== Warp Shuffle 规约详解 ==========\n");
    printf("数据量: %d 个 float\n\n", N);

    // 分配内存
    float* h_data = (float*)malloc(bytes);
    float *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    // 初始化
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // 创建计时事件
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // -------------------- Shuffle 演示 --------------------
    printf(">>> __shfl_down_sync 演示 (32 个线程)\n");
    printf("期望结果: 1+2+...+32 = %d\n\n", 32 * 33 / 2);
    shuffle_demo<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n>>> __shfl_xor_sync 演示\n");
    printf("XOR 方式：所有 lane 最终都会得到相同的结果\n\n");
    xor_demo<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n");

    int blockSize = 256;
    int gridSize = 256;
    float zero = 0.0f;
    float result;
    float ms;

    // -------------------- Warp Shuffle 版本测试 --------------------
    printf(">>> Warp Shuffle 规约\n");
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    reduce_warp_shuffle_grid_stride<<<gridSize, blockSize>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("结果: %.0f (期望 %d)\n", result, N);
    printf("时间: %.3f ms\n", ms);
    printf("原子操作次数: %d (GridSize * Warp 数)\n\n", gridSize * (blockSize / 32));

    // -------------------- 多级规约版本测试 --------------------
    printf(">>> 多级规约 (Warp + Block)\n");
    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    reduce_warp_shuffle_block<<<gridSize, blockSize>>>(d_data, d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("结果: %.0f (期望 %d)\n", result, N);
    printf("时间: %.3f ms\n", ms);
    printf("原子操作次数: %d (GridSize)\n\n", gridSize);

    // -------------------- 优势总结 --------------------
    printf("========== Warp Shuffle 优势 ==========\n");
    printf("1. 无需共享内存：直接在寄存器间交换数据\n");
    printf("2. 无需同步：Warp 内线程是锁步执行的\n");
    printf("3. 避免 Bank Conflict：不使用共享内存\n");
    printf("4. 更低的延迟：寄存器访问比共享内存更快\n");
    printf("5. 更高的带宽：充分利用寄存器带宽\n\n");

    printf("========== Shuffle 函数说明 ==========\n");
    printf("__shfl_sync(mask, var, srcLane):\n");
    printf("  从指定的 srcLane 获取 var 的值\n");
    printf("__shfl_up_sync(mask, var, delta):\n");
    printf("  从 lane - delta 获取 var 的值\n");
    printf("__shfl_down_sync(mask, var, delta):\n");
    printf("  从 lane + delta 获取 var 的值\n");
    printf("__shfl_xor_sync(mask, var, laneMask):\n");
    printf("  从 lane ^ laneMask 获取 var 的值\n");

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    free(h_data);

    return 0;
}