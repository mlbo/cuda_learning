/**
 * 04_warp_reduce.cu
 * Warp级别规约详解
 *
 * Warp Shuffle指令允许warp内的线程直接交换寄存器中的数据，
 * 无需使用共享内存，效率更高。
 *
 * 编译: nvcc -o 04_warp_reduce 04_warp_reduce.cu
 * 运行: ./04_warp_reduce
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Warp内规约 - 详细步骤演示
__global__ void warp_reduce_demo(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = idx & 31;  // 线程在warp内的位置 (0-31)

    // 每个线程读取一个元素
    float val = (idx < N) ? data[idx] : 0.0f;

    // 只在一个warp内打印（便于调试理解）
    if (blockIdx.x == 0 && lane < 8) {
        printf("初始: lane=%d, val=%.0f\n", lane, val);
    }

    // Warp Shuffle规约
    // __shfl_down_sync(mask, var, delta)
    // mask: 参与的线程掩码 (0xffffffff = 全部32个线程)
    // var: 要交换的变量
    // delta: 向下偏移量

    // 第1步: 偏移16，将后16个线程的值加到前16个线程
    val += __shfl_down_sync(0xffffffff, val, 16);

    if (blockIdx.x == 0 && lane < 8) {
        printf("偏移16后: lane=%d, val=%.0f\n", lane, val);
    }

    // 第2步: 偏移8
    val += __shfl_down_sync(0xffffffff, val, 8);

    // 第3步: 偏移4
    val += __shfl_down_sync(0xffffffff, val, 4);

    // 第4步: 偏移2
    val += __shfl_down_sync(0xffffffff, val, 2);

    // 第5步: 偏移1
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (blockIdx.x == 0 && lane == 0) {
        printf("规约完成: lane=0, val=%.0f (warp内32个元素的和)\n", val);
    }

    // 只有lane 0的线程拥有完整的warp规约结果
    if (lane == 0) {
        atomicAdd(result, val);
    }
}

// 通用的Warp规约函数
__device__ float warp_reduce_sum(float val) {
    // 使用Warp Shuffle进行规约
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// 使用Grid-Stride循环的Warp规约
__global__ void sum_warp_grid_stride(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = idx & 31;

    // Grid-stride循环：每个线程处理多个元素
    float sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        sum += data[i];
    }

    // Warp内规约
    sum = warp_reduce_sum(sum);

    // 只有lane 0写入结果
    if (lane == 0) {
        atomicAdd(result, sum);
    }
}

// __shfl_xor_sync 示例：另一种规约方式
__device__ float warp_reduce_xor(float val) {
    // __shfl_xor_sync: 与lane ID进行XOR操作来交换数据
    // 这种方式可以在规约的同时保持数据分布
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

int main() {
    int N = 32;  // 为了演示，使用较小的数据量
    size_t bytes = N * sizeof(float);

    float* h_data = (float*)malloc(bytes);
    float* d_data, *d_result;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_result, sizeof(float));

    // 初始化：1到32
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i + 1);
    }
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    printf("========== Warp规约详解 ==========\n");
    printf("数据: 1, 2, 3, ..., 32\n");
    printf("期望和: %d (1+2+...+32)\n\n", N * (N + 1) / 2);

    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    // 执行演示
    warp_reduce_demo<<<1, 32>>>(d_data, d_result, N);
    cudaDeviceSynchronize();

    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("\n最终结果: %.0f\n", h_result);

    printf("\n========== Shuffle指令说明 ==========\n");
    printf("__shfl_down_sync(mask, var, delta):\n");
    printf("  - 从lane ID + delta的线程获取var的值\n");
    printf("  - 如果超出warp范围，返回自己的值\n");
    printf("\n__shfl_xor_sync(mask, var, laneMask):\n");
    printf("  - 从lane ID ^ laneMask的线程获取var的值\n");
    printf("  - XOR操作确保数据交换的对称性\n");
    printf("\n__shfl_up_sync(mask, var, delta):\n");
    printf("  - 从lane ID - delta的线程获取var的值\n");
    printf("\n__shfl_sync(mask, var, srcLane):\n");
    printf("  - 从指定srcLane的线程获取var的值\n");

    // 清理
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);

    return 0;
}