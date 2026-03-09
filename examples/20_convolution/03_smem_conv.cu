/**
 * 03_smem_conv.cu
 * 共享内存优化的卷积实现
 *
 * 使用共享内存缓存输入数据，减少全局内存访问
 * 采用分块策略，每个block计算一块输出
 *
 * 编译: nvcc -o 03_smem_conv 03_smem_conv.cu
 * 运行: ./03_smem_conv
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define KERNEL_SIZE 3
#define PAD (KERNEL_SIZE / 2)

// 共享内存优化版本
__global__ void conv2d_smem(
    const float* input,
    const float* kernel,
    float* output,
    int H, int W
) {
    // 共享内存：存储输入的一个块（加上卷积核需要的额外边界）
    __shared__ float s_input[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];

    // 全局坐标
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 共享内存中的坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 输入坐标（考虑卷积核的偏移）
    int in_x = blockIdx.x * TILE_SIZE + tx;
    int in_y = blockIdx.y * TILE_SIZE + ty;

    // 加载数据到共享内存
    // 每个线程加载一个元素
    if (in_x < W && in_y < H) {
        s_input[ty][tx] = input[in_y * W + in_x];
    } else {
        s_input[ty][tx] = 0.0f;  // 边界填充
    }

    // 加载额外的边界元素（右侧和底部的halo区域）
    // 这些是卷积核需要的额外数据
    if (tx < KERNEL_SIZE - 1) {
        int extra_x = in_x + TILE_SIZE;
        if (extra_x < W && in_y < H) {
            s_input[ty][tx + TILE_SIZE] = input[in_y * W + extra_x];
        } else {
            s_input[ty][tx + TILE_SIZE] = 0.0f;
        }
    }

    if (ty < KERNEL_SIZE - 1) {
        int extra_y = in_y + TILE_SIZE;
        if (in_x < W && extra_y < H) {
            s_input[ty + TILE_SIZE][tx] = input[extra_y * W + in_x];
        } else {
            s_input[ty + TILE_SIZE][tx] = 0.0f;
        }
    }

    // 加载角落的额外元素
    if (tx < KERNEL_SIZE - 1 && ty < KERNEL_SIZE - 1) {
        int extra_x = in_x + TILE_SIZE;
        int extra_y = in_y + TILE_SIZE;
        if (extra_x < W && extra_y < H) {
            s_input[ty + TILE_SIZE][tx + TILE_SIZE] = input[extra_y * W + extra_x];
        } else {
            s_input[ty + TILE_SIZE][tx + TILE_SIZE] = 0.0f;
        }
    }

    __syncthreads();

    // 计算卷积
    float sum = 0.0f;
    for (int ky = 0; ky < KERNEL_SIZE; ky++) {
        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
            sum += s_input[ty + ky][tx + kx] * kernel[ky * KERNEL_SIZE + kx];
        }
    }

    // 写入输出
    int out_H = H - KERNEL_SIZE + 1;
    int out_W = W - KERNEL_SIZE + 1;

    if (out_x < out_W && out_y < out_H) {
        output[out_y * out_W + out_x] = sum;
    }
}

// 直接卷积（基准）
__global__ void conv2d_direct(
    const float* input,
    const float* kernel,
    float* output,
    int H, int W, int K
) {
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    int out_H = H - K + 1;
    int out_W = W - K + 1;

    if (out_y >= out_H || out_x >= out_W) return;

    float sum = 0.0f;
    for (int ky = 0; ky < K; ky++) {
        for (int kx = 0; kx < K; kx++) {
            int in_y = out_y + ky;
            int in_x = out_x + kx;
            sum += input[in_y * W + in_x] * kernel[ky * K + kx];
        }
    }

    output[out_y * out_W + out_x] = sum;
}

int main() {
    int H = 1024;
    int W = 1024;
    int K = KERNEL_SIZE;

    int out_H = H - K + 1;
    int out_W = W - K + 1;

    printf("========== 共享内存卷积优化 ==========\n");
    printf("输入尺寸: %d x %d\n", H, W);
    printf("卷积核: %d x %d\n", K, K);
    printf("输出尺寸: %d x %d\n\n", out_H, out_W);

    // 分配内存
    float* h_input = (float*)malloc(H * W * sizeof(float));
    float* h_kernel = (float*)malloc(K * K * sizeof(float));
    float* h_output = (float*)malloc(out_H * out_W * sizeof(float));

    // 初始化
    for (int i = 0; i < H * W; i++) h_input[i] = 1.0f;
    for (int i = 0; i < K * K; i++) h_kernel[i] = 1.0f / (K * K);

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, H * W * sizeof(float));
    cudaMalloc(&d_kernel, K * K * sizeof(float));
    cudaMalloc(&d_output, out_H * out_W * sizeof(float));

    cudaMemcpy(d_input, h_input, H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, K * K * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 测试直接卷积
    dim3 blockDim1(16, 16);
    dim3 gridDim1((out_W + 15) / 16, (out_H + 15) / 16);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        conv2d_direct<<<gridDim1, blockDim1>>>(d_input, d_kernel, d_output, H, W, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_direct;
    cudaEventElapsedTime(&ms_direct, start, stop);
    ms_direct /= 10;

    printf("直接卷积:\n");
    printf("  时间: %.4f ms\n", ms_direct);

    // 测试共享内存版本
    dim3 blockDim2(TILE_SIZE, TILE_SIZE);
    dim3 gridDim2((out_W + TILE_SIZE - 1) / TILE_SIZE,
                  (out_H + TILE_SIZE - 1) / TILE_SIZE);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        conv2d_smem<<<gridDim2, blockDim2>>>(d_input, d_kernel, d_output, H, W);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_smem;
    cudaEventElapsedTime(&ms_smem, start, stop);
    ms_smem /= 10;

    printf("\n共享内存卷积:\n");
    printf("  时间: %.4f ms\n", ms_smem);
    printf("  加速比: %.2fx\n", ms_direct / ms_smem);

    printf("\n========================================\n");
    printf("优化要点:\n");
    printf("  1. 使用共享内存缓存输入数据\n");
    printf("  2. 减少全局内存访问次数\n");
    printf("  3. 注意处理边界（halo区域）\n");

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output);

    return 0;
}