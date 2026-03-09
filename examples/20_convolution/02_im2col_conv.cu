/**
 * 02_im2col_conv.cu
 * im2col + GEMM 卷积实现
 *
 * im2col方法将卷积转换为矩阵乘法：
 * 1. 将输入按滑动窗口展开成矩阵（im2col）
 * 2. 用矩阵乘法计算输出
 *
 * 编译: nvcc -o 02_im2col_conv 02_im2col_conv.cu
 * 运行: ./02_im2col_conv
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// im2col：将输入图像展开为矩阵
// 输入: [H][W] 的图像
// 输出: [K*K][out_H*out_W] 的矩阵
// 每一列是一个滑动窗口的数据
__global__ void im2col_kernel(
    const float* input,
    float* col,
    int H, int W, int K
) {
    int out_H = H - K + 1;
    int out_W = W - K + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_H * out_W;

    if (idx >= total) return;

    int out_y = idx / out_W;
    int out_x = idx % out_W;

    // 将这个滑动窗口的数据填入col矩阵的对应列
    for (int ky = 0; ky < K; ky++) {
        for (int kx = 0; kx < K; kx++) {
            int in_y = out_y + ky;
            int in_x = out_x + kx;
            // col的行索引 = ky * K + kx（展平的卷积核索引）
            // col的列索引 = idx（输出像素索引）
            col[(ky * K + kx) * total + idx] = input[in_y * W + in_x];
        }
    }
}

// 矩阵乘法：output = kernel * col
// kernel: [1][K*K] (展平的卷积核)
// col: [K*K][out_H*out_W]
// output: [1][out_H*out_W]
__global__ void matmul_kernel(
    const float* kernel,  // [1 x K*K]
    const float* col,     // [K*K x N]
    float* output,        // [1 x N]
    int KK, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < KK; k++) {
        sum += kernel[k] * col[k * N + idx];
    }
    output[idx] = sum;
}

// 合并im2col和矩阵乘法的核函数（优化版本）
__global__ void im2col_conv_merged(
    const float* input,
    const float* kernel,
    float* output,
    int H, int W, int K
) {
    int out_H = H - K + 1;
    int out_W = W - K + 1;

    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_y >= out_H || out_x >= out_W) return;

    float sum = 0.0f;
    int kernel_idx = 0;

    // 直接计算，相当于on-the-fly的im2col
    for (int ky = 0; ky < K; ky++) {
        for (int kx = 0; kx < K; kx++) {
            int in_y = out_y + ky;
            int in_x = out_x + kx;
            sum += input[in_y * W + in_x] * kernel[kernel_idx++];
        }
    }

    output[out_y * out_W + out_x] = sum;
}

int main() {
    int H = 512;  // 较小尺寸演示
    int W = 512;
    int K = 3;

    int out_H = H - K + 1;
    int out_W = W - K + 1;
    int KK = K * K;
    int N = out_H * out_W;

    printf("========== im2col + GEMM 卷积 ==========\n");
    printf("输入尺寸: %d x %d\n", H, W);
    printf("卷积核: %d x %d\n", K, K);
    printf("输出尺寸: %d x %d\n\n", out_H, out_W);

    // 分配主机内存
    size_t input_size = H * W * sizeof(float);
    size_t kernel_size = KK * sizeof(float);
    size_t output_size = N * sizeof(float);
    size_t col_size = KK * N * sizeof(float);

    float* h_input = (float*)malloc(input_size);
    float* h_kernel = (float*)malloc(kernel_size);
    float* h_output = (float*)malloc(output_size);

    // 初始化
    for (int i = 0; i < H * W; i++) h_input[i] = 1.0f;
    for (int i = 0; i < KK; i++) h_kernel[i] = 1.0f / KK;

    // 分配设备内存
    float *d_input, *d_kernel, *d_output, *d_col;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_col, col_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("方法1: im2col + 矩阵乘法\n");
    // im2col
    cudaEventRecord(start);
    im2col_kernel<<<(N + 255) / 256, 256>>>(d_input, d_col, H, W, K);
    matmul_kernel<<<(N + 255) / 256, 256>>>(d_kernel, d_col, d_output, KK, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);
    printf("  时间: %.4f ms\n", ms1);
    printf("  临时内存: %.2f MB (im2col矩阵)\n", (float)col_size / 1024 / 1024);

    printf("\n方法2: 合并核函数（无临时内存）\n");
    cudaMemset(d_output, 0, output_size);
    dim3 blockDim(16, 16);
    dim3 gridDim((out_W + 15) / 16, (out_H + 15) / 16);

    cudaEventRecord(start);
    im2col_conv_merged<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, H, W, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms2;
    cudaEventElapsedTime(&ms2, start, stop);
    printf("  时间: %.4f ms\n", ms2);
    printf("  临时内存: 0 MB\n");

    printf("\n========================================\n");
    printf("结论: 合并核函数避免了im2col的大临时内存\n");

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_col);
    free(h_input);
    free(h_kernel);
    free(h_output);

    return 0;
}