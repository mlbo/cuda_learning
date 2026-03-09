/**
 * 01_direct_conv.cu
 * 直接卷积实现（Naive版本）
 *
 * 卷积操作：output = conv(input, kernel)
 * 直接实现：每个输出像素遍历所有卷积核元素
 *
 * 编译: nvcc -o 01_direct_conv 01_direct_conv.cu
 * 运行: ./01_direct_conv
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>

// 2D直接卷积核函数
// 假设：单通道输入、单通道输出、正方形卷积核
__global__ void conv2d_direct(
    const float* input,    // 输入图像 [H][W]
    const float* kernel,   // 卷积核 [K][K]
    float* output,         // 输出图像 [H-K+1][W-K+1]
    int H, int W,          // 输入高度和宽度
    int K                  // 卷积核大小
) {
    // 输出像素坐标
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    int out_H = H - K + 1;
    int out_W = W - K + 1;

    // 边界检查
    if (out_y >= out_H || out_x >= out_W) return;

    // 计算卷积
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

// CPU版本用于验证
void conv2d_cpu(const float* input, const float* kernel, float* output, int H, int W, int K) {
    int out_H = H - K + 1;
    int out_W = W - K + 1;

    for (int out_y = 0; out_y < out_H; out_y++) {
        for (int out_x = 0; out_x < out_W; out_x++) {
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
    }
}

int main() {
    // 输入尺寸
    int H = 1024;  // 输入高度
    int W = 1024;  // 输入宽度
    int K = 3;     // 卷积核大小（3x3）

    int out_H = H - K + 1;
    int out_W = W - K + 1;

    printf("========== 直接卷积实现 ==========\n");
    printf("输入尺寸: %d x %d\n", H, W);
    printf("卷积核: %d x %d\n", K, K);
    printf("输出尺寸: %d x %d\n\n", out_H, out_W);

    // 分配主机内存
    size_t input_size = H * W * sizeof(float);
    size_t kernel_size = K * K * sizeof(float);
    size_t output_size = out_H * out_W * sizeof(float);

    float* h_input = (float*)malloc(input_size);
    float* h_kernel = (float*)malloc(kernel_size);
    float* h_output = (float*)malloc(output_size);
    float* h_output_ref = (float*)malloc(output_size);

    // 初始化数据
    for (int i = 0; i < H * W; i++) {
        h_input[i] = 1.0f;  // 全1输入
    }
    // 3x3平均滤波器
    for (int i = 0; i < K * K; i++) {
        h_kernel[i] = 1.0f / (K * K);
    }

    // 分配设备内存
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    // 配置核函数
    dim3 blockDim(16, 16);
    dim3 gridDim((out_W + 15) / 16, (out_H + 15) / 16);

    // 创建事件计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 执行卷积
    cudaEventRecord(start);
    conv2d_direct<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, H, W, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // 复制结果
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // CPU验证
    conv2d_cpu(h_input, h_kernel, h_output_ref, H, W, K);

    // 检查正确性
    bool correct = true;
    for (int i = 0; i < out_H * out_W && correct; i++) {
        if (std::fabs(h_output[i] - h_output_ref[i]) > 1e-5) {
            correct = false;
            printf("错误: output[%d] = %f, 期望 %f\n", i, h_output[i], h_output_ref[i]);
        }
    }

    printf("执行时间: %.4f ms\n", ms);
    printf("结果正确: %s\n", correct ? "是" : "否");

    // 计算性能指标
    long long ops = 2LL * K * K * out_H * out_W;  // 乘加操作数
    printf("计算量: %lld FLOPs\n", ops);
    printf("性能: %.2f GFLOPS\n", ops / (ms * 1e6));

    printf("\n==================================\n");

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output);
    free(h_output_ref);

    return 0;
}
