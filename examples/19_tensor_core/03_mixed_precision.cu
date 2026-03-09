/**
 * 03_mixed_precision.cu
 * 混合精度计算示例：FP16存储 + FP32累加
 *
 * 混合精度是深度学习加速的关键技术
 * - FP16存储：减少显存占用和带宽压力
 * - FP32累加：保持数值精度
 *
 * 编译: nvcc -arch=sm_70 -o 03_mixed_precision 03_mixed_precision.cu
 * 运行: ./03_mixed_precision
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // FP16头文件

// FP32矩阵乘法（基准）
__global__ void matmul_fp32(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// FP16存储 + FP32计算（混合精度）
__global__ void matmul_mixed(const half* A, const half* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // 使用FP32累加器保持精度
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // 从FP16加载并转换为FP32进行计算
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[k * N + col]);
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

// 使用half2进行向量化访存（性能优化）
__global__ void matmul_half2(const half* A, const half* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // 使用half2一次加载A中的两个连续元素，对B保持标量读取避免越界
        int k = 0;
        for (; k + 1 < K; k += 2) {
            half2 a2 = __halves2half2(A[row * K + k], A[row * K + k + 1]);
            float2 a_f2 = __half22float2(a2);
            float b0 = __half2float(B[k * N + col]);
            float b1 = __half2float(B[(k + 1) * N + col]);
            sum += a_f2.x * b0;
            sum += a_f2.y * b1;
        }

        // 处理K为奇数时的尾元素
        if (k < K) {
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[k * N + col]);
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 1024, N = 1024, K = 1024;

    printf("========== 混合精度性能对比 ==========\n");
    printf("矩阵大小: %d x %d x %d\n\n", M, N, K);

    // 分配主机内存
    float* h_A_fp32 = (float*)malloc(M * K * sizeof(float));
    float* h_B_fp32 = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));

    half* h_A_fp16 = (half*)malloc(M * K * sizeof(half));
    half* h_B_fp16 = (half*)malloc(K * N * sizeof(half));

    // 初始化数据
    for (int i = 0; i < M * K; i++) {
        h_A_fp32[i] = 0.01f;
        h_A_fp16[i] = __float2half(0.01f);
    }
    for (int i = 0; i < K * N; i++) {
        h_B_fp32[i] = 0.01f;
        h_B_fp16[i] = __float2half(0.01f);
    }

    // 分配设备内存
    float *d_A_fp32, *d_B_fp32, *d_C;
    half *d_A_fp16, *d_B_fp16;

    cudaMalloc(&d_A_fp32, M * K * sizeof(float));
    cudaMalloc(&d_B_fp32, K * N * sizeof(float));
    cudaMalloc(&d_A_fp16, M * K * sizeof(half));
    cudaMalloc(&d_B_fp16, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A_fp32, h_A_fp32, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp32, h_B_fp32, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_fp16, h_A_fp16, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16, h_B_fp16, K * N * sizeof(half), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 测试FP32版本
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matmul_fp32<<<gridDim, blockDim>>>(d_A_fp32, d_B_fp32, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_fp32;
    cudaEventElapsedTime(&ms_fp32, start, stop);
    ms_fp32 /= 10;

    printf("FP32存储 + FP32计算:\n");
    printf("  时间: %.4f ms\n", ms_fp32);
    printf("  内存使用: %.2f MB\n",
           (M * K + K * N + M * N) * sizeof(float) / 1024.0 / 1024.0);

    // 测试混合精度版本
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matmul_mixed<<<gridDim, blockDim>>>(d_A_fp16, d_B_fp16, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_mixed;
    cudaEventElapsedTime(&ms_mixed, start, stop);
    ms_mixed /= 10;

    printf("\nFP16存储 + FP32计算 (混合精度):\n");
    printf("  时间: %.4f ms\n", ms_mixed);
    printf("  内存使用: %.2f MB (减少50%%)\n",
           (M * K + K * N) * sizeof(half) / 1024.0 / 1024.0 +
           M * N * sizeof(float) / 1024.0 / 1024.0);

    printf("\n========================================\n");
    printf("内存节省: %.1f%%\n",
           (1.0 - ((M * K + K * N) * sizeof(half) + M * N * sizeof(float)) /
            ((M * K + K * N + M * N) * sizeof(float))) * 100);

    printf("\n混合精度优势:\n");
    printf("  1. 内存占用减少50%%（FP16 vs FP32）\n");
    printf("  2. 内存带宽压力减半\n");
    printf("  3. 配合Tensor Core可获得更高加速\n");
    printf("  4. FP32累加保持数值精度\n");

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A_fp32);
    cudaFree(d_B_fp32);
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    cudaFree(d_C);
    free(h_A_fp32);
    free(h_B_fp32);
    free(h_A_fp16);
    free(h_B_fp16);
    free(h_C);

    return 0;
}
