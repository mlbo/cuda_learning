/**
 * 02_coalesced_gemm.cu
 * 内存合并优化的GEMM
 *
 * 这个示例展示了如何通过调整访存模式实现内存合并：
 * - 交换计算顺序，使B矩阵访问变为连续
 * - 大幅提高有效带宽利用率
 *
 * 编译: nvcc -o 02_coalesced_gemm 02_coalesced_gemm.cu
 * 运行: ./02_coalesced_gemm
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>

// 矩阵尺寸
#define M 1024
#define N 1024
#define K 1024

// Block尺寸
#define BLOCK_SIZE 16

// 内存合并优化的GEMM核函数
// 关键优化：调整线程映射，使Warp内线程在N方向连续
__global__ void coalesced_gemm(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;

        // 沿K维度累加
        // A[row][i]: 同一Warp内不同线程访问同一行不同列
        //            如果Warp内线程在row方向相同，则访问同一元素（广播）
        // B[i][col]: 同一Warp内不同线程访问同一行不同列
        //            地址连续，可以实现内存合并！
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// 更清晰的版本：显式展示内存合并原理
// 通过调整Block配置，确保Warp内线程在col方向连续
__global__ void coalesced_gemm_v2(float* A, float* B, float* C, int m, int n, int k) {
    // 使用1D Block配置，更容易理解
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 计算全局坐标
    // 关键：tx用于col方向（N方向），确保Warp内连续
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    if (row < m && col < n) {
        float sum = 0.0f;

        for (int i = 0; i < k; i++) {
            // A[row][i]: 同一Warp内所有线程访问相同地址 -> 广播
            // B[i][col]: 同一Warp内col连续 -> 合并访问
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CPU参考实现
void cpu_gemm(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

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

void init_matrix(float* mat, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = value;
    }
}

bool verify_result(float* C, float* ref, int m, int n, float eps = 1e-3) {
    for (int i = 0; i < m * n; i++) {
        if (std::fabs(C[i] - ref[i]) > eps) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("========== 内存合并优化GEMM演示 ==========\n");
    printf("矩阵尺寸: A(%dx%d) * B(%dx%d) = C(%dx%d)\n\n", M, K, K, N, M, N);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    float *h_ref = (float*)malloc(bytes_C);

    init_matrix(h_A, M, K, 1.0f);
    init_matrix(h_B, K, N, 1.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // CPU参考
    printf(">>> CPU参考实现\n");
    cpu_gemm(h_A, h_B, h_ref, M, N, K);
    printf("CPU计算完成\n\n");

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // -------------------- 内存合并版本 --------------------
    printf(">>> 内存合并优化版本\n");

    // 预热
    coalesced_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    coalesced_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float coalesced_ms;
    CUDA_CHECK(cudaEventElapsedTime(&coalesced_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    bool correct = verify_result(h_C, h_ref, M, N);
    printf("结果验证: %s\n", correct ? "通过" : "失败");
    printf("执行时间: %.3f ms\n", coalesced_ms);

    double flops = 2.0 * M * N * K;
    double gflops = flops / (coalesced_ms * 1e-3) / 1e9;
    printf("性能: %.2f GFLOPS\n", gflops);

    // -------------------- 性能对比 --------------------
    printf("\n========== 内存合并优化分析 ==========\n");
    printf("优化原理:\n");
    printf("  1. Warp内线程在col方向连续\n");
    printf("  2. B[i][col]访问变为连续地址\n");
    printf("  3. 32个线程访问连续128字节 -> 单次内存事务\n");
    printf("  4. 有效带宽利用率大幅提高\n\n");

    printf("访存模式对比:\n");
    printf("  朴素版本: B按列访问，跨度N，无法合并\n");
    printf("  合并版本: B按行访问，连续地址，完美合并\n");

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return 0;
}
