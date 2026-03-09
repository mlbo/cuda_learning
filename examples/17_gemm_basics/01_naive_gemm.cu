/**
 * 01_naive_gemm.cu
 * 朴素GEMM实现 - 展示访存问题
 *
 * 这个示例展示了最简单的GEMM实现及其性能问题：
 * - 每个线程计算C矩阵中的一个元素
 * - B矩阵按列访问，无法实现内存合并
 *
 * 编译: nvcc -o 01_naive_gemm 01_naive_gemm.cu
 * 运行: ./01_naive_gemm
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>

// 矩阵尺寸（可调整）
#define M 1024
#define N 1024
#define K 1024

// Block尺寸
#define BLOCK_SIZE 16

// 朴素GEMM核函数
// 每个线程计算C矩阵中的一个元素
__global__ void naive_gemm(float* A, float* B, float* C, int m, int n, int k) {
    // 使用2D线程索引映射到矩阵位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M方向
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N方向

    if (row < m && col < n) {
        float sum = 0.0f;
        // 沿K维度累加
        // A[row][k]: 按行访问，地址连续
        // B[k][col]: 按列访问，地址跨度为N，不连续！
        for (int i = 0; i < k; i++) {
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

// 初始化矩阵
void init_matrix(float* mat, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = value;
    }
}

// 验证结果
bool verify_result(float* C, float* ref, int m, int n, float eps = 1e-3) {
    for (int i = 0; i < m * n; i++) {
        if (std::fabs(C[i] - ref[i]) > eps) {
            printf("验证失败: C[%d] = %f, ref = %f\n", i, C[i], ref[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("========== 朴素GEMM演示 ==========\n");
    printf("矩阵尺寸: A(%dx%d) * B(%dx%d) = C(%dx%d)\n\n", M, K, K, N, M, N);

    // 计算内存大小
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // 分配主机内存
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    float *h_ref = (float*)malloc(bytes_C);

    // 初始化矩阵（使用小值避免溢出）
    init_matrix(h_A, M, K, 1.0f);
    init_matrix(h_B, K, N, 1.0f);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // 创建计时事件
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // -------------------- CPU参考实现 --------------------
    printf(">>> CPU参考实现\n");
    cpu_gemm(h_A, h_B, h_ref, M, N, K);
    printf("CPU计算完成\n\n");

    // -------------------- 朴素GPU实现 --------------------
    printf(">>> 朴素GPU实现\n");

    // 配置执行参数
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 预热
    naive_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    CUDA_CHECK(cudaEventRecord(start));
    naive_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float naive_ms;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));

    // 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // 验证结果
    bool correct = verify_result(h_C, h_ref, M, N);
    printf("结果验证: %s\n", correct ? "通过" : "失败");
    printf("执行时间: %.3f ms\n", naive_ms);

    // 计算性能指标
    double flops = 2.0 * M * N * K;  // 每个元素K次乘法和K次加法
    double gflops = flops / (naive_ms * 1e-3) / 1e9;
    printf("性能: %.2f GFLOPS\n", gflops);

    // -------------------- 性能分析 --------------------
    printf("\n========== 性能分析 ==========\n");
    printf("朴素实现的问题:\n");
    printf("  1. B矩阵按列访问，地址跨度为N\n");
    printf("  2. 同一Warp内线程访问B的不同列，地址不连续\n");
    printf("  3. 无法实现内存合并，每次访问触发独立内存事务\n");
    printf("  4. 有效带宽利用率低\n\n");

    printf("优化思路:\n");
    printf("  1. 调整访存模式，实现内存合并\n");
    printf("  2. 使用共享内存缓存数据\n");
    printf("  3. 分块计算，提高数据复用\n");

    // 清理资源
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
