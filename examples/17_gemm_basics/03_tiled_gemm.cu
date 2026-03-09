/**
 * 03_tiled_gemm.cu
 * 分块GEMM实现
 *
 * 这个示例展示了如何使用共享内存分块优化GEMM：
 * - 将矩阵分成小块（Tile）
 * - 利用共享内存缓存Tile数据
 * - 减少全局内存访问次数
 *
 * 编译: nvcc -o 03_tiled_gemm 03_tiled_gemm.cu
 * 运行: ./03_tiled_gemm
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>

// 矩阵尺寸
#define M 1024
#define N 1024
#define K 1024

// Tile尺寸（必须是16或32的倍数关系）
#define TILE_SIZE 32

// 分块GEMM核函数
__global__ void tiled_gemm(float* A, float* B, float* C, int m, int n, int k) {
    // 共享内存缓存Tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // 沿K方向迭代Tile
    int numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 加载A的Tile到共享内存
        int a_col = t * TILE_SIZE + tx;
        if (row < m && a_col < k) {
            As[ty][tx] = A[row * k + a_col];
        } else {
            As[ty][tx] = 0.0f;  // 边界处理
        }

        // 加载B的Tile到共享内存
        int b_row = t * TILE_SIZE + ty;
        if (b_row < k && col < n) {
            Bs[ty][tx] = B[b_row * n + col];
        } else {
            Bs[ty][tx] = 0.0f;  // 边界处理
        }

        // 同步：确保所有线程都加载完成
        __syncthreads();

        // 计算Tile内的部分结果
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        // 同步：确保所有线程都计算完成后再加载下一个Tile
        __syncthreads();
    }

    // 写回结果
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// 带Padding的分块GEMM（避免Bank Conflict）
__global__ void tiled_gemm_padded(float* A, float* B, float* C, int m, int n, int k) {
    // 使用Padding避免Bank Conflict
    // TILE_SIZE=32时，每行33个元素，避免同一列元素在同一Bank
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_SIZE + tx;
        if (row < m && a_col < k) {
            As[ty][tx] = A[row * k + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        int b_row = t * TILE_SIZE + ty;
        if (b_row < k && col < n) {
            Bs[ty][tx] = B[b_row * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
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
    printf("========== 分块GEMM演示 ==========\n");
    printf("矩阵尺寸: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("Tile尺寸: %d x %d\n\n", TILE_SIZE, TILE_SIZE);

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

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // -------------------- 基础分块版本 --------------------
    printf(">>> 基础分块GEMM\n");

    tiled_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    tiled_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float tiled_ms;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    bool correct = verify_result(h_C, h_ref, M, N);
    printf("结果验证: %s\n", correct ? "通过" : "失败");
    printf("执行时间: %.3f ms\n", tiled_ms);

    double flops = 2.0 * M * N * K;
    double gflops = flops / (tiled_ms * 1e-3) / 1e9;
    printf("性能: %.2f GFLOPS\n", gflops);

    // -------------------- Padding版本 --------------------
    printf("\n>>> Padding优化版本（避免Bank Conflict）\n");

    tiled_gemm_padded<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    tiled_gemm_padded<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float padded_ms;
    CUDA_CHECK(cudaEventElapsedTime(&padded_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    correct = verify_result(h_C, h_ref, M, N);
    printf("结果验证: %s\n", correct ? "通过" : "失败");
    printf("执行时间: %.3f ms\n", padded_ms);

    gflops = flops / (padded_ms * 1e-3) / 1e9;
    printf("性能: %.2f GFLOPS\n", gflops);

    // -------------------- 性能分析 --------------------
    printf("\n========== 分块优化分析 ==========\n");
    printf("分块优化原理:\n");
    printf("  1. 将矩阵分成 %d x %d 的小块\n", TILE_SIZE, TILE_SIZE);
    printf("  2. 每个Block处理一个Tile\n");
    printf("  3. Tile数据缓存到共享内存\n");
    printf("  4. 减少全局内存访问次数\n\n");

    printf("数据复用分析:\n");
    printf("  - 每个Tile加载: 2 * %d * %d = %d 个元素\n",
           TILE_SIZE, TILE_SIZE, 2 * TILE_SIZE * TILE_SIZE);
    printf("  - 每个Tile计算: %d * %d * %d = %d 次乘加\n",
           TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE * TILE_SIZE * TILE_SIZE);
    printf("  - 复用因子: %d\n", TILE_SIZE / 2);

    printf("\nBank Conflict优化:\n");
    printf("  - 基础版本: 可能存在Bank Conflict\n");
    printf("  - Padding版本: 每行增加1个元素，避免Bank Conflict\n");

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
