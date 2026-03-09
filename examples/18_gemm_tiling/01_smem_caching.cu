/**
 * 01_smem_caching.cu
 * 共享内存缓存优化的GEMM
 *
 * 这个示例展示了如何使用共享内存缓存Tile数据：
 * - 加载Tile到共享内存
 * - 从共享内存进行计算
 * - 减少全局内存访问
 *
 * 编译: nvcc -o 01_smem_caching 01_smem_caching.cu
 * 运行: ./01_smem_caching
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>

// 矩阵尺寸
#define M 1024
#define N 1024
#define K 1024

// Block/Tile尺寸
#define BLOCK_SIZE 32

// SMEM Caching GEMM核函数
// 每个线程计算C中的一个元素
// 使用共享内存缓存A和B的Tile
__global__ void smem_caching_gemm(float* A, float* B, float* C, int m, int n, int k) {
    // 共享内存缓存Tile
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // 沿K方向滑动Tile
    int numTiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 协作加载A的Tile到共享内存
        int a_col = t * BLOCK_SIZE + tx;
        if (row < m && a_col < k) {
            As[ty][tx] = A[row * k + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // 协作加载B的Tile到共享内存
        int b_row = t * BLOCK_SIZE + ty;
        if (b_row < k && col < n) {
            Bs[ty][tx] = B[b_row * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 同步：确保所有线程都加载完成
        __syncthreads();

        // 从共享内存计算Tile内的部分结果
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        // 同步：确保所有线程都计算完成
        __syncthreads();
    }

    // 写回结果
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// 带Padding避免Bank Conflict的版本
__global__ void smem_caching_gemm_padded(float* A, float* B, float* C, int m, int n, int k) {
    // Padding: 每行增加1个元素
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * BLOCK_SIZE + tx;
        if (row < m && a_col < k) {
            As[ty][tx] = A[row * k + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        int b_row = t * BLOCK_SIZE + ty;
        if (b_row < k && col < n) {
            Bs[ty][tx] = B[b_row * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
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
    printf("========== SMEM Caching GEMM演示 ==========\n");
    printf("矩阵尺寸: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("Block尺寸: %d x %d\n\n", BLOCK_SIZE, BLOCK_SIZE);

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

    // -------------------- 基础版本 --------------------
    printf(">>> SMEM Caching GEMM (基础版本)\n");

    smem_caching_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    smem_caching_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float basic_ms;
    CUDA_CHECK(cudaEventElapsedTime(&basic_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    bool correct = verify_result(h_C, h_ref, M, N);
    printf("结果验证: %s\n", correct ? "通过" : "失败");
    printf("执行时间: %.3f ms\n", basic_ms);

    double flops = 2.0 * M * N * K;
    double gflops = flops / (basic_ms * 1e-3) / 1e9;
    printf("性能: %.2f GFLOPS\n", gflops);

    // -------------------- Padding版本 --------------------
    printf("\n>>> SMEM Caching GEMM (Padding优化)\n");

    smem_caching_gemm_padded<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    smem_caching_gemm_padded<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
    printf("\n========== SMEM Caching分析 ==========\n");
    printf("访存分析:\n");
    printf("  - 每个线程计算C中1个元素\n");
    printf("  - 每个元素需要K次乘加\n");
    printf("  - 从共享内存读取As和Bs\n");
    printf("  - 每个元素约需要 %d 次共享内存读取\n", 2 * BLOCK_SIZE);
    printf("  - 实际每次迭代: %d + %d = %d 次SMEM读取\n",
           BLOCK_SIZE, BLOCK_SIZE, 2 * BLOCK_SIZE);

    printf("\nPadding优化效果:\n");
    printf("  - 避免Bank Conflict\n");
    printf("  - 提高共享内存访问效率\n");

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
