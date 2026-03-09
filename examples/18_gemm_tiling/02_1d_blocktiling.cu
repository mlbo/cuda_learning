/**
 * 02_1d_blocktiling.cu
 * 1D Block Tiling优化的GEMM
 *
 * 这个示例展示了1D Block Tiling优化：
 * - 每个线程计算M方向的多个元素
 * - B矩阵元素在寄存器中复用
 * - 减少共享内存访问次数
 *
 * 编译: nvcc -o 02_1d_blocktiling 02_1d_blocktiling.cu
 * 运行: ./02_1d_blocktiling
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
#define BLOCK_SIZE 32

// 每个线程在M方向处理的元素数
#define THREAD_M 4

// 1D Block Tiling GEMM核函数
// 每个线程计算THREAD_M个C元素（同列不同行）
__global__ void gemm_1d_blocktiling(float* A, float* B, float* C, int m, int n, int k) {
    // 共享内存
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每个线程处理的起始行
    int row_start = blockIdx.y * BLOCK_SIZE + ty * THREAD_M;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    // 寄存器缓存累加结果（每个线程THREAD_M个）
    float sum[THREAD_M];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        sum[i] = 0.0f;
    }

    int numTiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 协作加载B的Tile - 每个线程加载多个元素
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i += blockDim.y) {
            int b_row = t * BLOCK_SIZE + ty + i;
            if (b_row < k && col < n) {
                Bs[ty + i][tx] = B[b_row * n + col];
            } else {
                Bs[ty + i][tx] = 0.0f;
            }
        }

        // 加载A的Tile（需要加载THREAD_M行）
        #pragma unroll
        for (int i = 0; i < THREAD_M; i++) {
            int row = row_start + i;
            int a_col = t * BLOCK_SIZE + tx;
            if (row < m && a_col < k) {
                As[ty * THREAD_M + i][tx] = A[row * k + a_col];
            } else {
                As[ty * THREAD_M + i][tx] = 0.0f;
            }
        }

        __syncthreads();

        // 计算：B元素缓存到寄存器，在M方向复用
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            // 预取B元素到寄存器（关键优化！）
            float b_val = Bs[i][tx];

            // 对每个M方向的元素进行累加
            #pragma unroll
            for (int j = 0; j < THREAD_M; j++) {
                sum[j] += As[ty * THREAD_M + j][i] * b_val;
            }
        }

        __syncthreads();
    }

    // 写回结果
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        int row = row_start + i;
        if (row < m && col < n) {
            C[row * n + col] = sum[i];
        }
    }
}

// 带Padding的版本
__global__ void gemm_1d_blocktiling_padded(float* A, float* B, float* C, int m, int n, int k) {
    __shared__ float As[BLOCK_SIZE * THREAD_M][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = blockIdx.y * BLOCK_SIZE + ty * THREAD_M;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum[THREAD_M];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        sum[i] = 0.0f;
    }

    int numTiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 协作加载B的Tile - 每个线程加载多个元素
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i += blockDim.y) {
            int b_row = t * BLOCK_SIZE + ty + i;
            if (b_row < k && col < n) {
                Bs[ty + i][tx] = B[b_row * n + col];
            } else {
                Bs[ty + i][tx] = 0.0f;
            }
        }

        #pragma unroll
        for (int i = 0; i < THREAD_M; i++) {
            int row = row_start + i;
            int a_col = t * BLOCK_SIZE + tx;
            if (row < m && a_col < k) {
                As[ty * THREAD_M + i][tx] = A[row * k + a_col];
            } else {
                As[ty * THREAD_M + i][tx] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float b_val = Bs[i][tx];
            #pragma unroll
            for (int j = 0; j < THREAD_M; j++) {
                sum[j] += As[ty * THREAD_M + j][i] * b_val;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        int row = row_start + i;
        if (row < m && col < n) {
            C[row * n + col] = sum[i];
        }
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
    printf("========== 1D Block Tiling GEMM演示 ==========\n");
    printf("矩阵尺寸: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("Block尺寸: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("每线程处理: %d 个元素 (M方向)\n\n", THREAD_M);

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

    printf(">>> CPU参考实现\n");
    cpu_gemm(h_A, h_B, h_ref, M, N, K);
    printf("CPU计算完成\n\n");

    // 计算Grid配置（考虑THREAD_M）
    int blocks_m = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_n = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE / THREAD_M);
    dim3 gridDim(blocks_n, blocks_m);

    printf("Grid配置: (%d, %d), Block配置: (%d, %d)\n\n",
           blocks_n, blocks_m, BLOCK_SIZE, BLOCK_SIZE / THREAD_M);

    // -------------------- 1D Block Tiling版本 --------------------
    printf(">>> 1D Block Tiling GEMM\n");

    gemm_1d_blocktiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    gemm_1d_blocktiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float tiling_ms;
    CUDA_CHECK(cudaEventElapsedTime(&tiling_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    bool correct = verify_result(h_C, h_ref, M, N);
    printf("结果验证: %s\n", correct ? "通过" : "失败");
    printf("执行时间: %.3f ms\n", tiling_ms);

    double flops = 2.0 * M * N * K;
    double gflops = flops / (tiling_ms * 1e-3) / 1e9;
    printf("性能: %.2f GFLOPS\n", gflops);

    // -------------------- Padding版本 --------------------
    printf("\n>>> 1D Block Tiling GEMM (Padding优化)\n");

    gemm_1d_blocktiling_padded<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    gemm_1d_blocktiling_padded<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
    printf("\n========== 1D Block Tiling分析 ==========\n");
    printf("优化原理:\n");
    printf("  1. 每个线程计算THREAD_M=%d个元素\n", THREAD_M);
    printf("  2. B元素在寄存器中复用\n");
    printf("  3. 减少共享内存访问次数\n\n");

    printf("访存对比:\n");
    printf("  SMEM Caching: 每元素约 %d 次SMEM读取\n", 2 * BLOCK_SIZE);
    printf("  1D Tiling: 每元素约 %.1f 次SMEM读取\n",
           (float)(BLOCK_SIZE + BLOCK_SIZE) / THREAD_M);
    printf("  减少约 %.0f%% 的共享内存访问\n",
           (1.0 - 1.0 / THREAD_M) * 100);

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
