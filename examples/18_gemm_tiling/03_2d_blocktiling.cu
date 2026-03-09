/**
 * 03_2d_blocktiling.cu
 * 2D Block Tiling优化的GEMM
 *
 * 这个示例展示了2D Block Tiling优化：
 * - 每个线程计算一个THREAD_M x THREAD_N的小块
 * - A和B元素都在寄存器中复用
 * - 进一步减少共享内存访问
 *
 * 编译: nvcc -o 03_2d_blocktiling 03_2d_blocktiling.cu
 * 运行: ./03_2d_blocktiling
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

// 每个线程处理的M和N方向元素数
#define THREAD_M 4
#define THREAD_N 4

// 2D Block Tiling GEMM核函数
// 每个线程计算THREAD_M x THREAD_N个C元素
__global__ void gemm_2d_blocktiling(float* A, float* B, float* C, int m, int n, int k) {
    // 共享内存
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算线程负责的C矩阵区域起始位置
    int row_start = blockIdx.y * BLOCK_SIZE + ty * THREAD_M;
    int col_start = blockIdx.x * BLOCK_SIZE + tx * THREAD_N;

    // 寄存器缓存累加结果（THREAD_M x THREAD_N）
    float sum[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            sum[i][j] = 0.0f;
        }
    }

    int numTiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 协作加载A的Tile - 每个线程加载多个元素
        // blockDim是 (8, 8)，需要加载 32x32 的A tile
        #pragma unroll
        for (int load_row = 0; load_row < BLOCK_SIZE; load_row += blockDim.y) {
            #pragma unroll
            for (int load_col = 0; load_col < BLOCK_SIZE; load_col += blockDim.x) {
                int a_row = blockIdx.y * BLOCK_SIZE + ty + load_row;
                int a_col = t * BLOCK_SIZE + tx + load_col;
                if (a_row < m && a_col < k) {
                    As[ty + load_row][tx + load_col] = A[a_row * k + a_col];
                } else {
                    As[ty + load_row][tx + load_col] = 0.0f;
                }
            }
        }

        // 协作加载B的Tile - 每个线程加载多个元素
        #pragma unroll
        for (int load_row = 0; load_row < BLOCK_SIZE; load_row += blockDim.y) {
            #pragma unroll
            for (int load_col = 0; load_col < BLOCK_SIZE; load_col += blockDim.x) {
                int b_row = t * BLOCK_SIZE + ty + load_row;
                int b_col = blockIdx.x * BLOCK_SIZE + tx + load_col;
                if (b_row < k && b_col < n) {
                    Bs[ty + load_row][tx + load_col] = B[b_row * n + b_col];
                } else {
                    Bs[ty + load_row][tx + load_col] = 0.0f;
                }
            }
        }

        __syncthreads();

        // 计算：双重循环，利用寄存器缓存
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
            // 预取A和B的值到寄存器（关键优化！）
            float a_vals[THREAD_M];
            float b_vals[THREAD_N];

            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                a_vals[i] = As[ty * THREAD_M + i][kk];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                b_vals[j] = Bs[kk][tx * THREAD_N + j];
            }

            // 累加到寄存器
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    sum[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        __syncthreads();
    }

    // 写回结果
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int row = row_start + i;
            int col = col_start + j;
            if (row < m && col < n) {
                C[row * n + col] = sum[i][j];
            }
        }
    }
}

// 带Padding的版本
__global__ void gemm_2d_blocktiling_padded(float* A, float* B, float* C, int m, int n, int k) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = blockIdx.y * BLOCK_SIZE + ty * THREAD_M;
    int col_start = blockIdx.x * BLOCK_SIZE + tx * THREAD_N;

    float sum[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            sum[i][j] = 0.0f;
        }
    }

    int numTiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 协作加载A的Tile
        #pragma unroll
        for (int load_row = 0; load_row < BLOCK_SIZE; load_row += blockDim.y) {
            #pragma unroll
            for (int load_col = 0; load_col < BLOCK_SIZE; load_col += blockDim.x) {
                int a_row = blockIdx.y * BLOCK_SIZE + ty + load_row;
                int a_col = t * BLOCK_SIZE + tx + load_col;
                if (a_row < m && a_col < k) {
                    As[ty + load_row][tx + load_col] = A[a_row * k + a_col];
                } else {
                    As[ty + load_row][tx + load_col] = 0.0f;
                }
            }
        }

        // 协作加载B的Tile
        #pragma unroll
        for (int load_row = 0; load_row < BLOCK_SIZE; load_row += blockDim.y) {
            #pragma unroll
            for (int load_col = 0; load_col < BLOCK_SIZE; load_col += blockDim.x) {
                int b_row = t * BLOCK_SIZE + ty + load_row;
                int b_col = blockIdx.x * BLOCK_SIZE + tx + load_col;
                if (b_row < k && b_col < n) {
                    Bs[ty + load_row][tx + load_col] = B[b_row * n + b_col];
                } else {
                    Bs[ty + load_row][tx + load_col] = 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
            float a_vals[THREAD_M];
            float b_vals[THREAD_N];

            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                a_vals[i] = As[ty * THREAD_M + i][kk];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                b_vals[j] = Bs[kk][tx * THREAD_N + j];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    sum[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int row = row_start + i;
            int col = col_start + j;
            if (row < m && col < n) {
                C[row * n + col] = sum[i][j];
            }
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
    printf("========== 2D Block Tiling GEMM演示 ==========\n");
    printf("矩阵尺寸: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("Block尺寸: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("每线程处理: %d x %d = %d 个元素\n\n", THREAD_M, THREAD_N, THREAD_M * THREAD_N);

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

    // 计算Grid配置
    int blocks_m = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_n = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE / THREAD_N, BLOCK_SIZE / THREAD_M);
    dim3 gridDim(blocks_n, blocks_m);

    printf("Grid配置: (%d, %d), Block配置: (%d, %d)\n\n",
           blocks_n, blocks_m, BLOCK_SIZE / THREAD_N, BLOCK_SIZE / THREAD_M);

    // -------------------- 2D Block Tiling版本 --------------------
    printf(">>> 2D Block Tiling GEMM\n");

    gemm_2d_blocktiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    gemm_2d_blocktiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
    printf("\n>>> 2D Block Tiling GEMM (Padding优化)\n");

    gemm_2d_blocktiling_padded<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    gemm_2d_blocktiling_padded<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&tiling_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    correct = verify_result(h_C, h_ref, M, N);
    printf("结果验证: %s\n", correct ? "通过" : "失败");
    printf("执行时间: %.3f ms\n", tiling_ms);

    gflops = flops / (tiling_ms * 1e-3) / 1e9;
    printf("性能: %.2f GFLOPS\n", gflops);

    printf("\n========== 2D Block Tiling分析 ==========\n");
    printf("优化原理:\n");
    printf("  1. 每个线程计算 %d x %d = %d 个元素\n", THREAD_M, THREAD_N, THREAD_M * THREAD_N);
    printf("  2. A元素在N方向复用\n");
    printf("  3. B元素在M方向复用\n");
    printf("  4. 最大化寄存器利用\n");

    printf("\n访存对比:\n");
    printf("  SMEM Caching: 每元素约 64 次SMEM读取\n");
    printf("  1D Tiling: 每元素约 %.1f 次SMEM读取\n", (float)BLOCK_SIZE / THREAD_M);
    printf("  2D Tiling: 每元素约 %.2f 次SMEM读取\n", (float)BLOCK_SIZE / THREAD_M);

    // 清理
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
