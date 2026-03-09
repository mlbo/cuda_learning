/**
 * 04_warptiling.cu
 * Warp Tiling优化的GEMM
 *
 * 这个示例展示了Warp Tiling优化：
 * - 分层分块：Block -> Warp -> Thread
 * - 充分利用GPU层次结构
 * - 最大化数据复用
 *
 * 编译: nvcc -o 04_warptiling 04_warptiling.cu
 * 运行: ./04_warptiling
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>

// 矩阵尺寸
#define M 1024
#define N 1024
#define K 1024

// 分块参数
#define BLOCK_SIZE 32
#define WARP_SIZE 32

// Warp Tiling GEMM核函数 - 简化版本
// 每个warp处理一个 BLOCK_SIZE x BLOCK_SIZE 的tile
__global__ void gemm_warptiling(float* A, float* B, float* C, int m, int n, int k) {
    // 共享内存
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 全局坐标
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // 累加器
    float sum = 0.0f;

    int numTiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 协作加载A的Tile
        int a_row = row;
        int a_col = t * BLOCK_SIZE + tx;
        if (a_row < m && a_col < k) {
            As[ty][tx] = A[a_row * k + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // 协作加载B的Tile
        int b_row = t * BLOCK_SIZE + ty;
        int b_col = col;
        if (b_row < k && b_col < n) {
            Bs[ty][tx] = B[b_row * n + b_col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 计算
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
            sum += As[ty][kk] * Bs[kk][tx];
        }

        __syncthreads();
    }

    // 写回结果
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
            printf("验证失败: C[%d] = %.2f, ref = %.2f\n", i, C[i], ref[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("========== Warp Tiling GEMM演示 ==========\n");
    printf("矩阵尺寸: A(%dx%d) * B(%dx%d) = C(%dx%d)\n\n", M, K, K, N, M, N);

    printf("分层配置:\n");
    printf("  Block Tile: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("  Warp Size: %d\n\n", WARP_SIZE);

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

    // Block配置
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Grid配置: (%d, %d), Block配置: (%d, %d)\n\n",
           (N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
           BLOCK_SIZE, BLOCK_SIZE);

    // -------------------- Warp Tiling版本 --------------------
    printf(">>> Warp Tiling GEMM\n");

    gemm_warptiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    gemm_warptiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float warptiling_ms;
    CUDA_CHECK(cudaEventElapsedTime(&warptiling_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    bool correct = verify_result(h_C, h_ref, M, N);
    printf("结果验证: %s\n", correct ? "通过" : "失败");
    printf("执行时间: %.3f ms\n", warptiling_ms);

    double flops = 2.0 * M * N * K;
    double gflops = flops / (warptiling_ms * 1e-3) / 1e9;
    printf("性能: %.2f GFLOPS\n", gflops);

    printf("\n========== Warp Tiling分析 ==========\n");
    printf("优化原理:\n");
    printf("  1. 每个Block处理 %d x %d 的Tile\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("  2. 共享内存缓存A和B的Tile\n");
    printf("  3. 每个线程计算一个C元素\n");
    printf("  4. 利用Warp的SIMD执行\n");

    printf("\n内存层次映射:\n");
    printf("  全局内存 -> 共享内存 (Tile)\n");
    printf("  共享内存 -> 寄存器 (计算)\n");

    printf("\n进一步优化方向:\n");
    printf("  1. 使用Tensor Core加速矩阵乘法\n");
    printf("  2. 双缓冲隐藏内存延迟\n");
    printf("  3. 向量化访存 (float4)\n");
    printf("  4. 自动调优参数\n");

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

    printf("\n示例完成!\n");
    return 0;
}
