/**
 * @file 04_cutlass_gemm.cu
 * @brief CUTLASS GEMM示例
 *
 * 本示例展示：
 * 1. 基本CUTLASS GEMM配置
 * 2. FP16 Tensor Core GEMM
 * 3. 性能测量
 *
 * 注意：需要安装CUTLASS库
 * 编译: nvcc -I/path/to/cutlass -o cutlass_gemm 04_cutlass_gemm.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// CUTLASS头文件
// 如果CUTLASS未安装，将使用简化版本
#ifdef CUTLASS_ENABLED
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// 简单GEMM实现（用于演示，当CUTLASS不可用时）
// ============================================================================
__global__ void simple_gemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K,
                                   float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void simple_gemm(const float* d_A, const float* d_B, float* d_C,
                 int M, int N, int K, float alpha, float beta) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    simple_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

// ============================================================================
// 使用Tensor Core的GEMM (WMMA)
// ============================================================================
#include <mma.h>

using namespace nvcuda;

__global__ void wmma_gemm_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K,
                                 float alpha, float beta) {
    // WMMA矩阵分块大小
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    // 声明WMMA片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);

    // 循环计算
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // 存储结果
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

void wmma_gemm(const half* d_A, const half* d_B, float* d_C,
               int M, int N, int K, float alpha, float beta) {
    // 每个warp处理一个16x16输出分块
    dim3 block(32, 1);  // 一个warp
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    wmma_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

// ============================================================================
// 性能测试
// ============================================================================
void benchmark_gemm(const char* name,
                    void (*gemm_func)(const float*, const float*, float*, int, int, int, float, float),
                    const float* d_A, const float* d_B, float* d_C,
                    int M, int N, int K, float alpha, float beta,
                    int warmup = 5, int repeat = 20) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    for (int i = 0; i < warmup; i++) {
        gemm_func(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        gemm_func(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeat;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e6);

    printf("%-20s: %.4f ms, %.2f GFLOPS\n", name, ms, gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void benchmark_wmma_gemm(const half* d_A, const half* d_B, float* d_C,
                         int M, int N, int K, float alpha, float beta,
                         int warmup = 5, int repeat = 20) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    for (int i = 0; i < warmup; i++) {
        wmma_gemm(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        wmma_gemm(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeat;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e6);

    printf("%-20s: %.4f ms, %.2f GFLOPS\n", "WMMA (Tensor Core)", ms, gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    float alpha = 1.0f, beta = 0.0f;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("===== CUTLASS/WMMA GEMM示例 =====\n\n");
    printf("矩阵尺寸: A(%dx%d), B(%dx%d), C(%dx%d)\n\n", M, K, K, N, M, N);

    // 分配主机内存
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));

    // 初始化
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    printf("=== GEMM性能对比 ===\n");
    benchmark_gemm("Simple CUDA Kernel", simple_gemm, d_A, d_B, d_C, M, N, K, alpha, beta);

    // FP16 + WMMA (Tensor Core)
    half *d_A16, *d_B16;
    CUDA_CHECK(cudaMalloc(&d_A16, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B16, K * N * sizeof(half)));

    // 转换为FP16
    half* h_A16 = (half*)malloc(M * K * sizeof(half));
    half* h_B16 = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) h_A16[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++) h_B16[i] = __float2half(h_B[i]);

    CUDA_CHECK(cudaMemcpy(d_A16, h_A16, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B16, h_B16, K * N * sizeof(half), cudaMemcpyHostToDevice));

    benchmark_wmma_gemm(d_A16, d_B16, d_C, M, N, K, alpha, beta);

    // CUTLASS示例（如果可用）
#ifdef CUTLASS_ENABLED
    printf("\n=== CUTLASS GEMM ===\n");

    using GemmFP32 = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80
    >;

    GemmFP32 gemm_op;
    cutlass::Status status = gemm_op({
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_C, N},
        {alpha, beta}
    });

    if (status == cutlass::Status::kSuccess) {
        printf("CUTLASS FP32 GEMM成功\n");
    }
#else
    printf("\n注意: CUTLASS未安装，跳过CUTLASS示例\n");
    printf("如需使用CUTLASS，请从 https://github.com/NVIDIA/cutlass 下载\n");
#endif

    // 清理
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_A16);
    free(h_B16);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A16));
    CUDA_CHECK(cudaFree(d_B16));

    printf("\n=== CUTLASS使用建议 ===\n");
    printf("1. CUTLASS提供高度优化的GEMM实现\n");
    printf("2. 支持多种精度: FP64/FP32/FP16/BF16/INT8\n");
    printf("3. 可自定义: 分块大小、流水线深度等\n");
    printf("4. 自动利用Tensor Core\n");
    printf("5. 适合需要定制GEMM的场景\n");

    return 0;
}