/**
 * 02_wmma_gemm.cu
 * 使用WMMA API实现Tensor Core矩阵乘法
 *
 * WMMA (Warp Matrix Multiply Accumulate) 是Tensor Core的高层API
 * 每个warp协作计算一个矩阵块
 *
 * 编译: nvcc -arch=sm_70 -o 02_wmma_gemm 02_wmma_gemm.cu
 * 运行: ./02_wmma_gemm
 */

#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>  // WMMA头文件

using namespace nvcuda;  // WMMA命名空间

// 矩阵块大小（WMMA固定为16x16）
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 使用WMMA的FP16矩阵乘法
__global__ void wmma_gemm(half* A, half* B, float* C, int M, int N, int K) {
    // 计算这个warp负责的输出块位置
    constexpr int kWarpSize = 32;
    int warpsPerBlockN = blockDim.x / kWarpSize;
    int warpM = blockIdx.y * blockDim.y + threadIdx.y;
    int warpN = blockIdx.x * warpsPerBlockN + threadIdx.x / kWarpSize;

    // 边界检查
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    // 声明WMMA片段（fragment）
    // 这些片段会自动分配到线程的寄存器中
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器为0
    wmma::fill_fragment(c_frag, 0.0f);

    // 沿K维度迭代
    for (int k = 0; k < K; k += WMMA_K) {
        // 加载A和B的矩阵块
        // 每个warp中的一个线程负责加载不同的元素
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        // 检查边界
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // 执行矩阵乘累加（Tensor Core操作）
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

// FP32矩阵乘法（CUDA Core基准）
__global__ void matmul_fp32(float* A, float* B, float* C, int M, int N, int K) {
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

int main() {
    // 使用较小的矩阵进行演示
    int M = 512, N = 512, K = 512;

    printf("========== WMMA Tensor Core GEMM ==========\n");
    printf("矩阵大小: %d x %d x %d\n\n", M, N, K);

    // 分配主机内存
    half* h_A = (half*)malloc(M * K * sizeof(half));
    half* h_B = (half*)malloc(K * N * sizeof(half));
    float* h_C_wmma = (float*)malloc(M * N * sizeof(float));

    // 初始化矩阵
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(1.0f);

    // 分配设备内存
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // WMMA配置：每个block有多个warp
    // blockDim.x 应该是warpSize的倍数
    dim3 blockDim(128, 4);  // 4 warps in x, 4 warps in y
    constexpr int hostWarpSize = 32;
    dim3 gridDim((N + WMMA_N * (blockDim.x / hostWarpSize) - 1) / (WMMA_N * (blockDim.x / hostWarpSize)),
                 (M + WMMA_M * blockDim.y - 1) / (WMMA_M * blockDim.y));

    // 创建事件计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    wmma_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 正式测试
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        wmma_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_wmma;
    cudaEventElapsedTime(&ms_wmma, start, stop);
    ms_wmma /= 10;

    // 复制结果
    cudaMemcpy(h_C_wmma, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果（期望值应该是K，因为每个元素是1.0）
    float expected = (float)K;
    bool correct = true;
    for (int i = 0; i < M * N && correct; i++) {
        if (std::fabs(h_C_wmma[i] - expected) > 0.1f) {
            correct = false;
            printf("错误: C[%d] = %f, 期望 %f\n", i, h_C_wmma[i], expected);
        }
    }

    printf("WMMA Tensor Core GEMM:\n");
    printf("  时间: %.4f ms\n", ms_wmma);
    printf("  性能: %.2f GFLOPS\n",
           (2.0 * M * N * K) / (ms_wmma * 1e6));
    printf("  结果正确: %s\n\n", correct ? "是" : "否");

    printf("============================================\n");

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_wmma);

    return 0;
}
