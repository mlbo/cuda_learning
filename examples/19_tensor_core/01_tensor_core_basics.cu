/**
 * 01_tensor_core_basics.cu
 * Tensor Core基础：检查设备支持、性能对比
 *
 * 编译: nvcc -arch=sm_70 -o 01_tensor_core_basics 01_tensor_core_basics.cu
 * 运行: ./01_tensor_core_basics
 */

#include <stdio.h>
#include <cuda_runtime.h>

// 检查Tensor Core支持
void check_tensor_core_support() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("========== Tensor Core支持检查 ==========\n");
    printf("设备名称: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // Tensor Core从Volta (SM 7.0)开始支持
    bool tensor_core_supported = (prop.major >= 7);

    printf("Tensor Core支持: %s\n", tensor_core_supported ? "是" : "否");

    if (tensor_core_supported) {
        printf("\n支持的Tensor Core类型:\n");
        if (prop.major >= 7) printf("  - FP16/FP32 MMA (Volta+)\n");
        if (prop.major >= 7 && prop.minor >= 5) printf("  - INT8/INT32 MMA (Turing+)\n");
        if (prop.major >= 8) {
            printf("  - BF16/FP32 MMA (Ampere+)\n");
            printf("  - TF32 MMA (Ampere+)\n");
        }
        if (prop.major >= 9) printf("  - FP8 MMA (Hopper+)\n");
    }

    printf("==========================================\n\n");
}

// 使用CUDA Core的矩阵乘法（基准）
__global__ void matmul_cuda_core(float* A, float* B, float* C, int M, int N, int K) {
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

// 初始化矩阵
void init_matrix(float* mat, int rows, int cols, float value = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = value;
    }
}

// 性能对比测试
void performance_comparison() {
    int M = 1024, N = 1024, K = 1024;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(bytes_A);
    h_B = (float*)malloc(bytes_B);
    h_C = (float*)malloc(bytes_C);

    init_matrix(h_A, M, K, 1.0f);
    init_matrix(h_B, K, N, 1.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    // 创建事件计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("========== 性能对比 (CUDA Core vs Tensor Core) ==========\n");
    printf("矩阵大小: %d x %d x %d\n\n", M, N, K);

    // 测试CUDA Core版本
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matmul_cuda_core<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_cuda_core;
    cudaEventElapsedTime(&ms_cuda_core, start, stop);
    ms_cuda_core /= 10;

    printf("CUDA Core GEMM:\n");
    printf("  平均时间: %.4f ms\n", ms_cuda_core);
    printf("  性能: %.2f GFLOPS\n",
           (2.0 * M * N * K) / (ms_cuda_core * 1e6));

    // 注意：实际的Tensor Core版本需要使用WMMA或CUTLASS
    // 这里只是展示理论加速比
    printf("\nTensor Core GEMM (理论):\n");
    printf("  预期加速: 4-16x (取决于精度类型)\n");
    printf("  FP16 Tensor Core 比 FP32 CUDA Core 快约 8x\n");
    printf("  TF32 Tensor Core 比 FP32 CUDA Core 快约 8x\n");

    printf("========================================================\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    // 检查Tensor Core支持
    check_tensor_core_support();

    // 性能对比测试
    performance_comparison();

    printf("\n提示: 完整的Tensor Core GEMM实现请参考 02_wmma_gemm.cu\n");

    return 0;
}