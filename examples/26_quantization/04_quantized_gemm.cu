/**
 * 第26章示例4：量化GEMM实现
 *
 * 演示内容：
 * 1. INT8 GEMM原理
 * 2. 使用cuBLAS INT8 GEMM
 * 3. 完整量化GEMM流程
 * 4. 精度验证
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS错误 %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while(0)

/**
 * INT8 GEMM原理说明
 */
void int8_gemm_theory() {
    printf("========================================\n");
    printf("INT8 GEMM原理\n");
    printf("========================================\n\n");

    printf("量化GEMM流程:\n\n");

    printf("1. 量化阶段:\n");
    printf("   A_fp32 --[scale_A]--> A_int8\n");
    printf("   B_fp32 --[scale_B]--> B_int8\n\n");

    printf("2. INT8矩阵乘法:\n");
    printf("   C_int32 = A_int8 x B_int8\n");
    printf("   (使用INT32累加避免溢出)\n\n");

    printf("3. 反量化阶段:\n");
    printf("   C_fp32 = C_int32 * scale_A * scale_B\n\n");

    printf("关键点:\n");
    printf("  - INT8 x INT8 范围: [-128, 127]\n");
    printf("  - 乘积范围: [-16384, 16129]\n");
    printf("  - 多次累加需要INT32\n");
    printf("  - Tensor Core INT8: Volta+\n\n");
}

/**
 * 对称量化核函数
 */
__global__ void quantize_kernel(const float* src, int8_t* dst, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float q = roundf(src[idx] / scale);
        q = fmaxf(fminf(q, 127.0f), -128.0f);
        dst[idx] = (int8_t)q;
    }
}

/**
 * 反量化核函数
 */
__global__ void dequantize_kernel(const int32_t* src, float* dst, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (float)src[idx] * scale;
    }
}

/**
 * 计算scale
 */
float compute_scale(const float* data, int n) {
    float abs_max = 0.0f;
    for (int i = 0; i < n; i++) {
        abs_max = fmaxf(abs_max, fabsf(data[i]));
    }
    return abs_max / 127.0f;
}

/**
 * FP32 GEMM (cuBLAS)
 */
void fp32_gemm(cublasHandle_t handle, int M, int N, int K,
               const float* A, const float* B, float* C) {
    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K, &alpha, A, M, B, K, &beta, C, M));
}

/**
 * INT8 GEMM (cuBLAS)
 */
void int8_gemm(cublasHandle_t handle, int M, int N, int K,
               const int8_t* A, const int8_t* B, int32_t* C,
               int lda, int ldb, int ldc) {
    int32_t alpha = 1;
    int32_t beta = 0;

    // INT8 GEMM使用cublasGemmEx
    CHECK_CUBLAS(cublasGemmEx(handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               M, N, K,
                               &alpha,
                               A, CUDA_R_8I, lda,
                               B, CUDA_R_8I, ldb,
                               &beta,
                               C, CUDA_R_32I, ldc,
                               CUBLAS_COMPUTE_32I,
                               CUBLAS_GEMM_DEFAULT));
}

/**
 * 完整的量化GEMM示例
 */
void quantized_gemm_example() {
    printf("\n========================================\n");
    printf("完整量化GEMM示例\n");
    printf("========================================\n\n");

    // 矩阵大小
    const int M = 512;
    const int N = 512;
    const int K = 512;

    printf("矩阵大小: M=%d, N=%d, K=%d\n", M, N, K);

    // 创建cuBLAS句柄
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 分配FP32内存
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    CHECK_CUDA(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_fp32, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));

    // 分配INT8内存
    int8_t *d_A_int8, *d_B_int8;
    int32_t *d_C_int32;
    CHECK_CUDA(cudaMalloc(&d_A_int8, M * K * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_B_int8, K * N * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_C_int32, M * N * sizeof(int32_t)));

    // 初始化矩阵
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));

    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 200 - 100) / 1000.0f;  // -0.1到0.1
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 200 - 100) / 1000.0f;
    }

    CHECK_CUDA(cudaMemcpy(d_A_fp32, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp32, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 计算scale
    float scale_A = compute_scale(h_A, M * K);
    float scale_B = compute_scale(h_B, K * N);
    float scale_C = scale_A * scale_B;

    printf("量化参数:\n");
    printf("  Scale A: %.6f\n", scale_A);
    printf("  Scale B: %.6f\n", scale_B);
    printf("  Scale C: %.6f\n", scale_C);

    // 创建事件计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 10;

    // === FP32 GEMM ===
    printf("\n执行FP32 GEMM...\n");

    // 预热
    for (int i = 0; i < 3; i++) {
        fp32_gemm(handle, M, N, K, d_A_fp32, d_B_fp32, d_C_fp32);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        fp32_gemm(handle, M, N, K, d_A_fp32, d_B_fp32, d_C_fp32);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_fp32;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp32, start, stop));
    ms_fp32 /= iterations;

    double tflops_fp32 = 2.0 * M * N * K / (ms_fp32 * 1e9);
    printf("  时间: %.3f ms\n", ms_fp32);
    printf("  性能: %.2f TFLOPS\n", tflops_fp32);

    // === INT8 量化GEMM ===
    printf("\n执行INT8量化GEMM...\n");

    int block_size = 256;

    // 量化A和B
    quantize_kernel<<<(M * K + block_size - 1) / block_size, block_size>>>(
        d_A_fp32, d_A_int8, M * K, scale_A);
    quantize_kernel<<<(K * N + block_size - 1) / block_size, block_size>>>(
        d_B_fp32, d_B_int8, K * N, scale_B);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 预热
    for (int i = 0; i < 3; i++) {
        int8_gemm(handle, M, N, K, d_A_int8, d_B_int8, d_C_int32, M, K, M);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        int8_gemm(handle, M, N, K, d_A_int8, d_B_int8, d_C_int32, M, K, M);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_int8;
    CHECK_CUDA(cudaEventElapsedTime(&ms_int8, start, stop));
    ms_int8 /= iterations;

    double tflops_int8 = 2.0 * M * N * K / (ms_int8 * 1e9);
    printf("  INT8 GEMM时间: %.3f ms\n", ms_int8);
    printf("  性能: %.2f TFLOPS\n", tflops_int8);

    // 反量化结果
    dequantize_kernel<<<(M * N + block_size - 1) / block_size, block_size>>>(
        d_C_int32, d_C_fp32, M * N, scale_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("\n加速比: %.2fx\n", ms_fp32 / ms_int8);

    // 验证精度
    printf("\n精度验证:\n");

    // 保存FP32结果
    float* h_C_fp32 = (float*)malloc(M * N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_C_fp32, d_C_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 重新计算FP32 GEMM作为参考
    CHECK_CUDA(cudaMemset(d_C_fp32, 0, M * N * sizeof(float)));
    fp32_gemm(handle, M, N, K, d_A_fp32, d_B_fp32, d_C_fp32);

    float* h_C_ref = (float*)malloc(M * N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 计算误差
    float max_error = 0, avg_error = 0;
    for (int i = 0; i < M * N; i++) {
        float error = fabsf(h_C_ref[i] - h_C_fp32[i]);
        max_error = fmaxf(max_error, error);
        avg_error += error;
    }
    avg_error /= (M * N);

    printf("  最大误差: %.6f\n", max_error);
    printf("  平均误差: %.6f\n", avg_error);

    // 清理
    free(h_A);
    free(h_B);
    free(h_C_fp32);
    free(h_C_ref);
    CHECK_CUDA(cudaFree(d_A_fp32));
    CHECK_CUDA(cudaFree(d_B_fp32));
    CHECK_CUDA(cudaFree(d_C_fp32));
    CHECK_CUDA(cudaFree(d_A_int8));
    CHECK_CUDA(cudaFree(d_B_int8));
    CHECK_CUDA(cudaFree(d_C_int32));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
}

/**
 * 不同精度的GEMM性能对比
 */
void gemm_precision_comparison() {
    printf("\n========================================\n");
    printf("不同精度GEMM性能对比\n");
    printf("========================================\n\n");

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    printf("矩阵大小: M=N=K=%d\n", M);
    printf("计算量: %.2f GFLOPS\n", 2.0 * M * N * K / 1e9);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 分配内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    half *d_A_h, *d_B_h, *d_C_h;
    CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(half)));

    // 初始化
    float* h_init = (float*)malloc(M * K * sizeof(float));
    for (int i = 0; i < M * K; i++) {
        h_init[i] = 0.01f;
    }
    CHECK_CUDA(cudaMemcpy(d_A, h_init, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_init, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // FP16数据
    for (int i = 0; i < M * K; i++) {
        half h = __float2half(h_init[i]);
        CHECK_CUDA(cudaMemcpy(d_A_h + i, &h, sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B_h + i, &h, sizeof(half), cudaMemcpyHostToDevice));
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 20;

    printf("\n%-20s | %-12s | %-12s | %-10s\n", "精度", "时间(ms)", "TFLOPS", "加速比");
    printf("----------------------------------------------------------------\n");

    float alpha = 1.0f, beta = 0.0f;
    half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);

    // FP32标准
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    for (int i = 0; i < 3; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                                  &alpha, d_A, M, d_B, K, &beta, d_C, M));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                                  &alpha, d_A, M, d_B, K, &beta, d_C, M));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_fp32;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp32, start, stop));
    ms_fp32 /= iterations;
    double tflops_fp32 = 2.0 * M * N * K / (ms_fp32 * 1e9);
    printf("%-20s | %-10.3f | %-10.2f | %-10s\n", "FP32", ms_fp32, tflops_fp32, "1.00x");

    // TF32 (如果支持)
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                                  &alpha, d_A, M, d_B, K, &beta, d_C, M));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_tf32;
    CHECK_CUDA(cudaEventElapsedTime(&ms_tf32, start, stop));
    ms_tf32 /= iterations;
    double tflops_tf32 = 2.0 * M * N * K / (ms_tf32 * 1e9);
    printf("%-20s | %-10.3f | %-10.2f | %-10.2fx\n", "TF32", ms_tf32, tflops_tf32, ms_fp32 / ms_tf32);

    // FP16 Tensor Core
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                                  &alpha_h, d_A_h, M, d_B_h, K, &beta_h, d_C_h, M));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_fp16;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp16, start, stop));
    ms_fp16 /= iterations;
    double tflops_fp16 = 2.0 * M * N * K / (ms_fp16 * 1e9);
    printf("%-20s | %-10.3f | %-10.2f | %-10.2fx\n", "FP16 Tensor Core", ms_fp16, tflops_fp16, ms_fp32 / ms_fp16);

    // 清理
    free(h_init);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_h));
    CHECK_CUDA(cudaFree(d_B_h));
    CHECK_CUDA(cudaFree(d_C_h));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
}

/**
 * 量化最佳实践
 */
void quantization_best_practices() {
    printf("\n========================================\n");
    printf("量化最佳实践\n");
    printf("========================================\n\n");

    printf("1. 量化方法选择:\n");
    printf("   - 对称量化: 简单高效，适合权重\n");
    printf("   - 非对称量化: 精度更高，适合激活\n\n");

    printf("2. 量化粒度选择:\n");
    printf("   - 权重: 通道级量化\n");
    printf("   - 激活: 张量级量化\n");
    printf("   - 特殊场景: 组级量化\n\n");

    printf("3. 精度保护:\n");
    printf("   - 使用FP32累加\n");
    printf("   - 量化感知训练(QAT)\n");
    printf("   - 混合精度推理\n\n");

    printf("4. 硬件考虑:\n");
    printf("   - INT8 Tensor Core: Volta+\n");
    printf("   - FP16 Tensor Core: Volta+\n");
    printf("   - BF16 Tensor Core: Ampere+\n\n");

    printf("5. 调试技巧:\n");
    printf("   - 逐层比较FP32和INT8输出\n");
    printf("   - 检查量化参数是否合理\n");
    printf("   - 关注异常值和溢出\n");
}

int main() {
    printf("=============================================\n");
    printf("  第26章示例4：量化GEMM实现\n");
    printf("=============================================\n");

    // 检查设备
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("\n当前设备: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // INT8 GEMM原理
    int8_gemm_theory();

    // 完整量化GEMM示例
    quantized_gemm_example();

    // 不同精度对比
    gemm_precision_comparison();

    // 最佳实践
    quantization_best_practices();

    printf("\n示例完成！\n");
    return 0;
}