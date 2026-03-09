/**
 * @file 01_cublas_gemm.cu
 * @brief cuBLAS GEMM使用示例
 *
 * 本示例展示：
 * 1. 基本SGEMM（单精度GEMM）
 * 2. 混合精度GEMM
 * 3. cuBLASLt API
 * 4. 性能测量
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// 初始化矩阵
// ============================================================================
void init_matrix(float* mat, int rows, int cols, float val = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = val * (float)(i % 100) / 100.0f;
    }
}

// ============================================================================
// CPU GEMM参考实现
// ============================================================================
void cpu_gemm(const float* A, const float* B, float* C,
              int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// ============================================================================
// cuBLAS SGEMM示例
// ============================================================================
void cublas_sgemm_demo(cublasHandle_t handle,
                       float* d_A, float* d_B, float* d_C,
                       int M, int N, int K, float alpha, float beta) {
    printf("=== cuBLAS SGEMM ===\n");
    printf("矩阵尺寸: A(%dx%d), B(%dx%d), C(%dx%d)\n", M, K, K, N, M, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,  // 注意：cuBLAS列主序，交换M/N
                             &alpha,
                             d_B, N,   // B矩阵在前（因为列主序）
                             d_A, K,   // A矩阵在后
                             &beta,
                             d_C, N));

    cudaDeviceSynchronize();

    // 计时
    int warmup = 5, repeat = 20;
    for (int i = 0; i < warmup; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeat;

    // 计算GFLOPS
    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e6);

    printf("执行时间: %.4f ms\n", ms);
    printf("性能: %.2f GFLOPS\n", gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// cuBLASLt示例（支持行主序）
// ============================================================================
void cublaslt_gemm_demo(cublasLtHandle_t ltHandle,
                        float* d_A, float* d_B, float* d_C,
                        int M, int N, int K, float alpha, float beta) {
    printf("\n=== cuBLASLt GEMM ===\n");
    printf("矩阵尺寸: A(%dx%d), B(%dx%d), C(%dx%d)\n", M, K, K, N, M, N);

    // 创建矩阵布局描述符
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, K));  // 行主序: ld=K
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, N));

    // 创建矩阵乘法描述符
    cublasLtMatmulDesc_t matmulDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // 分配工作空间
    size_t workspaceSize = 4 * 1024 * 1024;  // 4MB
    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
                                &alpha, d_A, Adesc, d_B, Bdesc,
                                &beta, d_C, Cdesc, d_C, Cdesc,
                                NULL, d_workspace, workspaceSize, 0));
    cudaDeviceSynchronize();

    // 计时
    int repeat = 20;
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
                                    &alpha, d_A, Adesc, d_B, Bdesc,
                                    &beta, d_C, Cdesc, d_C, Cdesc,
                                    NULL, d_workspace, workspaceSize, 0));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeat;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e6);

    printf("执行时间: %.4f ms\n", ms);
    printf("性能: %.2f GFLOPS\n", gflops);

    // 清理
    cudaFree(d_workspace);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// 混合精度GEMM (FP16输入, FP32累加)
// ============================================================================
void cublas_hgemm_demo(cublasHandle_t handle,
                       half* d_A, half* d_B, float* d_C,
                       int M, int N, int K, float alpha, float beta) {
    printf("\n=== cuBLAS HGEMM (FP16/FP32) ===\n");
    printf("矩阵尺寸: A(%dx%d), B(%dx%d), C(%dx%d)\n", M, K, K, N, M, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_B, CUDA_R_16F, N,
                              d_A, CUDA_R_16F, K,
                              &beta,
                              d_C, CUDA_R_32F, N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    cudaDeviceSynchronize();

    // 计时
    int repeat = 20;
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        CUBLAS_CHECK(cublasGemmEx(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  d_B, CUDA_R_16F, N,
                                  d_A, CUDA_R_16F, K,
                                  &beta,
                                  d_C, CUDA_R_32F, N,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeat;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e6);

    printf("执行时间: %.4f ms\n", ms);
    printf("性能: %.2f GFLOPS\n", gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// 批量GEMM
// ============================================================================
void cublas_batched_gemm_demo(cublasHandle_t handle,
                              float** d_A_array, float** d_B_array, float** d_C_array,
                              int batchCount, int M, int N, int K, float alpha, float beta) {
    printf("\n=== cuBLAS Batched GEMM ===\n");
    printf("批次数: %d, 矩阵尺寸: %dx%d x %dx%d\n", batchCount, M, K, K, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    CUBLAS_CHECK(cublasSgemmBatched(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    N, M, K,
                                    &alpha,
                                    (const float**)d_B_array, N,
                                    (const float**)d_A_array, K,
                                    &beta,
                                    d_C_array, N,
                                    batchCount));
    cudaDeviceSynchronize();

    // 计时
    int repeat = 20;
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        CUBLAS_CHECK(cublasSgemmBatched(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        N, M, K,
                                        &alpha,
                                        (const float**)d_B_array, N,
                                        (const float**)d_A_array, K,
                                        &beta,
                                        d_C_array, N,
                                        batchCount));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeat;

    double flops = 2.0 * batchCount * M * N * K;
    double gflops = flops / (ms * 1e6);

    printf("执行时间: %.4f ms\n", ms);
    printf("性能: %.2f GFLOPS\n", gflops);

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

    printf("=== cuBLAS GEMM示例 ===\n");
    printf("矩阵尺寸: M=%d, N=%d, K=%d\n\n", M, N, K);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // 分配主机内存
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);

    init_matrix(h_A, M, K, 1.0f);
    init_matrix(h_B, K, N, 1.0f);
    init_matrix(h_C, M, N, 0.0f);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    // 拷贝数据
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    // 创建cuBLAS句柄
    cublasHandle_t handle;
    cublasLtHandle_t ltHandle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasLtCreate(&ltHandle));

    // 运行SGEMM示例
    cublas_sgemm_demo(handle, d_A, d_B, d_C, M, N, K, alpha, beta);

    // 重置C
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    // 运行cuBLASLt示例
    cublaslt_gemm_demo(ltHandle, d_A, d_B, d_C, M, N, K, alpha, beta);

    // FP16示例
    if (M == 1024 && N == 1024 && K == 1024) {
        half *d_A16, *d_B16;
        CUDA_CHECK(cudaMalloc(&d_A16, M * K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B16, K * N * sizeof(half)));

        // 转换为FP16
        for (int i = 0; i < M * K; i++) {
            ((half*)h_A)[i] = __float2half(h_A[i]);
        }
        for (int i = 0; i < K * N; i++) {
            ((half*)h_B)[i] = __float2half(h_B[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_A16, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B16, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));

        cublas_hgemm_demo(handle, d_A16, d_B16, d_C, M, N, K, alpha, beta);

        cudaFree(d_A16);
        cudaFree(d_B16);
    }

    // 批量GEMM示例
    int batchCount = 16;
    int batchM = 256, batchN = 256, batchK = 256;

    float **h_A_batch = (float**)malloc(batchCount * sizeof(float*));
    float **h_B_batch = (float**)malloc(batchCount * sizeof(float*));
    float **h_C_batch = (float**)malloc(batchCount * sizeof(float*));
    float **d_A_batch = (float**)malloc(batchCount * sizeof(float*));
    float **d_B_batch = (float**)malloc(batchCount * sizeof(float*));
    float **d_C_batch = (float**)malloc(batchCount * sizeof(float*));

    for (int b = 0; b < batchCount; b++) {
        h_A_batch[b] = (float*)malloc(batchM * batchK * sizeof(float));
        h_B_batch[b] = (float*)malloc(batchK * batchN * sizeof(float));
        h_C_batch[b] = (float*)malloc(batchM * batchN * sizeof(float));
        init_matrix(h_A_batch[b], batchM, batchK, 1.0f);
        init_matrix(h_B_batch[b], batchK, batchN, 1.0f);

        CUDA_CHECK(cudaMalloc(&d_A_batch[b], batchM * batchK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B_batch[b], batchK * batchN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_batch[b], batchM * batchN * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A_batch[b], h_A_batch[b], batchM * batchK * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_batch[b], h_B_batch[b], batchK * batchN * sizeof(float), cudaMemcpyHostToDevice));
    }

    // cuBLAS Batched GEMM 需要设备端的指针数组
    float **d_A_array, **d_B_array, **d_C_array;
    CUDA_CHECK(cudaMalloc(&d_A_array, batchCount * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_B_array, batchCount * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_C_array, batchCount * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(d_A_array, d_A_batch, batchCount * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_array, d_B_batch, batchCount * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_array, d_C_batch, batchCount * sizeof(float*), cudaMemcpyHostToDevice));

    cublas_batched_gemm_demo(handle, d_A_array, d_B_array, d_C_array,
                             batchCount, batchM, batchN, batchK, alpha, beta);

    // 清理
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);

    // 清理
    cublasDestroy(handle);
    cublasLtDestroy(ltHandle);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (int b = 0; b < batchCount; b++) {
        free(h_A_batch[b]);
        free(h_B_batch[b]);
        free(h_C_batch[b]);
        cudaFree(d_A_batch[b]);
        cudaFree(d_B_batch[b]);
        cudaFree(d_C_batch[b]);
    }
    free(h_A_batch);
    free(h_B_batch);
    free(h_C_batch);
    free(d_A_batch);
    free(d_B_batch);
    free(d_C_batch);

    printf("\n=== 提示 ===\n");
    printf("cuBLAS默认使用列主序，对于行主序矩阵需要调整参数或使用cuBLASLt\n");
    printf("使用 -DCUDA_ARCHITECTURES=80 或更高版本以启用Tensor Core\n");

    return 0;
}