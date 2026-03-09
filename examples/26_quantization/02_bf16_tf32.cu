/**
 * 第26章示例2：BF16和TF32使用
 *
 * 演示内容：
 * 1. BF16类型和操作
 * 2. BF16 vs FP16对比
 * 3. TF32 Tensor Core使用
 * 4. cuBLAS混合精度GEMM
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

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
 * BF16基础操作演示
 */
void bf16_basics_host() {
    printf("\n========================================\n");
    printf("BF16基础操作\n");
    printf("========================================\n\n");

    printf("BF16 (Brain Floating Point) 特点:\n");
    printf("  - 16位浮点格式\n");
    printf("  - 1位符号 + 8位指数 + 7位尾数\n");
    printf("  - 与FP32相同的数值范围\n");
    printf("  - 精度约为FP32的1/256\n\n");

    // BF16类型声明
    __nv_bfloat16 bf1 = __float2bfloat16(1.5f);
    __nv_bfloat16 bf2 = __float2bfloat16(2.5f);

    printf("1. BF16类型声明和转换:\n");
    printf("   bf1 = __float2bfloat16(1.5f)\n");
    printf("   bf2 = __float2bfloat16(2.5f)\n");

    // BF16转回FP32
    float f1 = __bfloat162float(bf1);
    float f2 = __bfloat162float(bf2);
    printf("   转回FP32: bf1=%.2f, bf2=%.2f\n", f1, f2);

    // BF16数值范围演示
    printf("\n2. BF16数值范围测试:\n");
    float test_values[] = {
        0.01f, 0.1f, 1.0f, 10.0f, 100.0f,
        1000.0f, 10000.0f, 100000.0f, 1e10f, 1e30f
    };

    printf("%-15s | %-15s | %-15s\n", "FP32原值", "BF16转换后", "相对误差");
    printf("-------------------------------------------------------\n");

    for (int i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++) {
        float val = test_values[i];
        __nv_bfloat16 bf = __float2bfloat16(val);
        float back = __bfloat162float(bf);
        float rel_error = fabsf((back - val) / val) * 100;
        printf("%-15.4e | %-15.4e | %-12.4f%%\n", val, back, rel_error);
    }

    // __nv_bfloat162类型
    printf("\n3. __nv_bfloat162类型 (两个BF16打包):\n");
    __nv_bfloat162 bf2_val = __floats2bfloat162_rn(1.0f, 2.0f);

    float f_low = __low2float(bf2_val);
    float f_high = __high2float(bf2_val);
    printf("   __nv_bfloat162内容: (%.2f, %.2f)\n", f_low, f_high);
}

/**
 * FP16 vs BF16 精度对比
 */
void fp16_vs_bf16_comparison() {
    printf("\n========================================\n");
    printf("FP16 vs BF16 精度对比\n");
    printf("========================================\n\n");

    printf("格式对比:\n");
    printf("  FP16: 1位符号 + 5位指数 + 10位尾数\n");
    printf("  BF16: 1位符号 + 8位指数 + 7位尾数\n\n");

    // 数值范围测试
    printf("数值范围测试:\n");
    printf("%-15s | %-15s | %-15s | %-15s\n",
           "FP32值", "FP16结果", "BF16结果", "差异");
    printf("-----------------------------------------------------------------\n");

    float range_tests[] = {
        1e-5f, 1e-3f, 1e-1f, 1.0f, 10.0f,
        1000.0f, 10000.0f, 60000.0f, 70000.0f, 1e10f
    };

    for (int i = 0; i < sizeof(range_tests) / sizeof(range_tests[0]); i++) {
        float val = range_tests[i];

        __half fp16 = __float2half(val);
        __nv_bfloat16 bf16 = __float2bfloat16(val);

        float fp16_back = __half2float(fp16);
        float bf16_back = __bfloat162float(bf16);

        // 检查是否溢出
        bool fp16_overflow = (val > 65504.0f);
        const char* fp16_status = fp16_overflow ? "OVERFLOW" : "";

        printf("%-15.4e | %-15.4e %s | %-15.4e | FP16:%.2f%% BF16:%.2f%%\n",
               val, fp16_back, fp16_status, bf16_back,
               fabsf((fp16_back - val) / val) * 100,
               fabsf((bf16_back - val) / val) * 100);
    }

    printf("\n结论:\n");
    printf("  - FP16: 适合小范围数值，精度更高\n");
    printf("  - BF16: 数值范围与FP32相同，适合大范围计算\n");
    printf("  - 训练时BF16更稳定（不易溢出）\n");
}

/**
 * 检查BF16硬件支持
 */
bool check_bf16_support() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // BF16需要Ampere或更新架构 (sm_80+)
    int major = prop.major;
    return (major >= 8);
}

/**
 * 检查TF32支持
 */
bool check_tf32_support() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // TF32需要Ampere或更新架构 (sm_80+)
    return (prop.major >= 8);
}

/**
 * TF32 Tensor Core说明
 */
void tf32_explanation() {
    printf("\n========================================\n");
    printf("TF32 (Tensor Float 32)\n");
    printf("========================================\n\n");

    printf("TF32特点:\n");
    printf("  - 19位内部格式: 1位符号 + 8位指数 + 10位尾数\n");
    printf("  - 专为Tensor Core设计\n");
    printf("  - FP32输入自动转换为TF32计算\n");
    printf("  - FP32累加\n");
    printf("  - 对程序员透明，无需代码修改\n\n");

    printf("TF32 vs FP32:\n");
    printf("  - 数值范围: 相同\n");
    printf("  - 精度: TF32约为FP32的1/64\n");
    printf("  - 性能: TF32在Tensor Core上快约8x\n\n");

    printf("启用TF32:\n");
    printf("  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);\n");
}

/**
 * 使用cuBLAS进行混合精度GEMM
 */
void cublas_mixed_precision_gemm() {
    printf("\n========================================\n");
    printf("cuBLAS混合精度GEMM\n");
    printf("========================================\n\n");

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    printf("矩阵大小: M=N=K=%d\n", M);

    // 创建cuBLAS句柄
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 分配内存
    float *d_A, *d_B, *d_C;
    half *d_A_fp16, *d_B_fp16, *d_C_fp16;

    size_t fp32_size = M * K * sizeof(float);
    size_t fp16_size = M * K * sizeof(half);

    CHECK_CUDA(cudaMalloc(&d_A, fp32_size));
    CHECK_CUDA(cudaMalloc(&d_B, fp32_size));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_A_fp16, fp16_size));
    CHECK_CUDA(cudaMalloc(&d_B_fp16, fp16_size));
    CHECK_CUDA(cudaMalloc(&d_C_fp16, M * N * sizeof(half)));

    // 初始化矩阵
    float* h_A = (float*)malloc(fp32_size);
    float* h_B = (float*)malloc(fp32_size);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = 0.01f;
        h_B[i] = 0.01f;
    }
    CHECK_CUDA(cudaMemcpy(d_A, h_A, fp32_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, fp32_size, cudaMemcpyHostToDevice));

    // 转换为FP16
    for (int i = 0; i < M * K; i++) {
        half h_val = __float2half(h_A[i]);
        CHECK_CUDA(cudaMemcpy(d_A_fp16 + i, &h_val, sizeof(half), cudaMemcpyHostToDevice));
        h_val = __float2half(h_B[i]);
        CHECK_CUDA(cudaMemcpy(d_B_fp16 + i, &h_val, sizeof(half), cudaMemcpyHostToDevice));
    }

    // 创建事件计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float alpha = 1.0f;
    float beta = 0.0f;
    const int iterations = 10;

    printf("\n%-25s | %-12s | %-12s\n", "GEMM类型", "时间(ms)", "TFLOPS");
    printf("---------------------------------------------------\n");

    // FP32 GEMM (标准)
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    // 预热
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_fp32;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp32, start, stop));
    ms_fp32 /= iterations;
    double tflops_fp32 = 2.0 * M * N * K / (ms_fp32 * 1e9);
    printf("%-25s | %-10.3f | %-10.2f\n", "FP32 (标准)", ms_fp32, tflops_fp32);

    // TF32 GEMM (如果支持)
    if (check_tf32_support()) {
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms_tf32;
        CHECK_CUDA(cudaEventElapsedTime(&ms_tf32, start, stop));
        ms_tf32 /= iterations;
        double tflops_tf32 = 2.0 * M * N * K / (ms_tf32 * 1e9);
        printf("%-25s | %-10.3f | %-10.2f\n", "TF32 Tensor Core", ms_tf32, tflops_tf32);

        printf("\nTF32加速比: %.2fx\n", ms_fp32 / ms_tf32);
    } else {
        printf("TF32不支持 (需要sm_80+)\n");
    }

    // FP16 Tensor Core GEMM
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    half alpha_h = __float2half(1.0f);
    half beta_h = __float2half(0.0f);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  M, N, K, &alpha_h, d_A_fp16, M, d_B_fp16, K,
                                  &beta_h, d_C_fp16, M));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_fp16;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp16, start, stop));
    ms_fp16 /= iterations;
    double tflops_fp16 = 2.0 * M * N * K / (ms_fp16 * 1e9);
    printf("%-25s | %-10.3f | %-10.2f\n", "FP16 Tensor Core", ms_fp16, tflops_fp16);

    printf("\nFP16加速比: %.2fx\n", ms_fp32 / ms_fp16);

    // 清理
    free(h_A);
    free(h_B);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_fp16));
    CHECK_CUDA(cudaFree(d_B_fp16));
    CHECK_CUDA(cudaFree(d_C_fp16));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
}

/**
 * 精度选择建议
 */
void precision_selection_guide() {
    printf("\n========================================\n");
    printf("精度选择指南\n");
    printf("========================================\n\n");

    printf("训练场景:\n");
    printf("  - BF16: 推荐用于大模型训练（Ampere+）\n");
    printf("    * 不易溢出，数值稳定\n");
    printf("    * 无需loss scaling\n");
    printf("  - FP16: 广泛支持，需要loss scaling\n");
    printf("  - 混合精度: FP16/BF16计算 + FP32累加\n\n");

    printf("推理场景:\n");
    printf("  - FP16: 平衡精度和性能\n");
    printf("  - INT8: 量化推理，最大性能\n");
    printf("  - FP32: 精度敏感场景\n\n");

    printf("硬件支持:\n");
    printf("  - FP16: Kepler及更新架构\n");
    printf("  - BF16: Ampere及更新架构 (sm_80+)\n");
    printf("  - TF32: Ampere及更新架构 (sm_80+)\n");
    printf("  - Tensor Core: Volta及更新架构 (sm_70+)\n");
}

int main() {
    printf("=============================================\n");
    printf("  第26章示例2：BF16和TF32使用\n");
    printf("=============================================\n");

    // 检查硬件支持
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("\n当前设备: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // BF16基础
    bf16_basics_host();

    // FP16 vs BF16对比
    fp16_vs_bf16_comparison();

    // TF32说明
    tf32_explanation();

    // cuBLAS混合精度GEMM
    cublas_mixed_precision_gemm();

    // 精度选择指南
    precision_selection_guide();

    printf("\n示例完成！\n");
    return 0;
}