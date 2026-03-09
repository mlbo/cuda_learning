/**
 * 第26章示例1：FP16半精度浮点基础
 *
 * 演示内容：
 * 1. FP16类型定义和基本操作
 * 2. FP16与FP32类型转换
 * 3. FP16向量运算
 * 4. __half2向量化操作
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

/**
 * FP16基础操作演示
 */
void fp16_basics_host() {
    printf("\n========================================\n");
    printf("FP16基础操作 (Host端)\n");
    printf("========================================\n\n");

    // FP16类型声明
    __half h1 = __float2half(1.5f);
    __half h2 = __float2half(2.5f);

    printf("1. FP16类型声明和转换:\n");
    printf("   h1 = __float2half(1.5f)\n");
    printf("   h2 = __float2half(2.5f)\n");

    // FP16转回FP32
    float f1 = __half2float(h1);
    float f2 = __half2float(h2);
    printf("   转回FP32: h1=%.2f, h2=%.2f\n", f1, f2);

    // FP16数值范围演示
    printf("\n2. FP16数值范围:\n");
    float test_values[] = {0.0001f, 1.0f, 100.0f, 1000.0f, 10000.0f, 65504.0f, 70000.0f};

    for (int i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++) {
        __half h = __float2half(test_values[i]);
        float back = __half2float(h);
        printf("   FP32: %.4f -> FP16: %.4f (误差: %.6f)\n",
               test_values[i], back, fabsf(test_values[i] - back));
    }

    // __half2类型 (SIMD操作)
    printf("\n3. __half2类型 (两个FP16打包):\n");
    __half2 h2_val = __floats2half2_rn(1.0f, 2.0f);

    // 解包
    float f_first = __low2float(h2_val);
    float f_second = __high2float(h2_val);
    printf("   __half2内容: (%.2f, %.2f)\n", f_first, f_second);
}

/**
 * FP32向量加法核函数 (基准)
 */
__global__ void fp32_vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * FP16向量加法核函数 (基本版本)
 */
__global__ void fp16_vector_add_basic(const __half* a, const __half* b, __half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 转为FP32计算，结果存为FP16
        c[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

/**
 * FP16向量加法核函数 (使用__half2向量化)
 */
__global__ void fp16_vector_add_vec2(const __half2* a, const __half2* b, __half2* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2) {
        // 一次处理两个FP16元素
        __half2 a_val = a[idx];
        __half2 b_val = b[idx];

        // FP16加法 (可能转为FP32计算)
        c[idx] = __hadd2(a_val, b_val);
    }
}

/**
 * FP16向量加法核函数 (纯FP16计算)
 */
__global__ void fp16_vector_add_native(const __half* a, const __half* b, __half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 原生FP16加法
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

/**
 * 性能测试函数
 */
template<typename KernelFunc, typename PtrType>
float benchmark_kernel(KernelFunc kernel, PtrType d_a, PtrType d_b, PtrType d_c,
                       int n, int block_size, int iterations, cudaStream_t stream) {
    int grid_size = (n + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    for (int i = 0; i < 3; i++) {
        kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_c, n);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 计时
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid_size, block_size, 0, stream>>>(d_a, d_b, d_c, n);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iterations;
}

/**
 * 性能对比测试
 */
void performance_comparison() {
    printf("\n========================================\n");
    printf("FP32 vs FP16 性能对比\n");
    printf("========================================\n\n");

    const int N = 4 * 1024 * 1024;  // 4M元素 (进一步减少以加快运行)
    const size_t fp32_size = N * sizeof(float);
    const size_t fp16_size = N * sizeof(__half);
    const int block_size = 256;
    const int iterations = 20;  // 减少迭代次数

    printf("数据大小: %d 元素 (%.2f MB for FP32, %.2f MB for FP16)\n",
           N, fp32_size / 1e6, fp16_size / 1e6);

    // 分配FP32内存
    float *d_a_fp32, *d_b_fp32, *d_c_fp32;
    CHECK_CUDA(cudaMalloc(&d_a_fp32, fp32_size));
    CHECK_CUDA(cudaMalloc(&d_b_fp32, fp32_size));
    CHECK_CUDA(cudaMalloc(&d_c_fp32, fp32_size));

    // 分配FP16内存
    __half *d_a_fp16, *d_b_fp16, *d_c_fp16;
    CHECK_CUDA(cudaMalloc(&d_a_fp16, fp16_size));
    CHECK_CUDA(cudaMalloc(&d_b_fp16, fp16_size));
    CHECK_CUDA(cudaMalloc(&d_c_fp16, fp16_size));

    // 分配__half2内存
    __half2 *d_a_h2, *d_b_h2, *d_c_h2;
    CHECK_CUDA(cudaMalloc(&d_a_h2, fp16_size));
    CHECK_CUDA(cudaMalloc(&d_b_h2, fp16_size));
    CHECK_CUDA(cudaMalloc(&d_c_h2, fp16_size));

    // 初始化数据
    float* h_init = (float*)malloc(fp32_size);
    for (int i = 0; i < N; i++) {
        h_init[i] = (float)(i % 100) / 100.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_a_fp32, h_init, fp32_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_fp32, h_init, fp32_size, cudaMemcpyHostToDevice));

    // 转换为FP16 (批量转换，避免逐元素拷贝)
    __half* h_fp16 = (__half*)malloc(fp16_size);
    for (int i = 0; i < N; i++) {
        h_fp16[i] = __float2half(h_init[i]);
    }
    CHECK_CUDA(cudaMemcpy(d_a_fp16, h_fp16, fp16_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_fp16, h_fp16, fp16_size, cudaMemcpyHostToDevice));
    free(h_fp16);

    // 转换为__half2
    CHECK_CUDA(cudaMemcpy(d_a_h2, d_a_fp16, fp16_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_h2, d_b_fp16, fp16_size, cudaMemcpyDeviceToDevice));

    // 创建流
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    printf("\n%-25s | %-12s | %-12s\n", "内核", "时间(ms)", "带宽(GB/s)");
    printf("---------------------------------------------------\n");

    // FP32测试
    float fp32_time = benchmark_kernel(
        fp32_vector_add, d_a_fp32, d_b_fp32, d_c_fp32,
        N, block_size, iterations, stream
    );
    float fp32_bw = 3.0f * fp32_size / fp32_time / 1e6;
    printf("%-25s | %-10.3f | %-10.2f\n", "FP32 Vector Add", fp32_time, fp32_bw);

    // FP16基本版本
    float fp16_basic_time = benchmark_kernel(
        fp16_vector_add_basic, d_a_fp16, d_b_fp16, d_c_fp16,
        N, block_size, iterations, stream
    );
    float fp16_basic_bw = 3.0f * fp16_size / fp16_basic_time / 1e6;
    printf("%-25s | %-10.3f | %-10.2f\n", "FP16 Vector Add (Basic)", fp16_basic_time, fp16_basic_bw);

    // FP16原生版本
    float fp16_native_time = benchmark_kernel(
        fp16_vector_add_native, d_a_fp16, d_b_fp16, d_c_fp16,
        N, block_size, iterations, stream
    );
    float fp16_native_bw = 3.0f * fp16_size / fp16_native_time / 1e6;
    printf("%-25s | %-10.3f | %-10.2f\n", "FP16 Vector Add (Native)", fp16_native_time, fp16_native_bw);

    // __half2向量化版本
    float fp16_vec2_time = benchmark_kernel(
        fp16_vector_add_vec2, d_a_h2, d_b_h2, d_c_h2,
        N, block_size, iterations, stream
    );
    float fp16_vec2_bw = 3.0f * fp16_size / fp16_vec2_time / 1e6;
    printf("%-25s | %-10.3f | %-10.2f\n", "FP16 Vector Add (half2 SIMD)", fp16_vec2_time, fp16_vec2_bw);

    printf("\n加速比:\n");
    printf("  FP16 Basic vs FP32:   %.2fx\n", fp32_time / fp16_basic_time);
    printf("  FP16 Native vs FP32:  %.2fx\n", fp32_time / fp16_native_time);
    printf("  FP16 half2 vs FP32:   %.2fx\n", fp32_time / fp16_vec2_time);

    // 验证结果
    printf("\n结果验证:\n");
    float h_fp32_result, h_fp16_result;
    CHECK_CUDA(cudaMemcpy(&h_fp32_result, d_c_fp32, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_fp16_result, d_c_fp16, sizeof(__half), cudaMemcpyDeviceToHost));
    float fp16_as_float = __half2float(*(__half*)&h_fp16_result);
    printf("  FP32结果: %.4f\n", h_fp32_result);
    printf("  FP16结果: %.4f (误差: %.6f)\n", fp16_as_float, fabsf(h_fp32_result - fp16_as_float));

    // 清理
    free(h_init);
    CHECK_CUDA(cudaFree(d_a_fp32));
    CHECK_CUDA(cudaFree(d_b_fp32));
    CHECK_CUDA(cudaFree(d_c_fp32));
    CHECK_CUDA(cudaFree(d_a_fp16));
    CHECK_CUDA(cudaFree(d_b_fp16));
    CHECK_CUDA(cudaFree(d_c_fp16));
    CHECK_CUDA(cudaFree(d_a_h2));
    CHECK_CUDA(cudaFree(d_b_h2));
    CHECK_CUDA(cudaFree(d_c_h2));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

/**
 * FP16精度范围测试
 */
__global__ void fp16_range_test_kernel(float* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 测试不同范围的值
        float val = powf(10.0f, (float)idx / 10.0f - 5.0f);  // 10^-5 到 10^5
        __half h = __float2half(val);
        results[idx] = __half2float(h);
    }
}

void fp16_precision_test() {
    printf("\n========================================\n");
    printf("FP16精度测试\n");
    printf("========================================\n\n");

    printf("FP16特性:\n");
    printf("  - 最小正规数: 2^-14 ≈ 6.1e-5\n");
    printf("  - 最大值: 2^15 * (2 - 2^-10) ≈ 65504\n");
    printf("  - 尾数精度: ~3位小数\n\n");

    // 测试数据
    float test_vals[] = {
        0.00001f, 0.0001f, 0.001f, 0.01f, 0.1f,
        1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f,
        60000.0f, 65000.0f, 65504.0f, 65505.0f
    };

    printf("FP32 -> FP16 -> FP32 转换测试:\n");
    printf("%-12s | %-12s | %-12s\n", "原值", "转换后", "相对误差");
    printf("----------------------------------------------\n");

    for (int i = 0; i < sizeof(test_vals) / sizeof(test_vals[0]); i++) {
        float val = test_vals[i];
        __half h = __float2half(val);
        float back = __half2float(h);
        float rel_error = (val != 0) ? fabsf((back - val) / val) : fabsf(back - val);
        printf("%-12.4f | %-12.4f | %-12.6f%%\n", val, back, rel_error * 100);
    }

    // 溢出测试
    printf("\n溢出测试:\n");
    float overflow_vals[] = {65504.0f, 65505.0f, 70000.0f, 100000.0f};
    for (int i = 0; i < sizeof(overflow_vals) / sizeof(overflow_vals[0]); i++) {
        float val = overflow_vals[i];
        __half h = __float2half(val);
        float back = __half2float(h);
        printf("  %.1f -> %.1f (溢出: %s)\n", val, back, val > 65504.0f ? "是" : "否");
    }
}

/**
 * __half2 SIMD操作演示
 */
void half2_operations_demo() {
    printf("\n========================================\n");
    printf("__half2 SIMD操作演示\n");
    printf("========================================\n\n");

    printf("__half2将两个FP16打包为32位，可以一次处理两个元素:\n\n");

    // 创建__half2
    __half2 a = __floats2half2_rn(1.5f, 2.5f);
    __half2 b = __floats2half2_rn(0.5f, 1.0f);

    printf("初始化:\n");
    printf("  a = (%.2f, %.2f)\n", __low2float(a), __high2float(a));
    printf("  b = (%.2f, %.2f)\n", __low2float(b), __high2float(b));

    // 常用操作
    printf("\n常用操作:\n");

    // 加法
    __half2 sum = __hadd2(a, b);
    printf("  加法: (%.2f, %.2f)\n", __low2float(sum), __high2float(sum));

    // 乘法
    __half2 prod = __hmul2(a, b);
    printf("  乘法: (%.2f, %.2f)\n", __low2float(prod), __high2float(prod));

    // 减法
    __half2 diff = __hsub2(a, b);
    printf("  减法: (%.2f, %.2f)\n", __low2float(diff), __high2float(diff));

    // 比较和选择
    __half2 sel = __hmax2(a, b);
    printf("  最大值: (%.2f, %.2f)\n", __low2float(sel), __high2float(sel));

    printf("\n__half2优势:\n");
    printf("  - 一次内存访问读取两个FP16值\n");
    printf("  - 一条指令处理两个元素\n");
    printf("  - 更好的内存带宽利用率\n");
}

int main() {
    printf("=============================================\n");
    printf("  第26章示例1：FP16半精度浮点基础\n");
    printf("=============================================\n");

    // 基础操作演示
    fp16_basics_host();

    // 精度测试
    fp16_precision_test();

    // __half2操作演示
    half2_operations_demo();

    // 性能对比
    performance_comparison();

    printf("\n示例完成！\n");
    return 0;
}