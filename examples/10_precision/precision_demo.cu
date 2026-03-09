/**
 * 第十章: CUDA 精度与性能
 *
 * 学习目标:
 *   1. 掌握 CUDA FP16 (半精度) 编程基础
 *   2. 理解 half 和 half2 类型的使用
 *   3. 学习 __hadd 和 __hadd2 等内建函数
 *   4. 对比 FP32 与 FP16 的性能差异
 *
 * 编译方法:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
 *   make
 *
 * 运行:
 *   ./precision_demo
 *
 * 注意:
 *   - FP16 需要 SM 5.3+ (Maxwell/Pascal 及更新架构)
 *   - __hadd2 需要 SM 7.0+ (Volta 及更新架构)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// ==================== 错误检查宏 ====================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA 错误 at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

// ==================== FP32 核函数 ====================

/**
 * FP32 标量版本 - 基准
 *
 * 特点:
 *   - 单精度浮点数 (32 位)
 *   - 每个线程处理 1 个元素
 *   - 精度最高，但访存量大
 */
__global__ void add_fp32(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * FP32 float4 向量化版本
 *
 * 特点:
 *   - 使用 float4 向量类型
 *   - 每个线程处理 4 个元素
 *   - 提高访存效率
 */
__global__ void add_fp32_vec4(float4* a, float4* b, float4* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;

    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];

        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        c[idx] = vc;
    }
}

// ==================== FP16 核函数 ====================

/**
 * FP16 标量版本
 *
 * 特点:
 *   - 半精度浮点数 (16 位)
 *   - 使用 half 类型
 *   - 使用 __hadd 内建函数
 *
 * half 类型说明:
 *   - 符号位: 1 位
 *   - 指数位: 5 位
 *   - 尾数位: 10 位
 *   - 范围: 约 6.1e-5 到 6.5e4
 *   - 精度: 约 3-4 位有效数字
 */
__global__ void add_fp16(half* a, half* b, half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // __hadd: 半精度加法 (标量)
        // 注意: 这不是真正的 FP16 计算，而是转换为 FP32 后计算
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

/**
 * FP16 half2 向量化版本 - 手动拆分
 *
 * 特点:
 *   - 使用 half2 类型 (包含 2 个 half)
 *   - 手动访问 x 和 y 分量
 *   - 使用 __hadd 对每个分量单独计算
 */
__global__ void add_fp16_vec2_split(half2* a, half2* b, half2* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2) {
        half2 va = a[idx];
        half2 vb = b[idx];

        half2 vc;
        // 手动拆分 - 每个分量单独调用 __hadd
        vc.x = __hadd(va.x, vb.x);
        vc.y = __hadd(va.y, vb.y);

        c[idx] = vc;
    }
}

/**
 * FP16 half2 SIMD 版本 - 最佳实践
 *
 * 特点:
 *   - 使用 __hadd2 内建函数
 *   - 一条指令同时处理 2 个 half 数据
 *   - 真正的 SIMD (Single Instruction Multiple Data)
 *
 * __hadd2 说明:
 *   - 输入: 两个 half2 向量
 *   - 输出: 一个 half2 向量
 *   - 同时计算: result.x = a.x + b.x, result.y = a.y + b.y
 *   - 性能: 比手动拆分快约 2 倍
 */
__global__ void add_fp16_vec2_simd(half2* a, half2* b, half2* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2) {
        // __hadd2: 一条指令同时处理 2 个 half
        // 这是真正的向量化 FP16 计算!
        c[idx] = __hadd2(a[idx], b[idx]);
    }
}

/**
 * FP16 混合精度版本
 *
 * 特点:
 *   - 输入输出为 FP16
 *   - 计算过程使用 FP32
 *   - 适合不支持 FP16 计算的架构
 *
 * 用途:
 *   - 在旧架构上存储 FP16 数据
 *   - 保证计算精度
 */
__global__ void add_fp16_fallback(half* a, half* b, half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 转换到 FP32 计算
        float fa = __half2float(a[idx]);
        float fb = __half2float(b[idx]);
        float fc = fa + fb;

        // 转换回 FP16 存储
        c[idx] = __float2half(fc);
    }
}

// ==================== 更多 FP16 运算示例 ====================

/**
 * FP16 乘法示例
 */
__global__ void mul_fp16(half* a, half* b, half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hmul(a[idx], b[idx]);  // FP16 乘法
    }
}

/**
 * FP16 half2 乘法 (SIMD)
 */
__global__ void mul_fp16_vec2(half2* a, half2* b, half2* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2) {
        c[idx] = __hmul2(a[idx], b[idx]);  // SIMD 乘法
    }
}

/**
 * FP16 FMA (Fused Multiply-Add) 示例
 * result = a * b + c
 */
__global__ void fma_fp16_vec2(half2* a, half2* b, half2* c, half2* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2) {
        // __hfma2: a * b + c (一条指令完成)
        result[idx] = __hfma2(a[idx], b[idx], c[idx]);
    }
}

// ==================== 精度对比函数 ====================

/**
 * 比较 FP32 和 FP16 的精度差异
 */
void compare_precision(float fp32_val, half fp16_val, const char* name) {
    float fp16_as_float = __half2float(fp16_val);
    float diff = std::fabs(fp32_val - fp16_as_float);
    float rel_error = diff / fp32_val * 100.0f;

    printf("  %s:\n", name);
    printf("    FP32: %.8f\n", fp32_val);
    printf("    FP16: %.8f\n", fp16_as_float);
    printf("    绝对误差: %.8f\n", diff);
    printf("    相对误差: %.4f%%\n\n", rel_error);
}

// ==================== 辅助函数 ====================

void init_vector_fp32(float* vec, int n, float value) {
    for (int i = 0; i < n; i++) {
        vec[i] = value;
    }
}

void init_vector_fp16(half* vec, int n, float value) {
    for (int i = 0; i < n; i++) {
        vec[i] = __float2half(value);
    }
}

bool verify_fp32(float* c, int n, float expected) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (std::fabs(c[i] - expected) > 1e-5) {
            if (errors < 5) {
                printf("  错误: c[%d] = %f, 期望 %f\n", i, c[i], expected);
            }
            errors++;
        }
    }
    return errors == 0;
}

bool verify_fp16(half* c, int n, float expected) {
    int errors = 0;
    float expected_half = __half2float(__float2half(expected));

    for (int i = 0; i < n; i++) {
        float val = __half2float(c[i]);
        if (std::fabs(val - expected_half) > 0.01) {  // FP16 精度较低
            errors++;
        }
    }
    return errors == 0;
}

// 计时模板
template<typename KernelFunc, typename... Args>
float benchmark_kernel(cudaEvent_t& start, cudaEvent_t& stop,
                       KernelFunc kernel, dim3 grid, dim3 block,
                       int iterations, Args... args) {
    // 预热
    kernel<<<grid, block>>>(args...);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(args...);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms / iterations;
}

// ==================== 主函数 ====================

int main(int argc, char** argv) {
    int n = 1 << 24;  // 16M 元素
    if (argc > 1) {
        n = 1 << atoi(argv[1]);
    }

    size_t bytes_fp32 = n * sizeof(float);
    size_t bytes_fp16 = n * sizeof(half);

    printf("========================================\n");
    printf("CUDA 精度与性能演示\n");
    printf("========================================\n");
    printf("元素数量: %d\n", n);
    printf("FP32 数据量: %.2f MB per vector\n", bytes_fp32 / 1024.0 / 1024.0);
    printf("FP16 数据量: %.2f MB per vector (减半!)\n\n", bytes_fp16 / 1024.0 / 1024.0);

    // ==================== 精度对比测试 ====================
    printf("========================================\n");
    printf("精度对比测试\n");
    printf("========================================\n\n");

    // 不同数值范围的精度测试
    compare_precision(1.0f, __float2half(1.0f), "数值 1.0");
    compare_precision(100.0f, __float2half(100.0f), "数值 100.0");
    compare_precision(0.001f, __float2half(0.001f), "数值 0.001");
    compare_precision(3.14159265f, __float2half(3.14159265f), "数值 PI");
    compare_precision(12345.6789f, __float2half(12345.6789f), "数值 12345.6789");

    printf("精度说明:\n");
    printf("  FP32: 约 7 位有效数字\n");
    printf("  FP16: 约 3-4 位有效数字\n");
    printf("  FP16 范围: 约 6.1e-5 到 6.5e4\n\n");

    // ==================== 性能测试 ====================

    // 分配内存
    float *h_a_fp32 = (float*)malloc(bytes_fp32);
    float *h_b_fp32 = (float*)malloc(bytes_fp32);
    float *h_c_fp32 = (float*)malloc(bytes_fp32);

    half *h_a_fp16 = (half*)malloc(bytes_fp16);
    half *h_b_fp16 = (half*)malloc(bytes_fp16);
    half *h_c_fp16 = (half*)malloc(bytes_fp16);

    init_vector_fp32(h_a_fp32, n, 1.0f);
    init_vector_fp32(h_b_fp32, n, 2.0f);
    init_vector_fp16(h_a_fp16, n, 1.0f);
    init_vector_fp16(h_b_fp16, n, 2.0f);

    float *d_a_fp32, *d_b_fp32, *d_c_fp32;
    half *d_a_fp16, *d_b_fp16, *d_c_fp16;

    CUDA_CHECK(cudaMalloc(&d_a_fp32, bytes_fp32));
    CUDA_CHECK(cudaMalloc(&d_b_fp32, bytes_fp32));
    CUDA_CHECK(cudaMalloc(&d_c_fp32, bytes_fp32));

    CUDA_CHECK(cudaMalloc(&d_a_fp16, bytes_fp16));
    CUDA_CHECK(cudaMalloc(&d_b_fp16, bytes_fp16));
    CUDA_CHECK(cudaMalloc(&d_c_fp16, bytes_fp16));

    CUDA_CHECK(cudaMemcpy(d_a_fp32, h_a_fp32, bytes_fp32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_fp32, h_b_fp32, bytes_fp32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_fp16, h_a_fp16, bytes_fp16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_fp16, h_b_fp16, bytes_fp16, cudaMemcpyHostToDevice));

    // 创建计时事件
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blockSize = 256;
    int iterations = 100;

    printf("========================================\n");
    printf("性能对比测试\n");
    printf("========================================\n\n");

    printf("%-22s %12s %12s %12s %10s\n",
           "版本", "时间(ms)", "带宽(GB/s)", "加速比", "数据量");
    printf("-------------------------------------------------------------------------\n");

    // -------------------- FP32 标量版本 --------------------
    float baseline_ms;
    {
        int gridSize = (n + blockSize - 1) / blockSize;
        dim3 grid(gridSize), block(blockSize);

        float ms = benchmark_kernel(start, stop, add_fp32, grid, block, iterations,
                                    d_a_fp32, d_b_fp32, d_c_fp32, n);

        baseline_ms = ms;
        float bandwidth = (3.0 * bytes_fp32 / ms) / 1e6;
        printf("%-22s %12.4f %12.1f %12s %10s\n",
               "FP32 标量", ms, bandwidth, "1.00x", "100%");
    }

    // -------------------- FP32 float4 版本 --------------------
    {
        int n4 = n / 4;
        int gridSize = (n4 + blockSize - 1) / blockSize;
        dim3 grid(gridSize), block(blockSize);

        float ms = benchmark_kernel(start, stop, add_fp32_vec4, grid, block, iterations,
                                    (float4*)d_a_fp32, (float4*)d_b_fp32, (float4*)d_c_fp32, n);

        float bandwidth = (3.0 * bytes_fp32 / ms) / 1e6;
        printf("%-22s %12.4f %12.1f %12.2fx %10s\n",
               "FP32 float4", ms, bandwidth, baseline_ms / ms, "100%");
    }

    // -------------------- FP16 标量版本 --------------------
    {
        int gridSize = (n + blockSize - 1) / blockSize;
        dim3 grid(gridSize), block(blockSize);

        float ms = benchmark_kernel(start, stop, add_fp16, grid, block, iterations,
                                    d_a_fp16, d_b_fp16, d_c_fp16, n);

        float bandwidth = (3.0 * bytes_fp16 / ms) / 1e6;
        printf("%-22s %12.4f %12.1f %12.2fx %10s\n",
               "FP16 标量", ms, bandwidth, baseline_ms / ms, "50%");
    }

    // -------------------- FP16 half2 (手动拆分) --------------------
    {
        int n2 = n / 2;
        int gridSize = (n2 + blockSize - 1) / blockSize;
        dim3 grid(gridSize), block(blockSize);

        float ms = benchmark_kernel(start, stop, add_fp16_vec2_split, grid, block, iterations,
                                    (half2*)d_a_fp16, (half2*)d_b_fp16, (half2*)d_c_fp16, n);

        float bandwidth = (3.0 * bytes_fp16 / ms) / 1e6;
        printf("%-22s %12.4f %12.1f %12.2fx %10s\n",
               "FP16 half2 (拆分)", ms, bandwidth, baseline_ms / ms, "50%");
    }

    // -------------------- FP16 half2 SIMD --------------------
    {
        int n2 = n / 2;
        int gridSize = (n2 + blockSize - 1) / blockSize;
        dim3 grid(gridSize), block(blockSize);

        float ms = benchmark_kernel(start, stop, add_fp16_vec2_simd, grid, block, iterations,
                                    (half2*)d_a_fp16, (half2*)d_b_fp16, (half2*)d_c_fp16, n);

        float bandwidth = (3.0 * bytes_fp16 / ms) / 1e6;
        printf("%-22s %12.4f %12.1f %12.2fx %10s\n",
               "FP16 half2 (SIMD)", ms, bandwidth, baseline_ms / ms, "50%");
        printf("  -> __hadd2: 一条指令处理 2 个 half 数据!\n");
    }

    // -------------------- FP16 混合精度 --------------------
    {
        int gridSize = (n + blockSize - 1) / blockSize;
        dim3 grid(gridSize), block(blockSize);

        float ms = benchmark_kernel(start, stop, add_fp16_fallback, grid, block, iterations,
                                    d_a_fp16, d_b_fp16, d_c_fp16, n);

        float bandwidth = (3.0 * bytes_fp16 / ms) / 1e6;
        printf("%-22s %12.4f %12.1f %12.2fx %10s\n",
               "FP16 混合精度", ms, bandwidth, baseline_ms / ms, "50%");
    }

    printf("-------------------------------------------------------------------------\n");

    // ==================== 验证结果 ====================
    printf("\n验证结果...\n");

    CUDA_CHECK(cudaMemcpy(h_c_fp32, d_c_fp32, bytes_fp32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_fp16, d_c_fp16, bytes_fp16, cudaMemcpyDeviceToHost));

    bool ok_fp32 = verify_fp32(h_c_fp32, n, 3.0f);
    bool ok_fp16 = verify_fp16(h_c_fp16, n, 3.0f);

    printf("  FP32: %s\n", ok_fp32 ? "通过" : "失败");
    printf("  FP16: %s\n", ok_fp16 ? "通过" : "失败");

    // ==================== 清理资源 ====================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_a_fp32));
    CUDA_CHECK(cudaFree(d_b_fp32));
    CUDA_CHECK(cudaFree(d_c_fp32));
    CUDA_CHECK(cudaFree(d_a_fp16));
    CUDA_CHECK(cudaFree(d_b_fp16));
    CUDA_CHECK(cudaFree(d_c_fp16));

    free(h_a_fp32);
    free(h_b_fp32);
    free(h_c_fp32);
    free(h_a_fp16);
    free(h_b_fp16);
    free(h_c_fp16);

    // ==================== 总结 ====================
    printf("\n========================================\n");
    printf("精度与性能要点总结\n");
    printf("========================================\n");
    printf("\n1. 数据类型对比:\n");
    printf("   FP32: 32位, 7位有效数字, 范围 ~1e38\n");
    printf("   FP16: 16位, 3-4位有效数字, 范围 ~6.5e4\n");
    printf("\n2. FP16 优势:\n");
    printf("   - 数据量减半 -> 访存量减半\n");
    printf("   - AI (计算强度) 翻倍\n");
    printf("   - Tensor Core 加速\n");
    printf("\n3. 常用 FP16 内建函数:\n");
    printf("   __hadd(a, b)     - 标量加法\n");
    printf("   __hadd2(a, b)    - 向量加法 (SIMD)\n");
    printf("   __hmul(a, b)     - 标量乘法\n");
    printf("   __hmul2(a, b)    - 向量乘法\n");
    printf("   __hfma2(a,b,c)   - 向量 FMA\n");
    printf("   __float2half(f)  - FP32 -> FP16\n");
    printf("   __half2float(h)  - FP16 -> FP32\n");
    printf("\n4. 适用场景:\n");
    printf("   - 深度学习推理/训练\n");
    printf("   - 图形渲染\n");
    printf("   - 对精度要求不高的计算\n");
    printf("========================================\n");

    return 0;
}
