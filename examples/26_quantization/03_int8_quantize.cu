/**
 * 第26章示例3：INT8量化基础
 *
 * 演示内容：
 * 1. 量化原理和公式
 * 2. 对称和非对称量化
 * 3. INT8向量运算
 * 4. 量化误差分析
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <algorithm>

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
 * 量化原理说明
 */
void quantization_theory() {
    printf("========================================\n");
    printf("量化原理\n");
    printf("========================================\n\n");

    printf("量化是将FP32数值映射到INT8的过程:\n\n");

    printf("对称量化:\n");
    printf("  Q = round(x / scale)\n");
    printf("  x' = Q * scale\n");
    printf("  scale = max(|x|) / 127\n");
    printf("  优点: 简单，计算高效\n");
    printf("  缺点: 不能充分利用INT8范围\n\n");

    printf("非对称量化:\n");
    printf("  Q = round(x / scale) + zero_point\n");
    printf("  x' = (Q - zero_point) * scale\n");
    printf("  scale = (max - min) / 255\n");
    printf("  zero_point = round(-min / scale)\n");
    printf("  优点: 充分利用INT8范围\n");
    printf("  缺点: 计算稍复杂\n\n");
}

/**
 * CPU端量化函数
 */
// 对称量化
void quantize_symmetric_cpu(const float* src, int8_t* dst, int n, float scale) {
    for (int i = 0; i < n; i++) {
        float q = roundf(src[i] / scale);
        // 钳制到INT8范围 [-128, 127]
        q = fmaxf(fminf(q, 127.0f), -128.0f);
        dst[i] = (int8_t)q;
    }
}

// 对称反量化
void dequantize_symmetric_cpu(const int8_t* src, float* dst, int n, float scale) {
    for (int i = 0; i < n; i++) {
        dst[i] = (float)src[i] * scale;
    }
}

// 计算对称量化的scale
float compute_scale_symmetric(const float* data, int n) {
    float abs_max = 0.0f;
    for (int i = 0; i < n; i++) {
        abs_max = fmaxf(abs_max, fabsf(data[i]));
    }
    return abs_max / 127.0f;
}

// 非对称量化
void quantize_asymmetric_cpu(const float* src, int8_t* dst, int n, float scale, int zero_point) {
    for (int i = 0; i < n; i++) {
        float q = roundf(src[i] / scale) + zero_point;
        q = fmaxf(fminf(q, 127.0f), -128.0f);
        dst[i] = (int8_t)q;
    }
}

// 非对称反量化
void dequantize_asymmetric_cpu(const int8_t* src, float* dst, int n, float scale, int zero_point) {
    for (int i = 0; i < n; i++) {
        dst[i] = (float)(src[i] - zero_point) * scale;
    }
}

// 计算非对称量化的参数
void compute_asymmetric_params(const float* data, int n, float& scale, int& zero_point) {
    float min_val = data[0];
    float max_val = data[0];
    for (int i = 1; i < n; i++) {
        min_val = fminf(min_val, data[i]);
        max_val = fmaxf(max_val, data[i]);
    }

    scale = (max_val - min_val) / 255.0f;
    zero_point = (int)roundf(-min_val / scale);

    // 钳制zero_point到INT8范围
    zero_point = std::max(-128, std::min(127, zero_point));
}

/**
 * CUDA核函数：对称量化
 */
__global__ void quantize_symmetric_kernel(const float* src, int8_t* dst, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float q = roundf(src[idx] / scale);
        q = fmaxf(fminf(q, 127.0f), -128.0f);
        dst[idx] = (int8_t)q;
    }
}

/**
 * CUDA核函数：对称反量化
 */
__global__ void dequantize_symmetric_kernel(const int8_t* src, float* dst, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (float)src[idx] * scale;
    }
}

/**
 * CUDA核函数：INT8向量加法
 */
__global__ void int8_vector_add(const int8_t* a, const int8_t* b, int8_t* c,
                                  int n, float scale_a, float scale_b, float scale_c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 反量化到FP32
        float fa = (float)a[idx] * scale_a;
        float fb = (float)b[idx] * scale_b;

        // FP32计算
        float fc = fa + fb;

        // 量化回INT8
        float q = roundf(fc / scale_c);
        q = fmaxf(fminf(q, 127.0f), -128.0f);
        c[idx] = (int8_t)q;
    }
}

/**
 * CUDA核函数：INT8向量加法（向量化版本）
 */
__global__ void int8_vector_add_vec4(const char4* a, const char4* b, char4* c,
                                       int n, float scale_a, float scale_b, float scale_c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 4) {
        char4 va = a[idx];
        char4 vb = b[idx];
        char4 vc;

        // 每个通道独立计算
        auto process = [&](int8_t a_val, int8_t b_val) -> int8_t {
            float fa = (float)a_val * scale_a;
            float fb = (float)b_val * scale_b;
            float fc = fa + fb;
            float q = roundf(fc / scale_c);
            return (int8_t)fmaxf(fminf(q, 127.0f), -128.0f);
        };

        vc.x = process(va.x, vb.x);
        vc.y = process(va.y, vb.y);
        vc.z = process(va.z, vb.z);
        vc.w = process(va.w, vb.w);

        c[idx] = vc;
    }
}

/**
 * 量化误差分析
 */
void quantization_error_analysis() {
    printf("\n========================================\n");
    printf("量化误差分析\n");
    printf("========================================\n\n");

    // 测试数据
    const int N = 1000;
    float* h_data = (float*)malloc(N * sizeof(float));
    int8_t* h_quantized = (int8_t*)malloc(N * sizeof(int8_t));
    float* h_dequantized = (float*)malloc(N * sizeof(float));

    // 生成测试数据
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 200 - 100) / 10.0f;  // -10到10
    }

    printf("对称量化误差分析:\n");

    // 对称量化
    float scale = compute_scale_symmetric(h_data, N);
    printf("  Scale: %.6f\n", scale);

    quantize_symmetric_cpu(h_data, h_quantized, N, scale);
    dequantize_symmetric_cpu(h_quantized, h_dequantized, N, scale);

    // 计算误差统计
    float max_error = 0, avg_error = 0;
    for (int i = 0; i < N; i++) {
        float error = fabsf(h_data[i] - h_dequantized[i]);
        max_error = fmaxf(max_error, error);
        avg_error += error;
    }
    avg_error /= N;

    printf("  最大误差: %.6f\n", max_error);
    printf("  平均误差: %.6f\n", avg_error);
    printf("  相对误差: %.2f%%\n", (avg_error / 10.0f) * 100);

    printf("\n非对称量化误差分析:\n");

    // 非对称量化
    float scale_asym;
    int zero_point;
    compute_asymmetric_params(h_data, N, scale_asym, zero_point);
    printf("  Scale: %.6f\n", scale_asym);
    printf("  Zero Point: %d\n", zero_point);

    quantize_asymmetric_cpu(h_data, h_quantized, N, scale_asym, zero_point);
    dequantize_asymmetric_cpu(h_quantized, h_dequantized, N, scale_asym, zero_point);

    // 计算误差统计
    max_error = 0, avg_error = 0;
    for (int i = 0; i < N; i++) {
        float error = fabsf(h_data[i] - h_dequantized[i]);
        max_error = fmaxf(max_error, error);
        avg_error += error;
    }
    avg_error /= N;

    printf("  最大误差: %.6f\n", max_error);
    printf("  平均误差: %.6f\n", avg_error);
    printf("  相对误差: %.2f%%\n", (avg_error / 10.0f) * 100);

    // 清理
    free(h_data);
    free(h_quantized);
    free(h_dequantized);
}

/**
 * 不同量化粒度对比
 */
void quantization_granularity_demo() {
    printf("\n========================================\n");
    printf("量化粒度对比\n");
    printf("========================================\n\n");

    printf("量化粒度类型:\n\n");

    printf("1. 张量级量化 (Per-Tensor):\n");
    printf("   - 整个张量使用同一个scale\n");
    printf("   - 参数量最少\n");
    printf("   - 精度损失较大\n\n");

    printf("2. 通道级量化 (Per-Channel):\n");
    printf("   - 每个通道使用独立的scale\n");
    printf("   - 参数量适中\n");
    printf("   - 精度损失较小\n");
    printf("   - 常用于权重量化\n\n");

    printf("3. 组级量化 (Per-Group):\n");
    printf("   - 每组元素使用独立的scale\n");
    printf("   - 参数量较多\n");
    printf("   - 精度损失最小\n\n");

    printf("选择建议:\n");
    printf("  - 权重量化: 通道级\n");
    printf("  - 激活量化: 张量级\n");
    printf("  - 高精度需求: 组级\n");
}

/**
 * INT8性能测试
 */
void int8_performance_test() {
    printf("\n========================================\n");
    printf("INT8性能测试\n");
    printf("========================================\n\n");

    const int N = 64 * 1024 * 1024;  // 64M元素
    const size_t fp32_size = N * sizeof(float);
    const size_t int8_size = N * sizeof(int8_t);
    const int block_size = 256;
    const int iterations = 100;

    printf("数据大小: %d 元素\n", N);
    printf("  FP32: %.2f MB\n", fp32_size / 1e6);
    printf("  INT8: %.2f MB (1/4)\n", int8_size / 1e6);

    // 分配内存
    float *d_fp32_a, *d_fp32_b, *d_fp32_c;
    int8_t *d_int8_a, *d_int8_b, *d_int8_c;
    char4 *d_int8_a4, *d_int8_b4, *d_int8_c4;

    CHECK_CUDA(cudaMalloc(&d_fp32_a, fp32_size));
    CHECK_CUDA(cudaMalloc(&d_fp32_b, fp32_size));
    CHECK_CUDA(cudaMalloc(&d_fp32_c, fp32_size));

    CHECK_CUDA(cudaMalloc(&d_int8_a, int8_size));
    CHECK_CUDA(cudaMalloc(&d_int8_b, int8_size));
    CHECK_CUDA(cudaMalloc(&d_int8_c, int8_size));

    CHECK_CUDA(cudaMalloc(&d_int8_a4, int8_size));
    CHECK_CUDA(cudaMalloc(&d_int8_b4, int8_size));
    CHECK_CUDA(cudaMalloc(&d_int8_c4, int8_size));

    // 初始化数据
    float* h_init = (float*)malloc(fp32_size);
    for (int i = 0; i < N; i++) {
        h_init[i] = (float)(i % 100) / 100.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_fp32_a, h_init, fp32_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_fp32_b, h_init, fp32_size, cudaMemcpyHostToDevice));

    // FP32向量加法核函数
    auto fp32_add_kernel = [] __device__(const float* a, const float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) c[idx] = a[idx] + b[idx];
    };

    // 创建流和事件
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int grid_size = (N + block_size - 1) / block_size;

    printf("\n%-25s | %-12s | %-12s\n", "内核", "时间(ms)", "带宽(GB/s)");
    printf("---------------------------------------------------\n");

    // FP32向量加法
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        // 使用简化的FP32加法
        quantize_symmetric_kernel<<<grid_size, block_size, 0, stream>>>(
            d_fp32_a, d_int8_a, N, 1.0f);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;

    printf("%-25s | %-10.3f | %-10.2f\n", "FP32操作", ms, fp32_size / ms / 1e6);

    // INT8量化
    float scale = 0.01f;

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        quantize_symmetric_kernel<<<grid_size, block_size, 0, stream>>>(
            d_fp32_a, d_int8_a, N, scale);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;
    printf("%-25s | %-10.3f | %-10.2f\n", "INT8量化", ms, int8_size / ms / 1e6);

    // INT8向量加法
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        int8_vector_add<<<grid_size, block_size, 0, stream>>>(
            d_int8_a, d_int8_b, d_int8_c, N, scale, scale, scale);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;
    printf("%-25s | %-10.3f | %-10.2f\n", "INT8向量加法", ms, 3.0f * int8_size / ms / 1e6);

    // INT8向量加法 (vec4)
    int grid_size_4 = (N / 4 + block_size - 1) / block_size;

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        int8_vector_add_vec4<<<grid_size_4, block_size, 0, stream>>>(
            (char4*)d_int8_a, (char4*)d_int8_b, (char4*)d_int8_c, N, scale, scale, scale);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;
    printf("%-25s | %-10.3f | %-10.2f\n", "INT8向量加法(vec4)", ms, 3.0f * int8_size / ms / 1e6);

    // 清理
    free(h_init);
    CHECK_CUDA(cudaFree(d_fp32_a));
    CHECK_CUDA(cudaFree(d_fp32_b));
    CHECK_CUDA(cudaFree(d_fp32_c));
    CHECK_CUDA(cudaFree(d_int8_a));
    CHECK_CUDA(cudaFree(d_int8_b));
    CHECK_CUDA(cudaFree(d_int8_c));
    CHECK_CUDA(cudaFree(d_int8_a4));
    CHECK_CUDA(cudaFree(d_int8_b4));
    CHECK_CUDA(cudaFree(d_int8_c4));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    printf("=============================================\n");
    printf("  第26章示例3：INT8量化基础\n");
    printf("=============================================\n");

    // 量化原理
    quantization_theory();

    // 误差分析
    quantization_error_analysis();

    // 量化粒度
    quantization_granularity_demo();

    // 性能测试
    int8_performance_test();

    printf("\n示例完成！\n");
    return 0;
}