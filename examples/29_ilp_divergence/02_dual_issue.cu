/**
 * @file 02_dual_issue.cu
 * @brief Dual Issue（双发射）示例
 *
 * 本示例展示：
 * 1. Dual Issue的概念
 * 2. 如何利用Dual Issue提高性能
 * 3. 使用PTX内联实现精确控制
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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
// 版本1: 无Dual Issue优化 - 顺序执行
// ============================================================================
__global__ void saxpy_no_dual(float* __restrict__ y,
                              const float* __restrict__ x,
                              float a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // 顺序执行：加载 -> 计算 -> 存储
        float x_val = x[idx];
        float y_val = y[idx];
        y[idx] = a * x_val + y_val;
    }
}

// ============================================================================
// 版本2: 手动展开以启用Dual Issue
// ============================================================================
__global__ void saxpy_dual_issue(float* __restrict__ y,
                                 const float* __restrict__ x,
                                 float a, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    // 加载两个元素
    float x0 = x[idx];
    float x1 = x[idx + 1];
    float y0 = y[idx];
    float y1 = y[idx + 1];

    // 独立计算（可能与加载重叠）
    float r0 = a * x0 + y0;
    float r1 = a * x1 + y1;

    // 存储
    if (idx < N) y[idx] = r0;
    if (idx + 1 < N) y[idx + 1] = r1;
}

// ============================================================================
// 版本3: 使用PTX内联实现精确的Dual Issue控制
// ============================================================================
__global__ void saxpy_dual_ptx(float* __restrict__ y,
                               const float* __restrict__ x,
                               float a, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (idx + 1 < N) {
        float x0, x1, y0, y1;
        float r0, r1;

        // 使用PTX显式控制加载
        // ld.global.ca.f32 表示全局内存加载，缓存到L1
        asm("ld.global.ca.f32 %0, [%1];" : "=f"(x0) : "l"(x + idx));
        asm("ld.global.ca.f32 %0, [%1];" : "=f"(x1) : "l"(x + idx + 1));
        asm("ld.global.ca.f32 %0, [%1];" : "=f"(y0) : "l"(y + idx));
        asm("ld.global.ca.f32 %0, [%1];" : "=f"(y1) : "l"(y + idx + 1));

        // FMA运算
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r0) : "f"(a), "f"(x0), "f"(y0));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r1) : "f"(a), "f"(x1), "f"(y1));

        // 存储
        asm("st.global.wt.f32 [%0], %1;" : : "l"(y + idx), "f"(r0));
        asm("st.global.wt.f32 [%0], %1;" : : "l"(y + idx + 1), "f"(r1));
    }
}

// ============================================================================
// 版本4: 更激进的展开（x4）
// ============================================================================
__global__ void saxpy_unroll_x4(float* __restrict__ y,
                                const float* __restrict__ x,
                                float a, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // 寄存器存储
    float x_vals[4], y_vals[4], results[4];

    // 加载阶段 - 多个独立加载可以利用Dual Issue
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int gi = idx + i;
        if (gi < N) {
            x_vals[i] = x[gi];
            y_vals[i] = y[gi];
        }
    }

    // 计算阶段 - 多个独立FMA可以利用Dual Issue
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        results[i] = a * x_vals[i] + y_vals[i];
    }

    // 存储阶段
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int gi = idx + i;
        if (gi < N) {
            y[gi] = results[i];
        }
    }
}

// ============================================================================
// 混合计算与内存访问的示例
// ============================================================================
__global__ void mixed_kernel(float* __restrict__ data,
                             float* __restrict__ temp,
                             int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // 这种模式下，编译器可能无法有效调度
        float val = data[idx];
        val = val * 2.0f;
        temp[idx] = val;         // 中间存储
        val = temp[idx] + 1.0f;   // 再次加载
        data[idx] = val;
    }
}

__global__ void mixed_kernel_optimized(float* __restrict__ data,
                                       float* __restrict__ temp,
                                       int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // 消除不必要的中间存储，增加ILP机会
        float val = data[idx];
        val = val * 2.0f + 1.0f;  // 合并计算
        data[idx] = val;
        // temp可能不需要，或者用于其他目的
        temp[idx] = val;
    }
}

// ============================================================================
// 计时辅助函数
// ============================================================================
template<typename Kernel>
float benchmark(Kernel kernel, int blocks, int threads,
                float* y, const float* x, float a, int N,
                int warmup = 5, int repeat = 20) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    for (int i = 0; i < warmup; i++) {
        kernel<<<blocks, threads>>>(y, x, a, N);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        kernel<<<blocks, threads>>>(y, x, a, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeat;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    int N = 16 * 1024 * 1024;  // 16M 元素
    size_t bytes = N * sizeof(float);
    float a = 2.5f;

    printf("=== Dual Issue示例 ===\n");
    printf("数据规模: %d 元素 (%.2f MB)\n\n", N, bytes / 1024.0 / 1024.0);

    // 分配内存
    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_y_backup = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i / N;
        h_y[i] = (float)(N - i) / N;
        h_y_backup[i] = h_y[i];
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    float time_ms;
    double bandwidth;

    // 测试各个版本
    printf("SAXPY性能测试 (y = a*x + y):\n");
    printf("----------------------------------------\n");

    // 版本1
    cudaMemcpy(d_y, h_y_backup, bytes, cudaMemcpyHostToDevice);
    int blocks1 = (N + threads - 1) / threads;
    time_ms = benchmark(saxpy_no_dual, blocks1, threads, d_y, d_x, a, N);
    bandwidth = 3.0 * bytes / time_ms / 1e6;
    printf("版本1 (无优化):        %.4f ms  (%.2f GB/s)\n", time_ms, bandwidth);

    // 版本2
    cudaMemcpy(d_y, h_y_backup, bytes, cudaMemcpyHostToDevice);
    int blocks2 = (N / 2 + threads - 1) / threads;
    time_ms = benchmark(saxpy_dual_issue, blocks2, threads, d_y, d_x, a, N);
    bandwidth = 3.0 * bytes / time_ms / 1e6;
    printf("版本2 (Dual Issue x2): %.4f ms  (%.2f GB/s)\n", time_ms, bandwidth);

    // 版本3
    cudaMemcpy(d_y, h_y_backup, bytes, cudaMemcpyHostToDevice);
    time_ms = benchmark(saxpy_dual_ptx, blocks2, threads, d_y, d_x, a, N);
    bandwidth = 3.0 * bytes / time_ms / 1e6;
    printf("版本3 (PTX控制):       %.4f ms  (%.2f GB/s)\n", time_ms, bandwidth);

    // 版本4
    cudaMemcpy(d_y, h_y_backup, bytes, cudaMemcpyHostToDevice);
    int blocks4 = (N / 4 + threads - 1) / threads;
    time_ms = benchmark(saxpy_unroll_x4, blocks4, threads, d_y, d_x, a, N);
    bandwidth = 3.0 * bytes / time_ms / 1e6;
    printf("版本4 (展开x4):        %.4f ms  (%.2f GB/s)\n", time_ms, bandwidth);

    // 验证结果
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        float expected = a * h_x[i] + h_y_backup[i];
        if (fabsf(h_y[i] - expected) > 1e-4) {
            correct = false;
        }
    }
    printf("\n结果验证: %s\n", correct ? "正确" : "错误");

    // 清理
    free(h_x);
    free(h_y);
    free(h_y_backup);
    cudaFree(d_x);
    cudaFree(d_y);

    printf("\n=== 分析建议 ===\n");
    printf("使用Nsight Compute分析Dual Issue:\n");
    printf("  ncu --set full --metrics \\\n");
    printf("    smsp__inst_issued.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("    smsp__pipelined.avg.pct_of_peak_sustained_elapsed \\\n");
    printf("    ./02_dual_issue\n");
    printf("\n关键指标:\n");
    printf("  - inst_issued: 发射指令数（接近200%%表示Dual Issue效果好）\n");
    printf("  - pipelined: 流水线利用率\n");

    return 0;
}