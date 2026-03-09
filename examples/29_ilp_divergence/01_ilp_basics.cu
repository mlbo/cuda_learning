/**
 * @file 01_ilp_basics.cu
 * @brief ILP（指令级并行）基础概念演示
 *
 * 本示例展示：
 * 1. 什么是ILP
 * 2. 如何通过增加每个线程的工作量来提高ILP
 * 3. ILP对性能的影响
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA错误检查宏
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
// 版本1: 无ILP - 每个线程处理一个元素
// ============================================================================
__global__ void add_no_ilp(float* __restrict__ a,
                           float* __restrict__ b,
                           float* __restrict__ c,
                           int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================================================
// 版本2: ILP x2 - 每个线程处理两个元素
// ============================================================================
__global__ void add_ilp_x2(float* __restrict__ a,
                           float* __restrict__ b,
                           float* __restrict__ c,
                           int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    // 处理两个元素，增加指令级并行机会
    if (idx < N) {
        float a0 = a[idx];
        float a1 = a[idx + 1];
        float b0 = b[idx];
        float b1 = b[idx + 1];

        // 这些独立的加载和计算可以并行
        c[idx] = a0 + b0;
        c[idx + 1] = a1 + b1;
    }
}

// ============================================================================
// 版本3: ILP x4 - 每个线程处理四个元素
// ============================================================================
__global__ void add_ilp_x4(float* __restrict__ a,
                           float* __restrict__ b,
                           float* __restrict__ c,
                           int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // 使用寄存器存储多个值
    float a_vals[4], b_vals[4], c_vals[4];

    // 加载阶段 - 多个独立加载可以ILP
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int gi = idx + i;
        if (gi < N) {
            a_vals[i] = a[gi];
            b_vals[i] = b[gi];
        }
    }

    // 计算阶段 - 多个独立计算可以ILP
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c_vals[i] = a_vals[i] + b_vals[i];
    }

    // 存储阶段
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int gi = idx + i;
        if (gi < N) {
            c[gi] = c_vals[i];
        }
    }
}

// ============================================================================
// 版本4: ILP x4 使用向量加载（更高效）
// ============================================================================
__global__ void add_ilp_x4_vector(float4* __restrict__ a,
                                  float4* __restrict__ b,
                                  float4* __restrict__ c,
                                  int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // 使用float4向量加载，一次加载4个float
        float4 va = a[idx];
        float4 vb = b[idx];

        // 独立计算
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        c[idx] = vc;
    }
}

// ============================================================================
// 计时函数
// ============================================================================
template <void (*Kernel)(float*, float*, float*, int)>
float time_kernel(float* d_a, float* d_b, float* d_c, int N,
                  int blocks, int threads, int warmup = 5, int repeat = 20) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    for (int i = 0; i < warmup; i++) {
        Kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        Kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / repeat;
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv) {
    int N = 32 * 1024 * 1024;  // 32M 元素
    size_t bytes = N * sizeof(float);

    printf("=== ILP基础示例 ===\n");
    printf("数据规模: %d 元素 (%.2f MB)\n\n", N, bytes / 1024.0 / 1024.0);

    // 分配主机内存
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    // 初始化数据
    srand(42);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 拷贝数据
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    float time_ms;

    // 版本1: 无ILP
    int blocks1 = (N + threads - 1) / threads;
    time_ms = time_kernel<add_no_ilp>(d_a, d_b, d_c, N, blocks1, threads);
    printf("版本1 (无ILP):      %.4f ms  (%.2f GB/s)\n",
           time_ms, 3.0 * bytes / time_ms / 1e6);

    // 版本2: ILP x2
    int blocks2 = (N / 2 + threads - 1) / threads;
    time_ms = time_kernel<add_ilp_x2>(d_a, d_b, d_c, N, blocks2, threads);
    printf("版本2 (ILP x2):     %.4f ms  (%.2f GB/s)\n",
           time_ms, 3.0 * bytes / time_ms / 1e6);

    // 版本3: ILP x4
    int blocks3 = (N / 4 + threads - 1) / threads;
    time_ms = time_kernel<add_ilp_x4>(d_a, d_b, d_c, N, blocks3, threads);
    printf("版本3 (ILP x4):     %.4f ms  (%.2f GB/s)\n",
           time_ms, 3.0 * bytes / time_ms / 1e6);

    // 版本4: ILP x4 向量化
    int N4 = N / 4;
    float4 *d_a4, *d_b4, *d_c4;
    CUDA_CHECK(cudaMalloc(&d_a4, N4 * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b4, N4 * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_c4, N4 * sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a4, d_a, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_b4, d_b, bytes, cudaMemcpyDeviceToDevice));

    int blocks4 = (N4 + threads - 1) / threads;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    for (int i = 0; i < 5; i++) {
        add_ilp_x4_vector<<<blocks4, threads>>>(d_a4, d_b4, d_c4, N4);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 20; i++) {
        add_ilp_x4_vector<<<blocks4, threads>>>(d_a4, d_b4, d_c4, N4);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    time_ms /= 20;

    printf("版本4 (ILP x4 向量): %.4f ms  (%.2f GB/s)\n",
           time_ms, 3.0 * bytes / time_ms / 1e6);

    // 验证结果
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabsf(h_c[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("\n结果验证: %s\n", correct ? "正确" : "错误");

    // 清理
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_a4));
    CUDA_CHECK(cudaFree(d_b4));
    CUDA_CHECK(cudaFree(d_c4));

    printf("\n提示: 使用 ncu 分析ILP效果:\n");
    printf("  ncu --set full --metrics smsp__inst_issued.sum ./01_ilp_basics\n");

    return 0;
}
