/**
 * @file 03_warp_divergence.cu
 * @brief Warp Divergence（线程束发散）问题演示
 *
 * 本示例展示：
 * 1. 什么是Warp Divergence
 * 2. Divergence对性能的影响
 * 3. 如何检测和分析Divergence
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
// 版本1: 严重Divergence - 随机条件分支
// ============================================================================
__global__ void divergence_severe(int* data, int* even_sum, int* odd_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // 每个线程根据数据值决定分支
        // 如果数据随机分布，Warp内线程将走不同分支
        if (data[idx] % 2 == 0) {
            atomicAdd(even_sum, data[idx]);
        } else {
            atomicAdd(odd_sum, data[idx]);
        }
    }
}

// ============================================================================
// 版本2: 中等Divergence - 基于索引的分支
// ============================================================================
__global__ void divergence_moderate(int* data, int* even_sum, int* odd_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // 基于索引奇偶分支
        // 每个Warp内大约一半走if，一半走else
        if (idx % 2 == 0) {
            atomicAdd(even_sum, data[idx]);
        } else {
            atomicAdd(odd_sum, data[idx]);
        }
    }
}

// ============================================================================
// 版本3: 无Divergence - Warp内统一分支
// ============================================================================
__global__ void divergence_none(int* data, int* even_sum, int* odd_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / 32;
    int lane = idx % 32;

    // 整个Warp走同一分支（基于Warp ID）
    if (idx < N) {
        if (warp_id % 2 == 0) {
            atomicAdd(even_sum, data[idx]);
        } else {
            atomicAdd(odd_sum, data[idx]);
        }
    }
}

// ============================================================================
// 版本4: 使用谓词执行（Masking）
// ============================================================================
__global__ void divergence_predicate(int* data, int* even_sum, int* odd_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // 使用条件表达式替代分支
        // 编译器会生成谓词指令，避免分支跳转
        int val = data[idx];
        int is_even = (val % 2 == 0);

        // 两条路径都执行，但只有符合条件的写入
        // 注意：这里仍然有原子操作的开销
        if (is_even) {
            atomicAdd(even_sum, val);
        }
        if (!is_even) {
            atomicAdd(odd_sum, val);
        }
    }
}

// ============================================================================
// 版本5: 使用Warp级别规约减少Divergence影响
// ============================================================================
__global__ void divergence_warp_reduce(int* data, int* even_sum, int* odd_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;

    int local_even = 0;
    int local_odd = 0;

    if (idx < N) {
        int val = data[idx];
        if (val % 2 == 0) {
            local_even = val;
        } else {
            local_odd = val;
        }
    }

    // Warp内规约偶数和
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_even += __shfl_down_sync(0xffffffff, local_even, offset);
        local_odd += __shfl_down_sync(0xffffffff, local_odd, offset);
    }

    // 只有lane 0写入全局结果
    if (lane == 0) {
        if (local_even != 0) atomicAdd(even_sum, local_even);
        if (local_odd != 0) atomicAdd(odd_sum, local_odd);
    }
}

// ============================================================================
// 多重分支示例
// ============================================================================
__global__ void multiple_divergence(float* data, int N,
                                    int* count1, int* count2,
                                    int* count3, int* count4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = data[idx];

        // 多重分支，更严重的Divergence
        if (val < 0.25f) {
            atomicAdd(count1, 1);
        } else if (val < 0.5f) {
            atomicAdd(count2, 1);
        } else if (val < 0.75f) {
            atomicAdd(count3, 1);
        } else {
            atomicAdd(count4, 1);
        }
    }
}

// ============================================================================
// 多重分支优化版本
// ============================================================================
__global__ void multiple_optimized(float* data, int N,
                                   int* count1, int* count2,
                                   int* count3, int* count4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = data[idx];

        // 使用整数计算代替多重分支
        int bucket = (int)(val * 4);
        bucket = min(max(bucket, 0), 3);

        // 使用数组指针简化
        int* counts[] = {count1, count2, count3, count4};
        atomicAdd(counts[bucket], 1);
    }
}

// ============================================================================
// 计时辅助函数
// ============================================================================
template<typename Kernel, typename... Args>
float benchmark(Kernel kernel, int blocks, int threads,
                Args... args, int warmup = 5, int repeat = 20) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    for (int i = 0; i < warmup; i++) {
        kernel<<<blocks, threads>>>(args...);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        kernel<<<blocks, threads>>>(args...);
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
    int N = 8 * 1024 * 1024;  // 8M 元素
    size_t bytes = N * sizeof(int);
    size_t float_bytes = N * sizeof(float);

    printf("=== Warp Divergence示例 ===\n");
    printf("数据规模: %d 元素\n\n", N);

    // 分配内存
    int *h_data = (int*)malloc(bytes);
    float *h_float_data = (float*)malloc(float_bytes);

    // 初始化数据 - 随机值
    srand(42);
    for (int i = 0; i < N; i++) {
        h_data[i] = rand();
        h_float_data[i] = (float)rand() / RAND_MAX;
    }

    // 设备内存
    int *d_data, *d_even_sum, *d_odd_sum;
    float *d_float_data;
    int *d_counts[4];

    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_even_sum, sizeof(int));
    cudaMalloc(&d_odd_sum, sizeof(int));
    cudaMalloc(&d_float_data, float_bytes);
    for (int i = 0; i < 4; i++) {
        cudaMalloc(&d_counts[i], sizeof(int));
    }

    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_float_data, h_float_data, float_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float time_ms;

    printf("=== 偶数/奇数求和测试 ===\n");
    printf("----------------------------------------\n");

    auto reset_sums = [&]() {
        int zero = 0;
        cudaMemcpy(d_even_sum, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_odd_sum, &zero, sizeof(int), cudaMemcpyHostToDevice);
    };

    // 版本1: 严重Divergence
    reset_sums();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 5; i++) divergence_severe<<<blocks, threads>>>(d_data, d_even_sum, d_odd_sum, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        reset_sums();
        divergence_severe<<<blocks, threads>>>(d_data, d_even_sum, d_odd_sum, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= 20;
    printf("版本1 (严重Divergence): %.4f ms\n", time_ms);

    // 版本2: 中等Divergence
    reset_sums();
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        reset_sums();
        divergence_moderate<<<blocks, threads>>>(d_data, d_even_sum, d_odd_sum, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= 20;
    printf("版本2 (中等Divergence): %.4f ms\n", time_ms);

    // 版本3: 无Divergence
    reset_sums();
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        reset_sums();
        divergence_none<<<blocks, threads>>>(d_data, d_even_sum, d_odd_sum, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= 20;
    printf("版本3 (无Divergence):   %.4f ms\n", time_ms);

    // 版本4: Warp规约
    reset_sums();
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        reset_sums();
        divergence_warp_reduce<<<blocks, threads>>>(d_data, d_even_sum, d_odd_sum, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= 20;
    printf("版本4 (Warp规约):       %.4f ms\n", time_ms);

    // 验证结果
    int h_even, h_odd;
    cudaMemcpy(&h_even, d_even_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_odd, d_odd_sum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n偶数和: %d, 奇数和: %d\n", h_even, h_odd);

    // 多重分支测试
    printf("\n=== 多重分支测试 ===\n");
    printf("----------------------------------------\n");

    auto reset_counts = [&]() {
        int zero = 0;
        for (int i = 0; i < 4; i++) {
            cudaMemcpy(d_counts[i], &zero, sizeof(int), cudaMemcpyHostToDevice);
        }
    };

    // 多重分支
    reset_counts();
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        reset_counts();
        multiple_divergence<<<blocks, threads>>>(d_float_data, N,
                                                  d_counts[0], d_counts[1],
                                                  d_counts[2], d_counts[3]);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= 20;
    printf("多重分支 (Divergence): %.4f ms\n", time_ms);

    // 优化版本
    reset_counts();
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        reset_counts();
        multiple_optimized<<<blocks, threads>>>(d_float_data, N,
                                                 d_counts[0], d_counts[1],
                                                 d_counts[2], d_counts[3]);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= 20;
    printf("优化版本 (无分支):     %.4f ms\n", time_ms);

    // 清理
    free(h_data);
    free(h_float_data);
    cudaFree(d_data);
    cudaFree(d_even_sum);
    cudaFree(d_odd_sum);
    cudaFree(d_float_data);
    for (int i = 0; i < 4; i++) cudaFree(d_counts[i]);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n=== 分析建议 ===\n");
    printf("使用Nsight Compute分析Divergence:\n");
    printf("  ncu --set full --metrics \\\n");
    printf("    smsp__sass_branch_targets.sum,\\\n");
    printf("    smsp__average_branch_targets_threads_uniform.pct \\\n");
    printf("    ./03_warp_divergence\n");
    printf("\n关键指标:\n");
    printf("  - branch_targets: 分支目标数量（越少越好）\n");
    printf("  - uniform百分比: 均匀分支百分比（越高越好）\n");

    return 0;
}