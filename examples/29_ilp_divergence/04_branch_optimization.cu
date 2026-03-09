/**
 * @file 04_branch_optimization.cu
 * @brief 分支优化策略示例
 *
 * 本示例展示：
 * 1. 数据重排优化
 * 2. 谓词执行
 * 3. Warp级别优化
 * 4. 条件过滤优化
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
// 问题：条件过滤 - 过滤出大于阈值的元素
// ============================================================================

// 朴素实现：有严重Divergence
__global__ void filter_naive(int* input, int* output, int* count,
                             int N, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        if (input[idx] > threshold) {
            int pos = atomicAdd(count, 1);
            output[pos] = input[idx];
        }
    }
}

// 优化1：Warp级别Stream Compaction
__global__ void filter_warp_compact(int* input, int* output, int* count,
                                    int N, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // 加载值和判断条件
    int val = 0;
    int valid = 0;

    if (idx < N) {
        val = input[idx];
        valid = (val > threshold) ? 1 : 0;
    }

    // 计算Warp内的ballot掩码
    unsigned int mask = __ballot_sync(0xffffffff, valid);

    // 计算当前线程在有效元素中的位置（前缀和）
    int warp_offset = __popc(mask & ((1 << lane) - 1));
    int warp_count = __popc(mask);

    // Warp内第一个线程负责申请全局位置
    __shared__ int block_base;
    if (threadIdx.x == 0) {
        block_base = atomicAdd(count, warp_count);
    }
    __syncthreads();

    // 有效线程写入输出
    if (valid) {
        output[block_base + warp_offset] = val;
    }
}

// 优化2：Block级别规约后再写入
__global__ void filter_block_reduce(int* input, int* output, int* count,
                                    int N, int threshold) {
    __shared__ int smem[256];  // 存储有效元素
    __shared__ int valid_count;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) valid_count = 0;
    __syncthreads();

    // 每个线程处理一个元素
    int local_pos = -1;
    if (idx < N && input[idx] > threshold) {
        local_pos = atomicAdd(&valid_count, 1);
        smem[local_pos] = input[idx];
    }
    __syncthreads();

    // 只有线程0负责写入全局内存
    if (tid == 0 && valid_count > 0) {
        int global_pos = atomicAdd(count, valid_count);
        for (int i = 0; i < valid_count; i++) {
            output[global_pos + i] = smem[i];
        }
    }
}

// ============================================================================
// 问题：条件计算 - 根据条件执行不同操作
// ============================================================================

// 朴素实现
__global__ void conditional_compute_naive(float* data, int N, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        if (data[idx] > threshold) {
            data[idx] = data[idx] * 2.0f;
        } else {
            data[idx] = data[idx] * 0.5f;
        }
    }
}

// 使用谓词执行
__global__ void conditional_compute_predicate(float* data, int N, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = data[idx];
        // 使用条件表达式，编译器生成谓词指令
        float factor = (val > threshold) ? 2.0f : 0.5f;
        data[idx] = val * factor;
    }
}

// 使用数学函数避免分支
__global__ void conditional_compute_math(float* data, int N, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = data[idx];
        // 使用copysignf等数学函数
        // 这里用一个技巧：条件为真时factor=2，为假时factor=0.5
        int cond = (val > threshold);
        float factor = 0.5f + 1.5f * cond;  // cond=1时factor=2, cond=0时factor=0.5
        data[idx] = val * factor;
    }
}

// 使用条件表达式（编译器会生成谓词指令）
__device__ __forceinline__ float conditional_mul_ptx(float val, float threshold) {
    // 现代CUDA编译器会自动优化条件表达式为谓词指令
    float factor = (val > threshold) ? 2.0f : 0.5f;
    return val * factor;
}

__global__ void conditional_compute_ptx(float* data, int N, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        data[idx] = conditional_mul_ptx(data[idx], threshold);
    }
}

// ============================================================================
// 问题：分段函数 - 多重条件
// ============================================================================

// 朴素实现：多重分支
__global__ void piecewise_naive(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = data[idx];

        if (x < 0.0f) {
            data[idx] = 0.0f;
        } else if (x < 1.0f) {
            data[idx] = x * x;
        } else if (x < 2.0f) {
            data[idx] = 2.0f - x;
        } else {
            data[idx] = 0.0f;
        }
    }
}

// 优化：使用条件表达式链
__global__ void piecewise_conditional(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = data[idx];

        // 使用嵌套条件表达式
        float result = (x < 0.0f) ? 0.0f :
                       (x < 1.0f) ? x * x :
                       (x < 2.0f) ? (2.0f - x) : 0.0f;

        data[idx] = result;
    }
}

// 优化：使用clamp和数学函数
__device__ __forceinline__ float piecewise_func(float x) {
    // 分段函数的数学表达
    // f(x) = x^2 for 0 <= x < 1
    // f(x) = 2-x for 1 <= x < 2
    // f(x) = 0 otherwise

    float x1 = fminf(fmaxf(x, 0.0f), 1.0f);  // clamp to [0, 1]
    float x2 = fminf(fmaxf(x, 1.0f), 2.0f);  // clamp to [1, 2]

    // 在[0,1]区间: x^2
    float f1 = x1 * x1;
    // 在[1,2]区间: 2-x
    float f2 = 2.0f - x2;

    // 选择正确的值
    float in_range1 = (x >= 0.0f && x < 1.0f);
    float in_range2 = (x >= 1.0f && x < 2.0f);

    return f1 * in_range1 + f2 * in_range2;
}

__global__ void piecewise_math(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        data[idx] = piecewise_func(data[idx]);
    }
}

// ============================================================================
// 性能测试
// ============================================================================
int main() {
    int N = 8 * 1024 * 1024;
    size_t int_bytes = N * sizeof(int);
    size_t float_bytes = N * sizeof(float);

    printf("=== 分支优化策略示例 ===\n");
    printf("数据规模: %d 元素\n\n", N);

    // 分配主机内存
    int* h_input = (int*)malloc(int_bytes);
    float* h_data = (float*)malloc(float_bytes);
    float* h_data_backup = (float*)malloc(float_bytes);

    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = rand();
        h_data[i] = (float)rand() / RAND_MAX * 3.0f;  // 0-3范围
        h_data_backup[i] = h_data[i];
    }

    // 设备内存
    int *d_input, *d_output, *d_count;
    float *d_data;
    cudaMalloc(&d_input, int_bytes);
    cudaMalloc(&d_output, int_bytes);
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_data, float_bytes);

    cudaMemcpy(d_input, h_input, int_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data, float_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms;

    // ========== 条件过滤测试 ==========
    printf("=== 条件过滤测试 ===\n");
    printf("----------------------------------------\n");

    int threshold = RAND_MAX / 2;
    int zero = 0;

    // 朴素版本
    cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
        filter_naive<<<blocks, threads>>>(d_input, d_output, d_count, N, threshold);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("朴素版本:           %.4f ms\n", time_ms / 20);

    // Warp Compaction
    cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
        filter_warp_compact<<<blocks, threads>>>(d_input, d_output, d_count, N, threshold);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("Warp Compaction:    %.4f ms\n", time_ms / 20);

    // Block Reduce
    cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
        filter_block_reduce<<<blocks, threads>>>(d_input, d_output, d_count, N, threshold);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("Block Reduce:       %.4f ms\n", time_ms / 20);

    // ========== 条件计算测试 ==========
    printf("\n=== 条件计算测试 ===\n");
    printf("----------------------------------------\n");

    float fthreshold = 1.5f;

    // 朴素版本
    cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
        conditional_compute_naive<<<blocks, threads>>>(d_data, N, fthreshold);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("朴素版本:           %.4f ms\n", time_ms / 20);

    // 谓词版本
    cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
        conditional_compute_predicate<<<blocks, threads>>>(d_data, N, fthreshold);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("谓词版本:           %.4f ms\n", time_ms / 20);

    // 数学函数版本
    cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
        conditional_compute_math<<<blocks, threads>>>(d_data, N, fthreshold);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("数学函数版本:       %.4f ms\n", time_ms / 20);

    // PTX版本
    cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
        conditional_compute_ptx<<<blocks, threads>>>(d_data, N, fthreshold);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("PTX谓词版本:        %.4f ms\n", time_ms / 20);

    // ========== 分段函数测试 ==========
    printf("\n=== 分段函数测试 ===\n");
    printf("----------------------------------------\n");

    // 朴素版本
    cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
        piecewise_naive<<<blocks, threads>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("朴素版本 (多重分支): %.4f ms\n", time_ms / 20);

    // 条件表达式版本
    cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
        piecewise_conditional<<<blocks, threads>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("条件表达式版本:     %.4f ms\n", time_ms / 20);

    // 数学函数版本
    cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) {
        cudaMemcpy(d_data, h_data_backup, float_bytes, cudaMemcpyHostToDevice);
        piecewise_math<<<blocks, threads>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("数学函数版本:       %.4f ms\n", time_ms / 20);

    // 清理
    free(h_input);
    free(h_data);
    free(h_data_backup);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n=== 优化策略总结 ===\n");
    printf("1. 数据重排: 预处理数据使Warp内分支统一\n");
    printf("2. 谓词执行: 使用条件表达式替代if-else\n");
    printf("3. 数学函数: 使用fmin/fmax/copysign等避免分支\n");
    printf("4. Warp级别: 使用shuffle/ballot等Warp原语\n");
    printf("5. PTX内联: 直接使用谓词寄存器\n");

    return 0;
}