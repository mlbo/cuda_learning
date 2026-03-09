/**
 * 第28章示例01：#pragma unroll 循环展开优化
 *
 * 本示例展示如何使用#pragma unroll进行循环展开优化
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ============================================================================
// 循环展开的收益与代价
// ============================================================================

/**
 * 不展开的循环
 * 编译器可能保持循环结构，产生循环控制开销
 */
__global__ void loop_no_unroll(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll 1  // 明确告诉编译器不要展开
    for (int i = 0; i < 8; i++) {
        int data_idx = idx * 8 + i;
        if (data_idx < N) {
            data[data_idx] = data[data_idx] * 2.0f;
        }
    }
}

/**
 * 完全展开的循环
 * 消除循环控制开销，提高ILP
 */
__global__ void loop_full_unroll(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll  // 完全展开
    for (int i = 0; i < 8; i++) {
        int data_idx = idx * 8 + i;
        if (data_idx < N) {
            data[data_idx] = data[data_idx] * 2.0f;
        }
    }
}

/**
 * 部分展开
 * 对于大循环，部分展开可以平衡代码大小和性能
 */
__global__ void loop_partial_unroll(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每次展开4次迭代
    #pragma unroll 4
    for (int i = 0; i < 32; i++) {
        int data_idx = idx * 32 + i;
        if (data_idx < N) {
            data[data_idx] = data[data_idx] * 2.0f;
        }
    }
}

// ============================================================================
// 实际应用案例
// ============================================================================

/**
 * 规约中的循环展开
 * 展开规约循环可以显著提升性能
 */

// 未展开的规约
__global__ void reduce_no_unroll(float* input, float* output, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // 未展开的规约循环
    #pragma unroll 1
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// 展开的规约
__global__ void reduce_with_unroll(float* input, float* output, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // 展开规约循环（假设blockDim.x = 256）
    if (blockDim.x >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
        __syncthreads();
    }

    // 最后64个线程的展开规约
    #pragma unroll
    for (int s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

/**
 * 向量运算中的循环展开
 */
__global__ void saxpy_unroll(float* y, const float* x, float a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 每个线程处理4个元素
    #pragma unroll 4
    for (int i = idx; i < N; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

// ============================================================================
// 寄存器压力分析示例
// ============================================================================

/**
 * 过度展开可能导致寄存器溢出
 * 这个例子展示寄存器压力增加的情况
 */
__global__ void heavy_register_usage(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 大量局部变量
    float a0, a1, a2, a3, a4, a5, a6, a7;
    float b0, b1, b2, b3, b4, b5, b6, b7;

    // 完全展开可能导致寄存器溢出
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int data_idx = idx * 16 + i;
        if (data_idx < N) {
            float val = data[data_idx];
            // 复杂计算增加寄存器压力
            switch (i) {
                case 0: a0 = val; break;
                case 1: a1 = val; break;
                case 2: a2 = val; break;
                case 3: a3 = val; break;
                case 4: a4 = val; break;
                case 5: a5 = val; break;
                case 6: a6 = val; break;
                case 7: a7 = val; break;
                case 8: b0 = val; break;
                case 9: b1 = val; break;
                case 10: b2 = val; break;
                case 11: b3 = val; break;
                case 12: b4 = val; break;
                case 13: b5 = val; break;
                case 14: b6 = val; break;
                case 15: b7 = val; break;
            }
        }
    }

    // 使用变量防止被优化掉
    data[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
                b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7;
}

// ============================================================================
// 性能测试函数
// ============================================================================

void test_loop_unroll_performance() {
    const int N = 1024 * 1024;
    const int BLOCK_SIZE = 256;

    float *d_data, *d_output;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // 初始化数据
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\n=== 循环展开性能测试 ===\n");

    // 测试不展开
    cudaEventRecord(start);
    loop_no_unroll<<<blocks, BLOCK_SIZE>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_no_unroll;
    cudaEventElapsedTime(&ms_no_unroll, start, stop);
    printf("不展开:       %.3f ms\n", ms_no_unroll);

    // 测试完全展开
    cudaEventRecord(start);
    loop_full_unroll<<<blocks, BLOCK_SIZE>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_full_unroll;
    cudaEventElapsedTime(&ms_full_unroll, start, stop);
    printf("完全展开:     %.3f ms\n", ms_full_unroll);

    // 测试部分展开
    cudaEventRecord(start);
    loop_partial_unroll<<<blocks / 4, BLOCK_SIZE>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_partial;
    cudaEventElapsedTime(&ms_partial, start, stop);
    printf("部分展开:     %.3f ms\n", ms_partial);

    printf("\n=== 规约性能测试 ===\n");

    // 测试规约
    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start);
    reduce_no_unroll<<<blocks, BLOCK_SIZE>>>(d_data, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_reduce_no_unroll;
    cudaEventElapsedTime(&ms_reduce_no_unroll, start, stop);
    printf("规约(不展开): %.3f ms\n", ms_reduce_no_unroll);

    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start);
    reduce_with_unroll<<<blocks, BLOCK_SIZE>>>(d_data, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_reduce_unroll;
    cudaEventElapsedTime(&ms_reduce_unroll, start, stop);
    printf("规约(展开):   %.3f ms\n", ms_reduce_unroll);

    // 清理
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== #pragma unroll 循环展开优化示例 ===\n");

    // 运行性能测试
    test_loop_unroll_performance();

    printf("\n=== 最佳实践 ===\n");
    printf("1. 小循环（< 32次迭代）可以完全展开\n");
    printf("2. 大循环使用部分展开（如 #pragma unroll 4）\n");
    printf("3. 注意寄存器压力，避免过度展开\n");
    printf("4. 规约循环展开可以显著提升性能\n");
    printf("5. 使用 -Xptxas -v 查看寄存器使用情况\n");

    printf("\n#pragma unroll 示例完成！\n");
    return 0;
}