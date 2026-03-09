/**
 * 第28章示例04：内存对齐优化示例
 *
 * 本示例展示内存对齐对性能的影响
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ============================================================================
// 对齐的重要性
// ============================================================================

/**
 * 自然对齐的结构体
 * 编译器会自动对齐
 */
struct AlignNatural {
    float x, y, z, w;  // 16字节，自然对齐
};

/**
 * 手动指定对齐
 */
struct __align__(16) AlignManual16 {
    float x, y;  // 8字节，但强制16字节对齐
};

struct __align__(32) AlignManual32 {
    float x, y, z, w;  // 16字节，但强制32字节对齐
};

/**
 * 未对齐的结构体（模拟）
 * 可能导致性能下降
 */
struct NotAligned {
    char a;
    float x, y, z;  // 可能在非对齐地址
};

// ============================================================================
// 向量化访问示例
// ============================================================================

/**
 * float4加载要求16字节对齐
 */
__global__ void load_float4_aligned(const float4* __restrict__ input,
                                    float4* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // float4加载，要求16字节对齐
        float4 val = input[idx];
        val.x *= 2.0f;
        val.y *= 2.0f;
        val.z *= 2.0f;
        val.w *= 2.0f;
        output[idx] = val;
    }
}

/**
 * float加载（无特殊对齐要求）
 */
__global__ void load_float_scalar(const float* __restrict__ input,
                                  float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
    }
}

/**
 * 处理未对齐的数据
 * 性能可能较差
 */
__global__ void load_unaligned(const float* __restrict__ input,
                               float* __restrict__ output, int N, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int data_idx = idx + offset;
    if (data_idx < N) {
        // 未对齐访问
        output[idx] = input[data_idx] * 2.0f;
    }
}

// ============================================================================
// 结构体对齐示例
// ============================================================================

/**
 * 使用对齐结构体
 */
__global__ void process_aligned_struct(const AlignNatural* __restrict__ input,
                                       AlignNatural* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        AlignNatural val = input[idx];
        val.x *= 2.0f;
        val.y *= 2.0f;
        val.z *= 2.0f;
        val.w *= 2.0f;
        output[idx] = val;
    }
}

// ============================================================================
// 共享内存对齐
// ============================================================================

/**
 * 对齐的共享内存
 */
__global__ void shared_mem_aligned(const float* __restrict__ input,
                                   float* __restrict__ output, int N) {
    // 声明对齐的共享内存
    __shared__ float sdata[256];  // 自动对齐到4字节边界

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // 处理数据
    sdata[tid] *= 2.0f;
    __syncthreads();

    // 写回
    if (idx < N) {
        output[idx] = sdata[tid];
    }
}

/**
 * 使用填充避免bank conflict
 */
__global__ void shared_mem_padded(const float* __restrict__ input,
                                  float* __restrict__ output, int N) {
    // 添加填充避免bank conflict
    __shared__ float sdata[257];  // 257 = 256 + 1 padding

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    sdata[tid] *= 2.0f;
    __syncthreads();

    if (idx < N) {
        output[idx] = sdata[tid];
    }
}

// ============================================================================
// 动态共享内存对齐
// ============================================================================

/**
 * 动态共享内存
 * 需要注意对齐
 */
__global__ void dynamic_shared_mem(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int N,
                                   size_t shared_mem_size) {
    // 动态共享内存
    extern __shared__ float sdata_dynamic[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_dynamic[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    sdata_dynamic[tid] *= 2.0f;
    __syncthreads();

    if (idx < N) {
        output[idx] = sdata_dynamic[tid];
    }
}

// ============================================================================
// 性能测试
// ============================================================================

void test_alignment_performance() {
    const int N = 1024 * 1024;
    const int BLOCK_SIZE = 256;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // 初始化
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("=== 内存对齐性能测试 ===\n\n");

    // 对齐访问测试
    printf("向量化访问对比:\n");

    // float4向量化加载（16字节对齐）
    float4 *d_input4, *d_output4;
    cudaMalloc(&d_input4, (N/4) * sizeof(float4));
    cudaMalloc(&d_output4, (N/4) * sizeof(float4));

    cudaEventRecord(start);
    load_float4_aligned<<<blocks/4, BLOCK_SIZE>>>(d_input4, d_output4, N/4);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_float4;
    cudaEventElapsedTime(&ms_float4, start, stop);
    printf("  float4对齐加载: %.3f ms\n", ms_float4);

    // 标量加载
    cudaEventRecord(start);
    load_float_scalar<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_scalar;
    cudaEventElapsedTime(&ms_scalar, start, stop);
    printf("  float标量加载:  %.3f ms\n", ms_scalar);

    // 未对齐访问测试
    printf("\n对齐 vs 未对齐访问:\n");

    // 对齐访问
    cudaEventRecord(start);
    load_unaligned<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_aligned;
    cudaEventElapsedTime(&ms_aligned, start, stop);
    printf("  对齐访问(offset=0): %.3f ms\n", ms_aligned);

    // 未对齐访问
    cudaEventRecord(start);
    load_unaligned<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_unaligned;
    cudaEventElapsedTime(&ms_unaligned, start, stop);
    printf("  未对齐(offset=1):   %.3f ms\n", ms_unaligned);

    // 共享内存测试
    printf("\n共享内存测试:\n");

    cudaEventRecord(start);
    shared_mem_aligned<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_shared_aligned;
    cudaEventElapsedTime(&ms_shared_aligned, start, stop);
    printf("  共享内存: %.3f ms\n", ms_shared_aligned);

    cudaEventRecord(start);
    shared_mem_padded<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_shared_padded;
    cudaEventElapsedTime(&ms_shared_padded, start, stop);
    printf("  共享内存(填充): %.3f ms\n", ms_shared_padded);

    // 清理
    delete[] h_data;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input4);
    cudaFree(d_output4);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== 内存对齐优化示例 ===\n\n");

    // 打印结构体大小和对齐信息
    printf("结构体对齐信息:\n");
    printf("  sizeof(AlignNatural) = %zu bytes\n", sizeof(AlignNatural));
    printf("  sizeof(AlignManual16) = %zu bytes\n", sizeof(AlignManual16));
    printf("  sizeof(AlignManual32) = %zu bytes\n", sizeof(AlignManual32));
    printf("  sizeof(NotAligned) = %zu bytes\n", sizeof(NotAligned));

    printf("\n对齐要求:\n");
    printf("  float: 4字节对齐\n");
    printf("  float2: 8字节对齐\n");
    printf("  float4: 16字节对齐\n");
    printf("  float8 (或使用__align__(32)): 32字节对齐\n");

    // 性能测试
    test_alignment_performance();

    printf("\n=== 对齐最佳实践 ===\n");
    printf("1. 使用cudaMalloc分配的内存自动对齐\n");
    printf("2. 结构体使用__align__(N)指定对齐\n");
    printf("3. 向量化访问(float4等)需要相应的对齐\n");
    printf("4. 共享内存可以使用填充避免bank conflict\n");
    printf("5. 动态共享内存注意对齐需求\n");

    printf("\n内存对齐示例完成！\n");
    return 0;
}