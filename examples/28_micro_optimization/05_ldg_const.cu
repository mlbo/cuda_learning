/**
 * 第28章示例05：__ldg 与常量内存使用示例
 *
 * 本示例展示如何使用__ldg和常量内存进行只读数据优化
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ============================================================================
// 常量内存声明
// ============================================================================

// 常量内存声明（最大64KB）
__constant__ float const_coefficients[1024];
__constant__ int const_lookup_table[256];

// ============================================================================
// __ldg 内置函数示例
// ============================================================================

/**
 * 普通全局内存加载
 */
__global__ void load_normal(const float* __restrict__ input,
                            float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 普通加载
        output[idx] = input[idx];
    }
}

/**
 * 使用__ldg通过只读缓存加载
 * __ldg使用纹理缓存（只读缓存）路径
 */
__global__ void load_ldg(const float* __restrict__ input,
                         float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 通过只读缓存加载
        output[idx] = __ldg(&input[idx]);
    }
}

/**
 * 不规则访问模式下__ldg的优势
 * 只读缓存对于不规则访问更有效
 */
__global__ void gather_normal(const float* __restrict__ input,
                              float* __restrict__ output,
                              const int* __restrict__ indices, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 不规则访问：通过索引数组访问
        int data_idx = indices[idx];
        output[idx] = input[data_idx];
    }
}

__global__ void gather_ldg(const float* __restrict__ input,
                           float* __restrict__ output,
                           const int* __restrict__ indices, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int data_idx = __ldg(&indices[idx]);
        output[idx] = __ldg(&input[data_idx]);
    }
}

// ============================================================================
// 常量内存示例
// ============================================================================

/**
 * 使用常量内存存储系数
 * 适合所有线程读取相同值的情况
 */
__global__ void use_constant_memory(const float* __restrict__ input,
                                    float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 从常量内存读取系数
        // 如果warp内所有线程读取相同地址，会被广播
        float coeff = const_coefficients[idx % 1024];
        output[idx] = input[idx] * coeff;
    }
}

/**
 * 常量查找表示例
 */
__global__ void use_lookup_table(const unsigned char* __restrict__ input,
                                 float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 使用常量查找表
        unsigned char val = input[idx];
        output[idx] = (float)const_lookup_table[val];
    }
}

/**
 * 常量内存广播示例
 * 所有线程读取相同值时，常量内存广播机制非常高效
 */
__global__ void constant_broadcast(float* output, int N, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 使用常量内存中的第一个值（所有线程读取相同地址）
    float broadcast_val = const_coefficients[0];

    if (idx < N) {
        output[idx] = broadcast_val * scale;
    }
}

// ============================================================================
// 对比示例：全局内存 vs 常量内存 vs __ldg
// ============================================================================

// 使用全局内存存储系数
__global__ void coeff_global(const float* __restrict__ input,
                             const float* __restrict__ coeff,
                             float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * coeff[idx % 1024];
    }
}

// 使用__ldg读取系数
__global__ void coeff_ldg(const float* __restrict__ input,
                          const float* __restrict__ coeff,
                          float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float c = __ldg(&coeff[idx % 1024]);
        output[idx] = input[idx] * c;
    }
}

// 使用常量内存存储系数
__global__ void coeff_constant(const float* __restrict__ input,
                               float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * const_coefficients[idx % 1024];
    }
}

// ============================================================================
// 性能测试
// ============================================================================

void test_ldg_performance() {
    const int N = 1024 * 1024;
    const int BLOCK_SIZE = 256;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // 初始化
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i % 1000);
    }
    cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("=== __ldg 性能测试 ===\n\n");

    // 规则访问
    printf("规则访问模式:\n");

    cudaEventRecord(start);
    load_normal<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_normal;
    cudaEventElapsedTime(&ms_normal, start, stop);
    printf("  普通加载: %.3f ms\n", ms_normal);

    cudaEventRecord(start);
    load_ldg<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_ldg;
    cudaEventElapsedTime(&ms_ldg, start, stop);
    printf("  __ldg加载: %.3f ms\n", ms_ldg);

    // 不规则访问
    printf("\n不规则访问模式 (gather):\n");

    int* d_indices;
    cudaMalloc(&d_indices, N * sizeof(int));

    // 创建随机索引
    int* h_indices = new int[N];
    for (int i = 0; i < N; i++) {
        h_indices[i] = rand() % N;
    }
    cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    gather_normal<<<blocks, BLOCK_SIZE>>>(d_input, d_output, d_indices, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_normal, start, stop);
    printf("  普通gather: %.3f ms\n", ms_normal);

    cudaEventRecord(start);
    gather_ldg<<<blocks, BLOCK_SIZE>>>(d_input, d_output, d_indices, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_ldg, start, stop);
    printf("  __ldg gather: %.3f ms\n", ms_ldg);

    delete[] h_indices;
    cudaFree(d_indices);
    delete[] h_data;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void test_constant_memory() {
    const int N = 1024 * 1024;
    const int BLOCK_SIZE = 256;

    float *d_input, *d_output, *d_coeff;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_coeff, 1024 * sizeof(float));

    // 初始化
    float* h_data = new float[N];
    float* h_coeff = new float[1024];
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    for (int i = 0; i < 1024; i++) {
        h_coeff[i] = (float)(i + 1) / 1024.0f;
    }

    cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeff, h_coeff, 1024 * sizeof(float), cudaMemcpyHostToDevice);

    // 复制到常量内存
    cudaMemcpyToSymbol(const_coefficients, h_coeff, 1024 * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\n=== 常量内存性能测试 ===\n\n");

    printf("系数访问对比:\n");

    // 全局内存
    cudaEventRecord(start);
    coeff_global<<<blocks, BLOCK_SIZE>>>(d_input, d_coeff, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_global;
    cudaEventElapsedTime(&ms_global, start, stop);
    printf("  全局内存: %.3f ms\n", ms_global);

    // __ldg
    cudaEventRecord(start);
    coeff_ldg<<<blocks, BLOCK_SIZE>>>(d_input, d_coeff, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_ldg;
    cudaEventElapsedTime(&ms_ldg, start, stop);
    printf("  __ldg: %.3f ms\n", ms_ldg);

    // 常量内存
    cudaEventRecord(start);
    coeff_constant<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_const;
    cudaEventElapsedTime(&ms_const, start, stop);
    printf("  常量内存: %.3f ms\n", ms_const);

    delete[] h_data;
    delete[] h_coeff;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_coeff);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// 使用指南
// ============================================================================

void print_guidelines() {
    printf("\n=== __ldg 与常量内存使用指南 ===\n\n");

    printf("__ldg 适用场景:\n");
    printf("  - 数据只读，不会被修改\n");
    printf("  - 访问模式不规则（如gather操作）\n");
    printf("  - 数据量超过常量内存限制(64KB)\n");
    printf("  - 需要利用纹理缓存特性\n\n");

    printf("常量内存适用场景:\n");
    printf("  - 数据量 <= 64KB\n");
    printf("  - 所有线程读取相同值（广播机制）\n");
    printf("  - 查找表、系数数组等\n");
    printf("  - 数据在内核执行期间不变\n\n");

    printf("选择建议:\n");
    printf("  1. 所有线程读取相同值 -> 常量内存\n");
    printf("  2. 小型只读数据 -> 常量内存\n");
    printf("  3. 大型只读数据 -> __ldg\n");
    printf("  4. 不规则访问的只读数据 -> __ldg\n");
    printf("  5. 规则访问的大型数据 -> 普通全局内存可能足够\n");
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== __ldg 与常量内存使用示例 ===\n\n");

    // __ldg性能测试
    test_ldg_performance();

    // 常量内存测试
    test_constant_memory();

    // 打印使用指南
    print_guidelines();

    printf("\n__ldg 与常量内存示例完成！\n");
    return 0;
}