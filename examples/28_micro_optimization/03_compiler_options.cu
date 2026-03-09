/**
 * 第28章示例03：编译器选项调优示例
 *
 * 本示例展示各种编译器选项对性能的影响
 *
 * 编译命令示例：
 *   标准优化: nvcc -O3 -arch=sm_80 03_compiler_options.cu
 *   快速数学: nvcc -O3 -arch=sm_80 -use_fast_math 03_compiler_options.cu
 *   限制寄存器: nvcc -O3 -arch=sm_80 -maxrregcount=32 03_compiler_options.cu
 *   查看详情: nvcc -O3 -arch=sm_80 -Xptxas -v 03_compiler_options.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// 快速数学选项示例
// ============================================================================

/**
 * 使用标准数学函数
 * 这些函数遵循IEEE 754精度要求
 */
__global__ void math_standard(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        // 标准数学函数
        output[idx] = sinf(x) + cosf(x) + expf(x) + logf(x + 1.0f);
    }
}

/**
 * 使用CUDA内置快速数学函数
 * __sinf, __cosf等是__device__函数，精度略低但速度更快
 */
__global__ void math_fast(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        // 快速数学函数
        // 注意：精度在 -pi ~ pi 范围内最好
        output[idx] = __sinf(x) + __cosf(x) + __expf(x) + __logf(x + 1.0f);
    }
}

/**
 * 除法精度选项示例
 */
__global__ void division_test(float* output, float a, float b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 标准除法（默认高精度）
        output[idx] = a / (b + idx);
    }
}

// ============================================================================
// 寄存器限制示例
// ============================================================================

/**
 * 使用较多寄存器的内核
 * 限制寄存器数量可能影响性能
 */
__global__ void register_heavy(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 使用多个局部变量增加寄存器压力
        float a0 = input[idx] * 1.0f;
        float a1 = input[idx] * 2.0f;
        float a2 = input[idx] * 3.0f;
        float a3 = input[idx] * 4.0f;
        float a4 = input[idx] * 5.0f;
        float a5 = input[idx] * 6.0f;
        float a6 = input[idx] * 7.0f;
        float a7 = input[idx] * 8.0f;

        // 复杂计算
        float result = a0 + a1 * sinf(a2) + a3 * cosf(a4) +
                       a5 * expf(a6 * 0.1f) + a7 * logf(a0 + 1.0f);

        output[idx] = result;
    }
}

/**
 * 寄存器使用较少的内核
 */
__global__ void register_light(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 简单计算
        output[idx] = input[idx] * 2.0f;
    }
}

// ============================================================================
// 缓存配置选项示例
// ============================================================================

/**
 * 默认缓存策略
 */
__global__ void cache_default(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 默认缓存行为
        output[idx] = input[idx];
    }
}

/**
 * 流式访问模式
 * 使用 cg (Cache Global) 绕过L1
 */
__global__ void cache_global(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 编译时使用 -Xptxas -dlcm=cg 可以影响这个行为
        output[idx] = input[idx];
    }
}

// ============================================================================
// 占用率分析
// ============================================================================

void print_occupancy_info(int block_size, int regs_per_thread, int shared_mem_per_block) {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                   register_heavy,
                                                   block_size,
                                                   shared_mem_per_block);

    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int active_threads = max_blocks_per_sm * block_size;
    float occupancy = (float)active_threads / max_threads_per_sm * 100;

    printf("  Block size: %d\n", block_size);
    printf("  Max blocks per SM: %d\n", max_blocks_per_sm);
    printf("  Active threads: %d / %d\n", active_threads, max_threads_per_sm);
    printf("  Occupancy: %.1f%%\n", occupancy);
}

// ============================================================================
// 性能测试函数
// ============================================================================

void test_math_options() {
    const int N = 1024 * 1024;
    const int BLOCK_SIZE = 256;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // 初始化输入
    float* h_input = new float[N];
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 100) / 100.0f * 3.14159f;
    }
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("=== 数学函数性能测试 ===\n");

    // 标准数学
    cudaEventRecord(start);
    math_standard<<<blocks, BLOCK_SIZE>>>(d_output, d_input, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_standard;
    cudaEventElapsedTime(&ms_standard, start, stop);
    printf("  标准数学函数: %.3f ms\n", ms_standard);

    // 快速数学
    cudaEventRecord(start);
    math_fast<<<blocks, BLOCK_SIZE>>>(d_output, d_input, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_fast;
    cudaEventElapsedTime(&ms_fast, start, stop);
    printf("  快速数学函数: %.3f ms\n", ms_fast);

    // 验证精度差异
    float* h_output_standard = new float[N];
    float* h_output_fast = new float[N];
    cudaMemcpy(h_output_standard, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    math_standard<<<blocks, BLOCK_SIZE>>>(d_output, d_input, N);
    math_fast<<<blocks, BLOCK_SIZE>>>(d_input, d_input, N);  // 复用输入缓冲

    cudaDeviceSynchronize();
    cudaMemcpy(h_output_fast, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_output_standard[i] - h_output_fast[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  最大精度差异: %.6f\n", max_diff);

    delete[] h_input;
    delete[] h_output_standard;
    delete[] h_output_fast;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void test_register_options() {
    printf("\n=== 寄存器限制影响测试 ===\n");
    printf("\n编译时使用以下选项查看寄存器使用情况:\n");
    printf("  nvcc -Xptxas -v ...\n");
    printf("\n使用以下选项限制寄存器数量:\n");
    printf("  nvcc -maxrregcount=N ...\n");

    printf("\n当前内核占用率分析:\n");
    print_occupancy_info(256, 32, 0);  // 假设32个寄存器/线程
}

// ============================================================================
// 编译选项说明
// ============================================================================

void print_compiler_options() {
    printf("\n=== 常用编译选项说明 ===\n\n");

    printf("优化级别:\n");
    printf("  -O0          无优化（调试用）\n");
    printf("  -O1          基本优化\n");
    printf("  -O2          标准优化\n");
    printf("  -O3          激进优化（推荐）\n\n");

    printf("精度相关:\n");
    printf("  -prec-div=false    快速除法（降低精度）\n");
    printf("  -ftz=true          非规格化数flush为零\n");
    printf("  -use_fast_math     快速数学库\n\n");

    printf("寄存器控制:\n");
    printf("  -maxrregcount=N    限制每线程寄存器数\n");
    printf("  -Xptxas -v         显示编译详情\n\n");

    printf("缓存控制:\n");
    printf("  -Xptxas -dlcm=ca   Cache All（默认）\n");
    printf("  -Xptxas -dlcm=cg   Cache Global（绕过L1）\n");
    printf("  -Xptxas -dlcm=cs   Cache Streaming\n\n");

    printf("架构指定:\n");
    printf("  -arch=sm_75        Turing架构\n");
    printf("  -arch=sm_80        Ampere架构（A100）\n");
    printf("  -arch=sm_86        Ampere架构（RTX 3090）\n");
    printf("  -arch=sm_89        Ada Lovelace架构\n");
    printf("  -arch=sm_90        Hopper架构\n");
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== 编译器选项调优示例 ===\n");

    // 数学函数测试
    test_math_options();

    // 寄存器选项说明
    test_register_options();

    // 打印编译选项说明
    print_compiler_options();

    printf("\n编译器选项示例完成！\n");
    return 0;
}