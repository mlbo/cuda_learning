/**
 * 第28章示例02：__restrict__ 限定符使用示例
 *
 * 本示例展示__restrict__限定符如何影响编译器优化
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ============================================================================
// 指针别名问题演示
// ============================================================================

/**
 * 没有使用__restrict__的版本
 * 编译器必须假设a和c可能指向同一内存
 */
__global__ void add_no_restrict(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 编译器无法确定a和c是否重叠
        // 必须保守处理，可能产生冗余内存访问
        a[idx] = b[idx] + c[idx];
    }
}

/**
 * 使用__restrict__的版本
 * 告诉编译器这些指针不会重叠
 */
__global__ void add_with_restrict(
    float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 编译器知道指针不重叠，可以优化内存访问
        a[idx] = b[idx] + c[idx];
    }
}

// ============================================================================
// 更复杂的例子：展示__restrict__的优化效果
// ============================================================================

/**
 * SAXPY: y = a * x + y
 * 如果没有__restrict__，编译器必须假设x和y可能重叠
 */
__global__ void saxpy_no_restrict(float* y, float* x, float a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 编译器必须假设写入y可能影响x的值
        // 可能产生冗余加载
        y[idx] = a * x[idx] + y[idx];
    }
}

__global__ void saxpy_with_restrict(
    float* __restrict__ y,
    const float* __restrict__ x,
    float a,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 编译器知道x和y不重叠
        // 可以安全地优化内存访问
        y[idx] = a * x[idx] + y[idx];
    }
}

/**
 * 向量点积：result = sum(a[i] * b[i])
 */
__global__ void dot_product_no_restrict(
    float* a, float* b, float* result, int N)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < N) {
        // 没有restrict，编译器可能需要多次加载
        val = a[idx] * b[idx];
    }
    sdata[tid] = val;
    __syncthreads();

    // 规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

__global__ void dot_product_with_restrict(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    int N)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < N) {
        // 使用restrict，编译器可以优化加载
        val = a[idx] * b[idx];
    }
    sdata[tid] = val;
    __syncthreads();

    // 规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// ============================================================================
// 多指针情况下的__restrict__使用
// ============================================================================

/**
 * 多个输出指针
 */
__global__ void multi_output_no_restrict(
    float* out1, float* out2, const float* in, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 编译器不知道out1和out2是否重叠
        float val = in[idx];
        out1[idx] = val * 2.0f;
        out2[idx] = val * 3.0f;
    }
}

__global__ void multi_output_with_restrict(
    float* __restrict__ out1,
    float* __restrict__ out2,
    const float* __restrict__ in,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = in[idx];
        out1[idx] = val * 2.0f;
        out2[idx] = val * 3.0f;
    }
}

// ============================================================================
// 正确性测试：展示aliasing问题
// ============================================================================

/**
 * 测试没有__restrict__时aliasing的影响
 * 这个例子中a和c指向同一内存
 */
void test_aliasing() {
    printf("\n=== Aliasing测试 ===\n");

    const int N = 16;
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // 初始化数据
    float h_data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    cudaMemcpy(d_b, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // 正常情况：a、b、c不重叠
    add_no_restrict<<<1, 16>>>(d_a, d_b, d_c, N);

    float h_result[16];
    cudaMemcpy(h_result, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("正常情况（指针不重叠）:\n");
    printf("  b: [1, 2, 3, ..., 16]\n");
    printf("  c: [1, 2, 3, ..., 16]\n");
    printf("  a = b + c: [% .1f, % .1f, ..., % .1f]\n", h_result[0], h_result[1], h_result[15]);

    // Aliasing情况：a和c指向同一内存
    // 注意：这是错误用法，展示为什么需要注意aliasing
    printf("\n警告：以下展示aliasing问题（错误用法）\n");
    printf("如果 a 和 c 指向同一内存，结果会不同\n");
    printf("使用 __restrict__ 时必须确保指针确实不重叠\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// ============================================================================
// 性能测试
// ============================================================================

void test_performance() {
    const int N = 1024 * 1024;
    const int BLOCK_SIZE = 256;

    float *d_a, *d_b, *d_c, *d_result;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // 初始化
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i % 100);
    }
    cudaMemcpy(d_a, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\n=== 性能对比测试 ===\n");

    // SAXPY测试
    printf("\nSAXPY (y = a*x + y):\n");

    cudaEventRecord(start);
    saxpy_no_restrict<<<blocks, BLOCK_SIZE>>>(d_a, d_b, 2.0f, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_no_restrict;
    cudaEventElapsedTime(&ms_no_restrict, start, stop);
    printf("  无 __restrict__: %.3f ms\n", ms_no_restrict);

    // 重置数据
    cudaMemcpy(d_a, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    saxpy_with_restrict<<<blocks, BLOCK_SIZE>>>(d_a, d_b, 2.0f, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_with_restrict;
    cudaEventElapsedTime(&ms_with_restrict, start, stop);
    printf("  有 __restrict__: %.3f ms\n", ms_with_restrict);

    // 点积测试
    printf("\n向量点积:\n");

    cudaMemset(d_result, 0, sizeof(float));
    cudaEventRecord(start);
    dot_product_no_restrict<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_no_restrict, start, stop);
    printf("  无 __restrict__: %.3f ms\n", ms_no_restrict);

    cudaMemset(d_result, 0, sizeof(float));
    cudaEventRecord(start);
    dot_product_with_restrict<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_with_restrict, start, stop);
    printf("  有 __restrict__: %.3f ms\n", ms_with_restrict);

    // 清理
    delete[] h_data;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== __restrict__ 限定符使用示例 ===\n");

    // Aliasing测试
    test_aliasing();

    // 性能测试
    test_performance();

    printf("\n=== __restrict__ 使用建议 ===\n");
    printf("1. 对所有不重叠的指针参数使用 __restrict__\n");
    printf("2. 特别是只读输入指针，使用 const float* __restrict__\n");
    printf("3. 确保指针确实不重叠，否则会产生未定义行为\n");
    printf("4. 编译器会据此消除冗余加载，提升性能\n");

    printf("\n__restrict__ 示例完成！\n");
    return 0;
}