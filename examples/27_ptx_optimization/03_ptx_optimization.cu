/**
 * 第27章示例03：PTX性能优化案例
 *
 * 本示例展示使用PTX进行性能优化的实际案例
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ============================================================================
// 案例一：舍入模式控制
// ============================================================================

/**
 * 不同舍入模式的除法实现
 * PTX支持4种舍入模式：
 * - rn: Round to Nearest Even（IEEE 754默认）
 * - rz: Round toward Zero
 * - rp: Round Up (toward +Infinity)
 * - rm: Round Down (toward -Infinity)
 */
__device__ float div_rn(float a, float b) {
    float result;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ float div_rz(float a, float b) {
    float result;
    asm("div.rz.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ float div_ru(float a, float b) {
    float result;
    asm("div.rp.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ float div_rd(float a, float b) {
    float result;
    asm("div.rm.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

// ============================================================================
// 案例二：内存访问优化
// ============================================================================

/**
 * 对比普通加载与只读缓存加载
 */

// 普通加载版本
__global__ void reduce_normal(const float* __restrict__ data, float* result, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 普通加载
    sdata[tid] = (idx < N) ? data[idx] : 0.0f;
    __syncthreads();

    // Block内规约
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

// 使用PTX只读缓存加载版本
__global__ void reduce_readonly_cache(const float* __restrict__ data, float* result, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 使用只读缓存加载
    float val = 0.0f;
    if (idx < N) {
        // 使用PTX ld.global.nc指令
        asm("ld.global.nc.f32 %0, [%1];" : "=f"(val) : "l"(&data[idx]));
    }
    sdata[tid] = val;
    __syncthreads();

    // Block内规约
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
// 案例三：Warp Divergence优化
// ============================================================================

/**
 * 使用谓词寄存器避免warp divergence
 */

// 普通条件分支版本
__global__ void conditional_op_branch(float* data, int* conditions, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (conditions[idx]) {
            data[idx] = data[idx] * 2.0f;  // true分支
        } else {
            data[idx] = data[idx] + 1.0f;  // false分支
        }
    }
}

// 使用PTX谓词寄存器版本
__global__ void conditional_op_predicate(float* data, int* conditions, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        float true_val = val * 2.0f;
        float false_val = val + 1.0f;

        // 使用条件选择，编译器会生成优化的PTX（使用谓词寄存器）
        float result = conditions[idx] ? true_val : false_val;

        data[idx] = result;
    }
}

// ============================================================================
// 案例四：SIMD优化示例
// ============================================================================

/**
 * 使用SIMD指令加速字节级操作
 */

// 普通版本：逐字节计算绝对差
__global__ void sad_normal(const unsigned char* A, const unsigned char* B,
                           unsigned int* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 4) {
        unsigned int sad = 0;
        for (int i = 0; i < 4; i++) {
            int byte_idx = idx * 4 + i;
            int diff = (int)A[byte_idx] - (int)B[byte_idx];
            sad += (diff > 0) ? diff : -diff;
        }
        result[idx] = sad;
    }
}

// PTX SIMD版本：一次处理4字节
__device__ unsigned int vabsdiff4_u32(unsigned int A, unsigned int B) {
    unsigned int result;
    asm("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(result) : "r"(A), "r"(B), "r"(0));
    return result;
}

__global__ void sad_simd(const unsigned int* A, const unsigned int* B,
                         unsigned int* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 4) {
        // 每次处理4个字节
        result[idx] = vabsdiff4_u32(A[idx], B[idx]);
    }
}

// ============================================================================
// 性能测试函数
// ============================================================================

void test_performance() {
    const int N = 1024 * 1024;
    const int BLOCK_SIZE = 256;

    float *d_data, *d_result;
    int *d_conditions;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMalloc(&d_conditions, N * sizeof(int));

    // 初始化数据
    float* h_data = new float[N];
    int* h_conditions = new int[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
        h_conditions[i] = i % 2;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conditions, h_conditions, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 测试规约性能
    printf("\n=== 规约性能测试 ===\n");

    cudaMemset(d_result, 0, sizeof(float));
    cudaEventRecord(start);
    reduce_normal<<<blocks, BLOCK_SIZE>>>(d_data, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_normal;
    cudaEventElapsedTime(&ms_normal, start, stop);
    printf("普通加载: %.3f ms\n", ms_normal);

    cudaMemset(d_result, 0, sizeof(float));
    cudaEventRecord(start);
    reduce_readonly_cache<<<blocks, BLOCK_SIZE>>>(d_data, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_readonly;
    cudaEventElapsedTime(&ms_readonly, start, stop);
    printf("只读缓存: %.3f ms\n", ms_readonly);

    // 测试条件操作性能
    printf("\n=== 条件操作性能测试 ===\n");

    cudaEventRecord(start);
    conditional_op_branch<<<blocks, BLOCK_SIZE>>>(d_data, d_conditions, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_branch;
    cudaEventElapsedTime(&ms_branch, start, stop);
    printf("普通分支: %.3f ms\n", ms_branch);

    cudaEventRecord(start);
    conditional_op_predicate<<<blocks, BLOCK_SIZE>>>(d_data, d_conditions, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_predicate;
    cudaEventElapsedTime(&ms_predicate, start, stop);
    printf("谓词寄存器: %.3f ms\n", ms_predicate);

    // 清理
    delete[] h_data;
    delete[] h_conditions;
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_conditions);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// 测试内核
// ============================================================================
__global__ void test_rounding_modes(float* results) {
    if (threadIdx.x == 0) {
        printf("\n=== 舍入模式测试 ===\n");

        float a = 1.0f;
        float b = 3.0f;

        printf("除法 1.0/3.0 的不同舍入模式:\n");
        printf("  rn (最近偶数): %.20f\n", div_rn(a, b));
        printf("  rz (向零):     %.20f\n", div_rz(a, b));
        printf("  ru (向上):     %.20f\n", div_ru(a, b));
        printf("  rd (向下):     %.20f\n", div_rd(a, b));

        results[0] = div_rn(a, b);
        results[1] = div_rz(a, b);
        results[2] = div_ru(a, b);
        results[3] = div_rd(a, b);

        // 测试负数
        float c = -1.0f;
        printf("\n除法 -1.0/3.0 的不同舍入模式:\n");
        printf("  rn (最近偶数): %.20f\n", div_rn(c, b));
        printf("  rz (向零):     %.20f\n", div_rz(c, b));
        printf("  ru (向上):     %.20f\n", div_ru(c, b));
        printf("  rd (向下):     %.20f\n", div_rd(c, b));
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== PTX性能优化案例 ===\n");

    // 测试舍入模式
    float* d_results;
    cudaMalloc(&d_results, 4 * sizeof(float));
    test_rounding_modes<<<1, 32>>>(d_results);
    cudaDeviceSynchronize();

    // 性能测试
    test_performance();

    cudaFree(d_results);

    printf("\nPTX性能优化示例完成！\n");
    return 0;
}