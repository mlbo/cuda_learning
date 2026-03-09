/**
 * =============================================================================
 * 第二章：Bank Conflict 演示 - 树状规约中的冲突分析
 * =============================================================================
 *
 * 本示例演示 Bank Conflict 的产生过程和检测方法：
 * - 跨步访问产生的 Bank Conflict
 * - 树状规约中的 Bank Conflict 分析
 * - 使用 NCU 检测 Bank Conflict
 *
 * 编译：nvcc -o 02_bank_conflict_demo 02_bank_conflict_demo.cu
 * 分析：ncu --set full ./02_bank_conflict_demo
 * 分析：ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./02_bank_conflict_demo
 *
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// =============================================================================
// 核函数 1：跨步访问演示（stride = 2）- 产生 2-way Bank Conflict
// =============================================================================
__global__ void stride_access_conflict(float* input, float* output, int n) {
    // 共享内存声明
    __shared__ float smem[64];

    int tid = threadIdx.x;

    // 初始化共享内存（连续访问，无冲突）
    if (tid < 64) {
        smem[tid] = input[tid];
    }
    __syncthreads();

    // 跨步访问：stride = 2
    // 这种访问模式会产生 2-way Bank Conflict
    // Thread 0  -> smem[0]  -> Bank 0
    // Thread 1  -> smem[2]  -> Bank 2
    // Thread 16 -> smem[32] -> Bank 0  <- 与 Thread 0 冲突！
    // Thread 17 -> smem[34] -> Bank 2  <- 与 Thread 1 冲突！
    if (tid < 32) {
        output[tid] = smem[tid * 2];  // 2-way Bank Conflict!
    }
}

// =============================================================================
// 核函数 2：跨步访问演示（stride = 4）- 产生 4-way Bank Conflict
// =============================================================================
__global__ void stride_access_4way_conflict(float* input, float* output, int n) {
    __shared__ float smem[128];

    int tid = threadIdx.x;

    // 初始化（无冲突）
    if (tid < 128) {
        smem[tid] = input[tid];
    }
    __syncthreads();

    // 跨步访问：stride = 4
    // Thread 0  -> smem[0]   -> Bank 0
    // Thread 8  -> smem[32]  -> Bank 0  <- 冲突
    // Thread 16 -> smem[64]  -> Bank 0  <- 冲突
    // Thread 24 -> smem[96]  -> Bank 0  <- 冲突
    // 结果：4-way Bank Conflict
    if (tid < 32) {
        output[tid] = smem[tid * 4];  // 4-way Bank Conflict!
    }
}

// =============================================================================
// 核函数 3：对称树状规约（有 Bank Conflict）
// =============================================================================
__global__ void reduce_symmetric_conflict(float* data, float* result, int N) {
    // 对称树状规约：从两端向中间合并
    // 这种方法会产生 Bank Conflict
    __shared__ float smem[128];

    int tid = threadIdx.x;

    // 加载数据到共享内存
    if (tid < N) {
        smem[tid] = data[tid];
    }
    __syncthreads();

    // 对称规约
    // 问题：tid 和 (N - 1 - tid) 可能映射到同一 Bank
    // 例如：N = 64 时
    // tid = 0 -> smem[0]   -> Bank 0
    // tid = 63 -> smem[63]  -> Bank 31
    // 但是 tid = 16 访问 smem[16] + smem[47]
    // smem[16] -> Bank 16
    // smem[47] -> Bank 15 (不冲突)
    // 问题在于规约的后续步骤
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            // 当 s = 32 时：
            // tid = 0: smem[0] + smem[32] -> Bank 0 + Bank 0 (2-way 冲突!)
            // 因为 32 % 32 = 0，和 0 映射到同一 Bank
            smem[tid] += smem[tid + s];  // Bank Conflict!
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = smem[0];
    }
}

// =============================================================================
// 核函数 4：优化版树状规约（减少 Bank Conflict）
// =============================================================================
__global__ void reduce_optimized(float* data, float* result, int N) {
    __shared__ float smem[128];

    int tid = threadIdx.x;

    // 加载数据
    if (tid < N) {
        smem[tid] = data[tid];
    }
    __syncthreads();

    // 对半向前规约
    // 这种模式虽然仍有 Bank Conflict，但程度较轻
    // 关键：同一个线程顺序访问两个地址，不是并行的多线程访问
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            // tid 访问 smem[tid] 和 smem[tid + s]
            // 虽然可能映射到同一 Bank，但同一线程的两次访问是串行的
            // 不会产生 Bank Conflict（Bank Conflict 是多线程并行访问同一 Bank）
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = smem[0];
    }
}

// =============================================================================
// 核函数 5：完全无冲突的树状规约（交错存储）
// =============================================================================
__global__ void reduce_no_conflict(float* data, float* result, int N) {
    // 使用 Padding 消除 Bank Conflict
    // 将 smem[128] 改为 smem[129]，改变 Bank 映射
    __shared__ float smem[129];  // 注意：129 而不是 128

    int tid = threadIdx.x;

    // 加载数据（使用交错索引）
    if (tid < N) {
        smem[tid] = data[tid];
    }
    __syncthreads();

    // 树状规约
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            // 现在 Bank ID 计算方式改变
            // smem[tid + s] 的 Bank ID = (tid + s) % 32
            // 由于数组大小为 129，映射关系改变
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = smem[0];
    }
}

// =============================================================================
// 核函数 6：矩阵列访问（严重的 Bank Conflict）
// =============================================================================
__global__ void matrix_column_access_conflict(float* matrix, float* output,
                                               int width, int height) {
    // 32x32 的共享内存块
    __shared__ float tile[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 按行加载（无冲突）
    if (ty < height && tx < width) {
        tile[ty][tx] = matrix[ty * width + tx];
    }
    __syncthreads();

    // 按列读取（严重的 Bank Conflict！）
    // tile[tx][ty] 表示：
    // - tx 固定，ty 变化
    // - 列方向访问
    // - Bank ID = (tx * 32 + ty) % 32 = tx
    // - 所有 ty 变化时 Bank ID 不变！
    // - 同一列的 32 个元素映射到同一 Bank
    // - 32-way Bank Conflict!
    if (ty < height && tx < width) {
        output[ty * width + tx] = tile[tx][ty];  // 32-way Bank Conflict!
    }
}

// =============================================================================
// 工具函数：初始化数据
// =============================================================================
void init_data(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (float)(rand() % 100) / 10.0f;
    }
}

// =============================================================================
// 工具函数：验证结果
// =============================================================================
bool verify_result(float* result, float expected, const char* name) {
    float diff = fabsf(*result - expected);
    if (diff < 1e-3f) {
        printf("  [PASS] %s: 结果 = %.4f (预期 = %.4f)\n", name, *result, expected);
        return true;
    } else {
        printf("  [FAIL] %s: 结果 = %.4f (预期 = %.4f)\n", name, *result, expected);
        return false;
    }
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("=================================================================\n");
    printf("第二章：Bank Conflict 演示\n");
    printf("=================================================================\n\n");

    srand((unsigned int)time(NULL));

    // -------------------------------------------------------------------------
    // 第一部分：跨步访问演示
    // -------------------------------------------------------------------------
    printf("【第一部分：跨步访问产生的 Bank Conflict】\n");
    printf("-----------------------------------------------------------------\n");

    float *d_input, *d_output;
    int size = 128;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    float h_input[128];
    init_data(h_input, size);
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // 2-way Bank Conflict
    printf("\n1. stride = 2 访问（2-way Bank Conflict）\n");
    stride_access_conflict<<<1, 64>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    printf("  NCU 分析命令：ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./02_bank_conflict_demo\n");

    // 4-way Bank Conflict
    printf("\n2. stride = 4 访问（4-way Bank Conflict）\n");
    stride_access_4way_conflict<<<1, 128>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    printf("  预期 Bank Conflict 数量更大\n");

    // -------------------------------------------------------------------------
    // 第二部分：树状规约 Bank Conflict 分析
    // -------------------------------------------------------------------------
    printf("\n【第二部分：树状规约中的 Bank Conflict】\n");
    printf("-----------------------------------------------------------------\n");

    float *d_data, *d_result;
    int N = 128;

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    float h_data[128];
    init_data(h_data, N);
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // 计算预期结果
    float expected = 0.0f;
    for (int i = 0; i < N; i++) {
        expected += h_data[i];
    }

    // 测试各个版本
    float h_result;

    printf("\n1. 对称树状规约（有 Bank Conflict）\n");
    reduce_symmetric_conflict<<<1, 128>>>(d_data, d_result, N);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(&h_result, expected, "Symmetric Reduce");

    printf("\n2. 优化版树状规约\n");
    reduce_optimized<<<1, 128>>>(d_data, d_result, N);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(&h_result, expected, "Optimized Reduce");

    printf("\n3. 无冲突树状规约（Padding）\n");
    reduce_no_conflict<<<1, 128>>>(d_data, d_result, N);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(&h_result, expected, "No-Conflict Reduce");

    // -------------------------------------------------------------------------
    // 第三部分：矩阵列访问 Bank Conflict
    // -------------------------------------------------------------------------
    printf("\n【第三部分：矩阵列访问（最严重的 Bank Conflict）】\n");
    printf("-----------------------------------------------------------------\n");

    float *d_matrix, *d_matrix_output;
    int matrix_size = 32 * 32;

    cudaMalloc(&d_matrix, matrix_size * sizeof(float));
    cudaMalloc(&d_matrix_output, matrix_size * sizeof(float));

    float h_matrix[1024];
    init_data(h_matrix, matrix_size);
    cudaMemcpy(d_matrix, h_matrix, matrix_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid(1, 1);

    printf("\n按列访问 tile[tx][ty]：\n");
    printf("  - tx 固定，ty 变化\n");
    printf("  - Bank ID = (tx * 32 + ty) %% 32 = tx\n");
    printf("  - 所有 ty 变化时 Bank ID 不变\n");
    printf("  - 产生 32-way Bank Conflict（最严重！）\n");

    matrix_column_access_conflict<<<grid, block>>>(d_matrix, d_matrix_output, 32, 32);
    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------
    // 第四部分：NCU 分析指南
    // -------------------------------------------------------------------------
    printf("\n【第四部分：NCU Bank Conflict 分析指南】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n推荐的分析命令：\n\n");
    printf("1. 查看 Bank Conflict 总数：\n");
    printf("   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \\\n");
    printf("       ./02_bank_conflict_demo\n\n");

    printf("2. 分别查看加载和存储的 Bank Conflict：\n");
    printf("   ncu --metrics \\\n");
    printf("       l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\\\n");
    printf("       l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \\\n");
    printf("       ./02_bank_conflict_demo\n\n");

    printf("3. 完整分析报告：\n");
    printf("   ncu --set full -o bank_conflict_report ./02_bank_conflict_demo\n");
    printf("   ncu-ui bank_conflict_report.ncu-rep\n\n");

    printf("4. 在 NCU 界面中查看：\n");
    printf("   - 打开 'Source' 页面\n");
    printf("   - 查找 'L1 Wavefronts Shared Excessive' 列\n");
    printf("   - 非零值表示存在 Bank Conflict\n");

    // -------------------------------------------------------------------------
    // 第五部分：关键概念总结
    // -------------------------------------------------------------------------
    printf("\n【关键概念总结】\n");
    printf("-----------------------------------------------------------------\n");
    printf("1. Bank Conflict 定义：\n");
    printf("   同一 Warp 中多个线程同时访问同一 Bank 的不同地址\n\n");

    printf("2. N-way Bank Conflict：\n");
    printf("   - N 个线程访问同一 Bank\n");
    printf("   - 访问被串行化，延迟增加 N 倍\n\n");

    printf("3. 常见 Bank Conflict 场景：\n");
    printf("   - 跨步访问（stride = 2^n）\n");
    printf("   - 矩阵列访问\n");
    printf("   - 树状规约的某些步骤\n\n");

    printf("4. 不产生 Bank Conflict 的情况：\n");
    printf("   - 广播：多线程访问同一地址\n");
    printf("   - 多播：同一缓存行的访问\n\n");

    // -------------------------------------------------------------------------
    // 清理资源
    // -------------------------------------------------------------------------
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_matrix);
    cudaFree(d_matrix_output);

    printf("=================================================================\n");
    printf("演示完成！\n");
    printf("=================================================================\n");

    return 0;
}
