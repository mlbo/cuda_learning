/**
 * =============================================================================
 * 第五章：矩阵转置优化案例 - 综合对比各种方法
 * =============================================================================
 *
 * 本示例综合对比 Bank Conflict 的各种优化方法：
 * - Naive 版本（无优化）
 * - Padding 解决方案
 * - XOR Swizzling 解决方案
 * - 性能测试与 NCU 分析
 *
 * 编译：nvcc -o 05_matrix_transpose 05_matrix_transpose.cu
 * 运行：./05_matrix_transpose
 * 分析：ncu --set full -o transpose_report ./05_matrix_transpose
 *
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// =============================================================================
// 常量定义
// =============================================================================
#define TILE_SIZE 32

// =============================================================================
// 版本 1：Naive 转置（无共享内存，直接全局内存）
// =============================================================================
__global__ void transpose_naive_global(float* input, float* output,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

// =============================================================================
// 版本 2：共享内存转置（无 Padding，有 Bank Conflict）
// =============================================================================
__global__ void transpose_shared_no_padding(float* input, float* output,
                                             int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 加载到共享内存（按行加载，合并访问）
    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    }
    __syncthreads();

    // 转置后写回全局内存
    // 注意：这里有 Bank Conflict！
    // tile[tx][ty] 表示访问同一列
    // Bank ID = (tx * 32 + ty) % 32 = tx
    // 同一 tx 的所有 ty 映射到同一 Bank
    int new_x = blockIdx.y * TILE_SIZE + tx;
    int new_y = blockIdx.x * TILE_SIZE + ty;

    if (new_x < height && new_y < width) {
        output[new_y * height + new_x] = tile[tx][ty];  // Bank Conflict!
    }
}

// =============================================================================
// 版本 3：共享内存转置（有 Padding，无 Bank Conflict）
// =============================================================================
__global__ void transpose_shared_padding(float* input, float* output,
                                          int width, int height) {
    // 关键：第二维加 1，消除 Bank Conflict
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 加载
    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    }
    __syncthreads();

    // 转置写回
    // Bank ID = (tx * 33 + ty) % 32 = (tx + ty) % 32
    // 不同 ty 映射到不同 Bank，无冲突
    int new_x = blockIdx.y * TILE_SIZE + tx;
    int new_y = blockIdx.x * TILE_SIZE + ty;

    if (new_x < height && new_y < width) {
        output[new_y * height + new_x] = tile[tx][ty];
    }
}

// =============================================================================
// 版本 4：共享内存转置（XOR Swizzling，无 Bank Conflict）
// =============================================================================
__global__ void transpose_shared_xor(float* input, float* output,
                                      int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    // XOR Swizzling
    int swizzled_row = ty ^ tx;

    // 加载
    if (x < width && y < height) {
        tile[swizzled_row][tx] = input[y * width + x];
    }
    __syncthreads();

    // 转置写回
    int read_swizzled_row = tx ^ ty;  // XOR 交换律：tx ^ ty = ty ^ tx

    int new_x = blockIdx.y * TILE_SIZE + tx;
    int new_y = blockIdx.x * TILE_SIZE + ty;

    if (new_x < height && new_y < width) {
        output[new_y * height + new_x] = tile[read_swizzled_row][ty];
    }
}

// =============================================================================
// 版本 5：优化版（Padding + 边界处理优化）
// =============================================================================
__global__ void transpose_optimized(float* input, float* output,
                                     int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    // 使用寄存器缓存
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算 Block 起始位置
    int block_x = blockIdx.x * TILE_SIZE;
    int block_y = blockIdx.y * TILE_SIZE;

    // 加载（使用向量加载可以进一步优化）
    int x = block_x + tx;
    int y = block_y + ty;

    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    }
    __syncthreads();

    // 写回
    int new_x = block_y + tx;
    int new_y = block_x + ty;

    if (new_x < height && new_y < width) {
        output[new_y * height + new_x] = tile[tx][ty];
    }
}

// =============================================================================
// 工具函数
// =============================================================================
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 1000) / 10.0f;
    }
}

bool verify_transpose(float* original, float* transposed, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float orig = original[i * cols + j];
            float trans = transposed[j * rows + i];
            if (fabsf(orig - trans) > 1e-3f) {
                printf("  验证失败：[%d,%d] 原始=%.2f, 转置=%.2f\n",
                       i, j, orig, trans);
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// 性能测试函数
// =============================================================================
void benchmark_all_versions(int size) {
    printf("\n【矩阵大小：%d x %d】\n", size, size);

    float *d_input, *d_output;
    size_t bytes = size * size * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // 初始化
    float* h_input = (float*)malloc(bytes);
    init_matrix(h_input, size, size);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size + TILE_SIZE - 1) / TILE_SIZE,
              (size + TILE_SIZE - 1) / TILE_SIZE);

    // 计时事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 带宽计算参数
    float total_bytes = 2.0f * size * size * sizeof(float);  // 读 + 写

    // 定义测试函数指针和名称
    typedef void (*kernel_func)(float*, float*, int, int);

    struct {
        const char* name;
        kernel_func func;
        const char* conflict_status;
    } kernels[] = {
        {"Naive (全局内存)",    transpose_naive_global,      "无共享内存"},
        {"共享内存无Padding",    transpose_shared_no_padding, "有 32-way 冲突"},
        {"共享内存+Padding",     transpose_shared_padding,    "无冲突"},
        {"共享内存+XOR Swizzle", transpose_shared_xor,        "无冲突"},
        {"优化版",              transpose_optimized,         "无冲突"}
    };

    printf("\n+--------------------------+------------+-----------+---------------+\n");
    printf("| 方法                     | 时间 (ms)  | 带宽(GB/s)| Bank Conflict |\n");
    printf("+--------------------------+------------+-----------+---------------+\n");

    for (int k = 0; k < 5; k++) {
        cudaMemset(d_output, 0, bytes);

        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) {  // 多次运行取平均
            void* args[] = { &d_input, &d_output, &size, &size };
            cudaLaunchKernel((const void*)kernels[k].func, grid, block, args, 0, 0);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= 10.0f;

        float bandwidth = total_bytes / (ms / 1000.0f) / 1e9f;

        printf("| %-24s | %10.3f | %9.2f | %-13s |\n",
               kernels[k].name, ms, bandwidth, kernels[k].conflict_status);
    }
    printf("+--------------------------+------------+-----------+---------------+\n");

    // 验证正确性
    cudaMemcpy(h_input, d_output, bytes, cudaMemcpyDeviceToHost);
    float* h_original = (float*)malloc(bytes);
    init_matrix(h_original, size, size);
    if (verify_transpose(h_original, h_input, size, size)) {
        printf("\n结果验证：通过\n");
    }

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_original);
}

// =============================================================================
// NCU 分析指南
// =============================================================================
void print_ncu_guide() {
    printf("\n【NCU 分析指南】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n1. 基本 Bank Conflict 分析：\n");
    printf("   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \\\n");
    printf("       ./05_matrix_transpose\n\n");

    printf("2. 分别查看加载和存储的 Bank Conflict：\n");
    printf("   ncu --metrics \\\n");
    printf("       l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\\\n");
    printf("       l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \\\n");
    printf("       ./05_matrix_transpose\n\n");

    printf("3. 查看共享内存效率指标：\n");
    printf("   ncu --metrics \\\n");
    printf("       l1tex__t_requests_pipe_lsu_mem_shared.sum,\\\n");
    printf("       l1tex__t_sectors_pipe_lsu_mem_shared.sum,\\\n");
    printf("       l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \\\n");
    printf("       ./05_matrix_transpose\n\n");

    printf("4. 生成完整报告：\n");
    printf("   ncu --set full -o transpose_report ./05_matrix_transpose\n");
    printf("   ncu-ui transpose_report.ncu-rep\n\n");

    printf("5. 在 NCU UI 中查看 Bank Conflict：\n");
    printf("   a. 打开 'Details' 页面\n");
    printf("   b. 找到 'Memory Workload Analysis' 部分\n");
    printf("   c. 查看 'Shared Memory' 的 Bank Conflict 指标\n");
    printf("   d. 打开 'Source' 页面定位到具体代码行\n");
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("=================================================================\n");
    printf("第五章：矩阵转置优化案例\n");
    printf("=================================================================\n");

    srand((unsigned int)time(NULL));

    // -------------------------------------------------------------------------
    // 第一部分：问题背景
    // -------------------------------------------------------------------------
    printf("\n【问题背景】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n矩阵转置是一个经典的 Bank Conflict 示例：\n");
    printf("  - 按行读取：合并访问，高效\n");
    printf("  - 按列写入：跨步访问，效率低\n");
    printf("  - 使用共享内存缓存 tile 可以优化\n");
    printf("  - 但共享内存的列访问会触发 Bank Conflict\n\n");

    // -------------------------------------------------------------------------
    // 第二部分：各版本原理
    // -------------------------------------------------------------------------
    printf("【各版本优化原理】\n");
    printf("-----------------------------------------------------------------\n");

    printf("\n版本 1 - Naive（全局内存）：\n");
    printf("  直接在全局内存中转置\n");
    printf("  优点：简单\n");
    printf("  缺点：写入不合并，性能最差\n\n");

    printf("版本 2 - 共享内存无 Padding：\n");
    printf("  使用 tile[TILE_SIZE][TILE_SIZE]\n");
    printf("  优点：利用共享内存缓存\n");
    printf("  缺点：32-way Bank Conflict\n\n");

    printf("版本 3 - 共享内存 + Padding：\n");
    printf("  使用 tile[TILE_SIZE][TILE_SIZE + 1]\n");
    printf("  优点：无 Bank Conflict\n");
    printf("  缺点：增加 3%% 共享内存\n\n");

    printf("版本 4 - 共享内存 + XOR Swizzle：\n");
    printf("  使用 row ^ col 索引映射\n");
    printf("  优点：无 Bank Conflict，无额外内存\n");
    printf("  缺点：实现稍复杂\n\n");

    printf("版本 5 - 优化版：\n");
    printf("  Padding + 其他优化\n");
    printf("  推荐用于生产环境\n\n");

    // -------------------------------------------------------------------------
    // 第三部分：性能对比
    // -------------------------------------------------------------------------
    printf("【性能对比测试】\n");
    printf("-----------------------------------------------------------------\n");

    benchmark_all_versions(1024);
    benchmark_all_versions(2048);
    benchmark_all_versions(4096);

    // -------------------------------------------------------------------------
    // 第四部分：NCU 分析指南
    // -------------------------------------------------------------------------
    print_ncu_guide();

    // -------------------------------------------------------------------------
    // 第五部分：总结
    // -------------------------------------------------------------------------
    printf("\n【总结与建议】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n1. 性能排名（从快到慢）：\n");
    printf("   优化版 ≈ Padding ≈ XOR Swizzle > 无 Padding > Naive\n\n");

    printf("2. 推荐方案：\n");
    printf("   - 一般场景：使用 Padding（tile[32][33]）\n");
    printf("   - 共享内存紧张：使用 XOR Swizzling\n");
    printf("   - 生产环境：使用优化版\n\n");

    printf("3. 关键指标：\n");
    printf("   - Bank Conflict 数量（越少越好，理想为 0）\n");
    printf("   - 有效带宽（越高越好）\n");
    printf("   - 共享内存使用量\n\n");

    printf("4. 验证方法：\n");
    printf("   - 使用 NCU 检查 Bank Conflict 指标\n");
    printf("   - 对比优化前后的性能数据\n");
    printf("   - 确保结果正确性\n");

    printf("\n=================================================================\n");
    printf("演示完成！\n");
    printf("=================================================================\n");

    return 0;
}
