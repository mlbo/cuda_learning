/**
 * =============================================================================
 * 第三章：Padding 解决方案 - 通过内存填充消除 Bank Conflict
 * =============================================================================
 *
 * 本示例演示如何通过 Padding（内存填充）技术消除 Bank Conflict：
 * - Padding 原理分析
 * - 矩阵转置中的 Padding 应用
 * - 不同 Padding 大小的效果对比
 *
 * 编译：nvcc -o 03_padding_solution 03_padding_solution.cu
 * 运行：./03_padding_solution
 * 分析：ncu --set full ./03_padding_solution
 *
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// =============================================================================
// 核函数 1：无 Padding 的矩阵转置（有 Bank Conflict）
// =============================================================================
__global__ void transpose_no_padding(float* input, float* output,
                                      int width, int height) {
    // 32x32 的共享内存块（无 Padding）
    // 问题：同一列的元素映射到同一 Bank
    __shared__ float tile[32][32];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 从全局内存加载到共享内存（按行加载，无 Bank Conflict）
    // tile[ty][tx] 的 Bank ID = (ty * 32 + tx) % 32 = tx
    // 不同 ty，相同 tx：访问不同 Bank（因为 tx 不同）
    // 但同一行内，tx 变化时 Bank 也变化
    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    }
    __syncthreads();

    // 从共享内存存储到全局内存（按列读取，有 Bank Conflict！）
    // 我们要读取 tile[tx][ty]，即转置后的位置
    // tile[tx][ty] 的 Bank ID = (tx * 32 + ty) % 32 = tx
    // 问题：当 tx 固定，ty 变化时（即同一列）
    // Bank ID = tx（固定不变！）
    // 所有 ty 变化时访问同一 Bank -> 32-way Bank Conflict!
    int new_x = blockIdx.y * blockDim.y + tx;
    int new_y = blockIdx.x * blockDim.x + ty;

    if (new_x < height && new_y < width) {
        output[new_y * height + new_x] = tile[tx][ty];  // 32-way Bank Conflict!
    }
}

// =============================================================================
// 核函数 2：使用 Padding 的矩阵转置（无 Bank Conflict）
// =============================================================================
__global__ void transpose_with_padding(float* input, float* output,
                                        int width, int height) {
    // 关键：在第二维添加 1 列 Padding
    // tile[32][33] 而不是 tile[32][32]
    // 这会改变 Bank 映射关系
    __shared__ float tile[32][33];  // 33 而不是 32！

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 加载到共享内存
    // tile[ty][tx] 的 Bank ID = (ty * 33 + tx) % 32
    // 因为每行有 33 个元素，而不是 32
    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    }
    __syncthreads();

    // 从共享内存读取（按列读取，现在无 Bank Conflict！）
    // tile[tx][ty] 的 Bank ID = (tx * 33 + ty) % 32
    //                      = (tx + ty) % 32  （因为 33 % 32 = 1）
    // 当 tx 固定，ty 变化时，Bank ID = (tx + ty) % 32
    // 随着 ty 变化而变化！不同的 ty 映射到不同的 Bank
    // 无 Bank Conflict！
    int new_x = blockIdx.y * blockDim.y + tx;
    int new_y = blockIdx.x * blockDim.x + ty;

    if (new_x < height && new_y < width) {
        output[new_y * height + new_x] = tile[tx][ty];  // 无 Bank Conflict!
    }
}

// =============================================================================
// 核函数 3：不同 Padding 大小的对比
// =============================================================================
__global__ void transpose_padding_size(float* input, float* output,
                                        int width, int height, int padding) {
    // 使用模板参数或动态共享内存可以实现不同 Padding 大小
    // 这里我们硬编码几种常见情况

    extern __shared__ float tile_dynamic[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // 计算带 Padding 的索引
    int row_stride = 32 + padding;  // 每行的实际宽度

    // 加载
    if (x < width && y < height) {
        tile_dynamic[ty * row_stride + tx] = input[y * width + x];
    }
    __syncthreads();

    // 存储（转置）
    int new_x = blockIdx.y * blockDim.y + tx;
    int new_y = blockIdx.x * blockDim.x + ty;

    if (new_x < height && new_y < width) {
        output[new_y * height + new_x] = tile_dynamic[tx * row_stride + ty];
    }
}

// =============================================================================
// 核函数 4：一维数组的 Padding 示例
// =============================================================================
__global__ void array_padding_demo(float* input, float* output, int n) {
    // 原始版本：访问 stride = 32 的元素，会产生 Bank Conflict
    // __shared__ float smem[128];

    // Padding 版本：每 32 个元素后添加 1 个 Padding
    // 这会改变 Bank 映射，消除冲突
    __shared__ float smem[128 + 4];  // 添加 4 个 Padding（每 32 元素 1 个）

    int tid = threadIdx.x;

    // 加载数据到 Padding 后的位置
    // 原始索引 i 映射到 Padding 索引：i + (i / 32)
    int padded_idx = tid + (tid / 32);
    smem[padded_idx] = input[tid];
    __syncthreads();

    // 访问时也使用 Padding 索引
    // stride = 32 的访问现在无冲突
    if (tid < 32) {
        int src_idx = tid * 32 + (tid * 32 / 32);  // 带 Padding 的索引
        output[tid] = smem[src_idx];
    }
}

// =============================================================================
// 核函数 5：GEMM 中的 Padding 应用
// =============================================================================
#define TILE_SIZE 32

__global__ void gemm_with_padding(float* A, float* B, float* C,
                                   int M, int N, int K) {
    // 使用 Padding 避免共享内存 Bank Conflict
    // A 的 tile：每行 K 个元素，但我们只加载 TILE_SIZE 个
    // 使用 TILE_SIZE + 1 作为行宽
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 Padding
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];  // +1 Padding

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 遍历 K 维度的 tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        // 加载 A 和 B 的 tile
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 计算点积
        // 由于 Padding，As[threadIdx.y][k] 和 Bs[k][threadIdx.x]
        // 的 Bank 映射不同，无 Bank Conflict
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// 工具函数：初始化矩阵
// =============================================================================
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

// =============================================================================
// 工具函数：验证转置结果
// =============================================================================
bool verify_transpose(float* original, float* transposed, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float orig = original[i * cols + j];
            float trans = transposed[j * rows + i];
            if (fabsf(orig - trans) > 1e-3f) {
                printf("错误：[%d,%d] 原始=%.2f, 转置=%.2f\n", i, j, orig, trans);
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// 性能测试函数
// =============================================================================
void benchmark_transpose(int size) {
    float *d_input, *d_output;
    size_t bytes = size * size * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // 初始化数据
    float* h_input = (float*)malloc(bytes);
    init_matrix(h_input, size, size);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((size + 31) / 32, (size + 31) / 32);

    // 创建事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms;

    // 测试无 Padding 版本
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        transpose_no_padding<<<grid, block>>>(d_input, d_output, size, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  无 Padding:      %.3f ms (有 Bank Conflict)\n", ms / 10);

    // 测试有 Padding 版本
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        transpose_with_padding<<<grid, block>>>(d_input, d_output, size, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  有 Padding:      %.3f ms (无 Bank Conflict)\n", ms / 10);

    // 验证结果
    float* h_output = (float*)malloc(bytes);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    if (verify_transpose(h_input, h_output, size, size)) {
        printf("  结果验证: 通过\n");
    }

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("=================================================================\n");
    printf("第三章：Padding 解决方案\n");
    printf("=================================================================\n\n");

    srand((unsigned int)time(NULL));

    // -------------------------------------------------------------------------
    // 第一部分：Padding 原理
    // -------------------------------------------------------------------------
    printf("【第一部分：Padding 原理】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n原始布局 tile[32][32]：\n");
    printf("  tile[i][j] 的 Bank ID = (i * 32 + j) %% 32 = j\n");
    printf("  同一列 j 的所有元素映射到 Bank j\n");
    printf("  按列访问时产生 32-way Bank Conflict\n\n");

    printf("Padding 布局 tile[32][33]：\n");
    printf("  tile[i][j] 的 Bank ID = (i * 33 + j) %% 32 = (i + j) %% 32\n");
    printf("  同一列 j 的元素映射到不同 Bank（因为 i 不同）\n");
    printf("  按列访问时无 Bank Conflict\n\n");

    // -------------------------------------------------------------------------
    // 第二部分：详细 Bank 映射分析
    // -------------------------------------------------------------------------
    printf("【第二部分：Bank 映射对比】\n");
    printf("-----------------------------------------------------------------\n");

    printf("\n无 Padding 时 tile[i][0] 列的 Bank 映射：\n");
    printf("  tile[0][0] -> Bank 0\n");
    printf("  tile[1][0] -> Bank 0  <- 冲突！\n");
    printf("  tile[2][0] -> Bank 0  <- 冲突！\n");
    printf("  ...\n");
    printf("  tile[31][0] -> Bank 0 <- 冲突！\n");
    printf("  结果：32-way Bank Conflict\n\n");

    printf("有 Padding 时 tile[i][0] 列的 Bank 映射：\n");
    printf("  tile[0][0] -> Bank (0*33+0)%%32 = 0\n");
    printf("  tile[1][0] -> Bank (1*33+0)%%32 = 1\n");
    printf("  tile[2][0] -> Bank (2*33+0)%%32 = 2\n");
    printf("  ...\n");
    printf("  tile[31][0] -> Bank (31*33+0)%%32 = 31\n");
    printf("  结果：无 Bank Conflict\n\n");

    // -------------------------------------------------------------------------
    // 第三部分：性能对比
    // -------------------------------------------------------------------------
    printf("【第三部分：性能对比（1024x1024 矩阵转置）】\n");
    printf("-----------------------------------------------------------------\n");
    benchmark_transpose(1024);

    // -------------------------------------------------------------------------
    // 第四部分：内存开销分析
    // -------------------------------------------------------------------------
    printf("\n【第四部分：内存开销分析】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n32x32 共享内存块：\n");
    printf("  无 Padding: 32 * 32 * 4 = 4096 bytes\n");
    printf("  有 Padding: 32 * 33 * 4 = 4224 bytes\n");
    printf("  额外开销: 128 bytes (3%%)\n\n");

    printf("不同 Padding 大小的开销：\n");
    printf("  Padding = 1: 3%% 额外开销（推荐）\n");
    printf("  Padding = 2: 6%% 额外开销\n");
    printf("  Padding = 4: 13%% 额外开销\n\n");

    // -------------------------------------------------------------------------
    // 第五部分：GEMM Padding 示例
    // -------------------------------------------------------------------------
    printf("【第五部分：GEMM 中的 Padding 应用】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\nGEMM 中的 Bank Conflict 来源：\n");
    printf("  - As[threadIdx.y][k]: 不同 y，相同 k 访问同一 Bank\n");
    printf("  - Bs[k][threadIdx.x]: 不同 k，相同 x 访问同一 Bank\n\n");

    printf("Padding 解决方案：\n");
    printf("  As[TILE_SIZE][TILE_SIZE + 1]\n");
    printf("  Bs[TILE_SIZE][TILE_SIZE + 1]\n\n");

    printf("优化后 Bank 映射：\n");
    printf("  As[y][k] -> Bank (y * 33 + k) %% 32 = (y + k) %% 32\n");
    printf("  不同 y 映射到不同 Bank\n\n");

    // -------------------------------------------------------------------------
    // 第六部分：NCU 分析命令
    // -------------------------------------------------------------------------
    printf("【第六部分：NCU 分析命令】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n对比分析两个版本的 Bank Conflict：\n\n");
    printf("1. 分析无 Padding 版本：\n");
    printf("   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \\\n");
    printf("       ./03_padding_solution\n\n");

    printf("2. 生成对比报告：\n");
    printf("   ncu --set full -o transpose_no_padding ./03_padding_solution\n");
    printf("   # 修改代码使用无 Padding 版本后重新编译\n");
    printf("   ncu --set full -o transpose_with_padding ./03_padding_solution\n");
    printf("   ncu-ui  # 打开两个报告对比\n\n");

    // -------------------------------------------------------------------------
    // 第七部分：最佳实践
    // -------------------------------------------------------------------------
    printf("【第七部分：Padding 最佳实践】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n1. Padding 大小选择：\n");
    printf("   - 通常 Padding = 1 即可消除 Bank Conflict\n");
    printf("   - 某些复杂访问模式可能需要更大的 Padding\n\n");

    printf("2. 适用场景：\n");
    printf("   - 2D 访问模式（矩阵、图像）\n");
    printf("   - 按列访问的场景\n");
    printf("   - 共享内存 tile 操作\n\n");

    printf("3. 注意事项：\n");
    printf("   - Padding 会增加共享内存使用量\n");
    printf("   - 可能影响 Occupancy（每个 SM 的 Block 数量）\n");
    printf("   - 需要在性能和资源之间权衡\n\n");

    printf("4. 与其他方法对比：\n");
    printf("   - Padding: 简单直接，有内存开销\n");
    printf("   - XOR Swizzling: 无内存开销，但更复杂\n");
    printf("   - 选择取决于具体场景和优化目标\n");

    printf("\n=================================================================\n");
    printf("演示完成！\n");
    printf("=================================================================\n");

    return 0;
}
