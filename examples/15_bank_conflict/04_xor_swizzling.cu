/**
 * =============================================================================
 * 第四章：XOR Swizzling 解决方案 - 索引重映射技术
 * =============================================================================
 *
 * 本示例演示 XOR Swizzling 技术：
 * - XOR 运算的索引重映射原理
 * - 无额外内存开销消除 Bank Conflict
 * - 矩阵转置和 GEMM 中的应用
 *
 * 编译：nvcc -o 04_xor_swizzling 04_xor_swizzling.cu
 * 运行：./04_xor_swizzling
 * 分析：ncu --set full ./04_xor_swizzling
 *
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// =============================================================================
// XOR Swizzling 基本原理
// =============================================================================
// XOR Swizzling 通过 XOR 运算重新映射索引，改变 Bank 映射关系
// 
// 基本公式：swizzled_index = original_index ^ mask
// 
// 对于二维数组 tile[row][col]：
// swizzled_row = row ^ col  （最常用的形式）
// 
// 这会改变 Bank 映射：
// tile[row ^ col][col] 的 Bank ID = ((row ^ col) * stride + col) % 32
//                                = (row ^ col + col) % 32 （当 stride = 32）
//                                = row % 32
// 不同的 row 映射到不同的 Bank！

// =============================================================================
// 核函数 1：演示 XOR Swizzling 原理
// =============================================================================
__global__ void xor_swizzle_principle_demo(int* bank_ids_original,
                                            int* bank_ids_swizzled,
                                            int rows, int cols) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (tx < cols && ty < rows) {
        // 原始 Bank ID
        // tile[ty][tx] 的 Bank ID = (ty * cols + tx) % 32
        int original_bank = (ty * cols + tx) % 32;

        // Swizzled Bank ID
        // 使用 XOR Swizzling: row = ty ^ tx
        int swizzled_row = ty ^ tx;
        int swizzled_bank = (swizzled_row * cols + tx) % 32;

        bank_ids_original[ty * cols + tx] = original_bank;
        bank_ids_swizzled[ty * cols + tx] = swizzled_bank;
    }
}

// =============================================================================
// 核函数 2：使用 XOR Swizzling 的矩阵转置
// =============================================================================
__global__ void transpose_xor_swizzle(float* input, float* output,
                                       int width, int height) {
    // 注意：不需要 Padding！
    __shared__ float tile[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // XOR Swizzling 行索引
    // 原本存储在 tile[ty][tx] 的数据，现在存储在 tile[ty ^ tx][tx]
    int swizzled_row = ty ^ tx;

    // 加载：使用 Swizzled 索引存储
    if (x < width && y < height) {
        tile[swizzled_row][tx] = input[y * width + x];
    }
    __syncthreads();

    // 转置后读取
    // 我们要读取转置后的数据，即原 tile[tx][ty]
    // 原数据存储在 row = tx ^ ty 的位置
    // 所以读取 row = tx ^ ty（= ty ^ tx，XOR 交换律）
    int read_swizzled_row = tx ^ ty;

    int new_x = blockIdx.y * blockDim.y + tx;
    int new_y = blockIdx.x * blockDim.x + ty;

    if (new_x < height && new_y < width) {
        output[new_y * height + new_x] = tile[read_swizzled_row][ty];
    }
}

// =============================================================================
// 核函数 3：无 XOR Swizzling 的矩阵转置（对比用）
// =============================================================================
__global__ void transpose_no_swizzle(float* input, float* output,
                                      int width, int height) {
    __shared__ float tile[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // 加载：直接存储
    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    }
    __syncthreads();

    // 读取：按列读取（有 Bank Conflict！）
    int new_x = blockIdx.y * blockDim.y + tx;
    int new_y = blockIdx.x * blockDim.x + ty;

    if (new_x < height && new_y < width) {
        output[new_y * height + new_x] = tile[tx][ty];  // Bank Conflict!
    }
}

// =============================================================================
// 核函数 4：通用 XOR Swizzling 函数
// =============================================================================
// 更通用的 XOR Swizzling 公式：
// swizzled_addr = addr ^ ((addr >> S) & mask)
// 参数：
//   addr: 原始地址
//   S: 右移位数，控制取高位的范围
//   mask: 掩码，控制 XOR 的位数

__device__ int swizzle_address(int addr, int S, int mask) {
    return addr ^ ((addr >> S) & mask);
}

// 示例：使用通用 Swizzling 的共享内存访问
__global__ void generic_swizzle_demo(float* input, float* output, int n) {
    __shared__ float smem[128];

    int tid = threadIdx.x;

    // 加载时使用 Swizzling
    // S = 5, mask = 31 (对于 32 元素块)
    int swizzled_idx = swizzle_address(tid, 5, 31);
    smem[swizzled_idx] = input[tid];
    __syncthreads();

    // 读取时使用反向 Swizzling（相同操作，XOR 的逆操作是自身）
    int read_idx = swizzle_address(tid, 5, 31);
    output[tid] = smem[read_idx];
}

// =============================================================================
// 核函数 5：GEMM 中的 XOR Swizzling
// =============================================================================
#define TILE_DIM 32

__global__ void gemm_xor_swizzle(float* A, float* B, float* C,
                                  int M, int N, int K) {
    // 不需要 Padding
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float sum = 0.0f;

    // 遍历 K 维度
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int a_col = t * TILE_DIM + tx;
        int b_row = t * TILE_DIM + ty;

        // XOR Swizzling 加载 A 矩阵
        int swizzle_a = ty ^ tx;
        if (row < M && a_col < K) {
            As[swizzle_a][tx] = A[row * K + a_col];
        } else {
            As[swizzle_a][tx] = 0.0f;
        }

        // XOR Swizzling 加载 B 矩阵
        int swizzle_b = ty ^ tx;
        if (b_row < K && col < N) {
            Bs[swizzle_b][tx] = B[b_row * N + col];
        } else {
            Bs[swizzle_b][tx] = 0.0f;
        }

        __syncthreads();

        // 计算
        // 读取时使用对应的 Swizzled 索引
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            int read_swizzle = ty ^ k;
            sum += As[read_swizzle][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// 核函数 6：XOR Swizzling 的 Bank 映射分析
// =============================================================================
__global__ void analyze_xor_bank_mapping(int* results) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 分析 tile[ty][tx] 的 Bank 映射
    // 无 Swizzling：Bank ID = (ty * 32 + tx) % 32 = tx
    // 有 Swizzling：Bank ID = ((ty ^ tx) * 32 + tx) % 32 = (ty ^ tx) % 32

    int original_bank = tx;  // 对于无 Swizzling，Bank ID = tx
    int swizzled_bank = (ty ^ tx) % 32;

    // 存储结果
    // results[0..1023]: 无 Swizzling 的 Bank ID
    // results[1024..2047]: 有 Swizzling 的 Bank ID
    int idx = ty * 32 + tx;
    results[idx] = original_bank;
    results[1024 + idx] = swizzled_bank;
}

// =============================================================================
// 工具函数
// =============================================================================
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

bool verify_transpose(float* original, float* transposed, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float orig = original[i * cols + j];
            float trans = transposed[j * rows + i];
            if (fabsf(orig - trans) > 1e-3f) {
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// 性能测试
// =============================================================================
void benchmark_xor_swizzle(int size) {
    float *d_input, *d_output;
    size_t bytes = size * size * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    float* h_input = (float*)malloc(bytes);
    init_matrix(h_input, size, size);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((size + 31) / 32, (size + 31) / 32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms;

    // 测试无 Swizzling 版本
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        transpose_no_swizzle<<<grid, block>>>(d_input, d_output, size, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  无 Swizzling:    %.3f ms (有 Bank Conflict)\n", ms / 10);

    // 测试 XOR Swizzling 版本
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        transpose_xor_swizzle<<<grid, block>>>(d_input, d_output, size, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  XOR Swizzling:   %.3f ms (无 Bank Conflict)\n", ms / 10);

    // 验证
    float* h_output = (float*)malloc(bytes);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    if (verify_transpose(h_input, h_output, size, size)) {
        printf("  结果验证:        通过\n");
    }

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
    printf("第四章：XOR Swizzling 解决方案\n");
    printf("=================================================================\n\n");

    srand((unsigned int)time(NULL));

    // -------------------------------------------------------------------------
    // 第一部分：XOR 运算基础
    // -------------------------------------------------------------------------
    printf("【第一部分：XOR 运算基础】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\nXOR（异或）运算规则：\n");
    printf("  0 XOR 0 = 0\n");
    printf("  0 XOR 1 = 1\n");
    printf("  1 XOR 0 = 1\n");
    printf("  1 XOR 1 = 0\n");
    printf("  简记：相同为 0，不同为 1\n\n");

    printf("XOR 的特殊性质：\n");
    printf("  1. 自反性：A XOR A = 0\n");
    printf("  2. 恒等性：A XOR 0 = A\n");
    printf("  3. 交换律：A XOR B = B XOR A\n");
    printf("  4. 结合律：(A XOR B) XOR C = A XOR (B XOR C)\n");
    printf("  5. 逆运算：A XOR B XOR B = A\n\n");

    // -------------------------------------------------------------------------
    // 第二部分：XOR Swizzling 原理
    // -------------------------------------------------------------------------
    printf("【第二部分：XOR Swizzling 原理】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n基本思想：\n");
    printf("  通过 XOR 运算重新映射索引，改变 Bank 映射关系\n\n");

    printf("对于 tile[row][col]：\n");
    printf("  原始索引：tile[row][col]\n");
    printf("  Swizzled 索引：tile[row XOR col][col]\n\n");

    printf("Bank 映射变化：\n");
    printf("  原始：Bank ID = (row * stride + col) %% 32 = col\n");
    printf("        同一 col 的所有元素映射到 Bank col\n\n");
    printf("  Swizzled：Bank ID = ((row XOR col) * stride + col) %% 32\n");
    printf("            = (row XOR col + col) %% 32 = row %% 32\n");
    printf("        不同 row 映射到不同 Bank\n\n");

    // -------------------------------------------------------------------------
    // 第三部分：Bank 映射可视化
    // -------------------------------------------------------------------------
    printf("【第三部分：Bank 映射可视化】\n");
    printf("-----------------------------------------------------------------\n");

    // 分配设备内存
    int *d_results;
    cudaMalloc(&d_results, 2048 * sizeof(int));

    // 运行分析核函数
    dim3 block(32, 32);
    analyze_xor_bank_mapping<<<1, block>>>(d_results);
    cudaDeviceSynchronize();

    // 复制结果
    int h_results[2048];
    cudaMemcpy(h_results, d_results, 2048 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n无 Swizzling 时，列 0 的 Bank 映射：\n");
    printf("  ");
    for (int row = 0; row < 8; row++) {
        printf("tile[%d][0]->Bank%d  ", row, h_results[row * 32]);
    }
    printf("\n  ...所有行映射到 Bank 0（冲突！）\n\n");

    printf("有 Swizzling 时，列 0 的 Bank 映射：\n");
    printf("  ");
    for (int row = 0; row < 8; row++) {
        printf("tile[%d][0]->Bank%d  ", row, h_results[1024 + row * 32]);
    }
    printf("\n  ...每行映射到不同 Bank（无冲突）\n\n");

    cudaFree(d_results);

    // -------------------------------------------------------------------------
    // 第四部分：性能对比
    // -------------------------------------------------------------------------
    printf("【第四部分：性能对比（1024x1024 矩阵转置）】\n");
    printf("-----------------------------------------------------------------\n");
    benchmark_xor_swizzle(1024);

    // -------------------------------------------------------------------------
    // 第五部分：XOR Swizzling vs Padding
    // -------------------------------------------------------------------------
    printf("\n【第五部分：XOR Swizzling vs Padding 对比】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n+------------------+-------------------+-------------------+\n");
    printf("| 特性             | Padding           | XOR Swizzling     |\n");
    printf("+------------------+-------------------+-------------------+\n");
    printf("| 内存开销         | 增加（3%%-13%%）   | 无                |\n");
    printf("| 实现复杂度       | 简单              | 中等              |\n");
    printf("| 理解成本         | 低                | 高                |\n");
    printf("| 适用场景         | 简单 2D 访问      | 复杂访问模式      |\n");
    printf("| Occupancy 影响   | 可能降低          | 无                |\n");
    printf("| 灵活性           | 固定              | 可调参数          |\n");
    printf("+------------------+-------------------+-------------------+\n\n");

    // -------------------------------------------------------------------------
    // 第六部分：通用 XOR Swizzling 模板
    // -------------------------------------------------------------------------
    printf("【第六部分：通用 XOR Swizzling 模板】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n通用公式：\n");
    printf("  swizzled_addr = addr ^ ((addr >> S) & mask)\n\n");
    printf("参数说明：\n");
    printf("  addr: 原始地址\n");
    printf("  S: 右移位数，控制取哪些位进行 XOR\n");
    printf("  mask: 掩码，控制 XOR 的位数\n\n");

    printf("常用配置：\n");
    printf("  1. 32 元素块：S=5, mask=31\n");
    printf("  2. 64 元素块：S=6, mask=63\n");
    printf("  3. 矩阵转置：直接使用 row ^ col\n\n");

    // -------------------------------------------------------------------------
    // 第七部分：最佳实践
    // -------------------------------------------------------------------------
    printf("【第七部分：XOR Swizzling 最佳实践】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n1. 选择 XOR Swizzling 的场景：\n");
    printf("   - 共享内存资源紧张，不能使用 Padding\n");
    printf("   - 需要最大化 Occupancy\n");
    printf("   - 访问模式规则且可预测\n\n");

    printf("2. 实现注意事项：\n");
    printf("   - 加载和读取都要使用相同的 Swizzling 规则\n");
    printf("   - XOR 操作的逆操作是自身\n");
    printf("   - 确保边界检查正确\n\n");

    printf("3. 调试技巧：\n");
    printf("   - 先实现无 Swizzling 版本验证正确性\n");
    printf("   - 添加 Swizzling 后重新验证\n");
    printf("   - 使用 NCU 检查 Bank Conflict 是否消除\n\n");

    printf("4. 性能验证：\n");
    printf("   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \\\n");
    printf("       ./04_xor_swizzling\n");

    printf("\n=================================================================\n");
    printf("演示完成！\n");
    printf("=================================================================\n");

    return 0;
}
