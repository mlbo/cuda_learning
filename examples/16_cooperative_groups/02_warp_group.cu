/**
 * 02_warp_group.cu - thread_block_tile 和 warp 级操作
 *
 * 本示例演示：
 * 1. 使用 tiled_partition 划分线程组
 * 2. Warp 级 shuffle 操作
 * 3. 使用 warp 进行高效规约
 * 4. coalesced_threads() 处理分支分歧
 *
 * 编译要求：CUDA 9.0+, CC 5.0+
 */

#include <stdio.h>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// ============================================================================
// 核函数：演示 tiled_partition 基础
// ============================================================================
__global__ void tile_partition_basics(int* output_ranks) {
    // 获取线程块组
    cg::thread_block block = cg::this_thread_block();

    // 将块划分为 32 线程的 tile
    // 这通常对应一个 warp
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // 获取线程在 warp 内的 rank
    int lane_id = warp.thread_rank();

    // 获取 warp 在块内的 rank（第几个 warp）
    int warp_id = warp.meta_group_rank();

    // 获取块内有多少个 warp
    int num_warps = warp.meta_group_size();

    // 存储结果以便验证
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    output_ranks[global_tid * 3] = warp_id;
    output_ranks[global_tid * 3 + 1] = lane_id;
    output_ranks[global_tid * 3 + 2] = num_warps;
}

// ============================================================================
// 核函数：演示 Shuffle 操作
// ============================================================================
__global__ void shuffle_demo(float* data, float* output) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int lane_id = warp.thread_rank();
    float val = data[lane_id];

    // ----------------------------------------------------------------
    // 1. shfl: 从指定 rank 获取值
    // 所有线程都从 src_rank 处获取值
    // ----------------------------------------------------------------
    float from_lane_0 = warp.shfl(val, 0);  // 所有线程获取 lane 0 的值

    // ----------------------------------------------------------------
    // 2. shfl_down: 从 rank + delta 处获取值
    // 用于规约操作
    // ----------------------------------------------------------------
    float from_down = warp.shfl_down(val, 1);  // 从 lane_id + 1 获取

    // ----------------------------------------------------------------
    // 3. shfl_up: 从 rank - delta 处获取值
    // ----------------------------------------------------------------
    float from_up = warp.shfl_up(val, 1);  // 从 lane_id - 1 获取

    // ----------------------------------------------------------------
    // 4. shfl_xor: XOR shuffle
    // 用于某些特殊算法
    // ----------------------------------------------------------------
    float from_xor = warp.shfl_xor(val, 16);  // XOR with 16

    // 存储结果
    output[lane_id * 4] = from_lane_0;
    output[lane_id * 4 + 1] = from_down;
    output[lane_id * 4 + 2] = from_up;
    output[lane_id * 4 + 3] = from_xor;
}

// ============================================================================
// 核函数：Warp 级规约
// ============================================================================
__device__ float warp_reduce_sum(cg::thread_block_tile<32>& warp, float val) {
    // 使用 shfl_down 进行规约
    // 每次迭代，活跃线程数量减半

    // 方法 1：循环展开
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }

    // 此时只有 lane 0 拥有完整的和
    return val;
}

__global__ void warp_reduce_kernel(float* input, float* output, int N) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int lane_id = warp.thread_rank();
    int warp_id = warp.meta_group_rank();

    // Grid-stride 循环累加
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Warp 内规约
    sum = warp_reduce_sum(warp, sum);

    // 只有每个 warp 的 lane 0 写入结果
    if (lane_id == 0) {
        atomicAdd(output, sum);
    }
}

// ============================================================================
// 核函数：coalesced_threads() 处理分支分歧
// ============================================================================
__global__ void coalesced_threads_demo(int* active_count, int* ranks) {
    int tid = threadIdx.x;

    // 模拟条件：只有偶数线程活跃
    if (tid % 2 == 0) {
        // 创建包含所有活跃线程的组
        cg::coalesced_group active = cg::coalesced_threads();

        // 获取在活跃组内的 rank
        int rank = active.thread_rank();
        int size = active.num_threads();

        // 存储信息
        active_count[0] = size;
        ranks[tid / 2] = rank;

        // 活跃组内同步
        active.sync();

        // 可以在活跃组内使用 shuffle
        int leader_val = active.shfl(rank, 0);
        ranks[tid / 2 + 16] = leader_val;
    }
}

// ============================================================================
// 核函数：使用更小的 tile 大小
// ============================================================================
__global__ void small_tiles_demo(int* output) {
    cg::thread_block block = cg::this_thread_block();

    // 划分为大小为 4 的 tile
    cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);

    int rank = tile4.thread_rank();
    int tile_id = tile4.meta_group_rank();

    // 每个 tile 内部计算
    int base = tile_id * 4;
    output[base + rank] = rank * 10 + tile_id;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== Cooperative Groups: Warp 级操作示例 ===\n\n");

    // ========================
    // 测试 1: tiled_partition 基础
    // ========================
    printf("--- 测试 tiled_partition 基础 ---\n");
    {
        const int N = 64;
        int *d_output, *h_output;
        cudaMalloc(&d_output, N * 3 * sizeof(int));
        h_output = new int[N * 3];

        tile_partition_basics<<<1, N>>>(d_output);
        cudaDeviceSynchronize();
        cudaMemcpy(h_output, d_output, N * 3 * sizeof(int), cudaMemcpyDeviceToHost);

        printf("前 4 个线程的信息:\n");
        printf("线程ID | warp_id | lane_id | num_warps\n");
        for (int i = 0; i < 4; i++) {
            printf("  %2d   |   %2d    |   %2d    |    %2d\n",
                   i, h_output[i*3], h_output[i*3+1], h_output[i*3+2]);
        }
        printf("线程 32-35 的信息（第二个 warp）:\n");
        for (int i = 32; i < 36; i++) {
            printf("  %2d   |   %2d    |   %2d    |    %2d\n",
                   i, h_output[i*3], h_output[i*3+1], h_output[i*3+2]);
        }

        cudaFree(d_output);
        delete[] h_output;
    }

    // ========================
    // 测试 2: Shuffle 操作
    // ========================
    printf("\n--- 测试 Shuffle 操作 ---\n");
    {
        float *d_data, *d_output;
        float h_data[32], h_output[32 * 4];

        for (int i = 0; i < 32; i++) h_data[i] = (float)i;

        cudaMalloc(&d_data, 32 * sizeof(float));
        cudaMalloc(&d_output, 32 * 4 * sizeof(float));
        cudaMemcpy(d_data, h_data, 32 * sizeof(float), cudaMemcpyHostToDevice);

        shuffle_demo<<<1, 32>>>(d_data, d_output);
        cudaDeviceSynchronize();
        cudaMemcpy(h_output, d_output, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

        printf("Shuffle 测试 (前 4 个 lane):\n");
        printf("lane | shfl(0) | shfl_down | shfl_up | shfl_xor(16)\n");
        for (int i = 0; i < 4; i++) {
            printf(" %2d  |  %5.0f  |   %5.0f   |  %5.0f  |    %5.0f\n",
                   i, h_output[i*4], h_output[i*4+1], h_output[i*4+2], h_output[i*4+3]);
        }

        cudaFree(d_data);
        cudaFree(d_output);
    }

    // ========================
    // 测试 3: Warp 规约
    // ========================
    printf("\n--- 测试 Warp 规约 ---\n");
    {
        const int N = 1024;
        float *d_input, *d_output;
        float* h_input = new float[N];

        for (int i = 0; i < N; i++) h_input[i] = 1.0f;

        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, sizeof(float));
        cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, sizeof(float));

        warp_reduce_kernel<<<4, 256>>>(d_input, d_output, N);
        cudaDeviceSynchronize();

        float h_output;
        cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

        printf("输入: %d 个 1.0f\n", N);
        printf("计算的和: %.0f\n", h_output);
        printf("期望的和: %d\n", N);
        printf("结果: %s\n", std::fabs(h_output - N) < 1e-5 ? "正确" : "错误");

        cudaFree(d_input);
        cudaFree(d_output);
        delete[] h_input;
    }

    // ========================
    // 测试 4: coalesced_threads
    // ========================
    printf("\n--- 测试 coalesced_threads ---\n");
    {
        int *d_active_count, *d_ranks;
        int h_active_count, h_ranks[32];

        cudaMalloc(&d_active_count, sizeof(int));
        cudaMalloc(&d_ranks, 32 * sizeof(int));

        coalesced_threads_demo<<<1, 32>>>(d_active_count, d_ranks);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_active_count, d_active_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ranks, d_ranks, 32 * sizeof(int), cudaMemcpyDeviceToHost);

        printf("活跃线程数（应为 16）: %d\n", h_active_count);
        printf("活跃线程在组内的 rank: ");
        for (int i = 0; i < h_active_count; i++) {
            printf("%d ", h_ranks[i]);
        }
        printf("\n");

        cudaFree(d_active_count);
        cudaFree(d_ranks);
    }

    printf("\n=== 示例完成 ===\n");
    return 0;
}
