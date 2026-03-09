/**
 * 04_reduce_with_cg.cu - 使用 Cooperative Groups 的高性能规约实现
 *
 * 本示例演示：
 * 1. 完整的 GPU 规约实现
 * 2. 使用 CG 的 warp 级规约
 * 3. 使用 CG 的 block 级规约
 * 4. 使用 grid.sync() 的跨块规约
 * 5. 性能对比
 *
 * 编译要求：CUDA 9.0+, CC 6.0+
 */

#include <stdio.h>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// ============================================================================
// Warp 级规约（使用 Cooperative Groups）
// ============================================================================
__device__ float warp_reduce(cg::thread_block_tile<32>& warp, float val) {
    // 使用 shfl_down 进行高效的 warp 内规约
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += warp.shfl_down(val, offset);
    }
    return val;
}

// ============================================================================
// Block 级规约
// ============================================================================
__device__ float block_reduce(cg::thread_block& block, float val) {
    // 划分为 warp
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Warp 内规约
    val = warp_reduce(warp, val);

    // 使用共享内存进行 warp 间规约
    __shared__ float warp_sums[32];  // 假设最多 32 个 warp

    int warp_id = block.thread_rank() / 32;
    int lane_id = block.thread_rank() % 32;
    int num_warps = (block.num_threads() + 31) / 32;

    // 每个 warp 的 lane 0 写入结果
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    block.sync();

    // 第一个 warp 读取并规约所有 warp 的结果
    float result = 0.0f;
    if (warp_id == 0) {
        result = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        result = warp_reduce(warp, result);
    }

    return result;
}

// ============================================================================
// 核函数 1：使用原子操作（最慢，作为基准）
// ============================================================================
__global__ void reduce_atomic(float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        sum += input[i];
    }

    atomicAdd(output, sum);
}

// ============================================================================
// 核函数 2：Warp 规约 + 原子操作
// ============================================================================
__global__ void reduce_warp_cg(float* input, float* output, int N) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride 循环累加
    float sum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        sum += input[i];
    }

    // Warp 内规约
    sum = warp_reduce(warp, sum);

    // 每个 warp 的 lane 0 执行原子操作
    if (warp.thread_rank() == 0) {
        atomicAdd(output, sum);
    }
}

// ============================================================================
// 核函数 3：Block 规约 + 原子操作
// ============================================================================
__global__ void reduce_block_cg(float* input, float* output, int N) {
    cg::thread_block block = cg::this_thread_block();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride 循环累加
    float sum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        sum += input[i];
    }

    // Block 内规约
    sum = block_reduce(block, sum);

    // 每个 block 的 rank 0 执行原子操作
    if (block.thread_rank() == 0) {
        atomicAdd(output, sum);
    }
}

// ============================================================================
// 核函数 4：Grid Sync 规约（无原子操作）
// ============================================================================
__global__ void reduce_grid_sync(float* input, float* partial, float* output, int N) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 第一阶段：Grid-stride 累加 + Block 规约
    float sum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        sum += input[i];
    }

    sum = block_reduce(block, sum);

    // 每个 block 的 rank 0 存储部分结果
    if (block.thread_rank() == 0) {
        partial[blockIdx.x] = sum;
    }

    // 跨块同步
    grid.sync();

    // 第二阶段：block 0 汇总
    if (blockIdx.x == 0) {
        float total = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            total += partial[i];
        }

        total = block_reduce(block, total);

        if (block.thread_rank() == 0) {
            *output = total;
        }
    }
}

// ============================================================================
// 辅助函数：检查协作启动支持
// ============================================================================
bool check_cooperative_launch(int device) {
    int supports = 0;
    cudaDeviceGetAttribute(&supports, cudaDevAttrCooperativeLaunch, device);
    return supports != 0;
}

// ============================================================================
// 辅助函数：计算 block 数量
// ============================================================================
int get_num_blocks(int device, void* kernel, int threads) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, threads, 0);

    return prop.multiProcessorCount * blocks_per_sm;
}

// ============================================================================
// 计时测试函数
// ============================================================================
void benchmark_reduce(const char* name, float* d_input, float* d_output,
                      void* kernel, int blocks, int threads, int N,
                      bool use_cooperative = false) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 清零输出
    cudaMemset(d_output, 0, sizeof(float));

    cudaEventRecord(start);

    if (use_cooperative) {
        float* d_partial;
        cudaMalloc(&d_partial, blocks * sizeof(float));

        void* args[] = { &d_input, &d_partial, &d_output, &N };
        cudaLaunchCooperativeKernel(kernel, blocks, threads, args, 0, 0);

        cudaFree(d_partial);
    } else {
        // 对函数指针统一使用 runtime launch API
        void* args[] = { &d_input, &d_output, &N };
        cudaLaunchKernel(kernel, dim3(blocks), dim3(threads), args, 0, 0);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // 验证结果
    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("%-20s: %.3f ms, 结果 = %.0f %s\n",
           name, ms, result, std::fabs(result - N) < 1e-3 ? "[正确]" : "[错误]");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== Cooperative Groups: 规约实现对比 ===\n\n");

    int device = 0;
    cudaSetDevice(device);

    // 检查设备支持
    bool supports_coop = check_cooperative_launch(device);
    printf("Cooperative Launch 支持: %s\n\n", supports_coop ? "是" : "否");

    // 数据设置
    const int N = 16 * 1024 * 1024;  // 16M 元素
    const int threads = 256;

    printf("数据规模: %d 元素 (%.1f MB)\n", N, N * sizeof(float) / 1024.0 / 1024.0);
    printf("期望结果: %d\n\n", N);

    // 分配内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // 初始化数据（全 1）
    float* h_input = new float[N];
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_input;

    printf("--- 性能测试 ---\n");

    // 测试 1：原子操作（基准）
    int blocks = 256;
    benchmark_reduce("原子操作", d_input, d_output,
                     (void*)reduce_atomic, blocks, threads, N);

    // 测试 2：Warp 规约
    benchmark_reduce("Warp 规约 (CG)", d_input, d_output,
                     (void*)reduce_warp_cg, blocks, threads, N);

    // 测试 3：Block 规约
    benchmark_reduce("Block 规约 (CG)", d_input, d_output,
                     (void*)reduce_block_cg, blocks, threads, N);

    // 测试 4：Grid Sync（如果支持）
    if (supports_coop) {
        int coop_blocks = get_num_blocks(device, (void*)reduce_grid_sync, threads);
        printf("\nGrid Sync 使用 %d blocks\n", coop_blocks);
        benchmark_reduce("Grid Sync (CG)", d_input, d_output,
                         (void*)reduce_grid_sync, coop_blocks, threads, N, true);
    } else {
        printf("\nGrid Sync: 设备不支持，跳过\n");
    }

    printf("\n=== 性能优化提示 ===\n");
    printf("1. 原子操作最慢，因为每个线程都竞争同一内存\n");
    printf("2. Warp 规约减少原子操作次数到 1/32\n");
    printf("3. Block 规约进一步减少原子操作次数\n");
    printf("4. Grid Sync 完全避免原子操作，但需要协作启动\n");
    printf("   适合数据量大、原子操作成为瓶颈的场景\n");

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);

    printf("\n=== 示例完成 ===\n");
    return 0;
}
