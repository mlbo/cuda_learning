/**
 * 03_grid_sync.cu - 跨块同步 grid.sync()
 *
 * 本示例演示：
 * 1. 使用 grid_group 进行跨块同步
 * 2. cudaLaunchCooperativeKernel 的正确用法
 * 3. 检查设备是否支持协作启动
 * 4. 计算合适的 block 数量
 *
 * 编译要求：CUDA 9.0+, CC 6.0+
 *
 * 平台限制：
 * - Linux（无 MPS 或 CC 7.0+ 有 MPS）
 * - 最新 Windows 版本
 */

#include <stdio.h>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// ============================================================================
// 核函数：使用 grid.sync() 进行两阶段计算
// ============================================================================
__global__ void two_phase_sum(float* input, float* partial_sums, float* result, int N) {
    // 获取 grid 组
    cg::grid_group grid = cg::this_grid();

    // 获取线程块组
    cg::thread_block block = cg::this_thread_block();

    int tid = block.thread_rank();
    int bid = blockIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    // ================================================================
    // 第一阶段：每个 block 计算自己的部分和
    // ================================================================
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += grid_size) {
        sum += input[i];
    }

    // Block 内规约
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    block.sync();

    atomicAdd(&s_sum, sum);
    block.sync();

    // 每个 block 的 lane 0 存储部分结果
    if (tid == 0) {
        partial_sums[bid] = s_sum;
        printf("Block %d: partial sum = %.0f\n", bid, s_sum);
    }

    // ================================================================
    // 跨块同步：等待所有 block 完成第一阶段
    // ================================================================
    grid.sync();

    // ================================================================
    // 第二阶段：block 0 汇总所有部分和
    // ================================================================
    if (bid == 0) {
        float total = 0.0f;
        for (int i = tid; i < gridDim.x; i += blockDim.x) {
            total += partial_sums[i];
        }

        // Block 内规约
        __shared__ float s_total;
        if (tid == 0) s_total = 0.0f;
        block.sync();

        atomicAdd(&s_total, total);
        block.sync();

        if (tid == 0) {
            *result = s_total;
            printf("Final result = %.0f\n", s_total);
        }
    }
}

// ============================================================================
// 核函数：演示 grid 属性查询
// ============================================================================
__global__ void grid_properties_demo(unsigned long long* output) {
    cg::grid_group grid = cg::this_grid();

    // 获取各种属性
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        output[0] = grid.num_threads();    // 总线程数
        output[1] = grid.num_blocks();     // 总 block 数
        output[2] = grid.thread_rank();    // 当前线程 rank
        output[3] = grid.is_valid() ? 1 : 0;  // 是否可以同步
    }
}

// ============================================================================
// 辅助函数：检查设备是否支持协作启动
// ============================================================================
bool check_cooperative_launch_support(int device) {
    int supports_coop = 0;
    cudaError_t err = cudaDeviceGetAttribute(&supports_coop,
                                              cudaDevAttrCooperativeLaunch,
                                              device);
    if (err != cudaSuccess) {
        printf("Error checking cooperative launch support: %s\n",
               cudaGetErrorString(err));
        return false;
    }
    return supports_coop != 0;
}

// ============================================================================
// 辅助函数：计算合适的 block 数量
// ============================================================================
int calculate_num_blocks(int device, void* kernel, int threads_per_block) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // 方法 1：每 SM 一个 block（最保守）
    int num_blocks_conservative = prop.multiProcessorCount;

    // 方法 2：使用 occupancy calculator（更高效）
    int num_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        kernel,
        threads_per_block,
        0  // dynamic shared memory
    );
    int num_blocks_optimal = prop.multiProcessorCount * num_blocks_per_sm;

    printf("GPU: %s\n", prop.name);
    printf("SM 数量: %d\n", prop.multiProcessorCount);
    printf("每 SM 最大活跃 block 数: %d\n", num_blocks_per_sm);
    printf("保守 block 数: %d\n", num_blocks_conservative);
    printf("最优 block 数: %d\n", num_blocks_optimal);

    return num_blocks_optimal;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== Cooperative Groups: Grid Sync 示例 ===\n\n");

    int device = 0;
    cudaSetDevice(device);

    // 检查 CUDA 版本
    int cuda_version = 0;
    cudaRuntimeGetVersion(&cuda_version);
    printf("CUDA Runtime 版本: %d\n\n", cuda_version);

    // 检查设备支持
    if (!check_cooperative_launch_support(device)) {
        printf("错误：当前设备不支持 cooperative launch！\n");
        printf("需要计算能力 6.0+ 的 GPU\n");
        printf("并且需要在支持的平台上运行\n");
        return 1;
    }
    printf("设备支持 cooperative launch\n\n");

    // 数据设置
    const int N = 1024 * 1024;  // 1M 元素
    const int threads_per_block = 256;

    // 计算合适的 block 数量
    int num_blocks = calculate_num_blocks(device, (void*)two_phase_sum, threads_per_block);
    printf("\n使用 %d blocks, %d threads/block\n", num_blocks, threads_per_block);

    // 分配内存
    float *d_input, *d_partial, *d_result;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_partial, num_blocks * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // 初始化数据
    float* h_input = new float[N];
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // 全 1，期望和为 N
    }
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 创建事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 准备核函数参数
    void* args[] = { &d_input, &d_partial, &d_result, (void*)&N };

    printf("\n--- 启动协作核函数 ---\n");

    // 使用协作启动 API
    cudaEventRecord(start);
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)two_phase_sum,
        dim3(num_blocks),
        dim3(threads_per_block),
        args,
        0,   // shared memory
        0    // stream
    );
    cudaEventRecord(stop);

    if (err != cudaSuccess) {
        printf("启动失败: %s\n", cudaGetErrorString(err));
        // cudaErrorCooperativeLaunchTooManyBlocks 在CUDA 12中已弃用
        // 使用通用错误处理
        printf("提示：block 数量可能太多，无法保证所有 block 同时驻留\n");
        printf("尝试减少 block 数量\n");
        return 1;
    }

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    // 获取结果
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // 验证结果
    float expected = (float)N;
    printf("\n结果验证:\n");
    printf("期望值: %.0f\n", expected);
    printf("计算值: %.0f\n", h_result);
    printf("误差: %.6f\n", std::fabs(h_result - expected));
    printf("耗时: %.3f ms\n", ms);

    bool correct = std::fabs(h_result - expected) < 1e-3;
    printf("结果: %s\n", correct ? "正确" : "错误");

    // 测试 grid 属性
    printf("\n--- 测试 grid 属性 ---\n");
    unsigned long long* d_props;
    unsigned long long h_props[4];
    cudaMalloc(&d_props, 4 * sizeof(unsigned long long));

    void* props_args[] = { &d_props };
    err = cudaLaunchCooperativeKernel(
        (void*)grid_properties_demo,
        dim3(num_blocks),
        dim3(threads_per_block),
        props_args,
        0, 0
    );

    if (err == cudaSuccess) {
        cudaDeviceSynchronize();
        cudaMemcpy(h_props, d_props, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        printf("总线程数: %llu\n", h_props[0]);
        printf("总 block 数: %llu\n", h_props[1]);
        printf("线程 rank (tid 0): %llu\n", h_props[2]);
        printf("is_valid: %s\n", h_props[3] ? "true" : "false");
    }

    // 清理
    cudaFree(d_input);
    cudaFree(d_partial);
    cudaFree(d_result);
    cudaFree(d_props);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;

    printf("\n=== 示例完成 ===\n");
    return 0;
}
