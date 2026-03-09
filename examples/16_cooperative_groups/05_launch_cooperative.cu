/**
 * 05_launch_cooperative.cu - cudaLaunchCooperativeKernel 详细用法
 *
 * 本示例演示：
 * 1. 检查设备是否支持协作启动
 * 2. 计算合适的 block 配置
 * 3. 正确使用 cudaLaunchCooperativeKernel
 * 4. 错误处理
 * 5. 与普通启动的对比
 *
 * 编译要求：CUDA 9.0+, CC 6.0+
 */

#include <stdio.h>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// ============================================================================
// 核函数：持久化线程块示例
// ============================================================================
__global__ void persistent_blocks(int* task_queue, int* results, int num_tasks) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // 持久化循环：每个 block 持续处理任务直到队列为空
    while (true) {
        // 使用原子操作获取任务 ID
        int task_id = -1;
        if (block.thread_rank() == 0) {
            task_id = atomicAdd(task_queue, 1);
        }

        // 广播任务 ID 到 block 内所有线程
        __shared__ int s_task_id;
        if (block.thread_rank() == 0) {
            s_task_id = task_id;
        }
        block.sync();
        task_id = s_task_id;

        // 检查是否完成
        if (task_id >= num_tasks) {
            break;
        }

        // 处理任务（这里简单地将任务 ID 乘以 2）
        results[task_id] = task_id * 2;

        // 可选：在这里进行跨块同步
        // grid.sync();
    }
}

// ============================================================================
// 核函数：生产者-消费者模式
// ============================================================================
__global__ void producer_consumer(float* data, float* results, int N, int iterations) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int iter = 0; iter < iterations; iter++) {
        // 生产阶段
        if (blockIdx.x < gridDim.x / 2) {
            // 前一半 block 作为生产者
            for (int i = tid; i < N; i += stride) {
                data[i] = data[i] * 1.1f;
            }
        }

        // 跨块同步
        grid.sync();

        // 消费阶段
        if (blockIdx.x >= gridDim.x / 2) {
            // 后一半 block 作为消费者
            float sum = 0.0f;
            for (int i = tid; i < N; i += stride) {
                sum += data[i];
            }

            if (block.thread_rank() == 0) {
                results[iter] = sum;
            }
        }

        // 跨块同步，准备下一轮
        grid.sync();
    }
}

// ============================================================================
// 辅助类：协作启动管理器
// ============================================================================
class CooperativeLaunchManager {
public:
    CooperativeLaunchManager(int device = 0) : device_(device), supports_coop_(false) {
        check_support();
    }

    bool is_supported() const { return supports_coop_; }

    int calculate_blocks(void* kernel, int threads, size_t shared_mem = 0) {
        if (!supports_coop_) return 0;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_);

        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, kernel, threads, shared_mem);

        return prop.multiProcessorCount * blocks_per_sm;
    }

    cudaError_t launch(void* kernel, dim3 grid, dim3 block,
                       void** args, size_t shared_mem = 0, cudaStream_t stream = 0) {
        if (!supports_coop_) {
            return cudaErrorNotSupported;
        }

        return cudaLaunchCooperativeKernel(kernel, grid, block, args, shared_mem, stream);
    }

    void print_info() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_);

        printf("设备信息:\n");
        printf("  名称: %s\n", prop.name);
        printf("  计算能力: %d.%d\n", prop.major, prop.minor);
        printf("  SM 数量: %d\n", prop.multiProcessorCount);
        printf("  协作启动支持: %s\n", supports_coop_ ? "是" : "否");

        // 检查平台限制
        printf("\n平台检查:\n");
        #ifdef _WIN32
            printf("  Windows 平台\n");
        #else
            printf("  Linux 平台\n");
            // 可以检查 MPS 状态
        #endif
    }

private:
    void check_support() {
        int supports = 0;
        cudaError_t err = cudaDeviceGetAttribute(&supports,
                                                  cudaDevAttrCooperativeLaunch,
                                                  device_);
        supports_coop_ = (err == cudaSuccess && supports != 0);
    }

    int device_;
    bool supports_coop_;
};

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== cudaLaunchCooperativeKernel 详细用法 ===\n\n");

    CooperativeLaunchManager manager(0);
    manager.print_info();

    if (!manager.is_supported()) {
        printf("\n错误：当前设备/平台不支持协作启动\n");
        printf("要求：\n");
        printf("  - 计算能力 6.0+\n");
        printf("  - Linux（无 MPS 或 CC 7.0+ 有 MPS）\n");
        printf("  - 或最新 Windows 版本\n");
        return 1;
    }

    // ================================================================
    // 示例 1：持久化线程块
    // ================================================================
    printf("\n--- 示例 1：持久化线程块 ---\n");

    const int num_tasks = 1000;
    const int threads_per_block = 128;

    int num_blocks = manager.calculate_blocks((void*)persistent_blocks, threads_per_block);
    printf("使用 %d blocks\n", num_blocks);

    int *d_task_queue, *d_results;
    cudaMalloc(&d_task_queue, sizeof(int));
    cudaMalloc(&d_results, num_tasks * sizeof(int));

    // 初始化任务队列为 0
    cudaMemset(d_task_queue, 0, sizeof(int));

    void* args1[] = { &d_task_queue, &d_results, (void*)&num_tasks };
    cudaError_t err = manager.launch((void*)persistent_blocks,
                                      dim3(num_blocks),
                                      dim3(threads_per_block),
                                      args1);

    if (err != cudaSuccess) {
        printf("启动失败: %s\n", cudaGetErrorString(err));
    } else {
        cudaDeviceSynchronize();

        // 验证结果
        int* h_results = new int[num_tasks];
        cudaMemcpy(h_results, d_results, num_tasks * sizeof(int), cudaMemcpyDeviceToHost);

        int correct = 0;
        for (int i = 0; i < num_tasks; i++) {
            if (h_results[i] == i * 2) correct++;
        }
        printf("任务完成: %d/%d 正确\n", correct, num_tasks);

        delete[] h_results;
    }

    cudaFree(d_task_queue);
    cudaFree(d_results);

    // ================================================================
    // 示例 2：生产者-消费者
    // ================================================================
    printf("\n--- 示例 2：生产者-消费者模式 ---\n");

    const int N = 1024;
    const int iterations = 5;

    // 确保偶数个 block
    num_blocks = manager.calculate_blocks((void*)producer_consumer, threads_per_block);
    num_blocks = (num_blocks / 2) * 2;  // 确保是偶数
    printf("使用 %d blocks (%d 生产者, %d 消费者)\n",
           num_blocks, num_blocks / 2, num_blocks / 2);

    float *d_data, *d_float_results;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_float_results, iterations * sizeof(float));

    // 初始化数据
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_data;

    void* args2[] = { &d_data, &d_float_results, (void*)&N, (void*)&iterations };
    err = manager.launch((void*)producer_consumer,
                         dim3(num_blocks),
                         dim3(threads_per_block),
                         args2);

    if (err != cudaSuccess) {
        printf("启动失败: %s\n", cudaGetErrorString(err));
    } else {
        cudaDeviceSynchronize();

        float* h_results = new float[iterations];
        cudaMemcpy(h_results, d_float_results, iterations * sizeof(float), cudaMemcpyDeviceToHost);

        printf("迭代结果:\n");
        for (int i = 0; i < iterations; i++) {
            printf("  迭代 %d: sum = %.0f (期望 ~%.0f)\n",
                   i, h_results[i], N * std::pow(1.1f, i + 1));
        }

        delete[] h_results;
    }

    cudaFree(d_data);
    cudaFree(d_float_results);

    // ================================================================
    // 错误处理示例
    // ================================================================
    printf("\n--- 错误处理示例 ---\n");

    // 尝试启动过多的 block
    int too_many_blocks = 100000;
    printf("尝试启动 %d blocks（预期失败）...\n", too_many_blocks);

    int num_tasks_err = 1;
    int *d_task_queue_err, *d_results_err;
    cudaMalloc(&d_task_queue_err, sizeof(int));
    cudaMalloc(&d_results_err, sizeof(int));
    cudaMemset(d_task_queue_err, 0, sizeof(int));

    void* args3[] = { &d_task_queue_err, &d_results_err, (void*)&num_tasks_err };
    err = manager.launch((void*)persistent_blocks,
                         dim3(too_many_blocks),
                         dim3(threads_per_block),
                         args3);

    if (err != cudaSuccess) {
        printf("预期错误: %s\n", cudaGetErrorString(err));
        // cudaErrorCooperativeLaunchTooManyBlocks 在CUDA 12中已弃用
        printf("  -> 原因: block 数量超出设备能同时驻留的限制\n");
    }
    cudaFree(d_task_queue_err);
    cudaFree(d_results_err);

    printf("\n=== 示例完成 ===\n");
    return 0;
}
