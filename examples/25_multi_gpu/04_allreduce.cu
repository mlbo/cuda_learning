/**
 * 第25章示例4：NCCL AllReduce操作
 *
 * 演示内容：
 * 1. AllReduce操作原理
 * 2. 单线程AllReduce
 * 3. 多线程AllReduce
 * 4. AllReduce性能测试
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <thread>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_NCCL(call) \
    do { \
        ncclResult_t res = call; \
        if (res != ncclSuccess) { \
            printf("NCCL错误 %s:%d: %s\n", __FILE__, __LINE__, \
                   ncclGetErrorString(res)); \
            exit(1); \
        } \
    } while(0)

/**
 * AllReduce原理说明
 */
void explain_allreduce() {
    printf("========================================\n");
    printf("AllReduce操作原理\n");
    printf("========================================\n\n");

    printf("AllReduce是集合通信操作，将所有进程的数据规约后\n");
    printf("广播给所有进程。\n\n");

    printf("示例: 4个GPU，Sum操作\n");
    printf("  输入: GPU0=[1], GPU1=[2], GPU2=[3], GPU3=[4]\n");
    printf("  规约: Sum = 1+2+3+4 = 10\n");
    printf("  输出: GPU0=[10], GPU1=[10], GPU2=[10], GPU3=[10]\n\n");

    printf("典型应用:\n");
    printf("  - 分布式训练梯度同步\n");
    printf("  - 参数聚合\n");
    printf("  - 全局统计计算\n\n");
}

/**
 * 单线程AllReduce示例
 */
void single_thread_allreduce_example(int num_gpus) {
    printf("\n========================================\n");
    printf("单线程AllReduce示例\n");
    printf("========================================\n\n");

    const int N = 1024 * 1024;  // 1M元素
    size_t size = N * sizeof(float);

    // 初始化通信组
    ncclComm_t comms[num_gpus];
    CHECK_NCCL(ncclCommInitAll(comms, num_gpus, NULL));

    // 分配内存
    float* d_send[num_gpus];
    float* d_recv[num_gpus];
    cudaStream_t streams[num_gpus];

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&d_send[i], size));
        CHECK_CUDA(cudaMalloc(&d_recv[i], size));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        // 初始化: 每个GPU填充不同的值
        float val = (float)(i + 1);
        CHECK_CUDA(cudaMemset(d_send[i], 0, size));
        // 简单填充第一个元素
        CHECK_CUDA(cudaMemcpy(d_send[i], &val, sizeof(float), cudaMemcpyHostToDevice));
    }

    printf("初始数据:\n");
    for (int i = 0; i < num_gpus; i++) {
        float h_val;
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMemcpy(&h_val, d_send[i], sizeof(float), cudaMemcpyDeviceToHost));
        printf("  GPU %d: %.1f\n", i, h_val);
    }

    // 创建计时事件
    CHECK_CUDA(cudaSetDevice(0));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("\n执行AllReduce (Sum)...\n");

    // 执行AllReduce
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < num_gpus; i++) {
        CHECK_NCCL(ncclAllReduce(
            d_send[i], d_recv[i], N,
            ncclFloat, ncclSum,
            comms[i], streams[i]
        ));
    }

    // 同步所有流
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("AllReduce完成, 耗时: %.3f ms\n", ms);
    printf("带宽: %.2f GB/s\n", (2.0 * num_gpus - 2.0) * size / ms / 1e6);

    // 验证结果
    printf("\n结果验证:\n");
    float expected = 0;
    for (int i = 0; i < num_gpus; i++) {
        expected += (float)(i + 1);
    }

    for (int i = 0; i < num_gpus; i++) {
        float h_val;
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMemcpy(&h_val, d_recv[i], sizeof(float), cudaMemcpyDeviceToHost));
        printf("  GPU %d: %.1f (期望 %.1f) %s\n",
               i, h_val, expected,
               (std::fabs(h_val - expected) < 0.01f) ? "✓" : "✗");
    }

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(d_send[i]));
        CHECK_CUDA(cudaFree(d_recv[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }
}

/**
 * 多线程AllReduce工作函数
 */
void allreduce_worker(int rank, int num_ranks, ncclUniqueId id,
                       int data_size, int iterations) {
    CHECK_CUDA(cudaSetDevice(rank));

    // 初始化通信组
    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, num_ranks, id, rank));

    // 分配内存
    float* d_data;
    size_t size = data_size * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_data, size));

    // 初始化数据
    float init_val = (float)(rank + 1);
    CHECK_CUDA(cudaMemset(d_data, 0, size));
    CHECK_CUDA(cudaMemcpy(d_data, &init_val, sizeof(float), cudaMemcpyHostToDevice));

    // 创建流和事件
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    for (int i = 0; i < 3; i++) {
        CHECK_NCCL(ncclAllReduce(d_data, d_data, data_size, ncclFloat,
                                  ncclSum, comm, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 计时
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        CHECK_NCCL(ncclAllReduce(d_data, d_data, data_size, ncclFloat,
                                  ncclSum, comm, stream));
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    if (rank == 0) {
        printf("  %d次AllReduce平均时间: %.3f ms\n", iterations, ms / iterations);
        float bandwidth = (2.0 * num_ranks - 2.0) * size / (ms / iterations) / 1e6;
        printf("  有效带宽: %.2f GB/s\n", bandwidth);
    }

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_data));
    ncclCommDestroy(comm);
}

/**
 * 多线程AllReduce示例
 */
void multi_thread_allreduce_example(int num_gpus) {
    printf("\n========================================\n");
    printf("多线程AllReduce示例\n");
    printf("========================================\n\n");

    // 获取唯一ID
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));

    const int N = 1024 * 1024;  // 1M元素
    const int iterations = 10;

    printf("启动 %d 个线程执行AllReduce...\n", num_gpus);
    printf("数据大小: %.2f MB\n", N * sizeof(float) / 1e6);
    printf("迭代次数: %d\n", iterations);

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < num_gpus; i++) {
        threads.emplace_back(allreduce_worker, i, num_gpus, id, N, iterations);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    printf("\n总执行时间: %ld ms\n", duration.count());
}

/**
 * 不同数据大小的AllReduce性能测试
 */
void allreduce_size_benchmark(int num_gpus) {
    printf("\n========================================\n");
    printf("AllReduce性能基准测试\n");
    printf("========================================\n\n");

    // 初始化通信组
    ncclComm_t comms[num_gpus];
    CHECK_NCCL(ncclCommInitAll(comms, num_gpus, NULL));

    // 测试不同大小
    int sizes[] = {
        1024,           // 4 KB
        16 * 1024,      // 64 KB
        256 * 1024,     // 1 MB
        1024 * 1024,    // 4 MB
        4 * 1024 * 1024,// 16 MB
        16 * 1024 * 1024// 64 MB
    };

    printf("%-12s | %-12s | %-12s\n", "数据大小", "时间(ms)", "带宽(GB/s)");
    printf("----------------------------------------------\n");

    for (int s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int N = sizes[s];
        size_t size = N * sizeof(float);

        // 分配内存
        float* d_data[num_gpus];
        cudaStream_t streams[num_gpus];

        for (int i = 0; i < num_gpus; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaMalloc(&d_data[i], size));
            CHECK_CUDA(cudaStreamCreate(&streams[i]));
            CHECK_CUDA(cudaMemset(d_data[i], 0, size));
        }

        // 预热
        for (int i = 0; i < num_gpus; i++) {
            CHECK_NCCL(ncclAllReduce(d_data[i], d_data[i], N, ncclFloat,
                                      ncclSum, comms[i], streams[i]));
        }
        for (int i = 0; i < num_gpus; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        }

        // 计时
        CHECK_CUDA(cudaSetDevice(0));
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        int iterations = 10;
        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < iterations; iter++) {
            for (int i = 0; i < num_gpus; i++) {
                CHECK_NCCL(ncclAllReduce(d_data[i], d_data[i], N, ncclFloat,
                                          ncclSum, comms[i], streams[i]));
            }
        }
        for (int i = 0; i < num_gpus; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        }
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / iterations;
        float bandwidth = (2.0 * num_gpus - 2.0) * size / avg_ms / 1e6;

        printf("%-10.2f MB | %-10.3f | %-10.2f\n",
               size / 1e6, avg_ms, bandwidth);

        // 清理
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        for (int i = 0; i < num_gpus; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaFree(d_data[i]));
            CHECK_CUDA(cudaStreamDestroy(streams[i]));
        }
    }

    // 清理通信组
    for (int i = 0; i < num_gpus; i++) {
        ncclCommDestroy(comms[i]);
    }
}

/**
 * 模拟分布式训练梯度同步
 */
void simulated_gradient_sync(int num_gpus) {
    printf("\n========================================\n");
    printf("模拟分布式训练梯度同步\n");
    printf("========================================\n\n");

    const int param_count = 1000000;  // 1M参数
    size_t size = param_count * sizeof(float);

    printf("模拟场景: 数据并行训练\n");
    printf("参数数量: %d\n", param_count);
    printf("GPU数量: %d\n", num_gpus);

    // 初始化通信组
    ncclComm_t comms[num_gpus];
    CHECK_NCCL(ncclCommInitAll(comms, num_gpus, NULL));

    // 分配梯度缓冲区
    float* d_gradients[num_gpus];
    float* d_params[num_gpus];
    cudaStream_t streams[num_gpus];

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&d_gradients[i], size));
        CHECK_CUDA(cudaMalloc(&d_params[i], size));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        // 模拟计算出的梯度
        float grad_val = (float)(i + 1) * 0.01f;
        CHECK_CUDA(cudaMemset(d_gradients[i], 0, size));
        CHECK_CUDA(cudaMemcpy(d_gradients[i], &grad_val, sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    printf("\n模拟训练循环 (5次迭代):\n");

    for (int iter = 0; iter < 5; iter++) {
        CHECK_CUDA(cudaSetDevice(0));
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));

        // 梯度同步 (AllReduce求平均)
        for (int i = 0; i < num_gpus; i++) {
            CHECK_NCCL(ncclAllReduce(
                d_gradients[i], d_gradients[i], param_count,
                ncclFloat, ncclAvg,  // 使用平均值
                comms[i], streams[i]
            ));
        }

        // 同步
        for (int i = 0; i < num_gpus; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        }

        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        printf("  迭代 %d: 梯度同步耗时 %.3f ms\n", iter + 1, ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // 清理
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(d_gradients[i]));
        CHECK_CUDA(cudaFree(d_params[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }
}

int main() {
    printf("=============================================\n");
    printf("  第25章示例4：NCCL AllReduce操作\n");
    printf("=============================================\n\n");

    // 检查设备
    int num_gpus;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));

    if (num_gpus == 0) {
        printf("未发现CUDA设备\n");
        return 0;
    }

    printf("发现 %d 个GPU\n", num_gpus);

    if (num_gpus < 2) {
        printf("\n注意: AllReduce需要多个GPU才能发挥效果\n");
        printf("单GPU测试将使用模拟数据\n");
    }

    // 解释AllReduce
    explain_allreduce();

    // 单线程示例
    if (num_gpus >= 2) {
        single_thread_allreduce_example(num_gpus);
    }

    // 多线程示例
    if (num_gpus >= 2) {
        multi_thread_allreduce_example(num_gpus);
    }

    // 性能基准
    if (num_gpus >= 2) {
        allreduce_size_benchmark(num_gpus);
    }

    // 模拟梯度同步
    if (num_gpus >= 2) {
        simulated_gradient_sync(num_gpus);
    }

    printf("\n示例完成！\n");
    return 0;
}
