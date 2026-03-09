/**
 * 第25章示例3：NCCL基础使用
 *
 * 演示内容：
 * 1. NCCL初始化
 * 2. 点对点Send/Recv通信
 * 3. NCCL Group操作
 * 4. 错误处理
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

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
 * 打印NCCL版本信息
 */
void print_nccl_version() {
    int version;
    ncclGetVersion(&version);

    int major = version / 1000;
    int minor = (version % 1000) / 100;
    int patch = version % 100;

    printf("NCCL版本: %d.%d.%d (构建版本: %d)\n", major, minor, patch, version);
}

/**
 * NCCL单线程初始化示例
 * 使用ncclCommInitAll初始化所有GPU的通信组
 */
void nccl_single_thread_init_example() {
    printf("\n========================================\n");
    printf("NCCL单线程初始化示例\n");
    printf("========================================\n\n");

    int num_gpus;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));

    if (num_gpus < 2) {
        printf("需要至少2个GPU运行此示例\n");
        return;
    }

    printf("使用 %d 个GPU\n", num_gpus);

    // 初始化通信组
    ncclComm_t comms[num_gpus];
    CHECK_NCCL(ncclCommInitAll(comms, num_gpus, NULL));
    printf("NCCL通信组初始化成功\n");

    // 打印每个rank的信息
    for (int i = 0; i < num_gpus; i++) {
        int rank = ncclCommCuDevice(comms[i]);
        printf("  Rank %d: GPU %d\n", i, rank);
    }

    // 清理
    for (int i = 0; i < num_gpus; i++) {
        ncclCommDestroy(comms[i]);
    }
    printf("NCCL通信组已销毁\n");
}

/**
 * NCCL点对点Send/Recv示例
 */
void nccl_p2p_sendrecv_example() {
    printf("\n========================================\n");
    printf("NCCL点对点Send/Recv示例\n");
    printf("========================================\n\n");

    int num_gpus;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));

    if (num_gpus < 2) {
        printf("需要至少2个GPU运行此示例\n");
        return;
    }

    const int N = 1024;  // 数据大小
    size_t size = N * sizeof(float);

    // 初始化通信组
    ncclComm_t comms[num_gpus];
    CHECK_NCCL(ncclCommInitAll(comms, num_gpus, NULL));

    // 分配设备内存
    float* d_send[num_gpus];
    float* d_recv[num_gpus];

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&d_send[i], size));
        CHECK_CUDA(cudaMalloc(&d_recv[i], size));

        // 初始化发送数据
        float val = (float)(i + 1) * 100.0f;
        CHECK_CUDA(cudaMemset(d_send[i], 0, size));
        // 简单填充
        CHECK_CUDA(cudaMemcpy(d_send[i], &val, sizeof(float), cudaMemcpyHostToDevice));
    }

    // 创建CUDA流
    cudaStream_t streams[num_gpus];
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    printf("执行点对点通信: GPU0 -> GPU1\n");

    // GPU 0 发送数据给 GPU 1
    CHECK_NCCL(ncclGroupStart());
    {
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_NCCL(ncclSend(d_send[0], N, ncclFloat, 1, comms[0], streams[0]));

        CHECK_CUDA(cudaSetDevice(1));
        CHECK_NCCL(ncclRecv(d_recv[1], N, ncclFloat, 0, comms[1], streams[1]));
    }
    CHECK_NCCL(ncclGroupEnd());

    // 同步
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaStreamSynchronize(streams[1]));

    // 验证接收到的数据
    float h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, d_recv[1], sizeof(float), cudaMemcpyDeviceToHost));
    printf("GPU 1 接收到的数据: %.1f (期望 100.0)\n", h_result);

    // 清理
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(d_send[i]));
        CHECK_CUDA(cudaFree(d_recv[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }
}

/**
 * NCCL多线程示例
 * 每个线程处理一个GPU
 */
void worker_thread(int rank, int num_ranks, ncclUniqueId id) {
    // 设置设备
    CHECK_CUDA(cudaSetDevice(rank));

    // 初始化通信组
    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, num_ranks, id, rank));

    printf("  Rank %d: 通信组初始化完成\n", rank);

    // 分配缓冲区
    const int N = 1024;
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    // 初始化数据
    float init_val = (float)(rank + 1) * 10.0f;
    CHECK_CUDA(cudaMemset(d_data, 0, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, &init_val, sizeof(float), cudaMemcpyHostToDevice));

    // 创建流
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 执行AllReduce
    CHECK_NCCL(ncclAllReduce(d_data, d_data, N, ncclFloat, ncclSum, comm, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 验证结果
    float h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, d_data, sizeof(float), cudaMemcpyDeviceToHost));

    float expected = 0;
    for (int i = 0; i < num_ranks; i++) {
        expected += (float)(i + 1) * 10.0f;
    }

    printf("  Rank %d: AllReduce结果 = %.1f (期望 %.1f)\n", rank, h_result, expected);

    // 清理
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);
}

void nccl_multithread_example() {
    printf("\n========================================\n");
    printf("NCCL多线程示例\n");
    printf("========================================\n\n");

    int num_gpus;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));

    if (num_gpus < 2) {
        printf("需要至少2个GPU运行此示例\n");
        return;
    }

    // 获取唯一ID
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    printf("NCCL Unique ID已生成\n");

    // 启动多线程
    printf("启动 %d 个工作线程...\n", num_gpus);

    std::vector<std::thread> threads;
    for (int i = 0; i < num_gpus; i++) {
        threads.emplace_back(worker_thread, i, num_gpus, id);
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    printf("所有工作线程已完成\n");
}

/**
 * NCCL错误处理示例
 */
void nccl_error_handling_example() {
    printf("\n========================================\n");
    printf("NCCL错误处理示例\n");
    printf("========================================\n\n");

    printf("NCCL错误码和描述:\n");

    const ncclResult_t errors[] = {
        ncclSuccess,
        ncclUnhandledCudaError,
        ncclSystemError,
        ncclInternalError,
        ncclInvalidArgument,
        ncclInvalidUsage
    };

    const char* error_names[] = {
        "ncclSuccess",
        "ncclUnhandledCudaError",
        "ncclSystemError",
        "ncclInternalError",
        "ncclInvalidArgument",
        "ncclInvalidUsage"
    };

    for (int i = 0; i < sizeof(errors) / sizeof(errors[0]); i++) {
        printf("  %s: %s\n", error_names[i], ncclGetErrorString(errors[i]));
    }

    printf("\nNCCL调试环境变量:\n");
    printf("  NCCL_DEBUG=INFO      - 打印详细信息\n");
    printf("  NCCL_DEBUG=WARN      - 只打印警告\n");
    printf("  NCCL_DEBUG=VERSION   - 打印版本信息\n");
    printf("  NCCL_SOCKET_IFNAME   - 指定网络接口\n");
    printf("  NCCL_P2P_DISABLE=1   - 禁用P2P\n");
}

/**
 * NCCL性能建议
 */
void nccl_performance_tips() {
    printf("\n========================================\n");
    printf("NCCL性能优化建议\n");
    printf("========================================\n\n");

    printf("1. 使用NCCL Group:\n");
    printf("   - 将多个通信操作打包\n");
    printf("   - 允许NCCL优化通信模式\n\n");

    printf("2. 通信与计算重叠:\n");
    printf("   - 使用独立的CUDA流\n");
    printf("   - 隐藏通信延迟\n\n");

    printf("3. 批量通信:\n");
    printf("   - 减少通信次数\n");
    printf("   - 增大每次通信的数据量\n\n");

    printf("4. 拓扑感知:\n");
    printf("   - 使用NVLink连接的GPU对\n");
    printf("   - 避免跨NUMA节点通信\n\n");

    printf("5. 内存布局:\n");
    printf("   - 确保数据在设备内存中连续\n");
    printf("   - 使用cudaMalloc分配\n");
}

int main() {
    printf("=============================================\n");
    printf("  第25章示例3：NCCL基础使用\n");
    printf("=============================================\n\n");

    // 检查CUDA设备
    int num_gpus;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    printf("系统GPU数量: %d\n", num_gpus);

    if (num_gpus == 0) {
        printf("未发现CUDA设备\n");
        return 0;
    }

    // 打印NCCL版本
    print_nccl_version();

    // 单线程初始化示例
    nccl_single_thread_init_example();

    // 点对点通信示例
    nccl_p2p_sendrecv_example();

    // 多线程示例
    nccl_multithread_example();

    // 错误处理示例
    nccl_error_handling_example();

    // 性能建议
    nccl_performance_tips();

    printf("\n示例完成！\n");
    return 0;
}