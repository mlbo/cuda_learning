/**
 * 第二十四章示例：CUDA Graph基础
 *
 * 本示例演示CUDA Graph的基本概念和操作
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 简单核函数
__global__ void simple_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// 无Graph的传统执行方式
void traditional_execution(float* d_data, float* h_data, float* h_original, int n, int iterations) {
    printf("\n=== 传统执行方式 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    size_t size = n * sizeof(float);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    CHECK_CUDA(cudaEventRecord(start, stream));

    for (int i = 0; i < iterations; i++) {
        // 每次迭代前重置数据，确保每次执行相同的工作
        std::memcpy(h_data, h_original, size);

        // 每次迭代都需要单独启动每个操作
        CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream));
        simple_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, n);
        CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("迭代次数: %d\n", iterations);
    printf("总执行时间: %.3f ms\n", ms);
    printf("每次迭代: %.3f ms\n", ms / iterations);

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 使用CUDA Graph执行
void graph_execution(float* d_data, float* h_data, float* h_original, int n, int iterations) {
    printf("\n=== CUDA Graph执行方式 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    size_t size = n * sizeof(float);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // === 步骤1: 创建图 ===
    cudaGraph_t graph;

    // 开始流捕获
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // 记录操作到图中
    CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream));
    simple_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, n);
    CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream));

    // 结束捕获
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    printf("图创建成功\n");

    // 获取图信息
    size_t numNodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));
    printf("图节点数量: %zu\n", numNodes);

    // === 步骤2: 实例化图 ===
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    printf("图实例化成功\n");

    // === 步骤3: 执行图 ===
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));

    for (int i = 0; i < iterations; i++) {
        // 每次迭代前重置数据，确保每次执行相同的工作
        std::memcpy(h_data, h_original, size);

        // 每次只需执行图，所有操作批量提交
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("迭代次数: %d\n", iterations);
    printf("总执行时间: %.3f ms\n", ms);
    printf("每次迭代: %.3f ms\n", ms / iterations);

    // 清理
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// CUDA Graph基本流程演示
void graph_basic_flow_demo() {
    printf("\n=== CUDA Graph基本流程演示 ===\n");

    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CHECK_CUDA(cudaStreamCreate(&stream));

    // 注意：cudaMalloc 不能在流捕获期间调用，必须提前分配
    float* d_tmp;
    CHECK_CUDA(cudaMalloc(&d_tmp, 1024 * sizeof(float)));

    // 1. 开始捕获
    printf("1. 开始流捕获...\n");
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // 2. 执行操作（被记录到图中）
    printf("2. 记录操作到图中...\n");
    simple_kernel<<<4, 256, 0, stream>>>(d_tmp, 1024);

    // 3. 结束捕获
    printf("3. 结束流捕获...\n");
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // 4. 实例化图
    printf("4. 实例化图...\n");
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // 5. 执行图
    printf("5. 执行图...\n");
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("CUDA Graph执行流程完成!\n");

    // 清理
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_tmp));
}

// 验证结果正确性
// 注意：每次迭代都会用原始数据执行核函数，所以结果应该是 data[i] * 2 + 1
// 但由于传统方式每次都从 h_data 复制到 d_data，处理后再复制回来
// 100次迭代后 h_data 的值应该是基于最后一次迭代的结果
void verify_results(float* h_data, int n, bool is_graph = false) {
    bool correct = true;
    // 对于传统执行和Graph执行，每次都是基于初始数据执行一次核函数
    // 核函数: data[idx] = data[idx] * 2.0f + 1.0f
    // 初始值: h_data[i] = i
    // 执行一次后的期望值: i * 2.0f + 1.0f
    for (int i = 0; i < 10; i++) {
        float expected = (float)i * 2.0f + 1.0f;
        if (std::fabs(h_data[i] - expected) > 1e-5) {
            printf("验证失败: h_data[%d] = %.2f, expected = %.2f\n",
                   i, h_data[i], expected);
            correct = false;
            break;
        }
    }
    printf("结果验证: %s\n", correct ? "通过" : "失败");
}

int main() {
    printf("========================================\n");
    printf("  CUDA Graph基础示例 - 第二十四章\n");
    printf("========================================\n");

    // 获取设备信息
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("\n设备: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // 准备数据
    int n = 1024 * 1024;  // 1M 元素
    size_t size = n * sizeof(float);

    float *h_data, *h_original, *d_data;
    CHECK_CUDA(cudaMallocHost(&h_data, size));
    CHECK_CUDA(cudaMallocHost(&h_original, size));  // 保存原始数据副本
    CHECK_CUDA(cudaMalloc(&d_data, size));

    // 初始化
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
        h_original[i] = (float)i;
    }

    // 演示基本流程
    graph_basic_flow_demo();

    // 性能对比
    int iterations = 100;

    // 传统执行方式
    std::memcpy(h_data, h_original, size);  // 重置数据
    traditional_execution(d_data, h_data, h_original, n, iterations);
    verify_results(h_data, n, false);

    // CUDA Graph执行方式
    std::memcpy(h_data, h_original, size);  // 重置数据
    graph_execution(d_data, h_data, h_original, n, iterations);
    verify_results(h_data, n, true);

    // 清理
    CHECK_CUDA(cudaFreeHost(h_data));
    CHECK_CUDA(cudaFreeHost(h_original));
    CHECK_CUDA(cudaFree(d_data));

    printf("\n========================================\n");
    printf("关键要点:\n");
    printf("  1. cudaStreamBeginCapture开始捕获\n");
    printf("  2. cudaStreamEndCapture结束捕获并创建图\n");
    printf("  3. cudaGraphInstantiate实例化图\n");
    printf("  4. cudaGraphLaunch执行图\n");
    printf("  5. 图只需创建一次，可多次执行\n");
    printf("========================================\n\n");

    return 0;
}
