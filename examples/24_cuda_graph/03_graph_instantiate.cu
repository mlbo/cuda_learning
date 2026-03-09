/**
 * 第二十四章示例：图实例化
 *
 * 本示例演示图的实例化和执行
 */

#include <cstdio>
#include <cstdlib>
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
__global__ void kernel_process(float* data, int n, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * value + 1.0f;
    }
}

// 基本实例化
void basic_instantiation(float* d_data, int n) {
    printf("\n=== 基本实例化 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec = nullptr;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 创建图
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    int value = 2;
    kernel_process<<<numBlocks, blockSize, 0, stream>>>(d_data, n, value);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    printf("图创建成功\n");

    // 实例化图
    cudaError_t err = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    if (err != cudaSuccess) {
        printf("实例化失败: %s\n", cudaGetErrorString(err));
    } else {
        printf("图实例化成功\n");

        // 执行图
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        printf("图执行成功\n");
    }

    // 清理
    if (graphExec != nullptr) {
        CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    }
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

// 实例化错误处理
void instantiation_error_handling(float* d_data, int n) {
    printf("\n=== 实例化错误处理 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec = nullptr;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 创建图
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    kernel_process<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 2);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // 使用错误回调进行实例化
    cudaGraphNode_t errorNode = nullptr;
    char errorBuffer[1024];
    errorBuffer[0] = '\0';

    cudaError_t err = cudaGraphInstantiate(&graphExec, graph, &errorNode, errorBuffer, sizeof(errorBuffer));
    errorBuffer[sizeof(errorBuffer) - 1] = '\0';

    if (err == cudaSuccess) {
        printf("实例化成功，无错误\n");
    } else if (err == cudaErrorInvalidDevicePointer) {
        printf("错误: 无效设备指针\n");
        if (errorNode) {
            printf("错误节点: %p\n", errorNode);
        }
        if (errorBuffer[0]) {
            printf("错误信息: %s\n", errorBuffer);
        }
    } else {
        printf("实例化错误: %s\n", cudaGetErrorString(err));
    }

    // 清理
    if (err == cudaSuccess) {
        CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    }
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

// 图执行和同步
void graph_execution_and_sync(float* d_data, int n) {
    printf("\n=== 图执行和同步 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 创建图
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    kernel_process<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 2);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // 实例化
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // 多次执行
    printf("多次执行图:\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < 5; i++) {
        CHECK_CUDA(cudaEventRecord(start, stream));

        // 执行图
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));

        // 同步方式1: 流同步
        CHECK_CUDA(cudaStreamSynchronize(stream));

        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("  执行 %d: %.3f ms\n", i + 1, ms);
    }

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

// 在不同流中执行
void execute_in_different_streams(float* d_data, int n) {
    printf("\n=== 在不同流中执行 ===\n");

    const int NUM_STREAMS = 3;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 创建图（在stream 0中捕获）
    CHECK_CUDA(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
    kernel_process<<<numBlocks, blockSize, 0, streams[0]>>>(d_data, n, 2);
    CHECK_CUDA(cudaStreamEndCapture(streams[0], &graph));

    // 实例化
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    printf("在不同流中执行同一个图实例:\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaEventRecord(start, streams[i]));
        CHECK_CUDA(cudaGraphLaunch(graphExec, streams[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        CHECK_CUDA(cudaEventRecord(stop, streams[i]));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("  Stream %d: %.3f ms\n", i, ms);
    }

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
}

// 图实例复用
void graph_instance_reuse(float* d_data, int n, int iterations) {
    printf("\n=== 图实例复用 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 创建并实例化图（只做一次）
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    kernel_process<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 2);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    printf("图创建和实例化完成，开始 %d 次执行:\n", iterations);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));

    // 多次复用同一个图实例
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("总执行时间: %.3f ms\n", ms);
    printf("每次执行: %.3f ms\n", ms / iterations);

    printf("\n关键: 图只需创建和实例化一次，可多次执行\n");

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

int main() {
    printf("========================================\n");
    printf("  图实例化示例 - 第二十四章\n");
    printf("========================================\n");

    // 获取设备信息
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("\n设备: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // 准备数据
    int n = 1024 * 1024;
    size_t size = n * sizeof(float);

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));

    // 演示实例化
    basic_instantiation(d_data, n);
    instantiation_error_handling(d_data, n);
    graph_execution_and_sync(d_data, n);
    execute_in_different_streams(d_data, n);
    graph_instance_reuse(d_data, n, 100);

    // 清理
    CHECK_CUDA(cudaFree(d_data));

    printf("\n========================================\n");
    printf("关键要点:\n");
    printf("  1. cudaGraphInstantiate实例化图\n");
    printf("  2. 实例化只需执行一次\n");
    printf("  3. cudaGraphLaunch执行图\n");
    printf("  4. 图实例可在不同流中执行\n");
    printf("  5. 复用图实例避免重复开销\n");
    printf("========================================\n\n");

    return 0;
}
