/**
 * 第二十四章示例：流捕获
 *
 * 本示例演示不同方式的流捕获
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
__global__ void kernel_a(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

__global__ void kernel_b(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

// 单流捕获
void single_stream_capture(float* d_data, int n) {
    printf("\n=== 单流捕获 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    size_t size = n * sizeof(float);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 开始捕获
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // 在单个流中记录操作
    kernel_a<<<numBlocks, blockSize, 0, stream>>>(d_data, n);
    kernel_b<<<numBlocks, blockSize, 0, stream>>>(d_data, n);

    // 结束捕获
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // 打印图信息
    size_t numNodes;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));
    printf("捕获的节点数: %zu\n", numNodes);

    // 清理
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

// 多流捕获 - 使用显式图API演示节点依赖
void multi_stream_capture(float* d_data, int n) {
    printf("\n=== 多流捕获 (显式图API) ===\n");
    printf("注: 多流捕获需要特殊的捕获模式，这里使用显式图API演示\n");

    cudaGraph_t graph;
    CHECK_CUDA(cudaGraphCreate(&graph, 0));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 创建内核节点参数
    void* kernelArgs[] = {&d_data, &n};
    cudaKernelNodeParams nodeParams = {0};
    nodeParams.gridDim = dim3(numBlocks);
    nodeParams.blockDim = dim3(blockSize);
    nodeParams.sharedMemBytes = 0;
    nodeParams.kernelParams = (void**)kernelArgs;
    nodeParams.extra = NULL;

    // 创建两个内核节点
    cudaGraphNode_t nodeA, nodeB;
    nodeParams.func = (void*)kernel_a;
    CHECK_CUDA(cudaGraphAddKernelNode(&nodeA, graph, NULL, 0, &nodeParams));

    nodeParams.func = (void*)kernel_b;
    CHECK_CUDA(cudaGraphAddKernelNode(&nodeB, graph, NULL, 0, &nodeParams));

    // 两个kernel读写同一块数据，必须显式建立依赖避免数据竞争
    CHECK_CUDA(cudaGraphAddDependencies(graph, &nodeA, &nodeB, nullptr, 1));

    // 打印图信息
    size_t numNodes;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));
    printf("捕获的节点数: %zu\n", numNodes);

    // 获取边信息 (CUDA 12+ API)
    size_t numEdges;
    CHECK_CUDA(cudaGraphGetEdges(graph, NULL, NULL, NULL, &numEdges));
    printf("图中的边数: %zu (nodeA -> nodeB 顺序执行)\n", numEdges);

    // 清理
    CHECK_CUDA(cudaGraphDestroy(graph));
}

// 捕获模式演示
void capture_modes_demo(float* d_data, int n) {
    printf("\n=== 捕获模式演示 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    size_t size = n * sizeof(float);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 捕获模式
    cudaStreamCaptureMode modes[] = {
        cudaStreamCaptureModeGlobal,
        cudaStreamCaptureModeThreadLocal,
        cudaStreamCaptureModeRelaxed
    };

    const char* modeNames[] = {
        "Global",
        "ThreadLocal",
        "Relaxed"
    };

    for (int i = 0; i < 3; i++) {
        cudaGraph_t graph;

        CHECK_CUDA(cudaStreamBeginCapture(stream, modes[i]));
        kernel_a<<<numBlocks, blockSize, 0, stream>>>(d_data, n);
        kernel_b<<<numBlocks, blockSize, 0, stream>>>(d_data, n);
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

        size_t numNodes;
        CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));
        printf("模式 %s: 节点数 = %zu\n", modeNames[i], numNodes);

        CHECK_CUDA(cudaGraphDestroy(graph));
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
}

// 捕获错误处理
void capture_error_handling(float* d_data, int n) {
    printf("\n=== 捕获错误处理 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 正常捕获
    printf("1. 正常捕获流程:\n");
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    kernel_a<<<numBlocks, blockSize, 0, stream>>>(d_data, n);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    printf("   捕获成功\n");
    CHECK_CUDA(cudaGraphDestroy(graph));

    // 尝试在捕获中执行不支持的操作
    printf("2. 捕获中执行同步操作:\n");
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    kernel_a<<<numBlocks, blockSize, 0, stream>>>(d_data, n);

    // 注意: 捕获期间调用同步操作通常不被允许，可能使捕获失效
    cudaError_t sync_result = cudaStreamSynchronize(stream);
    if (sync_result != cudaSuccess) {
        printf("   警告: 同步操作返回错误 (预期行为): %s\n", cudaGetErrorString(sync_result));
        cudaStreamCaptureStatus cap_status;
        cudaError_t cap_query = cudaStreamIsCapturing(stream, &cap_status);
        if (cap_query == cudaSuccess && cap_status == cudaStreamCaptureStatusInvalidated) {
            printf("   提示: 捕获已失效 (invalidated)\n");
        }
    }

    cudaError_t end_result = cudaStreamEndCapture(stream, &graph);
    if (end_result != cudaSuccess) {
        printf("   捕获失败: %s (某些操作在捕获期间不允许或导致捕获失效)\n", cudaGetErrorString(end_result));
        // 重置流状态
        CHECK_CUDA(cudaStreamDestroy(stream));
        CHECK_CUDA(cudaStreamCreate(&stream));
    } else {
        printf("   捕获成功（本次同步未导致捕获失败）\n");
        CHECK_CUDA(cudaGraphDestroy(graph));
    }

    // 捕获状态查询
    printf("3. 捕获状态查询:\n");
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    kernel_a<<<numBlocks, blockSize, 0, stream>>>(d_data, n);

    cudaStreamCaptureStatus status;
    CHECK_CUDA(cudaStreamIsCapturing(stream, &status));
    printf("   流状态: %s\n",
           status == cudaStreamCaptureStatusActive ? "正在捕获" :
           status == cudaStreamCaptureStatusInvalidated ? "已失效" : "其他");

    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphDestroy(graph));

    CHECK_CUDA(cudaStreamDestroy(stream));
}

// 捕获包含memcpy操作
void capture_with_memcpy(float* d_data, float* h_data, int n) {
    printf("\n=== 捕获包含Memcpy操作 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    size_t size = n * sizeof(float);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 开始捕获
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // H2D传输
    CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream));

    // 核函数
    kernel_a<<<numBlocks, blockSize, 0, stream>>>(d_data, n);
    kernel_b<<<numBlocks, blockSize, 0, stream>>>(d_data, n);

    // D2H传输
    CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream));

    // 结束捕获
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // 打印图信息
    size_t numNodes;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));
    printf("捕获的节点数: %zu (包含Memcpy节点)\n", numNodes);

    // 清理
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

int main() {
    printf("========================================\n");
    printf("  流捕获示例 - 第二十四章\n");
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

    float *h_data, *d_data;
    CHECK_CUDA(cudaMallocHost(&h_data, size));
    CHECK_CUDA(cudaMalloc(&d_data, size));

    // 初始化
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
    }

    // 演示不同的捕获方式
    single_stream_capture(d_data, n);
    multi_stream_capture(d_data, n);
    capture_modes_demo(d_data, n);
    capture_error_handling(d_data, n);
    capture_with_memcpy(d_data, h_data, n);

    // 清理
    CHECK_CUDA(cudaFreeHost(h_data));
    CHECK_CUDA(cudaFree(d_data));

    printf("\n========================================\n");
    printf("关键要点:\n");
    printf("  1. cudaStreamBeginCapture开始捕获\n");
    printf("  2. 支持单流和多流捕获\n");
    printf("  3. 不同捕获模式有不同语义\n");
    printf("  4. Memcpy操作也可以被捕获\n");
    printf("  5. 注意捕获中的错误处理\n");
    printf("========================================\n\n");

    return 0;
}
