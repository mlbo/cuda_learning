/**
 * 第二十四章示例：图优化
 *
 * 本示例演示CUDA Graph的性能优化技术
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
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
__global__ void kernel_op1(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

__global__ void kernel_op2(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

__global__ void kernel_op3(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]);
    }
}

// 无Graph执行
void no_graph_execution(float* d_data, float* h_data, int n, int num_kernels, int iterations) {
    printf("\n=== 无Graph执行 (%d 个核函数) ===\n", num_kernels);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    size_t size = n * sizeof(float);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));

    for (int iter = 0; iter < iterations; iter++) {
        // H2D
        CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream));

        // 多个核函数
        for (int k = 0; k < num_kernels; k++) {
            kernel_op1<<<numBlocks, blockSize, 0, stream>>>(d_data, n);
        }

        // D2H
        CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("迭代次数: %d\n", iterations);
    printf("总时间: %.3f ms\n", ms);
    printf("每次迭代: %.3f ms\n", ms / iterations);

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 有Graph执行
void with_graph_execution(float* d_data, float* h_data, int n, int num_kernels, int iterations) {
    printf("\n=== 有Graph执行 (%d 个核函数) ===\n", num_kernels);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    size_t size = n * sizeof(float);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 创建图
    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream));

    for (int k = 0; k < num_kernels; k++) {
        kernel_op1<<<numBlocks, blockSize, 0, stream>>>(d_data, n);
    }

    CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // 实例化
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // 执行
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));

    for (int iter = 0; iter < iterations; iter++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("迭代次数: %d\n", iterations);
    printf("总时间: %.3f ms\n", ms);
    printf("每次迭代: %.3f ms\n", ms / iterations);

    // 清理
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 性能对比
void performance_comparison(float* d_data, float* h_data, int n) {
    printf("\n=== 性能对比 ===\n");

    int kernel_counts[] = {1, 5, 10, 50, 100};
    int num_counts = sizeof(kernel_counts) / sizeof(kernel_counts[0]);
    int iterations = 50;

    printf("\n%-15s %-15s %-15s %-15s\n", "核函数数", "无Graph(ms)", "有Graph(ms)", "提升(%)");
    printf("----------------------------------------------------------\n");

    for (int i = 0; i < num_counts; i++) {
        int num_kernels = kernel_counts[i];

        // 重置数据
        for (int j = 0; j < n; j++) h_data[j] = (float)j;

        // 无Graph
        cudaStream_t stream1;
        CHECK_CUDA(cudaStreamCreate(&stream1));
        size_t size = n * sizeof(float);
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;

        cudaEvent_t start1, stop1;
        CHECK_CUDA(cudaEventCreate(&start1));
        CHECK_CUDA(cudaEventCreate(&stop1));
        CHECK_CUDA(cudaEventRecord(start1, stream1));

        for (int iter = 0; iter < iterations; iter++) {
            CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1));
            for (int k = 0; k < num_kernels; k++) {
                kernel_op1<<<numBlocks, blockSize, 0, stream1>>>(d_data, n);
            }
            CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream1));
            CHECK_CUDA(cudaStreamSynchronize(stream1));
        }

        CHECK_CUDA(cudaEventRecord(stop1, stream1));
        CHECK_CUDA(cudaEventSynchronize(stop1));
        float ms_no_graph;
        CHECK_CUDA(cudaEventElapsedTime(&ms_no_graph, start1, stop1));

        CHECK_CUDA(cudaStreamDestroy(stream1));
        CHECK_CUDA(cudaEventDestroy(start1));
        CHECK_CUDA(cudaEventDestroy(stop1));

        // 有Graph
        for (int j = 0; j < n; j++) h_data[j] = (float)j;

        cudaStream_t stream2;
        CHECK_CUDA(cudaStreamCreate(&stream2));

        cudaGraph_t graph;
        CHECK_CUDA(cudaStreamBeginCapture(stream2, cudaStreamCaptureModeGlobal));
        CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream2));
        for (int k = 0; k < num_kernels; k++) {
            kernel_op1<<<numBlocks, blockSize, 0, stream2>>>(d_data, n);
        }
        CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2));
        CHECK_CUDA(cudaStreamEndCapture(stream2, &graph));

        cudaGraphExec_t graphExec;
        CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

        cudaEvent_t start2, stop2;
        CHECK_CUDA(cudaEventCreate(&start2));
        CHECK_CUDA(cudaEventCreate(&stop2));
        CHECK_CUDA(cudaEventRecord(start2, stream2));

        for (int iter = 0; iter < iterations; iter++) {
            CHECK_CUDA(cudaGraphLaunch(graphExec, stream2));
            CHECK_CUDA(cudaStreamSynchronize(stream2));
        }

        CHECK_CUDA(cudaEventRecord(stop2, stream2));
        CHECK_CUDA(cudaEventSynchronize(stop2));
        float ms_with_graph;
        CHECK_CUDA(cudaEventElapsedTime(&ms_with_graph, start2, stop2));

        float improvement = (ms_no_graph - ms_with_graph) / ms_no_graph * 100;

        printf("%-15d %-15.3f %-15.3f %-15.1f\n", num_kernels, ms_no_graph, ms_with_graph, improvement);

        CHECK_CUDA(cudaGraphExecDestroy(graphExec));
        CHECK_CUDA(cudaGraphDestroy(graph));
        CHECK_CUDA(cudaStreamDestroy(stream2));
        CHECK_CUDA(cudaEventDestroy(start2));
        CHECK_CUDA(cudaEventDestroy(stop2));
    }
}

// 并行核函数优化
void parallel_kernels_optimization(float* d_data, float* h_data, int n) {
    printf("\n=== 并行核函数优化 ===\n");

    size_t size = n * sizeof(float);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 方式1: 串行执行
    printf("方式1: 串行执行核函数\n");
    cudaStream_t stream1;
    CHECK_CUDA(cudaStreamCreate(&stream1));

    cudaGraph_t graph1;
    CHECK_CUDA(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));
    kernel_op1<<<numBlocks, blockSize, 0, stream1>>>(d_data, n);
    kernel_op2<<<numBlocks, blockSize, 0, stream1>>>(d_data, n);
    kernel_op3<<<numBlocks, blockSize, 0, stream1>>>(d_data, n);
    CHECK_CUDA(cudaStreamEndCapture(stream1, &graph1));

    size_t numNodes1;
    CHECK_CUDA(cudaGraphGetNodes(graph1, NULL, &numNodes1));
    printf("  节点数: %zu (串行依赖)\n", numNodes1);

    CHECK_CUDA(cudaGraphDestroy(graph1));
    CHECK_CUDA(cudaStreamDestroy(stream1));

    // 方式2: 并行执行（使用显式图创建API）
    printf("方式2: 并行执行核函数 (显式图API)\n");

    cudaGraph_t graph2;
    CHECK_CUDA(cudaGraphCreate(&graph2, 0));

    // 创建参数
    void* kernelArgs[] = {&d_data, &n};

    // 创建内核节点（可以并行执行）
    cudaGraphNode_t node1, node2, node3;
    cudaKernelNodeParams nodeParams = {0};
    nodeParams.func = (void*)kernel_op1;
    nodeParams.gridDim = dim3(numBlocks);
    nodeParams.blockDim = dim3(blockSize);
    nodeParams.sharedMemBytes = 0;
    nodeParams.kernelParams = (void**)kernelArgs;
    nodeParams.extra = NULL;

    CHECK_CUDA(cudaGraphAddKernelNode(&node1, graph2, NULL, 0, &nodeParams));

    nodeParams.func = (void*)kernel_op2;
    CHECK_CUDA(cudaGraphAddKernelNode(&node2, graph2, NULL, 0, &nodeParams));

    nodeParams.func = (void*)kernel_op3;
    CHECK_CUDA(cudaGraphAddKernelNode(&node3, graph2, NULL, 0, &nodeParams));

    size_t numNodes2;
    CHECK_CUDA(cudaGraphGetNodes(graph2, NULL, &numNodes2));
    printf("  节点数: %zu (可并行执行)\n", numNodes2);
    printf("  注: 三个内核节点无依赖关系，可并行执行\n");

    CHECK_CUDA(cudaGraphDestroy(graph2));
}

// 最佳实践总结
void best_practices() {
    printf("\n=== CUDA Graph最佳实践 ===\n");

    printf("\n适用场景:\n");
    printf("  1. 重复执行的相同操作序列\n");
    printf("  2. 多个小核函数（启动开销占主导）\n");
    printf("  3. 深度学习推理场景\n");
    printf("  4. 需要减少CPU开销的场景\n");

    printf("\n不适合场景:\n");
    printf("  1. 操作序列每次不同\n");
    printf("  2. 单次执行\n");
    printf("  3. 需要频繁重建图\n");
    printf("  4. 大核函数（执行时间远大于启动开销）\n");

    printf("\n性能优化建议:\n");
    printf("  1. 图创建和实例化只做一次\n");
    printf("  2. 多次复用同一个图实例\n");
    printf("  3. 使用参数更新适应变化\n");
    printf("  4. 利用图结构实现并行执行\n");
    printf("  5. 核函数数量越多，收益越大\n");
}

int main() {
    printf("========================================\n");
    printf("  图优化示例 - 第二十四章\n");
    printf("========================================\n");

    // 获取设备信息
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("\n设备: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // 准备数据
    int n = 1024 * 256;  // 256K 元素
    size_t size = n * sizeof(float);

    float *h_data, *d_data;
    CHECK_CUDA(cudaMallocHost(&h_data, size));
    CHECK_CUDA(cudaMalloc(&d_data, size));

    // 初始化
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
    }

    // 性能对比
    no_graph_execution(d_data, h_data, n, 10, 100);
    with_graph_execution(d_data, h_data, n, 10, 100);

    // 详细对比
    performance_comparison(d_data, h_data, n);

    // 并行优化
    parallel_kernels_optimization(d_data, h_data, n);

    // 最佳实践
    best_practices();

    // 清理
    CHECK_CUDA(cudaFreeHost(h_data));
    CHECK_CUDA(cudaFree(d_data));

    printf("\n========================================\n");
    printf("关键要点:\n");
    printf("  1. CUDA Graph减少核函数启动开销\n");
    printf("  2. 核函数数量越多，优化效果越明显\n");
    printf("  3. 图结构可以实现并行执行\n");
    printf("  4. 适合深度学习等重复执行场景\n");
    printf("========================================\n\n");

    return 0;
}