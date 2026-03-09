/**
 * 第二十二章示例：CUDA流基础
 *
 * 本示例演示CUDA流的基本概念和操作
 * 包括默认流、命名流、非阻塞流的使用
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

// 简单的向量加法核函数
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 使用默认流
void default_stream_example(int n) {
    printf("\n=== 默认流示例 ===\n");

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // 分配主机内存
    h_a = (float*)malloc(n * sizeof(float));
    h_b = (float*)malloc(n * sizeof(float));
    h_c = (float*)malloc(n * sizeof(float));

    // 初始化
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 分配设备内存
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, n * sizeof(float)));

    // 创建事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 记录开始时间
    CHECK_CUDA(cudaEventRecord(start));

    // 使用默认流（不指定流参数或传入0/NULL）
    CHECK_CUDA(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    CHECK_CUDA(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // 记录结束时间
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("数据量: %d 元素\n", n);
    printf("执行时间: %.3f ms\n", ms);
    printf("验证结果: %s\n", (h_c[0] == h_a[0] + h_b[0]) ? "通过" : "失败");

    // 清理
    free(h_a); free(h_b); free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 使用命名流
void named_stream_example(int n) {
    printf("\n=== 命名流示例 ===\n");

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = (float*)malloc(n * sizeof(float));
    h_b = (float*)malloc(n * sizeof(float));
    h_c = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, n * sizeof(float)));

    // 创建命名流
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));

    // 使用命名流（需要锁页内存才能使用异步传输）
    // 这里演示同步传输
    CHECK_CUDA(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_add<<<numBlocks, blockSize, 0, stream>>>(d_a, d_b, d_c, n);

    CHECK_CUDA(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("命名流执行时间: %.3f ms\n", ms);
    printf("验证结果: %s\n", (h_c[0] == h_a[0] + h_b[0]) ? "通过" : "失败");

    // 销毁流
    CHECK_CUDA(cudaStreamDestroy(stream));

    free(h_a); free(h_b); free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 非阻塞流示例
void non_blocking_stream_example() {
    printf("\n=== 非阻塞流示例 ===\n");

    // 普通命名流 vs 非阻塞流
    cudaStream_t normal_stream, non_blocking_stream;

    // 创建普通命名流
    CHECK_CUDA(cudaStreamCreate(&normal_stream));

    // 创建非阻塞流
    CHECK_CUDA(cudaStreamCreateWithFlags(&non_blocking_stream, cudaStreamNonBlocking));

    printf("已创建两个流:\n");
    printf("  1. 普通命名流: 会与默认流隐式同步\n");
    printf("  2. 非阻塞流: 不会与默认流同步\n");

    // 查询流属性
    // 注意: CUDA没有直接查询流是否为非阻塞的API
    // 需要通过创建时记录或在nsys中观察

    // 简单的核函数调用
    int n = 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 在默认流中执行
    vector_add<<<numBlocks, blockSize, 0, 0>>>(d_data, d_data, d_data, n);

    // 在普通命名流中执行
    // 这会等待默认流完成
    vector_add<<<numBlocks, blockSize, 0, normal_stream>>>(d_data, d_data, d_data, n);

    // 在非阻塞流中执行
    // 这可以与默认流并发（如果资源允许）
    vector_add<<<numBlocks, blockSize, 0, non_blocking_stream>>>(d_data, d_data, d_data, n);

    CHECK_CUDA(cudaDeviceSynchronize());

    printf("\n使用提示:\n");
    printf("  - 使用 nsys 可以观察到流的并发执行情况\n");
    printf("  - 非阻塞流适合需要与默认流并发的场景\n");

    CHECK_CUDA(cudaStreamDestroy(normal_stream));
    CHECK_CUDA(cudaStreamDestroy(non_blocking_stream));
    CHECK_CUDA(cudaFree(d_data));
}

// 流查询示例
void stream_query_example() {
    printf("\n=== 流查询示例 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 发射核函数
    vector_add<<<numBlocks, blockSize, 0, stream>>>(d_data, d_data, d_data, n);

    // 非阻塞查询流状态
    cudaError_t result = cudaStreamQuery(stream);
    if (result == cudaSuccess) {
        printf("流已完成\n");
    } else if (result == cudaErrorNotReady) {
        printf("流尚未完成，继续其他工作...\n");

        // 可以在这里做其他CPU工作
        int count = 0;
        while ((result = cudaStreamQuery(stream)) == cudaErrorNotReady) {
            count++;
            if (count % 100000 == 0) {
                printf("  等待中... (count = %d)\n", count);
            }
        }
        printf("流已完成 (等待了 %d 次查询)\n", count);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_data));
}

// 流属性查询
void stream_attributes_example() {
    printf("\n=== 流属性示例 ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 查询流优先级范围
    int least_priority, greatest_priority;
    CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));

    printf("流优先级范围: [%d, %d]\n", least_priority, greatest_priority);
    printf("  - %d: 最低优先级\n", least_priority);
    printf("  - %d: 最高优先级\n", greatest_priority);

    // 创建带优先级的流
    cudaStream_t high_priority_stream;
    CHECK_CUDA(cudaStreamCreateWithPriority(&high_priority_stream,
                                            cudaStreamNonBlocking,
                                            greatest_priority));

    printf("\n已创建高优先级非阻塞流\n");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaStreamDestroy(high_priority_stream));
}

int main() {
    printf("========================================\n");
    printf("  CUDA流基础演示 - 第二十二章\n");
    printf("========================================\n");

    // 获取设备信息
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("\n设备: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("并发核函数: %s\n", prop.concurrentKernels ? "支持" : "不支持");

    // 运行示例
    default_stream_example(1024 * 1024);
    named_stream_example(1024 * 1024);
    non_blocking_stream_example();
    stream_query_example();
    stream_attributes_example();

    printf("\n========================================\n");
    printf("关键要点:\n");
    printf("  1. 默认流会与其他流隐式同步\n");
    printf("  2. 使用命名流可以实现并发执行\n");
    printf("  3. cudaStreamNonBlocking创建非阻塞流\n");
    printf("  4. cudaStreamQuery可非阻塞检查状态\n");
    printf("========================================\n\n");

    return 0;
}