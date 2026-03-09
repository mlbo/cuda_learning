/**
 * 第二十二章示例：流同步机制
 *
 * 本示例演示CUDA流的各种同步方式
 * 包括设备同步、流同步、事件同步等
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

// 模拟计算密集型核函数
__global__ void compute_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.0001f + 0.0001f;
        }
        data[idx] = val;
    }
}

// 1. cudaDeviceSynchronize() - 设备同步
void device_sync_example() {
    printf("\n=== 1. cudaDeviceSynchronize() ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 0, n * sizeof(float)));

    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 在多个流中执行核函数
    compute_kernel<<<numBlocks, blockSize, 0, stream1>>>(d_data, n, 1000);
    compute_kernel<<<numBlocks, blockSize, 0, stream2>>>(d_data, n, 1000);

    printf("  等待GPU上所有操作完成...\n");

    // 设备同步：等待所有流完成
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("  所有操作已完成\n");

    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaFree(d_data));
}

// 2. cudaStreamSynchronize() - 流同步
void stream_sync_example() {
    printf("\n=== 2. cudaStreamSynchronize() ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t streams[3];
    for (int i = 0; i < 3; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 在不同流中执行核函数
    compute_kernel<<<numBlocks, blockSize, 0, streams[0]>>>(d_data, n, 500);
    compute_kernel<<<numBlocks, blockSize, 0, streams[1]>>>(d_data, n, 1000);
    compute_kernel<<<numBlocks, blockSize, 0, streams[2]>>>(d_data, n, 1500);

    // 分别同步每个流
    for (int i = 0; i < 3; i++) {
        printf("  等待 Stream %d 完成...\n", i);
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        printf("  Stream %d 已完成\n", i);
    }

    for (int i = 0; i < 3; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaFree(d_data));
}

// 3. cudaStreamQuery() - 非阻塞查询
void stream_query_example() {
    printf("\n=== 3. cudaStreamQuery() ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    compute_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 5000);

    printf("  使用非阻塞查询检查流状态...\n");

    int poll_count = 0;
    cudaError_t result;

    while ((result = cudaStreamQuery(stream)) == cudaErrorNotReady) {
        poll_count++;
        if (poll_count % 100000 == 0) {
            printf("  轮询中... (count = %d)\n", poll_count);
        }
    }

    if (result == cudaSuccess) {
        printf("  流已完成，轮询了 %d 次\n", poll_count);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_data));
}

// 4. cudaStreamAddCallback() - 流回调
void CUDART_CB stream_callback(cudaStream_t stream, cudaError_t status, void *userdata) {
    int *counter = (int*)userdata;
    printf("  [回调] Stream %p 完成，状态: %s\n",
           stream, cudaGetErrorString(status));
    (*counter)++;
}

void stream_callback_example() {
    printf("\n=== 4. cudaStreamAddCallback() ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    int callback_counter = 0;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 第一个核函数
    compute_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 500);

    // 添加回调
    CHECK_CUDA(cudaStreamAddCallback(stream, stream_callback, &callback_counter, 0));

    // 第二个核函数
    compute_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 500);

    // 添加另一个回调
    CHECK_CUDA(cudaStreamAddCallback(stream, stream_callback, &callback_counter, 0));

    printf("  等待流完成...\n");
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("  回调执行次数: %d\n", callback_counter);

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_data));
}

// 5. 流间同步（使用空流）
void implicit_sync_example() {
    printf("\n=== 5. 流间隐式同步 ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 在命名流中执行
    CHECK_CUDA(cudaEventRecord(start));
    compute_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 1000);

    // 在默认流中执行 - 会隐式同步
    compute_kernel<<<numBlocks, blockSize>>>(d_data, n, 1000);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("  命名流 + 默认流 执行时间: %.3f ms\n", ms);
    printf("  注意: 默认流会隐式同步命名流\n");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
}

// 6. cudaStreamWaitEvent() - 显式流间同步
void explicit_sync_example() {
    printf("\n=== 6. cudaStreamWaitEvent() 显式同步 ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    cudaEvent_t event;
    CHECK_CUDA(cudaEventCreate(&event));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Stream 1 执行第一阶段
    CHECK_CUDA(cudaEventRecord(start, stream1));
    compute_kernel<<<numBlocks, blockSize, 0, stream1>>>(d_data, n, 1000);

    // 记录事件，表示 Stream 1 完成了第一阶段
    CHECK_CUDA(cudaEventRecord(event, stream1));

    // Stream 2 等待事件
    CHECK_CUDA(cudaStreamWaitEvent(stream2, event, 0));

    // Stream 2 执行（确保在 Stream 1 之后）
    compute_kernel<<<numBlocks, blockSize, 0, stream2>>>(d_data, n, 1000);

    CHECK_CUDA(cudaEventRecord(stop, stream2));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("  显式同步执行时间: %.3f ms\n", ms);
    printf("  Stream 2 正确等待 Stream 1 完成\n");

    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaEventDestroy(event));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
}

// 同步方式性能对比
void sync_performance_comparison() {
    printf("\n=== 同步方式性能对比 ===\n");

    const int n = 4 * 1024 * 1024;
    const int iterations = 10;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 测试 cudaDeviceSynchronize
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        compute_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 100);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float device_sync_time;
    CHECK_CUDA(cudaEventElapsedTime(&device_sync_time, start, stop));

    // 测试 cudaStreamSynchronize
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        compute_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 100);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float stream_sync_time;
    CHECK_CUDA(cudaEventElapsedTime(&stream_sync_time, start, stop));

    printf("  cudaDeviceSynchronize:  %.3f ms\n", device_sync_time / iterations);
    printf("  cudaStreamSynchronize:  %.3f ms\n", stream_sync_time / iterations);
    printf("  流同步更快: %.2fx\n", device_sync_time / stream_sync_time);

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
}

int main() {
    printf("========================================\n");
    printf("  流同步机制演示 - 第二十二章\n");
    printf("========================================\n");

    device_sync_example();
    stream_sync_example();
    stream_query_example();
    stream_callback_example();
    implicit_sync_example();
    explicit_sync_example();
    sync_performance_comparison();

    printf("\n========================================\n");
    printf("同步方式总结:\n");
    printf("  1. cudaDeviceSynchronize: 等待所有操作\n");
    printf("  2. cudaStreamSynchronize: 等待特定流\n");
    printf("  3. cudaStreamQuery: 非阻塞查询\n");
    printf("  4. cudaStreamAddCallback: 流中回调\n");
    printf("  5. cudaStreamWaitEvent: 流等待事件\n");
    printf("========================================\n\n");

    return 0;
}