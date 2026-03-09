/**
 * 第二十二章示例：多流并发执行
 *
 * 本示例演示如何使用多个CUDA流实现并发执行
 * 包括数据传输与计算的overlap
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

const int N_STREAMS = 4;

// 简单的向量操作核函数
__global__ void vector_op(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * scale + 1.0f;
    }
}

// 使用单流（同步版本）
float single_stream_version(float* h_data, float* d_data, int total_size, int iterations) {
    int chunk_size = total_size;
    size_t chunk_bytes = chunk_size * sizeof(float);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int iter = 0; iter < iterations; iter++) {
        // H2D
        CHECK_CUDA(cudaMemcpy(d_data, h_data, chunk_bytes, cudaMemcpyHostToDevice));

        // Kernel
        int blockSize = 256;
        int numBlocks = (chunk_size + blockSize - 1) / blockSize;
        vector_op<<<numBlocks, blockSize>>>(d_data, chunk_size, 2.0f);

        // D2H
        CHECK_CUDA(cudaMemcpy(h_data, d_data, chunk_bytes, cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

// 使用多流（异步版本）
float multi_stream_version(float* h_data, float* d_data, int total_size, int iterations) {
    int chunk_size = total_size / N_STREAMS;
    size_t chunk_bytes = chunk_size * sizeof(float);

    // 创建流
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, streams[0]));

    int blockSize = 256;
    int numBlocks = (chunk_size + blockSize - 1) / blockSize;

    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < N_STREAMS; i++) {
            int offset = i * chunk_size;

            // H2D (需要锁页内存)
            CHECK_CUDA(cudaMemcpyAsync(d_data + offset, h_data + offset,
                                       chunk_bytes, cudaMemcpyHostToDevice, streams[i]));

            // Kernel
            vector_op<<<numBlocks, blockSize, 0, streams[i]>>>(d_data + offset, chunk_size, 2.0f);

            // D2H
            CHECK_CUDA(cudaMemcpyAsync(h_data + offset, d_data + offset,
                                       chunk_bytes, cudaMemcpyDeviceToHost, streams[i]));
        }
    }

    // 同步所有流
    for (int i = 0; i < N_STREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaEventRecord(stop, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 销毁流
    for (int i = 0; i < N_STREAMS; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

// 使用锁页内存的多流版本
float pinned_multi_stream_version(float* h_data_pinned, float* d_data, int total_size, int iterations) {
    int chunk_size = total_size / N_STREAMS;
    size_t chunk_bytes = chunk_size * sizeof(float);

    // 创建非阻塞流
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, streams[0]));

    int blockSize = 256;
    int numBlocks = (chunk_size + blockSize - 1) / blockSize;

    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < N_STREAMS; i++) {
            int offset = i * chunk_size;

            // H2D (锁页内存 + 异步传输)
            CHECK_CUDA(cudaMemcpyAsync(d_data + offset, h_data_pinned + offset,
                                       chunk_bytes, cudaMemcpyHostToDevice, streams[i]));

            // Kernel
            vector_op<<<numBlocks, blockSize, 0, streams[i]>>>(d_data + offset, chunk_size, 2.0f);

            // D2H
            CHECK_CUDA(cudaMemcpyAsync(h_data_pinned + offset, d_data + offset,
                                       chunk_bytes, cudaMemcpyDeviceToHost, streams[i]));
        }
    }

    for (int i = 0; i < N_STREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaEventRecord(stop, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    for (int i = 0; i < N_STREAMS; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

// 演示默认流与命名流的同步行为
void demonstrate_sync_behavior() {
    printf("\n=== 默认流与命名流同步行为 ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 测试1: 命名流 + 命名流（可并发）
    printf("\n测试1: 命名流 + 命名流\n");
    CHECK_CUDA(cudaEventRecord(start));
    vector_op<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 1.0f);
    vector_op<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 2.0f);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("  执行时间: %.3f ms\n", ms);

    // 测试2: 默认流 + 命名流（隐式同步）
    printf("\n测试2: 默认流 + 命名流（会隐式同步）\n");
    CHECK_CUDA(cudaEventRecord(start));
    vector_op<<<numBlocks, blockSize, 0, 0>>>(d_data, n, 1.0f);  // 默认流
    vector_op<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 2.0f);  // 命名流
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("  执行时间: %.3f ms（比预期长，因为隐式同步）\n", ms);

    // 测试3: 非阻塞流
    printf("\n测试3: 非阻塞流 + 默认流\n");
    cudaStream_t non_block_stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&non_block_stream, cudaStreamNonBlocking));

    CHECK_CUDA(cudaEventRecord(start));
    vector_op<<<numBlocks, blockSize, 0, 0>>>(d_data, n, 1.0f);  // 默认流
    vector_op<<<numBlocks, blockSize, 0, non_block_stream>>>(d_data, n, 2.0f);  // 非阻塞流
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("  执行时间: %.3f ms（可以并发）\n", ms);

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaStreamDestroy(non_block_stream));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 性能对比测试
void performance_comparison(int total_size) {
    printf("\n=== 性能对比 (数据量: %d 元素) ===\n", total_size);

    // 分配分页内存
    float *h_data = (float*)malloc(total_size * sizeof(float));
    for (int i = 0; i < total_size; i++) {
        h_data[i] = (float)i;
    }

    // 分配锁页内存
    float *h_data_pinned;
    CHECK_CUDA(cudaMallocHost(&h_data_pinned, total_size * sizeof(float)));
    memcpy(h_data_pinned, h_data, total_size * sizeof(float));

    // 分配设备内存
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, total_size * sizeof(float)));

    int iterations = 10;
    int warmup = 3;

    // 预热
    single_stream_version(h_data, d_data, total_size, warmup);
    multi_stream_version(h_data, d_data, total_size, warmup);
    pinned_multi_stream_version(h_data_pinned, d_data, total_size, warmup);

    // 正式测试
    float single_time = single_stream_version(h_data, d_data, total_size, iterations);
    float multi_time = multi_stream_version(h_data, d_data, total_size, iterations);
    float pinned_time = pinned_multi_stream_version(h_data_pinned, d_data, total_size, iterations);

    printf("\n结果:\n");
    printf("  单流版本:        %.3f ms (基准)\n", single_time / iterations);
    printf("  多流版本:        %.3f ms (%.2fx)\n", multi_time / iterations, single_time / multi_time);
    printf("  多流+锁页内存:   %.3f ms (%.2fx)\n", pinned_time / iterations, single_time / pinned_time);

    // 清理
    free(h_data);
    CHECK_CUDA(cudaFreeHost(h_data_pinned));
    CHECK_CUDA(cudaFree(d_data));
}

int main() {
    printf("========================================\n");
    printf("  多流并发执行演示 - 第二十二章\n");
    printf("========================================\n");

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("\n设备: %s\n", prop.name);
    printf("并发核函数: %s\n", prop.concurrentKernels ? "支持" : "不支持");

    // 演示同步行为
    demonstrate_sync_behavior();

    // 性能对比
    performance_comparison(1024 * 1024);
    performance_comparison(4 * 1024 * 1024);

    printf("\n========================================\n");
    printf("关键要点:\n");
    printf("  1. 多流可以实现传输与计算的重叠\n");
    printf("  2. 锁页内存是异步传输的前提\n");
    printf("  3. 非阻塞流可以与默认流并发\n");
    printf("  4. 使用 nsys 可视化流并发\n");
    printf("========================================\n\n");

    printf("使用 nsys 分析:\n");
    printf("  nsys profile --stats=true -o multi_stream ./02_multi_stream\n");
    printf("  nsys-ui multi_stream.nsys-rep\n\n");

    return 0;
}