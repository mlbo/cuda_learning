/**
 * 第二十三章示例：异步传输
 *
 * 本示例演示异步数据传输的使用和性能优化
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

// 简单向量运算核函数
__global__ void vector_mult(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// 同步传输测试
void sync_transfer_test(int n) {
    printf("\n=== 同步传输测试 ===\n");

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    // 分配锁页内存（即使是同步传输也用锁页内存以获得更好性能）
    CHECK_CUDA(cudaMallocHost(&h_a, size));
    CHECK_CUDA(cudaMallocHost(&h_b, size));
    CHECK_CUDA(cudaMallocHost(&h_c, size));

    // 初始化
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 0.001f;
        h_b[i] = i * 0.002f;
    }

    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // 同步传输 - 阻塞CPU
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_mult<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("同步传输时间: %.3f ms\n", ms);

    // 清理
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 异步传输测试
void async_transfer_test(int n) {
    printf("\n=== 异步传输测试 ===\n");

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    // 分配锁页内存（异步传输必须使用锁页内存）
    CHECK_CUDA(cudaMallocHost(&h_a, size));
    CHECK_CUDA(cudaMallocHost(&h_b, size));
    CHECK_CUDA(cudaMallocHost(&h_c, size));

    // 初始化
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 0.001f;
        h_b[i] = i * 0.002f;
    }

    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    // 创建CUDA流
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));

    // 异步传输 - 非阻塞，立即返回
    CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream));

    // 核函数在同一流中执行，会等待前面的传输完成
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_mult<<<numBlocks, blockSize, 0, stream>>>(d_a, d_b, d_c, n);

    // 异步传输结果回主机
    CHECK_CUDA(cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("异步传输时间: %.3f ms\n", ms);

    // CPU可以在GPU执行时做其他工作
    printf("GPU执行期间CPU可以做其他工作\n");

    // 清理
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 多流异步传输测试
void multi_stream_async_test(int n, int n_streams) {
    printf("\n=== 多流异步传输测试 (%d 流) ===\n", n_streams);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);
    size_t chunk_size = size / n_streams;

    // 分配锁页内存
    CHECK_CUDA(cudaMallocHost(&h_a, size));
    CHECK_CUDA(cudaMallocHost(&h_b, size));
    CHECK_CUDA(cudaMallocHost(&h_c, size));

    // 初始化
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 0.001f;
        h_b[i] = i * 0.002f;
    }

    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    // 创建多个流
    cudaStream_t* streams = new cudaStream_t[n_streams];
    for (int i = 0; i < n_streams; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, streams[0]));

    // 分发到多个流
    int chunk_elements = n / n_streams;
    int blockSize = 256;
    int numBlocks = (chunk_elements + blockSize - 1) / blockSize;

    for (int i = 0; i < n_streams; i++) {
        size_t offset = i * chunk_size;
        int element_offset = i * chunk_elements;

        // H2D传输
        CHECK_CUDA(cudaMemcpyAsync(d_a + element_offset, h_a + element_offset,
                                   chunk_size, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_b + element_offset, h_b + element_offset,
                                   chunk_size, cudaMemcpyHostToDevice, streams[i]));

        // 核函数执行
        vector_mult<<<numBlocks, blockSize, 0, streams[i]>>>(
            d_a + element_offset, d_b + element_offset, d_c + element_offset, chunk_elements);

        // D2H传输
        CHECK_CUDA(cudaMemcpyAsync(h_c + element_offset, d_c + element_offset,
                                   chunk_size, cudaMemcpyDeviceToHost, streams[i]));
    }

    // 同步所有流
    for (int i = 0; i < n_streams; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaEventRecord(stop, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("多流异步传输时间: %.3f ms\n", ms);

    // 验证结果
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (std::fabs(h_c[i] - h_a[i] * h_b[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("结果验证: %s\n", correct ? "通过" : "失败");

    // 清理
    for (int i = 0; i < n_streams; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 异步内存分配测试 (CUDA 11.2+)
void async_malloc_test(int n) {
    printf("\n=== 异步内存分配测试 ===\n");

    float *d_a, *d_b, *d_c;

    // 创建流
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));

    // 异步分配设备内存
    size_t size = n * sizeof(float);
    CHECK_CUDA(cudaMallocAsync(&d_a, size, stream));
    CHECK_CUDA(cudaMallocAsync(&d_b, size, stream));
    CHECK_CUDA(cudaMallocAsync(&d_c, size, stream));

    // 注意：异步分配后需要初始化数据
    // 这里仅演示分配功能

    // 异步释放
    CHECK_CUDA(cudaFreeAsync(d_a, stream));
    CHECK_CUDA(cudaFreeAsync(d_b, stream));
    CHECK_CUDA(cudaFreeAsync(d_c, stream));

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("异步分配和释放时间: %.3f ms\n", ms);

    printf("异步内存分配优势:\n");
    printf("  - 在流中执行，不阻塞CPU\n");
    printf("  - 可以与其他操作重叠\n");
    printf("  - 适合频繁分配释放的场景\n");

    // 清理
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    printf("========================================\n");
    printf("  异步传输示例 - 第二十三章\n");
    printf("========================================\n");

    // 获取设备信息
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("\n设备: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    int n = 1024 * 1024 * 16;  // 16M 元素

    sync_transfer_test(n);
    async_transfer_test(n);
    multi_stream_async_test(n, 4);
    async_malloc_test(n);

    printf("\n========================================\n");
    printf("关键要点:\n");
    printf("  1. cudaMemcpyAsync需要锁页内存\n");
    printf("  2. 异步传输在流中执行\n");
    printf("  3. 多流可以实现传输与计算重叠\n");
    printf("  4. cudaMallocAsync支持异步内存分配\n");
    printf("========================================\n\n");

    return 0;
}
