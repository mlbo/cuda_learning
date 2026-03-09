/**
 * 第二十三章示例：锁页内存
 *
 * 本示例演示锁页内存(Pinned Memory)的使用和性能优势
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

// 简单向量乘法核函数
__global__ void vector_mult(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// 使用分页内存（普通malloc）
void pageable_memory_test(int n) {
    printf("\n=== 分页内存测试 ===\n");

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    // 使用普通malloc分配分页内存
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // 初始化
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 0.001f;
        h_b[i] = i * 0.002f;
    }

    // 分配设备内存
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    // 创建事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 记录开始时间
    CHECK_CUDA(cudaEventRecord(start));

    // 同步传输H2D
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 执行核函数
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_mult<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // 同步传输D2H
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // 记录结束时间
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("数据量: %d 元素 (%.2f MB)\n", n, (float)size / 1024 / 1024);
    printf("总执行时间: %.3f ms\n", ms);
    printf("验证结果: c[0] = %.3f (期望 %.3f)\n", h_c[0], h_a[0] * h_b[0]);

    // 清理
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 使用锁页内存
void pinned_memory_test(int n) {
    printf("\n=== 锁页内存测试 ===\n");

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    // 使用cudaMallocHost分配锁页内存
    CHECK_CUDA(cudaMallocHost(&h_a, size));
    CHECK_CUDA(cudaMallocHost(&h_b, size));
    CHECK_CUDA(cudaMallocHost(&h_c, size));

    // 初始化
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 0.001f;
        h_b[i] = i * 0.002f;
    }

    // 分配设备内存
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    // 创建事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 记录开始时间
    CHECK_CUDA(cudaEventRecord(start));

    // 同步传输H2D（锁页内存传输更快）
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 执行核函数
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_mult<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // 同步传输D2H
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // 记录结束时间
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("数据量: %d 元素 (%.2f MB)\n", n, (float)size / 1024 / 1024);
    printf("总执行时间: %.3f ms\n", ms);
    printf("验证结果: c[0] = %.3f (期望 %.3f)\n", h_c[0], h_a[0] * h_b[0]);

    // 清理 - 注意使用cudaFreeHost而不是free
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 使用cudaHostAlloc的不同标志
void host_alloc_flags_test() {
    printf("\n=== cudaHostAlloc标志测试 ===\n");

    size_t size = 1024 * 1024 * sizeof(float);  // 4MB

    // 1. 默认标志
    float *h_default;
    CHECK_CUDA(cudaHostAlloc(&h_default, size, cudaHostAllocDefault));
    printf("cudaHostAllocDefault: 成功分配 %zu 字节\n", size);
    CHECK_CUDA(cudaFreeHost(h_default));

    // 2. 可移植标志
    float *h_portable;
    CHECK_CUDA(cudaHostAlloc(&h_portable, size, cudaHostAllocPortable));
    printf("cudaHostAllocPortable: 可跨上下文使用\n");
    CHECK_CUDA(cudaFreeHost(h_portable));

    // 3. 映射标志（用于零拷贝）
    float *h_mapped;
    CHECK_CUDA(cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped));
    printf("cudaHostAllocMapped: 可映射到GPU地址空间\n");

    // 获取设备端指针
    float *d_ptr;
    CHECK_CUDA(cudaHostGetDevicePointer(&d_ptr, h_mapped, 0));
    printf("  设备端指针: %p\n", d_ptr);

    CHECK_CUDA(cudaFreeHost(h_mapped));

    // 4. 写组合标志
    float *h_writecombined;
    CHECK_CUDA(cudaHostAlloc(&h_writecombined, size, cudaHostAllocWriteCombined));
    printf("cudaHostAllocWriteCombined: 优化PCIe写性能\n");
    CHECK_CUDA(cudaFreeHost(h_writecombined));
}

// 锁页内存限制说明
void pinned_memory_limits() {
    printf("\n=== 锁页内存限制说明 ===\n");

    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));

    printf("GPU内存: %.2f GB 总计, %.2f GB 可用\n",
           (float)total_mem / 1e9, (float)free_mem / 1e9);

    printf("\n锁页内存限制:\n");
    printf("  - 锁页内存占用物理内存，不可换出到磁盘\n");
    printf("  - 过度使用可能导致系统性能下降\n");
    printf("  - 分配速度较慢（需要操作系统锁定页面）\n");
    printf("  - 数量受系统物理内存限制\n");

    printf("\n适用场景:\n");
    printf("  - 频繁的大数据传输\n");
    printf("  - 需要异步传输\n");
    printf("  - 零拷贝访问\n");
}

int main() {
    printf("========================================\n");
    printf("  锁页内存示例 - 第二十三章\n");
    printf("========================================\n");

    // 获取设备信息
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("\n设备: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // 测试不同内存类型
    int n = 1024 * 1024 * 32;  // 32M 元素

    pageable_memory_test(n);
    pinned_memory_test(n);
    host_alloc_flags_test();
    pinned_memory_limits();

    printf("\n========================================\n");
    printf("关键要点:\n");
    printf("  1. cudaMallocHost分配锁页内存\n");
    printf("  2. 锁页内存支持更快的DMA传输\n");
    printf("  3. 必须用cudaFreeHost释放锁页内存\n");
    printf("  4. 锁页内存是实现异步传输的前提\n");
    printf("========================================\n\n");

    return 0;
}