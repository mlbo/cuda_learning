/**
 * 第二十三章示例：统一内存
 *
 * 本示例演示统一内存(Unified Memory/Managed Memory)的使用
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
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_mult(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// 统一内存基础示例
void unified_memory_basics(int n) {
    printf("\n=== 统一内存基础示例 ===\n");

    float *a, *b, *c;
    size_t size = n * sizeof(float);

    // 分配统一内存
    CHECK_CUDA(cudaMallocManaged(&a, size));
    CHECK_CUDA(cudaMallocManaged(&b, size));
    CHECK_CUDA(cudaMallocManaged(&c, size));

    // CPU端初始化
    printf("CPU初始化数据...\n");
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // GPU端计算
    printf("GPU计算...\n");
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_add<<<numBlocks, blockSize>>>(a, b, c, n);

    // 等待GPU完成
    CHECK_CUDA(cudaDeviceSynchronize());

    // CPU端读取结果
    printf("CPU读取结果...\n");
    printf("c[0] = %.1f (期望 %.1f)\n", c[0], a[0] + b[0]);
    printf("c[100] = %.1f (期望 %.1f)\n", c[100], a[100] + b[100]);

    // 清理
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));

    printf("统一内存特点:\n");
    printf("  - CPU和GPU共享同一内存\n");
    printf("  - CUDA自动管理数据迁移\n");
    printf("  - 无需显式cudaMemcpy\n");
}

// 对比：统一内存 vs 显式传输
void compare_unified_vs_explicit(int n) {
    printf("\n=== 统一内存 vs 显式传输对比 ===\n");

    size_t size = n * sizeof(float);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 方式1: 显式传输
    float *h_a1, *h_b1, *h_c1;
    float *d_a1, *d_b1, *d_c1;

    h_a1 = (float*)malloc(size);
    h_b1 = (float*)malloc(size);
    h_c1 = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_a1[i] = i;
        h_b1[i] = i * 2;
    }

    CHECK_CUDA(cudaMalloc(&d_a1, size));
    CHECK_CUDA(cudaMalloc(&d_b1, size));
    CHECK_CUDA(cudaMalloc(&d_c1, size));

    cudaEvent_t start1, stop1;
    CHECK_CUDA(cudaEventCreate(&start1));
    CHECK_CUDA(cudaEventCreate(&stop1));

    CHECK_CUDA(cudaEventRecord(start1));
    CHECK_CUDA(cudaMemcpy(d_a1, h_a1, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1, size, cudaMemcpyHostToDevice));
    vector_add<<<numBlocks, blockSize>>>(d_a1, d_b1, d_c1, n);
    CHECK_CUDA(cudaMemcpy(h_c1, d_c1, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop1));
    CHECK_CUDA(cudaEventSynchronize(stop1));

    float ms_explicit;
    CHECK_CUDA(cudaEventElapsedTime(&ms_explicit, start1, stop1));

    free(h_a1); free(h_b1); free(h_c1);
    CHECK_CUDA(cudaFree(d_a1));
    CHECK_CUDA(cudaFree(d_b1));
    CHECK_CUDA(cudaFree(d_c1));
    CHECK_CUDA(cudaEventDestroy(start1));
    CHECK_CUDA(cudaEventDestroy(stop1));

    // 方式2: 统一内存
    float *a2, *b2, *c2;

    CHECK_CUDA(cudaMallocManaged(&a2, size));
    CHECK_CUDA(cudaMallocManaged(&b2, size));
    CHECK_CUDA(cudaMallocManaged(&c2, size));

    for (int i = 0; i < n; i++) {
        a2[i] = i;
        b2[i] = i * 2;
    }

    cudaEvent_t start2, stop2;
    CHECK_CUDA(cudaEventCreate(&start2));
    CHECK_CUDA(cudaEventCreate(&stop2));

    CHECK_CUDA(cudaEventRecord(start2));
    vector_add<<<numBlocks, blockSize>>>(a2, b2, c2, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop2));
    CHECK_CUDA(cudaEventSynchronize(stop2));

    float ms_unified;
    CHECK_CUDA(cudaEventElapsedTime(&ms_unified, start2, stop2));

    printf("显式传输时间: %.3f ms\n", ms_explicit);
    printf("统一内存时间: %.3f ms\n", ms_unified);
    printf("结果验证: c2[0] = %.1f (期望 %.1f)\n", c2[0], a2[0] + b2[0]);

    CHECK_CUDA(cudaFree(a2));
    CHECK_CUDA(cudaFree(b2));
    CHECK_CUDA(cudaFree(c2));
    CHECK_CUDA(cudaEventDestroy(start2));
    CHECK_CUDA(cudaEventDestroy(stop2));
}

// Page Fault演示
void page_fault_demo(int n) {
    printf("\n=== Page Fault演示 ===\n");

    float *data;
    size_t size = n * sizeof(float);

    CHECK_CUDA(cudaMallocManaged(&data, size));

    // CPU初始化 - 数据在CPU侧
    printf("CPU初始化...\n");
    for (int i = 0; i < n; i++) {
        data[i] = i;
    }

    // GPU访问 - 第一次会产生Page Fault，数据迁移到GPU
    printf("GPU第一次访问 (会有Page Fault)...\n");
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    vector_mult<<<numBlocks, blockSize>>>(data, data, data, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_first;
    CHECK_CUDA(cudaEventElapsedTime(&ms_first, start, stop));
    printf("第一次GPU执行时间: %.3f ms (含Page Fault开销)\n", ms_first);

    // GPU再次访问 - 数据已在GPU，无Page Fault
    printf("GPU第二次访问 (无Page Fault)...\n");
    CHECK_CUDA(cudaEventRecord(start));
    vector_mult<<<numBlocks, blockSize>>>(data, data, data, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_second;
    CHECK_CUDA(cudaEventElapsedTime(&ms_second, start, stop));
    printf("第二次GPU执行时间: %.3f ms\n", ms_second);

    printf("Page Fault影响:\n");
    printf("  - 首次访问会产生延迟\n");
    printf("  - Nsys中可以看到Page Fault事件\n");
    printf("  - 可通过预取优化避免\n");

    CHECK_CUDA(cudaFree(data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 统一内存优势
void unified_memory_benefits() {
    printf("\n=== 统一内存优势 ===\n");

    printf("\n编程便捷性:\n");
    printf("  - 无需手动管理数据传输\n");
    printf("  - CPU和GPU代码共享指针\n");
    printf("  - 减少代码复杂度\n");

    printf("\n自动内存管理:\n");
    printf("  - CUDA自动处理数据迁移\n");
    printf("  - 支持超额订阅(内存超过GPU容量)\n");
    printf("  - 自动处理一致性\n");

    printf("\n适用场景:\n");
    printf("  - 快速原型开发\n");
    printf("  - 代码简化优先\n");
    printf("  - 数据访问模式复杂\n");
    printf("  - 内存需求超过GPU容量\n");
}

int main() {
    printf("========================================\n");
    printf("  统一内存示例 - 第二十三章\n");
    printf("========================================\n");

    // 获取设备信息
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("\n设备: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("统一内存: %s\n", prop.managedMemory ? "支持" : "不支持");
    printf("并发管理访问: %s\n", prop.concurrentManagedAccess ? "支持" : "不支持");

    int n = 1024 * 1024 * 4;  // 4M 元素

    unified_memory_basics(n);
    compare_unified_vs_explicit(n);
    page_fault_demo(n);
    unified_memory_benefits();

    printf("\n========================================\n");
    printf("关键要点:\n");
    printf("  1. cudaMallocManaged分配统一内存\n");
    printf("  2. CPU和GPU共享同一内存指针\n");
    printf("  3. CUDA自动管理数据迁移\n");
    printf("  4. Page Fault首次访问会产生延迟\n");
    printf("========================================\n\n");

    return 0;
}