/**
 * 第二十三章示例：零拷贝内存
 *
 * 本示例演示零拷贝内存(Zero-Copy Memory)的使用
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

// 零拷贝访问的核函数
__global__ void zero_copy_read(float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 直接从主机内存读取
        out[idx] = in[idx] * 2.0f;
    }
}

// 零拷贝写入的核函数
__global__ void zero_copy_write(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 直接写入主机内存
        data[idx] = value * idx;
    }
}

// 传统方式：显式传输
void traditional_transfer_test(int n) {
    printf("\n=== 传统传输方式 ===\n");

    float *h_in, *h_out;
    float *d_in, *d_out;
    size_t size = n * sizeof(float);

    // 分配主机内存
    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);

    // 初始化
    for (int i = 0; i < n; i++) {
        h_in[i] = i * 0.001f;
    }

    // 分配设备内存
    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // H2D传输
    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // 核函数执行
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    zero_copy_read<<<numBlocks, blockSize>>>(d_in, d_out, n);

    // D2H传输
    CHECK_CUDA(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("数据量: %.2f MB\n", (float)size / 1024 / 1024);
    printf("总时间: %.3f ms\n", ms);

    // 清理
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 零拷贝方式：无显式传输
void zero_copy_test(int n) {
    printf("\n=== 零拷贝方式 ===\n");

    float *h_in, *h_out;
    float *d_in, *d_out;
    size_t size = n * sizeof(float);

    // 分配映射到GPU的主机内存
    CHECK_CUDA(cudaHostAlloc(&h_in, size, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostAlloc(&h_out, size, cudaHostAllocMapped));

    // 初始化
    for (int i = 0; i < n; i++) {
        h_in[i] = i * 0.001f;
    }

    // 获取设备端指针
    CHECK_CUDA(cudaHostGetDevicePointer(&d_in, h_in, 0));
    CHECK_CUDA(cudaHostGetDevicePointer(&d_out, h_out, 0));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // 无需H2D传输！核函数直接访问主机内存
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    zero_copy_read<<<numBlocks, blockSize>>>(d_in, d_out, n);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("数据量: %.2f MB\n", (float)size / 1024 / 1024);
    printf("总时间: %.3f ms (无显式传输)\n", ms);

    // 验证结果
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (std::fabs(h_out[i] - h_in[i] * 2.0f) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("结果验证: %s\n", correct ? "通过" : "失败");

    // 清理
    CHECK_CUDA(cudaFreeHost(h_in));
    CHECK_CUDA(cudaFreeHost(h_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 零拷贝写入测试
void zero_copy_write_test(int n) {
    printf("\n=== 零拷贝写入测试 ===\n");

    float *h_data;
    float *d_data;
    size_t size = n * sizeof(float);

    // 分配映射的主机内存
    CHECK_CUDA(cudaHostAlloc(&h_data, size, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostGetDevicePointer(&d_data, h_data, 0));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // 核函数直接写入主机内存
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    zero_copy_write<<<numBlocks, blockSize>>>(d_data, 1.0f, n);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("零拷贝写入时间: %.3f ms\n", ms);
    printf("主机端读取: h_data[0]=%.1f, h_data[100]=%.1f\n", h_data[0], h_data[100]);

    // 清理
    CHECK_CUDA(cudaFreeHost(h_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 不同数据大小的性能对比
void performance_comparison() {
    printf("\n=== 性能对比 (不同数据大小) ===\n");

    int sizes[] = {1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-15s %-20s %-20s\n", "数据大小", "传统传输(ms)", "零拷贝(ms)");
    printf("----------------------------------------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        size_t size = n * sizeof(float);

        // 传统方式
        float *h_in, *h_out;
        float *d_in, *d_out;

        h_in = (float*)malloc(size);
        h_out = (float*)malloc(size);
        for (int j = 0; j < n; j++) h_in[j] = j;

        CHECK_CUDA(cudaMalloc(&d_in, size));
        CHECK_CUDA(cudaMalloc(&d_out, size));

        cudaEvent_t start1, stop1;
        CHECK_CUDA(cudaEventCreate(&start1));
        CHECK_CUDA(cudaEventCreate(&stop1));
        CHECK_CUDA(cudaEventRecord(start1));
        CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        zero_copy_read<<<numBlocks, blockSize>>>(d_in, d_out, n);
        CHECK_CUDA(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop1));
        CHECK_CUDA(cudaEventSynchronize(stop1));
        float ms_trad;
        CHECK_CUDA(cudaEventElapsedTime(&ms_trad, start1, stop1));

        free(h_in);
        free(h_out);
        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
        CHECK_CUDA(cudaEventDestroy(start1));
        CHECK_CUDA(cudaEventDestroy(stop1));

        // 零拷贝方式
        float *h_in2, *h_out2;
        float *d_in2, *d_out2;

        CHECK_CUDA(cudaHostAlloc(&h_in2, size, cudaHostAllocMapped));
        CHECK_CUDA(cudaHostAlloc(&h_out2, size, cudaHostAllocMapped));
        for (int j = 0; j < n; j++) h_in2[j] = j;

        CHECK_CUDA(cudaHostGetDevicePointer(&d_in2, h_in2, 0));
        CHECK_CUDA(cudaHostGetDevicePointer(&d_out2, h_out2, 0));

        cudaEvent_t start2, stop2;
        CHECK_CUDA(cudaEventCreate(&start2));
        CHECK_CUDA(cudaEventCreate(&stop2));
        CHECK_CUDA(cudaEventRecord(start2));
        zero_copy_read<<<numBlocks, blockSize>>>(d_in2, d_out2, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(stop2));
        CHECK_CUDA(cudaEventSynchronize(stop2));
        float ms_zero;
        CHECK_CUDA(cudaEventElapsedTime(&ms_zero, start2, stop2));

        CHECK_CUDA(cudaFreeHost(h_in2));
        CHECK_CUDA(cudaFreeHost(h_out2));
        CHECK_CUDA(cudaEventDestroy(start2));
        CHECK_CUDA(cudaEventDestroy(stop2));

        printf("%-15.2f %-20.3f %-20.3f\n", (float)size / 1024 / 1024, ms_trad, ms_zero);
    }

    printf("\n观察:\n");
    printf("  - 小数据: 零拷贝可能更快（无传输开销）\n");
    printf("  - 大数据: 传统传输更快（更高带宽）\n");
}

// 零拷贝适用场景说明
void zero_copy_use_cases() {
    printf("\n=== 零拷贝适用场景 ===\n");

    printf("\n适合零拷贝:\n");
    printf("  1. 小数据量 - 传输开销占主导\n");
    printf("  2. 单次访问 - 每个数据只读或只写一次\n");
    printf("  3. 读或写为主 - 避免频繁的读写切换\n");
    printf("  4. 稀疏访问 - 只访问部分数据\n");

    printf("\n不适合零拷贝:\n");
    printf("  1. 大数据量 - PCIe延迟成为瓶颈\n");
    printf("  2. 多次访问 - 同一数据被多次读写\n");
    printf("  3. 计算密集 - 计算时间远大于传输时间\n");
    printf("  4. 带宽敏感 - 需要高带宽访问\n");
}

int main() {
    printf("========================================\n");
    printf("  零拷贝内存示例 - 第二十三章\n");
    printf("========================================\n");

    // 获取设备信息
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("\n设备: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("是否支持映射内存: %s\n", prop.canMapHostMemory ? "是" : "否");

    int n = 1024 * 1024;  // 1M 元素

    traditional_transfer_test(n);
    zero_copy_test(n);
    zero_copy_write_test(n);
    performance_comparison();
    zero_copy_use_cases();

    printf("\n========================================\n");
    printf("关键要点:\n");
    printf("  1. cudaHostAllocMapped分配映射内存\n");
    printf("  2. cudaHostGetDevicePointer获取设备指针\n");
    printf("  3. 零拷贝适合小数据单次访问场景\n");
    printf("  4. 大数据场景传统传输更优\n");
    printf("========================================\n\n");

    return 0;
}
