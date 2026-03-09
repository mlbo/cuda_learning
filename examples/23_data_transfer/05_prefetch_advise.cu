/**
 * 第二十三章示例：预取与建议
 *
 * 本示例演示统一内存的预取(Prefetch)和建议(Advise)优化
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
__global__ void vector_op(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx] + a[idx];
    }
}

// 无优化的统一内存
void no_optimization_test(int n) {
    printf("\n=== 无优化的统一内存 ===\n");

    float *a, *b, *c;
    size_t size = n * sizeof(float);

    CHECK_CUDA(cudaMallocManaged(&a, size));
    CHECK_CUDA(cudaMallocManaged(&b, size));
    CHECK_CUDA(cudaMallocManaged(&c, size));

    // CPU初始化
    for (int i = 0; i < n; i++) {
        a[i] = i * 0.001f;
        b[i] = i * 0.002f;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // GPU计算（首次访问会有Page Fault）
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_op<<<numBlocks, blockSize>>>(a, b, c, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("执行时间: %.3f ms (含Page Fault开销)\n", ms);

    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 使用预取优化
void prefetch_test(int n) {
    printf("\n=== 使用预取优化 ===\n");

    float *a, *b, *c;
    size_t size = n * sizeof(float);

    CHECK_CUDA(cudaMallocManaged(&a, size));
    CHECK_CUDA(cudaMallocManaged(&b, size));
    CHECK_CUDA(cudaMallocManaged(&c, size));

    // CPU初始化
    for (int i = 0; i < n; i++) {
        a[i] = i * 0.001f;
        b[i] = i * 0.002f;
    }

    // 创建流
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // CUDA 12+ 使用新的 cudaMemLocation API
// 创建 GPU 设备位置
    cudaMemLocation gpu_loc = {cudaMemLocationTypeDevice, 0};

    CHECK_CUDA(cudaEventRecord(start, stream));

    // 预取到GPU (设备ID 0)
    CHECK_CUDA(cudaMemPrefetchAsync(a, size, gpu_loc, 0, stream));
    CHECK_CUDA(cudaMemPrefetchAsync(b, size, gpu_loc, 0, stream));
    CHECK_CUDA(cudaMemPrefetchAsync(c, size, gpu_loc, 0, stream));

    // GPU计算（数据已在GPU，无Page Fault）
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_op<<<numBlocks, blockSize, 0, stream>>>(a, b, c, n);

    // 预取回CPU
    cudaMemLocation cpu_loc = {cudaMemLocationTypeHost, 0};
    CHECK_CUDA(cudaMemPrefetchAsync(c, size, cpu_loc, 0, stream));

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("执行时间: %.3f ms (无Page Fault)\n", ms);

    printf("预取优化要点:\n");
    printf("  - 在GPU访问前预取数据到GPU\n");
    printf("  - 在CPU访问前预取数据到CPU\n");
    printf("  - 避免Page Fault延迟\n");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 使用内存建议优化
void advise_test(int n) {
    printf("\n=== 使用内存建议优化 ===\n");

    float *a, *b, *c;
    size_t size = n * sizeof(float);

    CHECK_CUDA(cudaMallocManaged(&a, size));
    CHECK_CUDA(cudaMallocManaged(&b, size));
    CHECK_CUDA(cudaMallocManaged(&c, size));

    // 设置内存建议 (CUDA 12+ API)
    cudaMemLocation gpu_loc = {cudaMemLocationTypeDevice, 0};
    // 建议: 数据首选位置在GPU
    CHECK_CUDA(cudaMemAdvise(a, size, cudaMemAdviseSetPreferredLocation, gpu_loc));
    CHECK_CUDA(cudaMemAdvise(b, size, cudaMemAdviseSetPreferredLocation, gpu_loc));
    CHECK_CUDA(cudaMemAdvise(c, size, cudaMemAdviseSetPreferredLocation, gpu_loc));

    // 建议: 数据将被GPU访问
    CHECK_CUDA(cudaMemAdvise(a, size, cudaMemAdviseSetAccessedBy, gpu_loc));
    CHECK_CUDA(cudaMemAdvise(b, size, cudaMemAdviseSetAccessedBy, gpu_loc));
    CHECK_CUDA(cudaMemAdvise(c, size, cudaMemAdviseSetAccessedBy, gpu_loc));

    // CPU初始化
    for (int i = 0; i < n; i++) {
        a[i] = i * 0.001f;
        b[i] = i * 0.002f;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_op<<<numBlocks, blockSize>>>(a, b, c, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("执行时间: %.3f ms (有内存建议)\n", ms);

    printf("内存建议类型:\n");
    printf("  - cudaMemAdviseSetReadMostly: 数据主要被读取\n");
    printf("  - cudaMemAdviseSetPreferredLocation: 数据首选位置\n");
    printf("  - cudaMemAdviseSetAccessedBy: 数据将被某设备访问\n");

    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 预取+建议组合优化
void combined_optimization_test(int n) {
    printf("\n=== 预取+建议组合优化 ===\n");

    float *a, *b, *c;
    size_t size = n * sizeof(float);

    CHECK_CUDA(cudaMallocManaged(&a, size));
    CHECK_CUDA(cudaMallocManaged(&b, size));
    CHECK_CUDA(cudaMallocManaged(&c, size));

    // 设置建议 (CUDA 12+ API)
    cudaMemLocation gpu_loc = {cudaMemLocationTypeDevice, 0};
    CHECK_CUDA(cudaMemAdvise(a, size, cudaMemAdviseSetPreferredLocation, gpu_loc));
    CHECK_CUDA(cudaMemAdvise(b, size, cudaMemAdviseSetPreferredLocation, gpu_loc));
    CHECK_CUDA(cudaMemAdvise(c, size, cudaMemAdviseSetPreferredLocation, gpu_loc));

    // CPU初始化
    for (int i = 0; i < n; i++) {
        a[i] = i * 0.001f;
        b[i] = i * 0.002f;
    }

    // 创建流
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));

    // 预取到GPU (复用上面声明的gpu_loc)
    CHECK_CUDA(cudaMemPrefetchAsync(a, size, gpu_loc, 0, stream));
    CHECK_CUDA(cudaMemPrefetchAsync(b, size, gpu_loc, 0, stream));
    CHECK_CUDA(cudaMemPrefetchAsync(c, size, gpu_loc, 0, stream));

    // GPU计算
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_op<<<numBlocks, blockSize, 0, stream>>>(a, b, c, n);

    // 预取回CPU
    cudaMemLocation cpu_loc = {cudaMemLocationTypeHost, 0};
    CHECK_CUDA(cudaMemPrefetchAsync(c, size, cpu_loc, 0, stream));

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("执行时间: %.3f ms (最佳优化)\n", ms);

    // 验证结果
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        float expected = a[i] * b[i] + a[i];
        if (std::fabs(c[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("结果验证: %s\n", correct ? "通过" : "失败");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 性能对比总结
void performance_summary(int n) {
    printf("\n=== 性能对比总结 ===\n");

    size_t size = n * sizeof(float);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    struct Result {
        const char* name;
        float ms;
    };

    Result results[4];

    // 1. 无优化
    {
        float *a, *b, *c;
        CHECK_CUDA(cudaMallocManaged(&a, size));
        CHECK_CUDA(cudaMallocManaged(&b, size));
        CHECK_CUDA(cudaMallocManaged(&c, size));
        for (int i = 0; i < n; i++) { a[i] = i; b[i] = i * 2; }

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        vector_op<<<numBlocks, blockSize>>>(a, b, c, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&results[0].ms, start, stop));
        results[0].name = "无优化";

        CHECK_CUDA(cudaFree(a)); CHECK_CUDA(cudaFree(b)); CHECK_CUDA(cudaFree(c));
        CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    }

    // 2. 仅预取
    {
        float *a, *b, *c;
        CHECK_CUDA(cudaMallocManaged(&a, size));
        CHECK_CUDA(cudaMallocManaged(&b, size));
        CHECK_CUDA(cudaMallocManaged(&c, size));
        for (int i = 0; i < n; i++) { a[i] = i; b[i] = i * 2; }

        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start, stream));
        cudaMemLocation gpu_loc = {cudaMemLocationTypeDevice, 0};
        cudaMemLocation cpu_loc = {cudaMemLocationTypeHost, 0};
        CHECK_CUDA(cudaMemPrefetchAsync(a, size, gpu_loc, 0, stream));
        CHECK_CUDA(cudaMemPrefetchAsync(b, size, gpu_loc, 0, stream));
        CHECK_CUDA(cudaMemPrefetchAsync(c, size, gpu_loc, 0, stream));
        vector_op<<<numBlocks, blockSize, 0, stream>>>(a, b, c, n);
        CHECK_CUDA(cudaMemPrefetchAsync(c, size, cpu_loc, 0, stream));
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&results[1].ms, start, stop));
        results[1].name = "仅预取";

        CHECK_CUDA(cudaStreamDestroy(stream));
        CHECK_CUDA(cudaFree(a)); CHECK_CUDA(cudaFree(b)); CHECK_CUDA(cudaFree(c));
        CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    }

    // 3. 仅建议
    {
        float *a, *b, *c;
        CHECK_CUDA(cudaMallocManaged(&a, size));
        CHECK_CUDA(cudaMallocManaged(&b, size));
        CHECK_CUDA(cudaMallocManaged(&c, size));
        cudaMemLocation gpu_loc = {cudaMemLocationTypeDevice, 0};
        CHECK_CUDA(cudaMemAdvise(a, size, cudaMemAdviseSetPreferredLocation, gpu_loc));
        CHECK_CUDA(cudaMemAdvise(b, size, cudaMemAdviseSetPreferredLocation, gpu_loc));
        CHECK_CUDA(cudaMemAdvise(c, size, cudaMemAdviseSetPreferredLocation, gpu_loc));
        for (int i = 0; i < n; i++) { a[i] = i; b[i] = i * 2; }

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        vector_op<<<numBlocks, blockSize>>>(a, b, c, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&results[2].ms, start, stop));
        results[2].name = "仅建议";

        CHECK_CUDA(cudaFree(a)); CHECK_CUDA(cudaFree(b)); CHECK_CUDA(cudaFree(c));
        CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    }

    // 4. 预取+建议
    {
        float *a, *b, *c;
        CHECK_CUDA(cudaMallocManaged(&a, size));
        CHECK_CUDA(cudaMallocManaged(&b, size));
        CHECK_CUDA(cudaMallocManaged(&c, size));
        cudaMemLocation gpu_loc = {cudaMemLocationTypeDevice, 0};
        CHECK_CUDA(cudaMemAdvise(a, size, cudaMemAdviseSetPreferredLocation, gpu_loc));
        CHECK_CUDA(cudaMemAdvise(b, size, cudaMemAdviseSetPreferredLocation, gpu_loc));
        CHECK_CUDA(cudaMemAdvise(c, size, cudaMemAdviseSetPreferredLocation, gpu_loc));
        for (int i = 0; i < n; i++) { a[i] = i; b[i] = i * 2; }

        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start, stream));
        cudaMemLocation cpu_loc = {cudaMemLocationTypeHost, 0};
        CHECK_CUDA(cudaMemPrefetchAsync(a, size, gpu_loc, 0, stream));
        CHECK_CUDA(cudaMemPrefetchAsync(b, size, gpu_loc, 0, stream));
        CHECK_CUDA(cudaMemPrefetchAsync(c, size, gpu_loc, 0, stream));
        vector_op<<<numBlocks, blockSize, 0, stream>>>(a, b, c, n);
        CHECK_CUDA(cudaMemPrefetchAsync(c, size, cpu_loc, 0, stream));
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&results[3].ms, start, stop));
        results[3].name = "预取+建议";

        CHECK_CUDA(cudaStreamDestroy(stream));
        CHECK_CUDA(cudaFree(a)); CHECK_CUDA(cudaFree(b)); CHECK_CUDA(cudaFree(c));
        CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    }

    printf("\n%-15s %-15s %-15s\n", "优化方式", "时间(ms)", "相对提升");
    printf("----------------------------------------\n");
    float baseline = results[0].ms;
    for (int i = 0; i < 4; i++) {
        printf("%-15s %-15.3f %-15.1f%%\n",
               results[i].name, results[i].ms,
               (baseline - results[i].ms) / baseline * 100);
    }
}

int main() {
    printf("========================================\n");
    printf("  预取与建议优化示例 - 第二十三章\n");
    printf("========================================\n");

    // 获取设备信息
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("\n设备: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    int n = 1024 * 1024 * 8;  // 8M 元素

    no_optimization_test(n);
    prefetch_test(n);
    advise_test(n);
    combined_optimization_test(n);
    performance_summary(n);

    printf("\n========================================\n");
    printf("关键要点:\n");
    printf("  1. cudaMemPrefetchAsync预取数据避免Page Fault\n");
    printf("  2. cudaMemAdvise给CUDA提示优化迁移策略\n");
    printf("  3. 预取和建议可以组合使用获得最佳性能\n");
    printf("  4. 统一内存优化后可接近显式传输性能\n");
    printf("========================================\n\n");

    return 0;
}
