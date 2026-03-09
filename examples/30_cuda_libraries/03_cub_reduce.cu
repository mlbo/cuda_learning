/**
 * @file 03_cub_reduce.cu
 * @brief CUB并行原语库示例
 *
 * 本示例展示：
 * 1. Device级Reduce
 * 2. Device级Sort
 * 3. Device级Scan
 * 4. Block级操作
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Device级Reduce示例
// ============================================================================
void device_reduce_demo() {
    printf("=== Device级Reduce ===\n");

    int N = 16 * 1024 * 1024;  // 16M元素
    size_t bytes = N * sizeof(int);

    printf("数据规模: %d 元素 (%.2f MB)\n", N, bytes / 1024.0 / 1024.0);

    // 分配主机内存
    int* h_in = (int*)malloc(bytes);
    long long cpu_sum = 0;
    for (int i = 0; i < N; i++) {
        h_in[i] = i % 100;
        cpu_sum += h_in[i];
    }

    // 分配设备内存
    int* d_in;
    int* d_out;  // 单个输出
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // 确定临时存储大小
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           d_in, d_out, N);

    // 分配临时存储
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    printf("临时存储大小: %.2f MB\n", temp_storage_bytes / 1024.0 / 1024.0);

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    int repeat = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= repeat;

    // 验证结果
    int gpu_sum;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    printf("执行时间: %.4f ms\n", ms);
    printf("带宽: %.2f GB/s\n", bytes / ms / 1e6);
    printf("CPU结果: %lld, GPU结果: %d %s\n\n",
           cpu_sum, gpu_sum, (gpu_sum == (int)cpu_sum) ? "✓" : "✗");

    // 清理
    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Device级Reduce (多操作)
// ============================================================================
void device_reduce_multi_demo() {
    printf("=== Device级Reduce (多操作) ===\n");

    int N = 8 * 1024 * 1024;
    size_t bytes = N * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 1000) / 100.0f;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Sum
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CUDA_CHECK(cudaEventRecord(start));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float sum_result;
    CUDA_CHECK(cudaMemcpy(&sum_result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Sum: %.2f (%.4f ms)\n", sum_result, ms);

    // Max
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(start));
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float max_result;
    CUDA_CHECK(cudaMemcpy(&max_result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Max: %.2f (%.4f ms)\n", max_result, ms);

    // Min
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(start));
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float min_result;
    CUDA_CHECK(cudaMemcpy(&min_result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Min: %.2f (%.4f ms)\n", min_result, ms);

    // ArgMax - 需要单独的输出缓冲区存储KeyValuePair
    cub::KeyValuePair<int, float>* d_argmax_out;
    CUDA_CHECK(cudaMalloc(&d_argmax_out, sizeof(cub::KeyValuePair<int, float>)));

    // 先计算临时存储大小
    void* d_argmax_temp = nullptr;
    size_t argmax_temp_bytes = 0;
    cub::DeviceReduce::ArgMax(nullptr, argmax_temp_bytes, d_in, d_argmax_out, N);
    CUDA_CHECK(cudaMalloc(&d_argmax_temp, argmax_temp_bytes));

    CUDA_CHECK(cudaEventRecord(start));
    cub::DeviceReduce::ArgMax(d_argmax_temp, argmax_temp_bytes, d_in, d_argmax_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    cub::KeyValuePair<int, float> argmax_result;
    CUDA_CHECK(cudaMemcpy(&argmax_result, d_argmax_out, sizeof(argmax_result), cudaMemcpyDeviceToHost));
    printf("ArgMax: index=%d, value=%.2f (%.4f ms)\n\n",
           argmax_result.key, argmax_result.value, ms);

    // 清理
    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_argmax_out));
    CUDA_CHECK(cudaFree(d_argmax_temp));
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Device级Sort示例
// ============================================================================
void device_sort_demo() {
    printf("=== Device级Sort ===\n");

    int N = 4 * 1024 * 1024;  // 4M元素
    size_t bytes = N * sizeof(int);

    printf("数据规模: %d 元素\n", N);

    // 生成随机数据
    int* h_keys = (int*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) {
        h_keys[i] = rand();
    }

    int* d_keys_in;
    int* d_keys_out;
    CUDA_CHECK(cudaMalloc(&d_keys_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_keys_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys, bytes, cudaMemcpyHostToDevice));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // 确定临时存储大小
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                   d_keys_in, d_keys_out, N);

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    printf("临时存储大小: %.2f MB\n", temp_storage_bytes / 1024.0 / 1024.0);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                   d_keys_in, d_keys_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    int repeat = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                       d_keys_in, d_keys_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= repeat;

    printf("执行时间: %.4f ms\n", ms);
    printf("吞吐量: %.2f M元素/秒\n\n", N / ms / 1000.0);

    // 清理
    free(h_keys);
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Device级Scan示例
// ============================================================================
void device_scan_demo() {
    printf("=== Device级Scan ===\n");

    int N = 8 * 1024 * 1024;
    size_t bytes = N * sizeof(int);

    printf("数据规模: %d 元素\n", N);

    int* h_in = (int*)malloc(bytes);
    int* h_out = (int*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_in[i] = 1;  // 每个元素都是1，扫描结果应该是0,1,2,3,...
    }

    int* d_in;
    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_in, d_out, N);

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    int repeat = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                      d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= repeat;

    printf("执行时间: %.4f ms\n", ms);
    printf("带宽: %.2f GB/s\n", 2.0 * bytes / ms / 1e6);

    // 验证
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("前10个结果: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_out[i]);
    }
    printf("...\n\n");

    // 清理
    free(h_in);
    free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Block级Reduce示例
// ============================================================================
__global__ void block_reduce_kernel(int* d_in, int* d_out, int N) {
    typedef cub::BlockReduce<int, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (gid < N) ? d_in[gid] : 0;

    // Block内规约
    int block_sum = BlockReduce(temp_storage).Sum(val);

    // Block 0写入结果
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = block_sum;
    }
}

void block_reduce_demo() {
    printf("=== Block级Reduce ===\n");

    int N = 256 * 1024;  // 1024个Block
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    printf("数据规模: %d 元素, %d blocks\n", N, blocks);

    size_t bytes = N * sizeof(int);
    int* h_in = (int*)malloc(bytes);
    long long total = 0;
    for (int i = 0; i < N; i++) {
        h_in[i] = i % 100;
        total += h_in[i];
    }

    int* d_in;
    int* d_block_sums;
    int* d_final;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_final, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 执行Block级规约
    CUDA_CHECK(cudaEventRecord(start));
    block_reduce_kernel<<<blocks, threads>>>(d_in, d_block_sums, N);

    // 第二阶段：规约Block结果
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           d_block_sums, d_final, blocks);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           d_block_sums, d_final, blocks);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    int result;
    CUDA_CHECK(cudaMemcpy(&result, d_final, sizeof(int), cudaMemcpyDeviceToHost));

    printf("执行时间: %.4f ms\n", ms);
    printf("CPU结果: %lld, GPU结果: %d %s\n\n",
           total, result, (result == (int)total) ? "✓" : "✗");

    // 清理
    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_final));
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Select/Filter示例
// ============================================================================
void device_select_demo() {
    printf("=== Device级Select (条件过滤) ===\n");

    int N = 4 * 1024 * 1024;
    size_t bytes = N * sizeof(int);

    int* h_in = (int*)malloc(bytes);
    int threshold = 500;
    int expected_count = 0;
    for (int i = 0; i < N; i++) {
        h_in[i] = rand() % 1000;
        if (h_in[i] > threshold) expected_count++;
    }

    int* d_in;
    int* d_out;
    int* d_num_selected;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_num_selected, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // 定义选择条件
    auto select_op = [] __device__(int val) { return val > 500; };

    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          d_in, d_out, d_num_selected, N, select_op);

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          d_in, d_out, d_num_selected, N, select_op);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    int num_selected;
    CUDA_CHECK(cudaMemcpy(&num_selected, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost));

    printf("总元素: %d, 阈值: %d\n", N, threshold);
    printf("选择元素: %d (期望: %d) %s\n",
           num_selected, expected_count,
           num_selected == expected_count ? "✓" : "✗");
    printf("执行时间: %.4f ms\n\n", ms);

    // 清理
    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_num_selected));
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("===== CUB并行原语库示例 =====\n\n");

    device_reduce_demo();
    device_reduce_multi_demo();
    device_sort_demo();
    device_scan_demo();
    block_reduce_demo();
    device_select_demo();

    printf("=== CUB使用建议 ===\n");
    printf("1. Device级操作最简单，自动处理所有优化\n");
    printf("2. 临时存储可以复用，避免频繁分配\n");
    printf("3. Block级操作需要手动管理共享内存和同步\n");
    printf("4. CUB性能通常接近或超过手动优化版本\n");

    return 0;
}