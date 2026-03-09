/**
 * 07_histogram.cu
 * 实战案例：并行直方图
 *
 * 直方图是原子操作的典型应用场景
 *
 * 编译: nvcc -o 07_histogram 07_histogram.cu
 * 运行: ./07_histogram
 * 分析: ncu ./07_histogram
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_BINS 256  // 直方图bin数量

// 朴素版本：每个线程直接对bin做原子操作
__global__ void histogram_naive(unsigned char* data, int* histogram, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 数据值即为bin索引
        int bin = data[idx];
        atomicAdd(&histogram[bin], 1);
    }
}

// 优化版本1：使用共享内存作为局部直方图
__global__ void histogram_shared(unsigned char* data, int* histogram, int N) {
    // 每个block的局部直方图
    __shared__ int local_hist[NUM_BINS];

    // 初始化局部直方图
    int tid = threadIdx.x;
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // 计算局部直方图
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int bin = data[idx];
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();

    // 合并到全局直方图
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        if (local_hist[i] > 0) {
            atomicAdd(&histogram[i], local_hist[i]);
        }
    }
}

// 优化版本2：每个线程处理多个数据
__global__ void histogram_multi_element(unsigned char* data, int* histogram, int N) {
    __shared__ int local_hist[NUM_BINS];

    int tid = threadIdx.x;

    // 初始化局部直方图
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride循环：每个线程处理多个元素
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        int bin = data[i];
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();

    // 合并到全局
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        if (local_hist[i] > 0) {
            atomicAdd(&histogram[i], local_hist[i]);
        }
    }
}

// CPU版本用于验证
void histogram_cpu(unsigned char* data, int* histogram, int N) {
    for (int i = 0; i < NUM_BINS; i++) {
        histogram[i] = 0;
    }
    for (int i = 0; i < N; i++) {
        histogram[data[i]]++;
    }
}

// 检查结果正确性
bool check_result(int* h_hist, int* d_hist, int num_bins) {
    for (int i = 0; i < num_bins; i++) {
        if (h_hist[i] != d_hist[i]) {
            printf("Mismatch at bin %d: CPU=%d, GPU=%d\n", i, h_hist[i], d_hist[i]);
            return false;
        }
    }
    return true;
}

// 计时函数
template <void (*Kernel)(unsigned char*, int*, int)>
float time_kernel(unsigned char* d_data, int* d_hist, int N,
                  int runs = 10) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 初始化直方图
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) {
        cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));
        Kernel<<<blocks, threads>>>(d_data, d_hist, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= runs;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    int N = 1024 * 1024;  // 1M 数据

    // 分配主机内存
    unsigned char* h_data = (unsigned char*)malloc(N);
    int* h_hist_cpu = (int*)malloc(NUM_BINS * sizeof(int));
    int* h_hist_gpu = (int*)malloc(NUM_BINS * sizeof(int));

    // 初始化随机数据
    srand(42);
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % NUM_BINS;
    }

    // 计算CPU参考结果
    histogram_cpu(h_data, h_hist_cpu, N);

    // 分配设备内存
    unsigned char* d_data;
    int* d_hist;
    cudaMalloc(&d_data, N);
    cudaMalloc(&d_hist, NUM_BINS * sizeof(int));

    cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);

    printf("========== 并行直方图性能对比 ==========\n");
    printf("数据量: %d (%.2f MB)\n", N, (float)N / 1024 / 1024);
    printf("直方图bin数: %d\n\n", NUM_BINS);

    float ms;

    // 测试朴素版本
    ms = time_kernel<histogram_naive>(d_data, d_hist, N);
    cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    printf("1. 朴素版本:\n");
    printf("   时间: %.4f ms\n", ms);
    printf("   正确: %s\n\n", check_result(h_hist_cpu, h_hist_gpu, NUM_BINS) ? "是" : "否");

    // 测试共享内存版本
    ms = time_kernel<histogram_shared>(d_data, d_hist, N);
    cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    printf("2. 共享内存版本:\n");
    printf("   时间: %.4f ms\n", ms);
    printf("   正确: %s\n\n", check_result(h_hist_cpu, h_hist_gpu, NUM_BINS) ? "是" : "否");

    // 测试多元素版本
    ms = time_kernel<histogram_multi_element>(d_data, d_hist, N);
    cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    printf("3. 多元素处理版本:\n");
    printf("   时间: %.4f ms\n", ms);
    printf("   正确: %s\n", check_result(h_hist_cpu, h_hist_gpu, NUM_BINS) ? "是" : "否");

    printf("\n========================================\n");
    printf("优化要点:\n");
    printf("1. 使用共享内存减少全局内存原子操作次数\n");
    printf("2. 每个线程处理多个数据提高效率\n");
    printf("3. 只有非零bin才合并到全局\n");

    // 打印部分直方图结果
    printf("\n部分直方图结果:\n");
    printf("Bin\tCount\tCPU\n");
    for (int i = 0; i < 10; i++) {
        printf("%d\t%d\t%d\n", i, h_hist_gpu[i], h_hist_cpu[i]);
    }

    // 清理
    cudaFree(d_data);
    cudaFree(d_hist);
    free(h_data);
    free(h_hist_cpu);
    free(h_hist_gpu);

    return 0;
}
