/**
 * 第二十二章示例：并发核函数
 *
 * 本示例演示多个核函数的并发执行
 * 分析并发条件和限制
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

// 不同计算强度的核函数
__global__ void light_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void medium_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.001f + 0.001f;
        }
        data[idx] = val;
    }
}

__global__ void heavy_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.0001f + 0.0001f;
        }
        data[idx] = val;
    }
}

// 打印设备信息
void print_device_info() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("=== 设备信息 ===\n");
    printf("  设备名称:          %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM数量:            %d\n", prop.multiProcessorCount);
    printf("  最大并发核函数:    %d\n", prop.concurrentKernels);
    printf("  每Block最大线程:   %d\n", prop.maxThreadsPerBlock);
    printf("  每SM最大线程:      %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  寄存器/Block:      %d\n", prop.regsPerBlock);
    printf("  共享内存/Block:    %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");
}

// 1. 简单并发核函数
void simple_concurrent_kernels() {
    printf("=== 1. 简单并发核函数 ===\n");

    const int n = 1024 * 1024;
    const int num_kernels = 4;

    // 为每个核函数分配独立的设备内存
    float *d_data[num_kernels];
    for (int i = 0; i < num_kernels; i++) {
        CHECK_CUDA(cudaMalloc(&d_data[i], n * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_data[i], i + 1, n * sizeof(float)));
    }

    // 创建流
    cudaStream_t streams[num_kernels];
    for (int i = 0; i < num_kernels; i++) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 串行执行（单流）
    printf("\n  串行执行:\n");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_kernels; i++) {
        light_kernel<<<numBlocks, blockSize>>>(d_data[i], n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float serial_time;
    CHECK_CUDA(cudaEventElapsedTime(&serial_time, start, stop));
    printf("    执行时间: %.3f ms\n", serial_time);

    // 并发执行（多流）
    printf("\n  并发执行 (%d 流):\n", num_kernels);
    CHECK_CUDA(cudaEventRecord(start, streams[0]));
    for (int i = 0; i < num_kernels; i++) {
        light_kernel<<<numBlocks, blockSize, 0, streams[i]>>>(d_data[i], n);
    }
    for (int i = 0; i < num_kernels; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    CHECK_CUDA(cudaEventRecord(stop, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float concurrent_time;
    CHECK_CUDA(cudaEventElapsedTime(&concurrent_time, start, stop));
    printf("    执行时间: %.3f ms\n", concurrent_time);
    printf("    加速比:   %.2fx\n", serial_time / concurrent_time);

    // 清理
    for (int i = 0; i < num_kernels; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        CHECK_CUDA(cudaFree(d_data[i]));
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// 2. 不同大小的核函数并发
void variable_size_kernels() {
    printf("\n=== 2. 不同大小核函数并发 ===\n");

    // 不同大小的数据
    int sizes[] = {256 * 1024, 512 * 1024, 1024 * 1024, 2 * 1024 * 1024};
    const int num_kernels = 4;

    float *d_data[num_kernels];
    cudaStream_t streams[num_kernels];

    for (int i = 0; i < num_kernels; i++) {
        CHECK_CUDA(cudaMalloc(&d_data[i], sizes[i] * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_data[i], 1, sizes[i] * sizeof(float)));
        CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    cudaEvent_t events[num_kernels + 1];
    for (int i = 0; i <= num_kernels; i++) {
        CHECK_CUDA(cudaEventCreate(&events[i]));
    }

    int blockSize = 256;

    printf("\n  并发执行不同大小的核函数:\n");
    CHECK_CUDA(cudaEventRecord(events[0], streams[0]));

    for (int i = 0; i < num_kernels; i++) {
        int numBlocks = (sizes[i] + blockSize - 1) / blockSize;
        medium_kernel<<<numBlocks, blockSize, 0, streams[i]>>>(d_data[i], sizes[i], 1000);
        CHECK_CUDA(cudaEventRecord(events[i + 1], streams[i]));
    }

    // 等待所有完成
    for (int i = 0; i < num_kernels; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // 打印各核函数时间
    for (int i = 0; i < num_kernels; i++) {
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, events[0], events[i + 1]));
        printf("    Kernel %d (%dK 元素): %.3f ms\n", i, sizes[i] / 1024, ms);
    }

    float total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, events[0], events[num_kernels]));
    printf("    总时间: %.3f ms\n", total_ms);

    // 清理
    for (int i = 0; i < num_kernels; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        CHECK_CUDA(cudaFree(d_data[i]));
        CHECK_CUDA(cudaEventDestroy(events[i]));
    }
    CHECK_CUDA(cudaEventDestroy(events[num_kernels]));
}

// 3. 资源限制分析
void resource_limit_analysis() {
    printf("\n=== 3. 资源限制分析 ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int max_concurrent = prop.concurrentKernels;
    printf("  理论最大并发核函数数: %d\n", max_concurrent);

    // 测试不同数量的并发核函数
    for (int num_streams = 1; num_streams <= 8; num_streams *= 2) {
        cudaStream_t *streams = new cudaStream_t[num_streams];
        for (int i = 0; i < num_streams; i++) {
            CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        }

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;

        CHECK_CUDA(cudaEventRecord(start, streams[0]));

        for (int i = 0; i < num_streams; i++) {
            light_kernel<<<numBlocks, blockSize, 0, streams[i]>>>(d_data, n);
        }

        for (int i = 0; i < num_streams; i++) {
            CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        }

        CHECK_CUDA(cudaEventRecord(stop, streams[0]));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("    %d 流: %.3f ms\n", num_streams, ms);

        for (int i = 0; i < num_streams; i++) {
            CHECK_CUDA(cudaStreamDestroy(streams[i]));
        }
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        delete[] streams;
    }

    CHECK_CUDA(cudaFree(d_data));
}

// 4. 寄存器和共享内存对并发的影响
void register_smem_impact() {
    printf("\n=== 4. 寄存器/共享内存对并发的影响 ===\n");

    // 核函数使用不同资源量
    const int n = 256 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 创建多个流
    const int num_streams = 8;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 轻量核函数（低资源使用）
    printf("\n  轻量核函数 (低资源):\n");
    CHECK_CUDA(cudaEventRecord(start, streams[0]));
    for (int i = 0; i < num_streams; i++) {
        light_kernel<<<numBlocks, blockSize, 0, streams[i]>>>(d_data, n);
    }
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    CHECK_CUDA(cudaEventRecord(stop, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float light_time;
    CHECK_CUDA(cudaEventElapsedTime(&light_time, start, stop));
    printf("    执行时间: %.3f ms\n", light_time);

    // 中等核函数
    printf("\n  中等核函数:\n");
    CHECK_CUDA(cudaEventRecord(start, streams[0]));
    for (int i = 0; i < num_streams; i++) {
        medium_kernel<<<numBlocks, blockSize, 0, streams[i]>>>(d_data, n, 100);
    }
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    CHECK_CUDA(cudaEventRecord(stop, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float medium_time;
    CHECK_CUDA(cudaEventElapsedTime(&medium_time, start, stop));
    printf("    执行时间: %.3f ms\n", medium_time);

    // 重量核函数（高资源使用）
    printf("\n  重量核函数 (高资源):\n");
    CHECK_CUDA(cudaEventRecord(start, streams[0]));
    for (int i = 0; i < num_streams; i++) {
        heavy_kernel<<<numBlocks, blockSize, 0, streams[i]>>>(d_data, n, 1000);
    }
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    CHECK_CUDA(cudaEventRecord(stop, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float heavy_time;
    CHECK_CUDA(cudaEventElapsedTime(&heavy_time, start, stop));
    printf("    执行时间: %.3f ms\n", heavy_time);

    printf("\n  分析: 资源使用越高，并发度可能降低\n");

    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
}

// 5. 计算与传输重叠
void compute_transfer_overlap() {
    printf("\n=== 5. 计算与传输重叠 ===\n");

    const int n = 4 * 1024 * 1024;

    // 使用锁页内存
    float *h_data_in, *h_data_out;
    CHECK_CUDA(cudaMallocHost(&h_data_in, n * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_data_out, n * sizeof(float)));

    for (int i = 0; i < n; i++) {
        h_data_in[i] = (float)i;
    }

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    // 创建流
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int chunk_size = n / 2;
    int blockSize = 256;
    int numBlocks = (chunk_size + blockSize - 1) / blockSize;

    // 不重叠版本
    printf("\n  串行版本:\n");
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDA(cudaMemcpy(d_data, h_data_in, n * sizeof(float), cudaMemcpyHostToDevice));
    heavy_kernel<<<(n + blockSize - 1) / blockSize, blockSize>>>(d_data, n, 500);
    CHECK_CUDA(cudaMemcpy(h_data_out, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float serial_time;
    CHECK_CUDA(cudaEventElapsedTime(&serial_time, start, stop));
    printf("    执行时间: %.3f ms\n", serial_time);

    // 重叠版本
    printf("\n  重叠版本:\n");
    CHECK_CUDA(cudaEventRecord(start, stream1));

    // Stream 1: H2D传输前半部分 + 计算
    CHECK_CUDA(cudaMemcpyAsync(d_data, h_data_in, chunk_size * sizeof(float),
                               cudaMemcpyHostToDevice, stream1));
    heavy_kernel<<<numBlocks, blockSize, 0, stream1>>>(d_data, chunk_size, 500);
    CHECK_CUDA(cudaMemcpyAsync(h_data_out, d_data, chunk_size * sizeof(float),
                               cudaMemcpyDeviceToHost, stream1));

    // Stream 2: H2D传输后半部分 + 计算
    CHECK_CUDA(cudaMemcpyAsync(d_data + chunk_size, h_data_in + chunk_size,
                               chunk_size * sizeof(float), cudaMemcpyHostToDevice, stream2));
    heavy_kernel<<<numBlocks, blockSize, 0, stream2>>>(d_data + chunk_size, chunk_size, 500);
    CHECK_CUDA(cudaMemcpyAsync(h_data_out + chunk_size, d_data + chunk_size,
                               chunk_size * sizeof(float), cudaMemcpyDeviceToHost, stream2));

    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));

    CHECK_CUDA(cudaEventRecord(stop, stream1));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float overlap_time;
    CHECK_CUDA(cudaEventElapsedTime(&overlap_time, start, stop));
    printf("    执行时间: %.3f ms\n", overlap_time);
    printf("    加速比:   %.2fx\n", serial_time / overlap_time);

    // 清理
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFreeHost(h_data_in));
    CHECK_CUDA(cudaFreeHost(h_data_out));
}

int main() {
    printf("========================================\n");
    printf("  并发核函数演示 - 第二十二章\n");
    printf("========================================\n\n");

    print_device_info();
    simple_concurrent_kernels();
    variable_size_kernels();
    resource_limit_analysis();
    register_smem_impact();
    compute_transfer_overlap();

    printf("\n========================================\n");
    printf("并发核函数要点:\n");
    printf("  1. 使用非阻塞流实现真正并发\n");
    printf("  2. 每个核函数需要独立的内存\n");
    printf("  3. 资源使用影响并发度\n");
    printf("  4. 可以实现计算与传输重叠\n");
    printf("========================================\n");
    printf("\n使用 nsys 可视化并发执行:\n");
    printf("  nsys profile --stats=true -o concurrent ./05_concurrent_kernels\n");
    printf("  nsys-ui concurrent.nsys-rep\n\n");

    return 0;
}