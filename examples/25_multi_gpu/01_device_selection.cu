/**
 * 第25章示例1：设备枚举与选择
 *
 * 演示内容：
 * 1. 枚举系统中的CUDA设备
 * 2. 获取设备属性信息
 * 3. 选择合适的设备
 * 4. 多设备并行执行
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/**
 * 枚举并打印所有设备信息
 */
void enumerate_devices() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    printf("\n========================================\n");
    printf("系统CUDA设备信息\n");
    printf("========================================\n");
    printf("发现 %d 个CUDA设备\n\n", deviceCount);

    if (deviceCount == 0) {
        printf("未发现CUDA设备！\n");
        return;
    }

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("----------------------------------------\n");
        printf("设备 %d: %s\n", dev, prop.name);
        printf("----------------------------------------\n");
        printf("  计算能力:          %d.%d\n", prop.major, prop.minor);
        printf("  全局显存:          %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  常量内存:          %zu KB\n", prop.totalConstMem / 1024);
        printf("  共享内存/Block:    %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  寄存器数量/Block:  %d\n", prop.regsPerBlock);
        printf("  Warp大小:          %d\n", prop.warpSize);
        printf("  最大线程/Block:    %d\n", prop.maxThreadsPerBlock);
        printf("  最大Block维度:     (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  最大Grid维度:      (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  SM数量:            %d\n", prop.multiProcessorCount);
        printf("  显存总线位宽:      %d bits\n", prop.memoryBusWidth);
        printf("  L2缓存大小:        %d KB\n", prop.l2CacheSize / 1024);
        printf("  最大线程/SM:       %d\n", prop.maxThreadsPerMultiProcessor);

        // 计算理论带宽 (使用固定参考值，因为memoryClockRate在新API中已移除)
        // HBM2E ~3.2 GHz, HBM3 ~4.0 GHz, GDDR6X ~2.1 GHz
        double bandwidth = 2.0 * prop.memoryBusWidth * 3200.0 * 1000.0 / 8.0 / 1e9;
        printf("  理论内存带宽(估计): %.1f GB/s\n", bandwidth);

        // P2P能力
        printf("  支持P2P:           %s\n", prop.managedMemory ? "Yes" : "No");
        printf("  支持统一内存:      %s\n", prop.managedMemory ? "Yes" : "No");
        printf("  支持并发内核:      %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  支持流优先级:      %s\n", prop.streamPrioritiesSupported ? "Yes" : "No");
        printf("\n");
    }
}

/**
 * 根据显存大小选择最佳设备
 */
int select_best_device_by_memory() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        return -1;
    }

    int bestDevice = 0;
    size_t maxMemory = 0;

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        if (prop.totalGlobalMem > maxMemory) {
            maxMemory = prop.totalGlobalMem;
            bestDevice = dev;
        }
    }

    return bestDevice;
}

/**
 * 根据计算能力选择最佳设备
 */
int select_best_device_by_compute() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        return -1;
    }

    int bestDevice = 0;
    int maxCompute = 0;

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        int compute = prop.major * 10 + prop.minor;
        if (compute > maxCompute) {
            maxCompute = compute;
            bestDevice = dev;
        }
    }

    return bestDevice;
}

/**
 * 打印P2P访问能力矩阵
 */
void print_p2p_matrix() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
        printf("需要至少2个GPU来测试P2P能力\n");
        return;
    }

    printf("\n========================================\n");
    printf("P2P访问能力矩阵\n");
    printf("========================================\n\n");

    // 打印表头
    printf("        ");
    for (int i = 0; i < deviceCount; i++) {
        printf("  GPU%d ", i);
    }
    printf("\n");

    for (int i = 0; i < deviceCount; i++) {
        printf("GPU%d    ", i);
        for (int j = 0; j < deviceCount; j++) {
            if (i == j) {
                printf("   -   ");
            } else {
                int canAccess;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess, i, j));
                printf("  %s  ", canAccess ? "Yes" : "No ");
            }
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * 简单向量加法核函数
 */
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * 在指定设备上执行向量加法
 */
void vector_add_on_device(int device, float* h_a, float* h_b, float* h_c, int n) {
    CHECK_CUDA(cudaSetDevice(device));

    size_t size = n * sizeof(float);

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 创建事件计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 执行核函数
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    CHECK_CUDA(cudaEventRecord(start));
    vector_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("设备 %d: 向量加法完成, 耗时 %.3f ms\n", device, ms);

    // 复制结果回主机
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // 清理
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

/**
 * 多GPU并行执行示例
 */
void multi_gpu_parallel_example() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
        printf("需要至少2个GPU来运行多GPU并行示例\n");
        return;
    }

    printf("\n========================================\n");
    printf("多GPU并行执行示例\n");
    printf("========================================\n\n");

    const int N = 1024 * 1024;  // 1M元素
    size_t size = N * sizeof(float);

    // 每个GPU分配的数据量
    int chunk_size = N / deviceCount;

    // 准备主机数据
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // 在每个GPU上并行执行
    printf("使用 %d 个GPU并行计算...\n", deviceCount);

    // 使用事件计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int dev = 0; dev < deviceCount; dev++) {
        int offset = dev * chunk_size;
        vector_add_on_device(dev, h_a.data() + offset, h_b.data() + offset,
                             h_c.data() + offset, chunk_size);
    }

    // 同步所有设备
    for (int dev = 0; dev < deviceCount; dev++) {
        CHECK_CUDA(cudaSetDevice(dev));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    printf("\n多GPU并行总耗时: %.3f ms\n", total_ms);

    // 验证结果
    bool correct = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("结果验证: %s\n", correct ? "正确" : "错误");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    printf("=============================================\n");
    printf("  第25章示例1：设备枚举与选择\n");
    printf("=============================================\n");

    // 1. 枚举设备
    enumerate_devices();

    // 2. P2P能力矩阵
    print_p2p_matrix();

    // 3. 设备选择示例
    printf("========================================\n");
    printf("设备选择示例\n");
    printf("========================================\n");

    int best_memory = select_best_device_by_memory();
    int best_compute = select_best_device_by_compute();

    printf("显存最大的设备: GPU %d\n", best_memory);
    printf("计算能力最高的设备: GPU %d\n", best_compute);

    // 4. 多GPU并行示例
    multi_gpu_parallel_example();

    printf("\n示例完成！\n");
    return 0;
}
