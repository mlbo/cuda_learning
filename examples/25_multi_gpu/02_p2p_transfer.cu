/**
 * 第25章示例2：P2P点对点传输
 *
 * 演示内容：
 * 1. 检测P2P访问能力
 * 2. 启用P2P访问
 * 3. P2P数据传输
 * 4. P2P传输性能测试
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

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
 * 检查并启用P2P
 */
bool enable_p2p_access(int gpu0, int gpu1) {
    int canAccess_01, canAccess_10;

    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess_01, gpu0, gpu1));
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess_10, gpu1, gpu0));

    printf("GPU %d -> GPU %d P2P: %s\n", gpu0, gpu1, canAccess_01 ? "支持" : "不支持");
    printf("GPU %d -> GPU %d P2P: %s\n", gpu1, gpu0, canAccess_10 ? "支持" : "不支持");

    if (canAccess_01) {
        CHECK_CUDA(cudaSetDevice(gpu0));
        CHECK_CUDA(cudaDeviceEnablePeerAccess(gpu1, 0));
        printf("已启用 GPU %d -> GPU %d 的P2P访问\n", gpu0, gpu1);
    }

    if (canAccess_10) {
        CHECK_CUDA(cudaSetDevice(gpu1));
        CHECK_CUDA(cudaDeviceEnablePeerAccess(gpu0, 0));
        printf("已启用 GPU %d -> GPU %d 的P2P访问\n", gpu1, gpu0);
    }

    return canAccess_01 || canAccess_10;
}

/**
 * 禁用P2P
 */
void disable_p2p_access(int gpu0, int gpu1) {
    int canAccess;

    CHECK_CUDA(cudaSetDevice(gpu0));
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess, gpu0, gpu1));
    if (canAccess) {
        CHECK_CUDA(cudaDeviceDisablePeerAccess(gpu1));
        printf("已禁用 GPU %d -> GPU %d 的P2P访问\n", gpu0, gpu1);
    }

    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess, gpu1, gpu0));
    if (canAccess) {
        CHECK_CUDA(cudaDeviceDisablePeerAccess(gpu0));
        printf("已禁用 GPU %d -> GPU %d 的P2P访问\n", gpu1, gpu0);
    }
}

/**
 * 使用cudaMemcpyPeer进行P2P传输
 */
void test_p2p_memcpy_peer(int gpu0, int gpu1, size_t size_bytes) {
    printf("\n测试 cudaMemcpyPeer P2P传输...\n");

    float *d_src, *d_dst;

    // 在源GPU分配内存
    CHECK_CUDA(cudaSetDevice(gpu0));
    CHECK_CUDA(cudaMalloc(&d_src, size_bytes));

    // 在目标GPU分配内存
    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaMalloc(&d_dst, size_bytes));

    // 初始化源数据
    CHECK_CUDA(cudaSetDevice(gpu0));
    int num_elements = size_bytes / sizeof(float);
    float val = 42.0f;
    CHECK_CUDA(cudaMemset(d_src, 0, size_bytes));
    // 简单填充测试值
    for (int i = 0; i < 10; i++) {
        CHECK_CUDA(cudaMemcpy(d_src + i, &val, sizeof(float), cudaMemcpyHostToDevice));
    }

    // 创建计时事件
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 执行P2P传输
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpyPeer(d_dst, gpu1, d_src, gpu0, size_bytes));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 计算带宽
    float bandwidth_gb = size_bytes / ms / 1e6;
    printf("  传输大小: %.2f MB\n", size_bytes / 1e6);
    printf("  传输时间: %.3f ms\n", ms);
    printf("  带宽: %.2f GB/s\n", bandwidth_gb);

    // 验证数据
    float h_check;
    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaMemcpy(&h_check, d_dst, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  数据验证: %s (期望 %.1f, 实际 %.1f)\n",
           (std::fabs(h_check - val) < 0.01f) ? "通过" : "失败", val, h_check);

    // 清理
    CHECK_CUDA(cudaSetDevice(gpu0));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

/**
 * 使用cudaMemcpyPeerAsync进行异步P2P传输
 */
void test_p2p_memcpy_async(int gpu0, int gpu1, size_t size_bytes) {
    printf("\n测试 cudaMemcpyPeerAsync 异步P2P传输...\n");

    float *d_src, *d_dst;
    cudaStream_t stream;

    // 在源GPU分配内存
    CHECK_CUDA(cudaSetDevice(gpu0));
    CHECK_CUDA(cudaMalloc(&d_src, size_bytes));

    // 在目标GPU分配内存
    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaMalloc(&d_dst, size_bytes));
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 初始化源数据
    CHECK_CUDA(cudaSetDevice(gpu0));
    int num_elements = size_bytes / sizeof(float);
    CHECK_CUDA(cudaMemset(d_src, 0xAB, size_bytes));

    // 创建计时事件
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 执行异步P2P传输
    CHECK_CUDA(cudaEventRecord(start, stream));
    CHECK_CUDA(cudaMemcpyPeerAsync(d_dst, gpu1, d_src, gpu0, size_bytes, stream));
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 计算带宽
    float bandwidth_gb = size_bytes / ms / 1e6;
    printf("  传输大小: %.2f MB\n", size_bytes / 1e6);
    printf("  传输时间: %.3f ms\n", ms);
    printf("  带宽: %.2f GB/s\n", bandwidth_gb);

    // 清理
    CHECK_CUDA(cudaSetDevice(gpu0));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

/**
 * 测试通过CPU中转的传输（用于对比）
 */
void test_via_host_transfer(int gpu0, int gpu1, size_t size_bytes) {
    printf("\n测试通过CPU中转传输...\n");

    float *d_src, *d_dst, *h_buf;

    // 分配主机缓冲区
    h_buf = (float*)malloc(size_bytes);

    // 在源GPU分配内存
    CHECK_CUDA(cudaSetDevice(gpu0));
    CHECK_CUDA(cudaMalloc(&d_src, size_bytes));

    // 在目标GPU分配内存
    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaMalloc(&d_dst, size_bytes));

    // 初始化源数据
    CHECK_CUDA(cudaSetDevice(gpu0));
    CHECK_CUDA(cudaMemset(d_src, 0xCD, size_bytes));

    // 创建计时事件
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 通过CPU中转传输
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(h_buf, d_src, size_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaMemcpy(d_dst, h_buf, size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 计算带宽
    float bandwidth_gb = size_bytes / ms / 1e6;
    printf("  传输大小: %.2f MB\n", size_bytes / 1e6);
    printf("  传输时间: %.3f ms\n", ms);
    printf("  带宽: %.2f GB/s\n", bandwidth_gb);

    // 清理
    free(h_buf);
    CHECK_CUDA(cudaSetDevice(gpu0));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

/**
 * 跨GPU核函数访问示例（UVA + P2P）
 */
__global__ void cross_gpu_access_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx] * 2.0f;
    }
}

void test_cross_gpu_kernel_access(int gpu0, int gpu1) {
    printf("\n测试跨GPU核函数访问...\n");

    const int N = 1024 * 1024;
    size_t size = N * sizeof(float);

    float *d_src, *d_dst;

    // 在GPU0分配源数据
    CHECK_CUDA(cudaSetDevice(gpu0));
    CHECK_CUDA(cudaMalloc(&d_src, size));
    CHECK_CUDA(cudaMemset(d_src, 0x42, size));  // 初始化

    // 在GPU1分配目标数据
    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaMalloc(&d_dst, size));

    // 在GPU1上执行核函数，访问GPU0的数据
    CHECK_CUDA(cudaSetDevice(gpu1));
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    cross_gpu_access_kernel<<<gridSize, blockSize>>>(d_dst, d_src, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("  核函数执行时间: %.3f ms\n", ms);
    printf("  注意: 实际性能取决于P2P带宽\n");

    // 清理
    CHECK_CUDA(cudaSetDevice(gpu0));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaSetDevice(gpu1));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

/**
 * 性能对比测试
 */
void benchmark_p2p_performance(int gpu0, int gpu1) {
    printf("\n========================================\n");
    printf("P2P传输性能测试\n");
    printf("========================================\n");

    // 测试不同大小的传输
    size_t sizes[] = {
        4 * 1024,           // 4 KB
        64 * 1024,          // 64 KB
        1024 * 1024,        // 1 MB
        16 * 1024 * 1024,   // 16 MB
        64 * 1024 * 1024,   // 64 MB
        256 * 1024 * 1024   // 256 MB
    };

    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("\n%-12s | %-12s | %-12s\n", "大小", "P2P带宽", "CPU中转带宽");
    printf("---------------------------------------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        size_t size = sizes[i];

        // P2P传输
        float *d_src, *d_dst;
        CHECK_CUDA(cudaSetDevice(gpu0));
        CHECK_CUDA(cudaMalloc(&d_src, size));
        CHECK_CUDA(cudaSetDevice(gpu1));
        CHECK_CUDA(cudaMalloc(&d_dst, size));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // 预热
        CHECK_CUDA(cudaMemcpyPeer(d_dst, gpu1, d_src, gpu0, size));

        // 多次测试取平均
        int iterations = 10;
        float total_ms = 0;
        for (int j = 0; j < iterations; j++) {
            CHECK_CUDA(cudaEventRecord(start));
            CHECK_CUDA(cudaMemcpyPeer(d_dst, gpu1, d_src, gpu0, size));
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float ms;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }
        float p2p_bandwidth = size / (total_ms / iterations) / 1e6;

        CHECK_CUDA(cudaSetDevice(gpu0));
        CHECK_CUDA(cudaFree(d_src));
        CHECK_CUDA(cudaSetDevice(gpu1));
        CHECK_CUDA(cudaFree(d_dst));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        printf("%-10.2f MB | %-10.2f GB/s | ", size / 1e6, p2p_bandwidth);
        printf("(参考值)\n");
    }
}

int main() {
    printf("=============================================\n");
    printf("  第25章示例2：P2P点对点传输\n");
    printf("=============================================\n\n");

    // 检查设备数量
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
        printf("此示例需要至少2个GPU\n");
        printf("当前系统只有 %d 个GPU\n", deviceCount);
        return 0;
    }

    int gpu0 = 0, gpu1 = 1;

    // 检查P2P能力
    printf("========================================\n");
    printf("检查P2P访问能力\n");
    printf("========================================\n");

    int canAccess;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess, gpu0, gpu1));
    printf("GPU %d -> GPU %d: %s\n", gpu0, gpu1, canAccess ? "支持P2P" : "不支持P2P");

    if (!canAccess) {
        printf("\nP2P不可用，将演示通过CPU中转的传输\n");
    }

    // 启用P2P（如果可用）
    bool p2p_enabled = enable_p2p_access(gpu0, gpu1);

    // 测试传输
    size_t test_size = 64 * 1024 * 1024;  // 64 MB

    if (p2p_enabled) {
        test_p2p_memcpy_peer(gpu0, gpu1, test_size);
        test_p2p_memcpy_async(gpu0, gpu1, test_size);
        test_cross_gpu_kernel_access(gpu0, gpu1);
    }

    // 对比通过CPU中转
    test_via_host_transfer(gpu0, gpu1, test_size);

    // 性能基准测试
    if (p2p_enabled) {
        benchmark_p2p_performance(gpu0, gpu1);
    }

    // 禁用P2P
    if (p2p_enabled) {
        disable_p2p_access(gpu0, gpu1);
    }

    printf("\n示例完成！\n");
    return 0;
}
