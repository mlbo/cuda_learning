/**
 * 第八章: CUDA 性能分析 (Profiling)
 *
 * 学习目标:
 *   1. 掌握 CUDA 事件 (Events) 计时方法
 *   2. 了解 Nsight Systems (nsys) 系统级分析
 *   3. 了解 Nsight Compute (ncu) 内核级分析
 *   4. 理解性能瓶颈定位方法
 *
 * 编译方法:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
 *   make
 *
 * 或者直接使用 nvcc:
 *   nvcc -arch=sm_80 profiling_demo.cu -o profiling_demo
 *
 * 运行:
 *   ./profiling_demo
 *
 * 使用 Nsight Systems 分析:
 *   nsys profile --stats=true ./profiling_demo
 *   nsys-ui report.nsys-rep
 *
 * 使用 Nsight Compute 分析:
 *   ncu --set full -o report ./profiling_demo
 *   ncu-ui report.ncu-rep
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// ==================== 错误检查宏 ====================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA 错误 at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

// ==================== 核函数定义 ====================

/**
 * 简单向量加法核函数 - 用于演示基本计时
 *
 * 参数:
 *   a, b: 输入向量
 *   c: 输出向量
 *   n: 元素数量
 */
__global__ void vector_add(float* a, float* b, float* c, int n) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * 使用 Grid-Strided Loop 的向量加法
 * 适用于数据量远大于线程数的情况
 */
__global__ void vector_add_grid_stride(float* a, float* b, float* c, int n) {
    // Grid-Strided Loop: 每个线程处理多个元素
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * 使用 float4 向量化的向量加法
 * 每个线程处理 4 个元素，提高访存效率
 */
__global__ void vector_add_float4(float4* a, float4* b, float4* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;  // float4 元素数量

    if (idx < n4) {
        // 向量化加载 - 128位对齐
        float4 va = a[idx];
        float4 vb = b[idx];

        // 向量化计算
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        // 向量化存储
        c[idx] = vc;
    }
}

/**
 * 计算密集型核函数 - 用于对比访存密集型
 * 每个元素进行多次计算
 */
__global__ void compute_intensive(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float val = a[idx];
        float sum = 0.0f;

        // 进行多次计算增加计算强度
        // 10 次 FMA (Fused Multiply-Add) 操作
        #pragma unroll
        for (int i = 0; i < 10; i++) {
            sum += val * val + b[idx];
        }

        c[idx] = sum;
    }
}

// ==================== 计时辅助函数 ====================

/**
 * 使用 CUDA Events 进行精确计时
 *
 * CUDA Events 的优势:
 *   1. GPU 端计时，不受 CPU-GPU 同步影响
 *   2. 精度可达微秒级
 *   3. 可以测量异步操作的时间
 *
 * 参数:
 *   kernel: 核函数指针
 *   grid, block: 执行配置
 *   iterations: 重复次数
 *   args: 核函数参数
 *
 * 返回:
 *   平均执行时间 (毫秒)
 */
template<typename KernelFunc, typename... Args>
float benchmark_with_events(KernelFunc kernel,
                            dim3 grid, dim3 block,
                            int iterations,
                            Args... args) {
    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热 - 确保所有资源已分配
    kernel<<<grid, block>>>(args...);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 开始计时
    CUDA_CHECK(cudaEventRecord(start));

    // 多次执行核函数
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(args...);
    }

    // 结束计时
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 计算经过的时间
    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 返回平均时间
    return total_ms / iterations;
}

// ==================== 主函数 ====================

int main(int argc, char** argv) {
    // 设置数据量 (默认 16M 元素)
    int n = 1 << 24;
    if (argc > 1) {
        n = 1 << atoi(argv[1]);
    }

    size_t bytes = n * sizeof(float);

    printf("========================================\n");
    printf("CUDA 性能分析演示\n");
    printf("========================================\n");
    printf("元素数量: %d (%.2f MB)\n", n, bytes / 1024.0 / 1024.0);
    printf("线程块大小: 256\n\n");

    // -------------------- 查询设备信息 --------------------
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU 信息:\n");
    printf("  型号: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM 数量: %d\n", prop.multiProcessorCount);
    // 获取内存时钟频率 (CUDA 13.0+ 需要使用 cudaDeviceGetAttribute)
    int mem_clock_khz = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    printf("  内存带宽: %.2f GB/s (理论峰值)\n",
           (float)prop.memoryBusWidth * 2 * mem_clock_khz * 2 / 8 / 1e6);
    printf("\n");

    // -------------------- 分配内存 --------------------
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // 初始化主机数据
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // -------------------- 数据传输计时 --------------------
    printf("数据传输性能:\n");
    printf("----------------------------------------\n");

    cudaEvent_t h2d_start, h2d_stop, d2h_start, d2h_stop;
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));

    // Host to Device
    CUDA_CHECK(cudaEventRecord(h2d_start));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2d_stop));
    CUDA_CHECK(cudaEventSynchronize(h2d_stop));

    float h2d_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));
    float h2d_bandwidth = (2.0 * bytes / h2d_ms) / 1e6;  // GB/s
    printf("  H2D 传输: %.3f ms (%.1f GB/s)\n", h2d_ms, h2d_bandwidth);

    // Device to Host
    CUDA_CHECK(cudaEventRecord(d2h_start));
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(d2h_stop));

    float d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));
    float d2h_bandwidth = (bytes / d2h_ms) / 1e6;  // GB/s
    printf("  D2H 传输: %.3f ms (%.1f GB/s)\n", d2h_ms, d2h_bandwidth);
    printf("----------------------------------------\n\n");

    // -------------------- 核函数性能测试 --------------------
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    int iterations = 100;

    printf("核函数性能对比:\n");
    printf("-----------------------------------------------------------------\n");
    printf("%-25s %12s %12s %12s\n", "核函数", "时间(ms)", "带宽(GB/s)", "加速比");
    printf("-----------------------------------------------------------------\n");

    // 1. 基础向量加法
    {
        float ms = benchmark_with_events(vector_add,
                                         dim3(gridSize), dim3(blockSize),
                                         iterations,
                                         d_a, d_b, d_c, n);

        float bandwidth = (3.0 * bytes / ms) / 1e6;  // 2读+1写
        printf("%-25s %12.4f %12.1f %12s\n",
               "vector_add (基础)", ms, bandwidth, "1.00x");
    }

    // 2. Grid-Strided Loop 版本
    {
        float ms = benchmark_with_events(vector_add_grid_stride,
                                         dim3(gridSize), dim3(blockSize),
                                         iterations,
                                         d_a, d_b, d_c, n);

        float bandwidth = (3.0 * bytes / ms) / 1e6;
        printf("%-25s %12.4f %12.1f %12.2fx\n",
               "grid_stride", ms, bandwidth, 1.0f);
    }

    // 3. float4 向量化版本
    {
        int n4 = n / 4;
        int gridSize4 = (n4 + blockSize - 1) / blockSize;

        float ms = benchmark_with_events(vector_add_float4,
                                         dim3(gridSize4), dim3(blockSize),
                                         iterations,
                                         (float4*)d_a, (float4*)d_b, (float4*)d_c, n);

        float bandwidth = (3.0 * bytes / ms) / 1e6;
        printf("%-25s %12.4f %12.1f %12.2fx\n",
               "float4 向量化", ms, bandwidth, 1.0f);
    }

    // 4. 计算密集型
    {
        float ms = benchmark_with_events(compute_intensive,
                                         dim3(gridSize), dim3(blockSize),
                                         iterations,
                                         d_a, d_b, d_c, n);

        // 计算密度更高: 20 FLOP/元素, 3次访存
        float flops = 20.0 * n;  // 每元素 20 次浮点运算
        float gflops = flops / (ms * 1e-3) / 1e9;
        float bandwidth = (3.0 * bytes / ms) / 1e6;
        printf("%-25s %12.4f %12.1f %12.2fx\n",
               "compute_intensive", ms, bandwidth, 1.0f);
        printf("  -> 性能: %.1f GFLOPS, AI: %.2f\n", gflops, flops / (3.0 * bytes));
    }

    printf("-----------------------------------------------------------------\n\n");

    // -------------------- 验证结果 --------------------
    printf("验证结果...\n");
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < n && errors < 5; i++) {
        if (std::fabs(h_c[i] - 30.0f) > 1e-3) {  // compute_intensive 的结果
            errors++;
        }
    }
    printf("验证: %s\n", errors == 0 ? "通过" : "失败");

    // -------------------- 清理资源 --------------------
    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);

    // -------------------- 分析工具说明 --------------------
    printf("\n========================================\n");
    printf("性能分析工具使用说明\n");
    printf("========================================\n");
    printf("\n1. Nsight Systems (nsys) - 系统级分析:\n");
    printf("   用途: 查看整体时间线、API调用、内核执行顺序\n");
    printf("   命令: nsys profile --stats=true ./profiling_demo\n");
    printf("   查看: nsys-ui report.nsys-rep\n");
    printf("\n2. Nsight Compute (ncu) - 内核级分析:\n");
    printf("   用途: 详细分析单个内核的性能指标\n");
    printf("   命令: ncu --set full -o report ./profiling_demo\n");
    printf("   查看: ncu-ui report.ncu-rep\n");
    printf("\n3. 关键指标:\n");
    printf("   - 内存吞吐量: 是否接近带宽峰值?\n");
    printf("   - 计算吞吐量: 是否接近算力峰值?\n");
    printf("   - Warp 执行效率: 是否有分支发散?\n");
    printf("   - 内存访问模式: 是否合并访问?\n");
    printf("========================================\n");

    return 0;
}
