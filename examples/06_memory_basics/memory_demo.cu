/*
 * memory_demo.cu - 第六章：CUDA 内存管理基础
 *
 * 本示例演示 CUDA 内存管理的基本操作：
 * 1. cudaMalloc - 在 GPU 上分配内存
 * 2. cudaMemcpy - 在 CPU 和 GPU 之间传输数据
 * 3. cudaFree - 释放 GPU 内存
 * 4. 错误处理 - 使用 CUDA_CHECK 宏进行错误检查
 *
 * 这是 CUDA 编程中最基础也是最重要的操作
 *
 * 编译方法：
 *   使用 CMake: mkdir build && cd build && cmake .. && make
 *   或直接编译: nvcc -arch=sm_80 memory_demo.cu -o memory_demo
 *
 * 适合：想要学习 GPU 内存管理的开发者
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// ==================== 错误检查宏 ====================

/*
 * CUDA_CHECK 宏说明：
 * ==================
 *
 * 这个宏用于检查 CUDA 函数调用是否成功。
 * 如果调用失败，会打印详细的错误信息并退出程序。
 *
 * 使用方法：
 * CUDA_CHECK(cudaMalloc(&d_ptr, size));
 * CUDA_CHECK(cudaMemcpy(dst, src, size, kind));
 *
 * 为什么需要错误检查？
 * 1. CUDA 函数可能因各种原因失败（内存不足、设备错误等）
 * 2. 默认情况下 CUDA 不会抛出异常
 * 3. 及早发现错误可以避免更难调试的问题
 */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "\n[CUDA 错误] 文件: %s, 行号: %d\n",              \
                    __FILE__, __LINE__);                                        \
            fprintf(stderr, "错误码: %d\n", err);                               \
            fprintf(stderr, "错误信息: %s\n", cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

// ==================== 内核函数定义 ====================

/*
 * 向量加法内核
 * =============
 *
 * 参数：
 * - a: 第一个输入数组（在 GPU 内存中）
 * - b: 第二个输入数组（在 GPU 内存中）
 * - c: 输出数组（在 GPU 内存中），存储 a + b 的结果
 * - n: 数组长度
 *
 * 每个线程处理数组中的一个元素
 */
__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n)
{
    /*
     * 计算当前线程负责处理的元素索引
     * 每个线程处理数组中的一个元素
     *
     * 索引计算公式：
     * idx = blockIdx.x * blockDim.x + threadIdx.x
     *
     * 解释：
     * - blockIdx.x: 当前块的索引（第几个块）
     * - blockDim.x: 每个块有多少线程
     * - threadIdx.x: 当前线程在块内的索引
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * 边界检查非常重要！
     * ==================
     *
     * 为什么需要边界检查？
     * - 启动的线程数可能大于数组长度
     * - 如果不检查，线程会访问非法内存，导致程序崩溃或结果错误
     *
     * 例如：数组有 100 个元素，启动了 128 个线程
     * 索引 100-127 的线程不应该访问数组
     */
    if (idx < n)
    {
        // 执行加法运算
        // 注意：这里的 a, b, c 都在 GPU 内存中
        c[idx] = a[idx] + b[idx];
    }
}

// ==================== 辅助函数 ====================

/*
 * 初始化数组
 * 在 CPU 上执行
 */
void init_array(float *arr, int n, float value)
{
    for (int i = 0; i < n; i++)
    {
        arr[i] = value;
    }
}

/*
 * 打印数组的一部分
 * 在 CPU 上执行
 */
void print_array(const char *name, float *arr, int n, int max_print)
{
    printf("%s = [", name);
    int print_count = (n < max_print) ? n : max_print;
    for (int i = 0; i < print_count; i++)
    {
        printf("%.1f", arr[i]);
        if (i < print_count - 1) printf(", ");
    }
    if (n > max_print) printf(", ...");
    printf("] (共 %d 个元素)\n", n);
}

/*
 * 验证计算结果
 * 在 CPU 上执行
 */
bool verify_result(float *c, int n, float expected)
{
    for (int i = 0; i < n; i++)
    {
        // 使用浮点数比较的容差
        if (std::fabs(c[i] - expected) > 1e-5)
        {
            printf("验证失败: c[%d] = %.2f, 期望 %.2f\n", i, c[i], expected);
            return false;
        }
    }
    return true;
}

// ==================== 主函数 ====================

int main()
{
    printf("=========================================================\n");
    printf("   第六章：CUDA 内存管理基础\n");
    printf("=========================================================\n\n");

    // ==================== 配置参数 ====================
    const int N = 1024;  // 数组长度
    const size_t bytes = N * sizeof(float);

    printf("数组配置:\n");
    printf("  元素数量: %d\n", N);
    printf("  每个数组大小: %.2f KB\n", bytes / 1024.0);
    printf("\n");

    // ==================== 第一步：分配主机内存 ====================
    printf("【步骤 1】分配主机（CPU）内存\n");
    printf("---------------------------------------------------------\n");

    /*
     * 使用 malloc 在 CPU 内存中分配空间
     * 这是标准的 C 语言内存分配
     */
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);  // 用于存储结果

    if (h_a == NULL || h_b == NULL || h_c == NULL)
    {
        fprintf(stderr, "主机内存分配失败！\n");
        return EXIT_FAILURE;
    }

    // 初始化输入数据
    // h_a 数组每个元素设为 1.0
    // h_b 数组每个元素设为 2.0
    // 期望结果：h_c 数组每个元素应该是 3.0
    init_array(h_a, N, 1.0f);
    init_array(h_b, N, 2.0f);

    print_array("h_a", h_a, N, 5);
    print_array("h_b", h_b, N, 5);
    printf("主机内存分配完成\n\n");

    // ==================== 第二步：分配设备内存 ====================
    printf("【步骤 2】分配设备（GPU）内存\n");
    printf("---------------------------------------------------------\n");

    /*
     * cudaMalloc 说明：
     * ================
     *
     * 函数原型：
     * cudaError_t cudaMalloc(void **devPtr, size_t size);
     *
     * 参数：
     * - devPtr: 指向 GPU 内存指针的指针（二级指针）
     * - size: 要分配的字节数
     *
     * 返回值：
     * - cudaSuccess 表示成功
     * - 其他值表示错误（如 cudaErrorMemoryAllocation）
     *
     * 重要特点：
     * 1. cudaMalloc 分配的是 GPU 上的全局内存（Global Memory）
     * 2. 分配的内存需要用 cudaFree 释放
     * 3. 分配的内存地址对齐到 256 字节边界
     * 4. GPU 内存有独立的地址空间，不能在 CPU 代码中直接访问
     * 5. 分配的大小受 GPU 显存限制
     */
    float *d_a = NULL;  // GPU 上的数组 a
    float *d_b = NULL;  // GPU 上的数组 b
    float *d_c = NULL;  // GPU 上的数组 c（结果）

    // 分配 GPU 内存
    // 注意：我们传递的是指针的地址 (&d_a)，而不是指针本身
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    printf("GPU 内存分配成功:\n");
    printf("  d_a: %p\n", (void*)d_a);
    printf("  d_b: %p\n", (void*)d_b);
    printf("  d_c: %p\n", (void*)d_c);
    printf("  总共分配: %.2f KB\n", 3.0 * bytes / 1024.0);
    printf("\n");

    // ==================== 第三步：传输数据到 GPU ====================
    printf("【步骤 3】传输数据到 GPU（Host -> Device）\n");
    printf("---------------------------------------------------------\n");

    /*
     * cudaMemcpy 说明：
     * ================
     *
     * 函数原型：
     * cudaError_t cudaMemcpy(void *dst, const void *src,
     *                        size_t count, cudaMemcpyKind kind);
     *
     * 参数：
     * - dst: 目标地址
     * - src: 源地址
     * - count: 要复制的字节数
     * - kind: 复制方向
     *
     * cudaMemcpyKind 枚举值：
     * - cudaMemcpyHostToDevice: CPU -> GPU (H2D)
     * - cudaMemcpyDeviceToHost: GPU -> CPU (D2H)
     * - cudaMemcpyDeviceToDevice: GPU -> GPU (D2D)
     * - cudaMemcpyHostToHost: CPU -> CPU (H2H)
     *
     * 重要特点：
     * 1. 这是一个同步操作，会阻塞 CPU 直到复制完成
     * 2. 对于大量数据，可以考虑使用 cudaMemcpyAsync 进行异步复制
     * 3. 复制速度受 PCIe 带宽限制
     * 4. 数据传输是 CUDA 程序的主要性能瓶颈之一
     */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    printf("数据传输完成:\n");
    printf("  h_a -> d_a (%zu 字节)\n", bytes);
    printf("  h_b -> d_b (%zu 字节)\n", bytes);
    printf("\n");

    // ==================== 第四步：执行内核 ====================
    printf("【步骤 4】在 GPU 上执行向量加法内核\n");
    printf("---------------------------------------------------------\n");

    /*
     * 内核启动配置说明：
     * ==================
     *
     * 我们需要确定启动多少个线程来处理所有数据
     *
     * 每个块的最大线程数：1024（大多数现代 GPU）
     *
     * 计算需要的块数：
     * blocks = (N + threads_per_block - 1) / threads_per_block
     *
     * 这是向上取整的标准技巧
     */
    int threads_per_block = 256;  // 每块 256 个线程
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    printf("内核配置:\n");
    printf("  每块线程数: %d\n", threads_per_block);
    printf("  块数量: %d\n", blocks);
    printf("  总线程数: %d\n", blocks * threads_per_block);
    printf("  实际需要的线程数: %d\n", N);
    printf("\n");

    /*
     * 内核启动语法：
     * kernel<<<grid_size, block_size>>>(arguments...)
     */
    vector_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);

    /*
     * 内核启动后的错误检查：
     * =====================
     *
     * cudaGetLastError(): 获取最后一次 CUDA 错误
     * 这个函数可以检测内核启动是否成功
     *
     * 注意：内核启动是异步的，这个函数只能检测启动错误
     * 要检测执行错误，需要使用 cudaDeviceSynchronize()
     */
    CUDA_CHECK(cudaGetLastError());

    /*
     * cudaDeviceSynchronize(): 等待 GPU 完成所有任务
     * 这会阻塞 CPU，直到 GPU 完成之前的所有操作
     */
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("内核执行完成\n\n");

    // ==================== 第五步：传输结果回 CPU ====================
    printf("【步骤 5】传输结果回 CPU（Device -> Host）\n");
    printf("---------------------------------------------------------\n");

    // 将 d_c 复制到 h_c（GPU -> CPU）
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    print_array("h_c (结果)", h_c, N, 5);
    printf("结果传输完成\n\n");

    // ==================== 第六步：验证结果 ====================
    printf("【步骤 6】验证计算结果\n");
    printf("---------------------------------------------------------\n");

    // 期望结果：1.0 + 2.0 = 3.0
    bool correct = verify_result(h_c, N, 3.0f);

    if (correct)
    {
        printf("验证通过！所有结果正确。\n");
        printf("  输入: a = 1.0, b = 2.0\n");
        printf("  输出: c = a + b = 3.0\n\n");
    }
    else
    {
        printf("验证失败！请检查代码。\n\n");
    }

    // ==================== 第七步：释放内存 ====================
    printf("【步骤 7】释放内存\n");
    printf("---------------------------------------------------------\n");

    /*
     * cudaFree 说明：
     * ==============
     *
     * 函数原型：
     * cudaError_t cudaFree(void *devPtr);
     *
     * 参数：
     * - devPtr: 要释放的 GPU 内存地址
     *
     * 注意事项：
     * 1. 只能释放由 cudaMalloc 分配的内存
     * 2. 传入 NULL 是安全的（不会做任何操作）
     * 3. 释放后指针不会自动置 NULL，建议手动置 NULL
     * 4. 重复释放同一内存会导致错误
     */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // 释放 CPU 内存
    free(h_a);
    free(h_b);
    free(h_c);

    printf("GPU 内存释放完成\n");
    printf("CPU 内存释放完成\n\n");

    // ==================== 总结 ====================
    printf("=========================================================\n");
    printf("          CUDA 内存管理流程总结\n");
    printf("=========================================================\n\n");

    printf("完整的数据流程:\n\n");
    printf("  ┌─────────────┐\n");
    printf("  │  CPU 内存    │\n");
    printf("  │  (h_a, h_b) │\n");
    printf("  └──────┬──────┘\n");
    printf("         │ cudaMemcpy(H2D)\n");
    printf("         ▼\n");
    printf("  ┌─────────────┐\n");
    printf("  │  GPU 内存    │\n");
    printf("  │  (d_a, d_b) │\n");
    printf("  └──────┬──────┘\n");
    printf("         │ kernel 执行\n");
    printf("         ▼\n");
    printf("  ┌─────────────┐\n");
    printf("  │  GPU 内存    │\n");
    printf("  │  (d_c)      │\n");
    printf("  └──────┬──────┘\n");
    printf("         │ cudaMemcpy(D2H)\n");
    printf("         ▼\n");
    printf("  ┌─────────────┐\n");
    printf("  │  CPU 内存    │\n");
    printf("  │  (h_c)      │\n");
    printf("  └─────────────┘\n\n");

    printf("关键函数:\n");
    printf("  1. cudaMalloc:   在 GPU 上分配内存\n");
    printf("  2. cudaMemcpy:    在 CPU 和 GPU 之间传输数据\n");
    printf("  3. cudaFree:      释放 GPU 内存\n\n");

    printf("内存管理最佳实践:\n");
    printf("  1. 尽量减少 CPU-GPU 数据传输次数\n");
    printf("  2. 传输大块数据比多次传输小块数据更高效\n");
    printf("  3. 使用 CUDA_CHECK 宏检查所有 CUDA 调用\n");
    printf("  4. 及时释放不再使用的 GPU 内存\n");
    printf("  5. 考虑使用 cudaMallocManaged 统一内存管理\n\n");

    return 0;
}
