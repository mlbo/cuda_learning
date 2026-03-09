/**
 * =============================================================================
 * 第一章：Bank 结构演示 - 展示 32 个 Bank 的工作原理
 * =============================================================================
 *
 * 本示例演示共享内存 Bank 的基本结构：
 * - 32 个 Bank 的组织方式
 * - Bank ID 的计算方法
 * - 连续访问与 Bank 映射的关系
 *
 * 编译：nvcc -o 01_bank_structure 01_bank_structure.cu
 * 运行：./01_bank_structure
 *
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <stdio.h>

// =============================================================================
// 全局常量定义
// =============================================================================
#define NUM_BANKS 32          // Bank 数量
#define BANK_SIZE 4           // 每个 Bank 宽度（字节）

// =============================================================================
// 工具函数：计算 Bank ID
// =============================================================================
// Bank ID = (字节地址 / 4) % 32
// 对于 float 数组：Bank ID = (数组索引) % 32
// 因为每个 float 占 4 字节，正好是一个 Bank 宽度
__host__ __device__ int get_bank_id(int byte_address) {
    return (byte_address / BANK_SIZE) % NUM_BANKS;
}

// 对于 float 数组的简化版本
__host__ __device__ int get_bank_id_float(int index) {
    return index % NUM_BANKS;
}

// =============================================================================
// 核函数：演示 Bank 结构
// =============================================================================
__global__ void demonstrate_bank_structure(int* bank_ids, int size) {
    int tid = threadIdx.x;

    // 每个线程计算一个元素对应的 Bank ID
    if (tid < size) {
        bank_ids[tid] = get_bank_id_float(tid);
    }
}

// =============================================================================
// 核函数：演示连续访问（无 Bank Conflict）
// =============================================================================
__global__ void sequential_access_demo(float* output) {
    // 声明共享内存
    __shared__ float smem[128];

    int tid = threadIdx.x;

    // 初始化共享内存
    smem[tid] = (float)tid;
    __syncthreads();

    // 连续访问：每个线程访问不同 Bank，无冲突
    // Thread 0 -> smem[0] -> Bank 0
    // Thread 1 -> smem[1] -> Bank 1
    // Thread 2 -> smem[2] -> Bank 2
    // ...
    // Thread 31 -> smem[31] -> Bank 31
    // Thread 32 -> smem[32] -> Bank 0 (循环回来)
    float val = smem[tid];
    output[tid] = val;
}

// =============================================================================
// 核函数：演示 Bank ID 计算的详细映射
// =============================================================================
__global__ void bank_mapping_demo(int* bank_ids, int* addresses, int size) {
    int tid = threadIdx.x;

    if (tid < size) {
        addresses[tid] = tid * 4;  // 字节地址（每个 float 4 字节）
        bank_ids[tid] = get_bank_id(tid * 4);
    }
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("=================================================================\n");
    printf("第一章：Bank 结构演示\n");
    printf("=================================================================\n\n");

    // -------------------------------------------------------------------------
    // 第一部分：Bank 基本结构
    // -------------------------------------------------------------------------
    printf("【第一部分：Bank 基本结构】\n");
    printf("-----------------------------------------------------------------\n");
    printf("共享内存配置：\n");
    printf("  - Bank 数量：%d\n", NUM_BANKS);
    printf("  - 每个 Bank 宽度：%d 字节\n", BANK_SIZE);
    printf("  - 总 Bank 宽度：%d × %d = %d 字节/次访问\n\n",
           NUM_BANKS, BANK_SIZE, NUM_BANKS * BANK_SIZE);

    // -------------------------------------------------------------------------
    // 第二部分：Bank ID 映射演示
    // -------------------------------------------------------------------------
    printf("【第二部分：Bank ID 映射演示】\n");
    printf("-----------------------------------------------------------------\n");
    printf("对于 float 数组 smem[]，Bank ID 的映射关系：\n");
    printf("  Bank ID = (元素索引) %% %d\n\n", NUM_BANKS);

    // 在主机上演示映射关系
    printf("元素索引 -> 字节地址 -> Bank ID\n");
    printf("----------------------------------------\n");
    for (int i = 0; i < 40; i++) {
        int byte_addr = i * 4;
        int bank_id = get_bank_id(byte_addr);
        printf("  smem[%2d] -> 0x%04X    -> Bank %2d", i, byte_addr, bank_id);
        if (i >= 32) {
            printf("  (与 smem[%d] 同 Bank)", i - 32);
        }
        printf("\n");
    }
    printf("\n");

    // -------------------------------------------------------------------------
    // 第三部分：无冲突访问模式
    // -------------------------------------------------------------------------
    printf("【第三部分：无冲突访问模式】\n");
    printf("-----------------------------------------------------------------\n");
    printf("理想访问模式：连续访问\n");
    printf("  Thread 0  -> smem[0]  -> Bank 0\n");
    printf("  Thread 1  -> smem[1]  -> Bank 1\n");
    printf("  Thread 2  -> smem[2]  -> Bank 2\n");
    printf("  ...\n");
    printf("  Thread 31 -> smem[31] -> Bank 31\n");
    printf("\n");
    printf("结果：32 个线程访问 32 个不同 Bank，可并行完成！\n");
    printf("延迟：1 个时钟周期\n\n");

    // -------------------------------------------------------------------------
    // 第四部分：GPU 核函数验证
    // -------------------------------------------------------------------------
    printf("【第四部分：GPU 核函数验证】\n");
    printf("-----------------------------------------------------------------\n");

    // 分配设备内存
    int *d_bank_ids, *d_addresses;
    float *d_output;
    int size = 64;

    cudaMalloc(&d_bank_ids, size * sizeof(int));
    cudaMalloc(&d_addresses, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(float));

    // 执行核函数
    demonstrate_bank_structure<<<1, 64>>>(d_bank_ids, size);
    cudaDeviceSynchronize();

    // 复制结果回主机
    int h_bank_ids[64];
    cudaMemcpy(h_bank_ids, d_bank_ids, size * sizeof(int), cudaMemcpyDeviceToHost);

    printf("GPU 验证结果（前 64 个元素的 Bank ID）：\n");
    printf("  ");
    for (int i = 0; i < size; i++) {
        printf("%2d ", h_bank_ids[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n  ");
        }
    }
    printf("\n");

    // -------------------------------------------------------------------------
    // 第五部分：关键概念总结
    // -------------------------------------------------------------------------
    printf("【关键概念总结】\n");
    printf("-----------------------------------------------------------------\n");
    printf("1. Bank ID 计算公式：Bank ID = (字节地址 / 4) %% 32\n");
    printf("2. 对于 float 数组：Bank ID = 数组索引 %% 32\n");
    printf("3. 连续元素映射到连续 Bank（模 32 循环）\n");
    printf("4. 一个 Warp（32 线程）正好对应 32 个 Bank\n");
    printf("5. 无冲突访问：每个线程访问不同 Bank\n");
    printf("\n");

    // -------------------------------------------------------------------------
    // 第六部分：特殊访问模式
    // -------------------------------------------------------------------------
    printf("【第六部分：特殊访问模式分析】\n");
    printf("-----------------------------------------------------------------\n");
    printf("\n模式 1：所有线程访问同一地址（广播）\n");
    printf("  Thread 0-31 -> smem[0] -> Bank 0\n");
    printf("  结果：广播机制，无 Bank Conflict，1 个周期完成\n\n");

    printf("模式 2：跨步访问（stride = 2）\n");
    printf("  Thread 0  -> smem[0]  -> Bank 0\n");
    printf("  Thread 1  -> smem[2]  -> Bank 2\n");
    printf("  Thread 16 -> smem[32] -> Bank 0  <- 与 Thread 0 冲突！\n");
    printf("  结果：2-way Bank Conflict\n\n");

    printf("模式 3：跨步访问（stride = 4）\n");
    printf("  Thread 0  -> smem[0]  -> Bank 0\n");
    printf("  Thread 8  -> smem[32] -> Bank 0\n");
    printf("  Thread 16 -> smem[64] -> Bank 0\n");
    printf("  Thread 24 -> smem[96] -> Bank 0\n");
    printf("  结果：4-way Bank Conflict\n\n");

    // -------------------------------------------------------------------------
    // 清理资源
    // -------------------------------------------------------------------------
    cudaFree(d_bank_ids);
    cudaFree(d_addresses);
    cudaFree(d_output);

    printf("=================================================================\n");
    printf("演示完成！\n");
    printf("=================================================================\n");

    return 0;
}
