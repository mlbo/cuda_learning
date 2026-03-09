/**
 * 第27章示例04：查看和分析PTX代码
 *
 * 本示例展示如何查看和分析CUDA编译生成的PTX代码
 *
 * 编译命令：
 *   生成PTX: nvcc -ptx -arch=sm_80 04_inspect_ptx.cu -o kernel.ptx
 *   生成SASS: nvcc -cubin -arch=sm_80 04_inspect_ptx.cu -o kernel.cubin
 *   保留中间文件: nvcc -keep -arch=sm_80 04_inspect_ptx.cu
 *
 * 分析命令：
 *   cuobjdump -ptx kernel.exe  # 查看PTX
 *   cuobjdump -sass kernel.exe # 查看SASS
 *   nvdisasm kernel.cubin      # 反汇编
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ============================================================================
// 示例内核：展示不同的PTX代码模式
// ============================================================================

/**
 * 简单向量加法
 * 对比PTX代码可以看到编译器如何处理基本运算
 */
__global__ void simple_add(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * 使用__restrict__的版本
 * 对比PTX代码可以看到__restrict__如何影响内存访问优化
 */
__global__ void simple_add_restrict(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * 循环展开示例
 * 对比PTX代码可以看到# pragma unroll如何影响指令生成
 */
__global__ void unrolled_loop(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 展开的循环
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int data_idx = idx * 4 + i;
        if (data_idx < N) {
            data[data_idx] = data[data_idx] * 2.0f;
        }
    }
}

/**
 * 不展开的循环
 */
__global__ void normal_loop(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 不展开的循环
    #pragma unroll 1
    for (int i = 0; i < 4; i++) {
        int data_idx = idx * 4 + i;
        if (data_idx < N) {
            data[data_idx] = data[data_idx] * 2.0f;
        }
    }
}

/**
 * 条件分支示例
 * 查看PTX代码中的bra指令和谓词寄存器使用
 */
__global__ void with_branch(float* data, int* flags, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (flags[idx] > 0) {
            data[idx] *= 2.0f;
        } else {
            data[idx] += 1.0f;
        }
    }
}

/**
 * 使用谓词寄存器的版本
 * 对比PTX代码可以看到selp指令的使用
 */
__global__ void with_predicate(float* data, int* flags, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        float result;

        // 使用谓词寄存器进行条件选择
        // 注意：这里用普通C++代码，编译器会生成优化的PTX
        bool pred = flags[idx] > 0;
        result = pred ? (val * 2.0f) : (val + 1.0f);

        data[idx] = result;
    }
}

/**
 * 共享内存使用示例
 * 查看PTX代码中的.shared空间访问
 */
__global__ void shared_memory_example(float* input, float* output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载到共享内存
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // 规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 写回
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * 使用内联PTX的内核
 */
__global__ void inline_ptx_example(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 使用PTX进行加载和存储
        float val;
        asm("ld.global.nc.f32 %0, [%1];" : "=f"(val) : "l"(&data[idx]));

        float result;
        asm("add.f32 %0, %1, %2;" : "=f"(result) : "f"(val), "f"(1.0f));

        asm("st.global.f32 [%0], %1;" :: "l"(&data[idx]), "f"(result));
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== PTX代码分析示例 ===\n\n");

    printf("本示例展示如何查看和分析PTX代码\n\n");

    printf("=== 生成PTX文件的方法 ===\n");
    printf("1. 使用 nvcc -ptx:\n");
    printf("   nvcc -ptx -arch=sm_80 04_inspect_ptx.cu -o kernel.ptx\n\n");

    printf("2. 保留中间文件:\n");
    printf("   nvcc -keep -arch=sm_80 04_inspect_ptx.cu\n");
    printf("   这会生成 .ptx 和 .cubin 文件\n\n");

    printf("3. 生成cubin文件:\n");
    printf("   nvcc -cubin -arch=sm_80 04_inspect_ptx.cu -o kernel.cubin\n\n");

    printf("=== 查看PTX/SASS代码的方法 ===\n");
    printf("1. 使用 cuobjdump:\n");
    printf("   cuobjdump -ptx kernel.ptx    # 查看PTX\n");
    printf("   cuobjdump -sass kernel.exe   # 查看SASS\n\n");

    printf("2. 使用 nvdisasm:\n");
    printf("   nvdisasm -b SM_80 kernel.cubin\n\n");

    printf("3. 使用 Nsight Compute:\n");
    printf("   ncu --set basic -k simple_add ./your_program\n\n");

    printf("=== PTX代码分析要点 ===\n");
    printf("1. 指令类型:\n");
    printf("   - 算术指令: add.f32, mul.f32, fma.rn.f32\n");
    printf("   - 内存指令: ld.global, st.global\n");
    printf("   - 控制流: bra, @pred bra, selp\n\n");

    printf("2. 状态空间:\n");
    printf("   - .reg: 寄存器\n");
    printf("   - .global: 全局内存\n");
    printf("   - .shared: 共享内存\n");
    printf("   - .const: 常量内存\n\n");

    printf("3. 谓词寄存器:\n");
    printf("   - setp: 设置谓词\n");
    printf("   - @pred: 条件执行\n");
    printf("   - selp: 条件选择\n\n");

    printf("4. 优化提示:\n");
    printf("   - 比较 __restrict__ 和普通指针的PTX\n");
    printf("   - 比较展开和未展开循环的PTX\n");
    printf("   - 比较分支和谓词的PTX\n\n");

    // 简单运行以确保内核被编译
    float* d_data;
    int* d_flags;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    cudaMalloc(&d_flags, 1024 * sizeof(int));

    simple_add<<<1, 256>>>(d_data, d_data, d_data, 1024);
    simple_add_restrict<<<1, 256>>>(d_data, d_data, d_data, 1024);
    unrolled_loop<<<1, 256>>>(d_data, 1024);
    normal_loop<<<1, 256>>>(d_data, 1024);
    with_branch<<<1, 256>>>(d_data, d_flags, 1024);
    with_predicate<<<1, 256>>>(d_data, d_flags, 1024);
    shared_memory_example<<<4, 256>>>(d_data, d_data, 1024);
    inline_ptx_example<<<1, 256>>>(d_data, 1024);

    cudaDeviceSynchronize();

    cudaFree(d_data);
    cudaFree(d_flags);

    printf("PTX代码分析示例完成！\n");
    printf("请使用上述命令生成并查看PTX代码进行学习。\n");

    return 0;
}