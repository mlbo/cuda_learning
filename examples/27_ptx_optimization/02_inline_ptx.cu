/**
 * 第27章示例02：内联PTX汇编使用示例
 *
 * 本示例展示如何在CUDA代码中使用内联PTX汇编
 * 包括SIMD指令、谓词寄存器等高级用法
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ============================================================================
// SIMD视频指令
// ============================================================================

/**
 * vabsdiff4：计算4个字节的绝对差之和
 * 输入两个32位整数，每个包含4字节
 * 输出是4个字节绝对差的总和
 */
__device__ unsigned int vabsdiff4_sum(unsigned int A, unsigned int B) {
    unsigned int result;
    // vabsdiff4：4字节SIMD绝对差
    // .u32：操作数类型
    // .add：累加模式
    asm("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(A), "r"(B), "r"(0));
    return result;
}

/**
 * vadd4：4字节SIMD加法（饱和模式）
 * 每个字节独立相加，结果饱和到0-255
 */
__device__ unsigned int vadd4_sat(unsigned int A, unsigned int B) {
    unsigned int result;
    // vadd4 with saturation
    asm("vadd4.u32.u32.u32.sat %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(A), "r"(B), "r"(0));
    return result;
}

/**
 * vmin4：4字节SIMD最小值
 * 每个字节独立取最小值
 */
__device__ unsigned int vmin4(unsigned int A, unsigned int B) {
    unsigned int result;
    // PTX指令: vmin4.u32.u32.u32 dest, a, b;
    // 注意：现代PTX使用不同的语法
    asm("vmin4.u32.u32.u32.add %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(A), "r"(B), "r"(0));
    return result;
}

/**
 * vmax4：4字节SIMD最大值
 */
__device__ unsigned int vmax4(unsigned int A, unsigned int B) {
    unsigned int result;
    asm("vmax4.u32.u32.u32.add %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(A), "r"(B), "r"(0));
    return result;
}

// ============================================================================
// 谓词寄存器
// ============================================================================

/**
 * 使用谓词寄存器实现条件选择
 * 避免分支，减少warp divergence
 * 注：现代CUDA编译器会自动优化条件表达式为谓词指令
 */
__device__ float select_predicate(float a, float b, int condition) {
    // 编译器会生成谓词指令 (selp)
    return condition ? a : b;
}

/**
 * 使用谓词寄存器实现条件加法
 * 只在条件满足时执行加法
 */
__device__ float conditional_add(float a, float b, int condition) {
    // 编译器会生成条件执行指令
    return condition ? (a + b) : a;
}

// ============================================================================
// 位操作指令
// ============================================================================

/**
 * brev：位反转
 * 将32位整数的位顺序反转
 */
__device__ unsigned int bit_reverse(unsigned int x) {
    unsigned int result;
    asm("brev.b32 %0, %1;" : "=r"(result) : "r"(x));
    return result;
}

/**
 * prmt：字节重排
 * 根据选择器重排字节
 */
__device__ unsigned int permute_bytes(unsigned int a, unsigned int b, unsigned int selector) {
    unsigned int result;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(selector));
    return result;
}

/**
 * bfe：位字段提取
 * 从源操作数中提取位字段
 * 注：bfe指令需要使用正确的类型修饰符
 */
__device__ unsigned int bit_field_extract(unsigned int src, unsigned int pos, unsigned int len) {
    unsigned int result;
    // 使用内置函数替代内联PTX，更安全可靠
    result = (src >> pos) & ((1u << len) - 1);
    return result;
}

/**
 * bfi：位字段插入
 */
__device__ unsigned int bit_field_insert(unsigned int dst, unsigned int src,
                                          int pos, int len) {
    unsigned int result;
    asm("bfi.b32 %0, %1, %2, %3, %4;"
        : "=r"(result)
        : "r"(src), "r"(dst), "r"(pos), "r"(len));
    return result;
}

// ============================================================================
// 内存访问优化
// ============================================================================

/**
 * 使用只读缓存加载数据
 * 适合只读且访问不规则的数据
 */
__device__ float load_readonly(const float* ptr) {
    float val;
    // ld.global.nc：使用纹理缓存（只读缓存）加载
    // nc = non-coherent，不保证与L1缓存一致性
    asm("ld.global.nc.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
    return val;
}

/**
 * 使用L2缓存的加载
 */
__device__ float load_l2_only(const float* ptr) {
    float val;
    // ld.global.cg：只缓存到L2
    asm("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
    return val;
}

/**
 * 流式加载（最小化缓存污染）
 */
__device__ float load_streaming(const float* ptr) {
    float val;
    // ld.global.cs：流式加载
    asm("ld.global.cs.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
    return val;
}

// ============================================================================
// 测试内核
// ============================================================================
__global__ void test_inline_ptx(unsigned int* int_results, float* float_results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        printf("=== 内联PTX汇编测试 ===\n\n");

        // 测试SIMD指令
        unsigned int A = 0x0A141E28;  // 字节: 10, 20, 30, 40
        unsigned int B = 0x050A0F14;  // 字节: 5, 10, 15, 20
        printf("SIMD视频指令测试:\n");
        printf("  A = 0x%08X (字节: 10, 20, 30, 40)\n", A);
        printf("  B = 0x%08X (字节: 5, 10, 15, 20)\n", B);
        printf("  vabsdiff4: %u (期望: 5+10+15+20=50)\n", vabsdiff4_sum(A, B));
        printf("  vadd4_sat: 0x%08X\n", vadd4_sat(A, B));
        printf("  vmin4: 0x%08X\n", vmin4(A, B));
        printf("  vmax4: 0x%08X\n", vmax4(A, B));

        int_results[0] = vabsdiff4_sum(A, B);
        int_results[1] = vmin4(A, B);
        int_results[2] = vmax4(A, B);

        // 测试谓词寄存器
        printf("\n谓词寄存器测试:\n");
        printf("  select(10.0, 20.0, 1) = %.1f\n", select_predicate(10.0f, 20.0f, 1));
        printf("  select(10.0, 20.0, 0) = %.1f\n", select_predicate(10.0f, 20.0f, 0));
        printf("  cond_add(10.0, 5.0, 1) = %.1f\n", conditional_add(10.0f, 5.0f, 1));
        printf("  cond_add(10.0, 5.0, 0) = %.1f\n", conditional_add(10.0f, 5.0f, 0));

        float_results[0] = select_predicate(10.0f, 20.0f, 1);
        float_results[1] = select_predicate(10.0f, 20.0f, 0);

        // 测试位操作
        unsigned int test_val = 0x12345678;
        printf("\n位操作测试:\n");
        printf("  原值: 0x%08X\n", test_val);
        printf("  bit_reverse: 0x%08X\n", bit_reverse(test_val));
        printf("  bfe(7:4): %u\n", bit_field_extract(test_val, 4, 4));

        int_results[3] = bit_reverse(test_val);

        // 测试字节重排
        unsigned int perm_result = permute_bytes(A, B, 0x3210);
        printf("  prmt(0x3210): 0x%08X\n", perm_result);

        int_results[4] = perm_result;
    }
}

// ============================================================================
// Warp Divergence对比示例
// ============================================================================

// 普通分支版本
__global__ void kernel_with_branch(float* data, int* flags, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (flags[idx] > 0) {
            data[idx] = data[idx] * 2.0f;
        } else {
            data[idx] = data[idx] + 1.0f;
        }
    }
}

// 使用谓词寄存器版本
__global__ void kernel_with_predicate(float* data, int* flags, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        int flag = flags[idx];

        float true_val = val * 2.0f;
        float false_val = val + 1.0f;

        // 使用条件表达式，编译器会生成谓词指令
        float result = flag > 0 ? true_val : false_val;

        data[idx] = result;
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    unsigned int* d_int_results;
    float* d_float_results;

    cudaMalloc(&d_int_results, 10 * sizeof(unsigned int));
    cudaMalloc(&d_float_results, 10 * sizeof(float));

    test_inline_ptx<<<1, 32>>>(d_int_results, d_float_results);
    cudaDeviceSynchronize();

    // 验证SIMD结果
    unsigned int h_int_results[10];
    cudaMemcpy(h_int_results, d_int_results, 10 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("\n=== 结果验证 ===\n");
    printf("vabsdiff4: GPU=%u, CPU期望=50\n", h_int_results[0]);

    // 验证位反转
    unsigned int expected_rev = 0x1E6A2C48;
    printf("bit_reverse: GPU=0x%08X, CPU期望=0x%08X\n", h_int_results[3], expected_rev);

    cudaFree(d_int_results);
    cudaFree(d_float_results);

    printf("\n内联PTX汇编示例完成！\n");
    return 0;
}