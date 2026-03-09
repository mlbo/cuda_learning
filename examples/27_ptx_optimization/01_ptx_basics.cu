/**
 * 第27章示例01：PTX基础语法演示
 *
 * 本示例展示PTX的基本语法和常用指令
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ============================================================================
// 使用PTX获取线程索引
// ============================================================================
__device__ int get_tid_x() {
    int tid;
    // mov指令：将特殊寄存器%tid.x的值移动到通用寄存器
    asm("mov.u32 %0, %tid.x;" : "=r"(tid));
    return tid;
}

__device__ int get_ctaid_x() {
    int ctaid;
    // 获取block索引
    asm("mov.u32 %0, %ctaid.x;" : "=r"(ctaid));
    return ctaid;
}

__device__ int get_ntid_x() {
    int ntid;
    // 获取block维度
    asm("mov.u32 %0, %ntid.x;" : "=r"(ntid));
    return ntid;
}

// ============================================================================
// 基本算术运算
// ============================================================================
__device__ float ptx_add(float a, float b) {
    float result;
    // add.f32：32位浮点加法
    asm("add.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ float ptx_mul(float a, float b) {
    float result;
    // mul.f32：32位浮点乘法
    asm("mul.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ float ptx_fma(float a, float b, float c) {
    float result;
    // fma.rn.f32：融合乘加，rn表示舍入到最近偶数
    // result = a * b + c
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

__device__ int ptx_add_int(int a, int b) {
    int result;
    // add.s32：32位有符号整数加法
    asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
    return result;
}

// ============================================================================
// 比较和选择操作
// ============================================================================
__device__ float ptx_min(float a, float b) {
    float result;
    // min.f32：浮点最小值
    asm("min.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ float ptx_max(float a, float b) {
    float result;
    // max.f32：浮点最大值
    asm("max.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ float ptx_abs(float a) {
    float result;
    // abs.f32：浮点绝对值
    asm("abs.f32 %0, %1;" : "=f"(result) : "f"(a));
    return result;
}

// ============================================================================
// 类型转换
// ============================================================================
__device__ int ptx_float_to_int(float a) {
    int result;
    // cvt.rni.s32.f32：浮点转整数，rni表示舍入到最近整数
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(result) : "f"(a));
    return result;
}

__device__ float ptx_int_to_float(int a) {
    float result;
    // cvt.rn.f32.s32：整数转浮点
    asm("cvt.rn.f32.s32 %0, %1;" : "=f"(result) : "r"(a));
    return result;
}

// ============================================================================
// 测试内核
// ============================================================================
__global__ void test_ptx_basics(float* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        printf("=== PTX基础测试 ===\n\n");

        // 测试线程索引获取
        int tid_x = get_tid_x();
        int ctaid_x = get_ctaid_x();
        int ntid_x = get_ntid_x();
        printf("线程索引: tid.x=%d, ctaid.x=%d, ntid.x=%d\n", tid_x, ctaid_x, ntid_x);

        // 测试算术运算
        float a = 3.14f, b = 2.0f;
        printf("\n算术运算测试 (a=%.2f, b=%.2f):\n", a, b);
        printf("  add: %.4f\n", ptx_add(a, b));
        printf("  mul: %.4f\n", ptx_mul(a, b));
        printf("  fma(a,b,a): %.4f\n", ptx_fma(a, b, a));

        // 测试整数运算
        int ia = 10, ib = 20;
        printf("\n整数运算测试 (ia=%d, ib=%d):\n", ia, ib);
        printf("  add: %d\n", ptx_add_int(ia, ib));

        // 测试比较操作
        printf("\n比较操作测试:\n");
        printf("  min(%.2f, %.2f) = %.4f\n", a, b, ptx_min(a, b));
        printf("  max(%.2f, %.2f) = %.4f\n", a, b, ptx_max(a, b));
        printf("  abs(-%.2f) = %.4f\n", a, ptx_abs(-a));

        // 测试类型转换
        printf("\n类型转换测试:\n");
        printf("  float_to_int(%.2f) = %d\n", 3.7f, ptx_float_to_int(3.7f));
        printf("  int_to_float(%d) = %.4f\n", 42, ptx_int_to_float(42));

        // 存储结果
        results[0] = ptx_add(a, b);
        results[1] = ptx_mul(a, b);
        results[2] = ptx_fma(a, b, a);
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    float* d_results;
    cudaMalloc(&d_results, 3 * sizeof(float));

    test_ptx_basics<<<1, 32>>>(d_results);
    cudaDeviceSynchronize();

    // 检查结果
    float h_results[3];
    cudaMemcpy(h_results, d_results, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n=== 结果验证 ===\n");
    printf("GPU计算结果: add=%.4f, mul=%.4f, fma=%.4f\n",
           h_results[0], h_results[1], h_results[2]);
    printf("CPU期望结果: add=%.4f, mul=%.4f, fma=%.4f\n",
           3.14f + 2.0f, 3.14f * 2.0f, 3.14f * 2.0f + 3.14f);

    cudaFree(d_results);

    printf("\nPTX基础示例完成！\n");
    return 0;
}