/**
 * 06_atomic_cas.cu
 * atomicCAS：比较并交换，构建自定义原子操作的基础
 *
 * atomicCAS是最基础的原子操作，可以用来实现其他原子操作
 *
 * 编译: nvcc -o 06_atomic_cas 06_atomic_cas.cu
 * 运行: ./06_atomic_cas
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// atomicCAS 原型：
// int atomicCAS(int* address, int compare, int val);
// 语义：
//   old = *address;
//   if (old == compare) *address = val;
//   return old;

// 使用atomicCAS实现原子乘法（float版本）
__device__ float atomicMul(float* address, float val) {
    // float和int需要通过类型转换来处理
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;

    do {
        assumed = old;
        // 将int解释为float，计算乘法，再转回int
        float old_float = __int_as_float(assumed);
        float new_float = old_float * val;
        int new_int = __float_as_int(new_float);

        // 尝试原子更新
        old = atomicCAS(address_as_int, assumed, new_int);

        // 如果old != assumed，说明其他线程修改了，需要重试
    } while (assumed != old);

    return __int_as_float(old);
}

// 使用atomicCAS实现原子除法
__device__ float atomicDiv(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;

    do {
        assumed = old;
        float old_float = __int_as_float(assumed);
        float new_float = old_float / val;
        int new_int = __float_as_int(new_float);

        old = atomicCAS(address_as_int, assumed, new_int);
    } while (assumed != old);

    return __int_as_float(old);
}

// 使用atomicCAS实现无锁原子最大值（float版本）
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;

    do {
        assumed = old;
        float old_float = __int_as_float(assumed);
        float new_float = fmaxf(old_float, val);
        int new_int = __float_as_int(new_float);

        old = atomicCAS(address_as_int, assumed, new_int);
    } while (assumed != old && __int_as_float(old) < val);

    return __int_as_float(old);
}

// 演示atomicCAS的基本用法
__global__ void cas_demo(int* data, int* result) {
    int idx = threadIdx.x;

    // 示例1: 简单的比较并交换
    // 如果*data等于10，则更新为20
    int old = atomicCAS(data, 10, 20);
    printf("Thread %d: atomicCAS returned %d\n", idx, old);
}

// 使用atomicCAS实现自增并返回旧值
__device__ int atomicIncrement(int* address, int limit) {
    int old = *address;
    int assumed;

    do {
        assumed = old;
        int new_val = (old >= limit) ? 0 : old + 1;
        old = atomicCAS(address, assumed, new_val);
    } while (assumed != old);

    return old;
}

// 使用atomicCAS实现简单的自旋锁
struct SpinLock {
    int* lock;

    __device__ void acquire() {
        while (atomicCAS(lock, 0, 1) != 0) {
            // 自旋等待
        }
    }

    __device__ void release() {
        atomicExch(lock, 0);
    }
};

__global__ void lock_demo(int* shared_counter, int* lock) {
    SpinLock spin_lock;
    spin_lock.lock = lock;

    // 使用锁保护临界区
    spin_lock.acquire();
    (*shared_counter)++;
    spin_lock.release();
}

// 测试自定义原子操作
__global__ void test_custom_atomic(float* data, float* result_mul, float* result_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 测试原子乘法
    atomicMul(result_mul, 2.0f);

    // 测试原子最大值
    atomicMaxFloat(result_max, data[idx]);
}

int main() {
    printf("========== atomicCAS详解 ==========\n\n");

    // 演示atomicCAS基本行为
    int h_data = 10;
    int *d_data;
    cudaMalloc(&d_data, sizeof(int));
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

    printf("初始值: %d\n", h_data);
    printf("执行: atomicCAS(data, 10, 20) - 如果值是10，则更新为20\n");

    cas_demo<<<1, 4>>>(d_data, nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    printf("执行后值: %d\n\n", h_data);

    // 测试自定义原子操作
    printf("========== 自定义原子操作测试 ==========\n");

    float h_result_mul = 1.0f;
    float h_result_max = 0.0f;
    float* d_result_mul, *d_result_max;
    float* d_values;

    cudaMalloc(&d_result_mul, sizeof(float));
    cudaMalloc(&d_result_max, sizeof(float));
    cudaMalloc(&d_values, 1000 * sizeof(float));

    cudaMemcpy(d_result_mul, &h_result_mul, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_max, &h_result_max, sizeof(float), cudaMemcpyHostToDevice);

    // 初始化测试数据
    float* h_values = (float*)malloc(1000 * sizeof(float));
    for (int i = 0; i < 1000; i++) {
        h_values[i] = (float)(i % 100);
    }
    cudaMemcpy(d_values, h_values, 1000 * sizeof(float), cudaMemcpyHostToDevice);

    test_custom_atomic<<<10, 100>>>(d_values, d_result_mul, d_result_max);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result_mul, d_result_mul, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_result_max, d_result_max, sizeof(float), cudaMemcpyDeviceToHost);

    printf("原子乘法: %.0f (2的1000次方的近似值)\n", h_result_mul);
    printf("原子最大值: %.0f (期望99)\n", h_result_max);

    printf("\n========== atomicCAS应用场景 ==========\n");
    printf("1. 实现CUDA未提供的原子操作（如原子乘法）\n");
    printf("2. 实现无锁数据结构（如无锁队列）\n");
    printf("3. 实现自旋锁\n");
    printf("4. 实现条件更新（只有满足条件才更新）\n");

    printf("\n========== 注意事项 ==========\n");
    printf("1. CAS循环可能导致活锁（多个线程反复失败）\n");
    printf("2. ABA问题：值从A变到B再变回A，CAS会认为未改变\n");
    printf("3. float的位操作需要__int_as_float和__float_as_int\n");

    // 清理
    cudaFree(d_data);
    cudaFree(d_result_mul);
    cudaFree(d_result_max);
    cudaFree(d_values);
    free(h_values);

    return 0;
}