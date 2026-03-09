/**
 * 02_static_vs_dynamic.cu
 * 静态共享内存 vs 动态共享内存对比示例
 *
 * 本示例演示：
 * 1. 静态共享内存的声明和使用
 * 2. 动态共享内存的声明和使用
 * 3. 多个动态共享内存数组的管理
 * 4. 两种方式的性能对比
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// 示例1: 静态共享内存
// ============================================================================

// 静态共享内存 - 编译时大小固定
__global__ void static_smem_demo(float *input, float *output, int n) {
  // 静态共享内存声明
  __shared__ float s_data[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 加载数据
  if (idx < n) {
    s_data[tid] = input[idx];
  }
  __syncthreads();

  // 简单处理：将数据翻倍
  if (idx < n) {
    output[idx] = s_data[tid] * 2.0f;
  }
}

// 多个静态共享内存变量
__global__ void multi_static_smem(float *a, float *b, float *c, int n) {
  // 可以声明多个静态共享内存变量
  __shared__ float s_a[128];
  __shared__ float s_b[128];
  __shared__ float s_c[128];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    s_a[tid] = a[idx];
    s_b[tid] = b[idx];
  }
  __syncthreads();

  if (idx < n) {
    s_c[tid] = s_a[tid] + s_b[tid];
    c[idx] = s_c[tid];
  }
}

// ============================================================================
// 示例2: 动态共享内存
// ============================================================================

// 动态共享内存 - 运行时确定大小
extern __shared__ float dynamic_smem[];

__global__ void dynamic_smem_demo(float *input, float *output, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 使用动态共享内存
  if (idx < n) {
    dynamic_smem[tid] = input[idx];
  }
  __syncthreads();

  if (idx < n) {
    output[idx] = dynamic_smem[tid] * 2.0f;
  }
}

// 动态共享内存可以根据运行时参数调整大小
__global__ void dynamic_smem_variable_size(float *input, float *output, int n,
                                           int smem_elements) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n && tid < smem_elements) {
    dynamic_smem[tid] = input[idx];
  }
  __syncthreads();

  if (idx < n && tid < smem_elements) {
    output[idx] = dynamic_smem[tid] * 2.0f;
  }
}

// ============================================================================
// 示例3: 多个动态共享内存数组
// ============================================================================

// 使用一个动态共享内存缓冲区存储多个数组
__global__ void multi_dynamic_smem(float *a, float *b, float *c, int n) {
  // 使用extern声明单个缓冲区
  extern __shared__ char smem_buffer[];

  // 手动划分共享内存
  int smem_size = blockDim.x;
  float *s_a = (float *)smem_buffer;
  int *s_b = (int *)(smem_buffer + smem_size * sizeof(float));
  float *s_c = (float *)(smem_buffer + smem_size * sizeof(float) +
                         smem_size * sizeof(int));

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    s_a[tid] = a[idx];
    s_b[tid] = (int)(b[idx]); // 转换为整数
  }
  __syncthreads();

  if (idx < n) {
    s_c[tid] = s_a[tid] + (float)s_b[tid];
    c[idx] = s_c[tid];
  }
}

// ============================================================================
// 示例4: 使用模板实现灵活的共享内存
// ============================================================================

template <int SMEM_SIZE>
__global__ void templated_smem(float *input, float *output, int n) {
  __shared__ float s_data[SMEM_SIZE];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n && tid < SMEM_SIZE) {
    s_data[tid] = input[idx];
  }
  __syncthreads();

  if (idx < n && tid < SMEM_SIZE) {
    output[idx] = s_data[tid] * 2.0f;
  }
}

// ============================================================================
// 性能测试
// ============================================================================

__global__ void reduction_static(float *input, float *output, int n) {
  __shared__ float s_data[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 加载并累加（grid stride loop）
  float sum = 0.0f;
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    sum += input[i];
  }
  s_data[tid] = sum;
  __syncthreads();

  // 树状规约
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = s_data[0];
  }
}

__global__ void reduction_dynamic(float *input, float *output, int n) {
  extern __shared__ float s_data[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f;
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    sum += input[i];
  }
  s_data[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = s_data[0];
  }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
  printf("========================================\n");
  printf("静态 vs 动态共享内存对比示例\n");
  printf("========================================\n\n");

  int N = 1024;
  size_t bytes = N * sizeof(float);

  // 分配主机内存
  float *h_input = (float *)malloc(bytes);
  float *h_output = (float *)malloc(bytes);
  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_c = (float *)malloc(bytes);

  // 初始化
  for (int i = 0; i < N; i++) {
    h_input[i] = (float)i;
    h_a[i] = (float)i;
    h_b[i] = (float)(i + 1);
  }

  // 分配设备内存
  float *d_input, *d_output, *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // ---------- 测试1: 静态共享内存 ----------
  printf("--- 测试1: 静态共享内存 ---\n");

  static_smem_demo<<<gridSize, blockSize>>>(d_input, d_output, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

  printf("输入: [");
  for (int i = 0; i < 5; i++)
    printf("%.0f ", h_input[i]);
  printf("...]\n");

  printf("输出: [");
  for (int i = 0; i < 5; i++)
    printf("%.0f ", h_output[i]);
  printf("...]\n\n");

  // ---------- 测试2: 动态共享内存 ----------
  printf("--- 测试2: 动态共享内存 ---\n");

  // 动态共享内存大小作为核函数启动的第三个参数
  size_t smem_size = blockSize * sizeof(float);
  dynamic_smem_demo<<<gridSize, blockSize, smem_size>>>(d_input, d_output, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

  printf("动态共享内存大小: %zu 字节\n", smem_size);
  printf("输出: [");
  for (int i = 0; i < 5; i++)
    printf("%.0f ", h_output[i]);
  printf("...]\n\n");

  // ---------- 测试3: 多个动态共享内存数组 ----------
  printf("--- 测试3: 多个动态共享内存数组 ---\n");

  size_t multi_smem_size =
      blockSize * sizeof(float) + blockSize * sizeof(int) + blockSize * sizeof(float);
  printf("多数组共享内存布局:\n");
  printf("  s_a: %zu 字节\n", blockSize * sizeof(float));
  printf("  s_b: %zu 字节\n", blockSize * sizeof(int));
  printf("  s_c: %zu 字节\n", blockSize * sizeof(float));
  printf("  总计: %zu 字节\n\n", multi_smem_size);

  multi_dynamic_smem<<<gridSize, blockSize, multi_smem_size>>>(d_a, d_b, d_c, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  printf("a: [");
  for (int i = 0; i < 5; i++)
    printf("%.0f ", h_a[i]);
  printf("...]\n");
  printf("b: [");
  for (int i = 0; i < 5; i++)
    printf("%.0f ", h_b[i]);
  printf("...]\n");
  printf("c=a+b: [");
  for (int i = 0; i < 5; i++)
    printf("%.0f ", h_c[i]);
  printf("...]\n\n");

  // ---------- 测试4: 模板化共享内存 ----------
  printf("--- 测试4: 模板化共享内存 ---\n");

  templated_smem<256><<<gridSize, blockSize>>>(d_input, d_output, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

  printf("模板参数 SMEM_SIZE=256\n");
  printf("输出: [");
  for (int i = 0; i < 5; i++)
    printf("%.0f ", h_output[i]);
  printf("...]\n\n");

  // ---------- 性能对比 ----------
  printf("--- 性能对比: 规约操作 ---\n");

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  int iterations = 1000;
  float ms_static, ms_dynamic;

  // 静态共享内存规约
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; i++) {
    reduction_static<<<gridSize, blockSize>>>(d_input, d_output, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms_static, start, stop));

  // 动态共享内存规约
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; i++) {
    reduction_dynamic<<<gridSize, blockSize, smem_size>>>(d_input, d_output, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms_dynamic, start, stop));

  printf("规约操作 (%d 次迭代):\n", iterations);
  printf("  静态共享内存: %.3f ms\n", ms_static);
  printf("  动态共享内存: %.3f ms\n", ms_dynamic);
  printf("  性能差异: %.2f%%\n",
         std::fabs(ms_static - ms_dynamic) / ms_static * 100);

  // ---------- 设备属性查询 ----------
  printf("\n--- 设备共享内存属性 ---\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  printf("设备名称: %s\n", prop.name);
  printf("每个线程块最大共享内存: %zu 字节 (%.1f KB)\n",
         prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
  printf("每个 SM 最大共享内存: %zu 字节 (%.1f KB)\n",
         prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor / 1024.0);

  // 清理
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  free(h_input);
  free(h_output);
  free(h_a);
  free(h_b);
  free(h_c);

  printf("\n========================================\n");
  printf("示例完成\n");
  printf("========================================\n");

  return 0;
}
