/**
 * 01_smem_basics.cu
 * 共享内存基础示例
 *
 * 本示例演示：
 * 1. 共享内存的声明方式
 * 2. 共享内存的基本访问
 * 3. 共享内存与全局内存的性能对比
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// 错误检查宏
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
// 示例1: 共享内存基本使用
// ============================================================================

// 使用全局内存的向量加法
__global__ void vector_add_global(const float *a, const float *b, float *c,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// 使用共享内存的向量加法（演示目的，实际场景中简单向量加法不需要共享内存）
__global__ void vector_add_shared(const float *a, const float *b, float *c,
                                  int n) {
  // 声明共享内存
  __shared__ float s_a[256]; // 静态共享内存
  __shared__ float s_b[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 从全局内存加载到共享内存
  if (idx < n) {
    s_a[tid] = a[idx];
    s_b[tid] = b[idx];
  }
  __syncthreads(); // 同步，确保所有线程完成加载

  // 从共享内存读取并计算
  if (idx < n) {
    c[idx] = s_a[tid] + s_b[tid];
  }
}

// ============================================================================
// 示例2: 共享内存用于数据复用
// ============================================================================

// 每个线程计算相邻元素的和（滑动窗口）
// 不使用共享内存
__global__ void sliding_sum_global(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx > 0 && idx < n - 1) {
    // 每个线程需要读取3个全局内存元素
    // 相邻线程会重复读取相同的元素
    output[idx] = input[idx - 1] + input[idx] + input[idx + 1];
  }
}

// 使用共享内存减少全局内存访问
__global__ void sliding_sum_shared(const float *input, float *output, int n) {
  // 共享内存需要额外2个元素处理边界
  __shared__ float s_data[258]; // blockDim.x + 2

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 加载主数据
  if (idx < n) {
    s_data[tid + 1] = input[idx]; // 偏移1个位置存储
  }

  // 处理边界元素（halo region）
  if (tid == 0 && idx > 0) {
    s_data[0] = input[idx - 1]; // 左边界
  }
  if (tid == blockDim.x - 1 && idx < n - 1) {
    s_data[blockDim.x + 1] = input[idx + 1]; // 右边界
  }

  __syncthreads();

  // 计算
  if (idx > 0 && idx < n - 1) {
    output[idx] = s_data[tid] + s_data[tid + 1] + s_data[tid + 2];
  }
}

// ============================================================================
// 示例3: 共享内存用于线程间通信
// ============================================================================

// 使用共享内存进行块内数据交换
__global__ void reverse_block(float *data, int n) {
  __shared__ float s_data[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 加载数据到共享内存
  if (idx < n) {
    s_data[tid] = data[idx];
  }
  __syncthreads();

  // 反向写回（块内反转）
  if (idx < n) {
    data[idx] = s_data[blockDim.x - 1 - tid];
  }
}

// ============================================================================
// 主函数
// ============================================================================

void print_array(const char *name, const float *arr, int n) {
  printf("%s: [", name);
  for (int i = 0; i < (n < 10 ? n : 10); i++) {
    printf("%.1f ", arr[i]);
  }
  if (n > 10)
    printf("...]");
  else
    printf("]");
  printf("\n");
}

int main() {
  printf("========================================\n");
  printf("共享内存基础示例\n");
  printf("========================================\n\n");

  int N = 1024;
  size_t bytes = N * sizeof(float);

  // 分配主机内存
  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_c = (float *)malloc(bytes);
  float *h_input = (float *)malloc(bytes);
  float *h_output = (float *)malloc(bytes);

  // 初始化数据
  for (int i = 0; i < N; i++) {
    h_a[i] = (float)i;
    h_b[i] = (float)(i * 2);
    h_input[i] = (float)i;
  }

  // 分配设备内存
  float *d_a, *d_b, *d_c, *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));

  // 拷贝数据到设备
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // ---------- 测试1: 向量加法对比 ----------
  printf("--- 测试1: 向量加法 ---\n");

  // 全局内存版本
  vector_add_global<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
  print_array("全局内存结果", h_c, N);

  // 共享内存版本
  vector_add_shared<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
  print_array("共享内存结果", h_c, N);

  printf("\n");

  // ---------- 测试2: 滑动窗口求和 ----------
  printf("--- 测试2: 滑动窗口求和 ---\n");

  // 重置输出
  CUDA_CHECK(cudaMemset(d_output, 0, bytes));

  sliding_sum_shared<<<gridSize, blockSize>>>(d_input, d_output, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
  print_array("输入数据", h_input, N);
  print_array("滑动求和", h_output, N);

  // 验证: 输出[i] = input[i-1] + input[i] + input[i+1]
  bool correct = true;
  for (int i = 1; i < N - 1 && correct; i++) {
    float expected = (i - 1) + i + (i + 1);
    if (std::fabs(h_output[i] - expected) > 1e-5) {
      printf("错误: h_output[%d] = %.1f, 期望 %.1f\n", i, h_output[i],
             expected);
      correct = false;
    }
  }
  printf("验证: %s\n", correct ? "通过" : "失败");

  printf("\n");

  // ---------- 测试3: 块内反转 ----------
  printf("--- 测试3: 块内数据反转 ---\n");

  // 重置数据
  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

  reverse_block<<<gridSize, blockSize>>>(d_input, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_output, d_input, bytes, cudaMemcpyDeviceToHost));
  print_array("原始数据", h_input, N);
  print_array("反转后", h_output, N);

  printf("\n");

  // ---------- 性能对比 ----------
  printf("--- 性能对比 ---\n");

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  int iterations = 1000;
  float ms_global, ms_shared;

  // 全局内存版本计时
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; i++) {
    sliding_sum_global<<<gridSize, blockSize>>>(d_input, d_output, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms_global, start, stop));

  // 共享内存版本计时
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; i++) {
    sliding_sum_shared<<<gridSize, blockSize>>>(d_input, d_output, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms_shared, start, stop));

  printf("滑动窗口求和 (%d 次迭代):\n", iterations);
  printf("  全局内存: %.3f ms\n", ms_global);
  printf("  共享内存: %.3f ms\n", ms_shared);
  printf("  加速比: %.2fx\n", ms_global / ms_shared);

  // 清理
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_input);
  free(h_output);

  printf("\n========================================\n");
  printf("示例完成\n");
  printf("========================================\n");

  return 0;
}
