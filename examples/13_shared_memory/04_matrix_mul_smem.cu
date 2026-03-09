/**
 * 04_matrix_mul_smem.cu
 * 使用共享内存优化矩阵乘法示例
 *
 * 本示例演示：
 * 1. 朴素矩阵乘法实现
 * 2. 使用共享内存的分块矩阵乘法
 * 3. 不同分块大小的性能对比
 * 4. Bank Conflict 优化
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
// 矩阵大小配置
// ============================================================================
#define N 1024 // 矩阵维度
#define TILE_SIZE 32

// ============================================================================
// 版本1: 朴素矩阵乘法（全局内存）
// ============================================================================

__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
      sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
  }
}

// ============================================================================
// 版本2: 使用共享内存的分块矩阵乘法
// ============================================================================

__global__ void matmul_shared(const float *A, const float *B, float *C, int n) {
  // 共享内存缓存矩阵块
  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  float sum = 0.0f;

  // 循环处理所有子块
  for (int t = 0; t < n / TILE_SIZE; t++) {
    // 从全局内存加载到共享内存
    sA[ty][tx] = A[row * n + t * TILE_SIZE + tx];
    sB[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];

    // 同步确保数据加载完成
    __syncthreads();

    // 计算子块乘积
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += sA[ty][k] * sB[k][tx];
    }

    // 同步确保所有线程完成计算后再加载下一块
    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
}

// ============================================================================
// 版本3: 优化版 - 减少Bank Conflict
// ============================================================================

// 通过填充避免Bank Conflict
__global__ void matmul_shared_optimized(const float *A, const float *B, float *C,
                                        int n) {
  // 添加一列填充以避免Bank Conflict
  __shared__ float sA[TILE_SIZE][TILE_SIZE + 1]; // +1 padding
  __shared__ float sB[TILE_SIZE][TILE_SIZE + 1];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  float sum = 0.0f;

  for (int t = 0; t < n / TILE_SIZE; t++) {
    sA[ty][tx] = A[row * n + t * TILE_SIZE + tx];
    sB[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += sA[ty][k] * sB[k][tx];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
}

// ============================================================================
// 版本4: 一维线程块版本
// ============================================================================

__global__ void matmul_shared_1d(const float *A, const float *B, float *C,
                                 int n) {
  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sB[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + tx / TILE_SIZE;
  int col = blockIdx.x * TILE_SIZE + tx % TILE_SIZE;

  float sum = 0.0f;

  for (int t = 0; t < n / TILE_SIZE; t++) {
    // 1D线程加载
    sA[tx / TILE_SIZE][tx % TILE_SIZE] =
        A[row * n + t * TILE_SIZE + tx % TILE_SIZE];
    sB[tx / TILE_SIZE][tx % TILE_SIZE] =
        B[(t * TILE_SIZE + tx / TILE_SIZE) * n + col];

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; k++) {
      sum += sA[tx / TILE_SIZE][k] * sB[k][tx % TILE_SIZE];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
}

// ============================================================================
// 版本5: 处理非 TILE_SIZE 整数倍的矩阵
// ============================================================================

__global__ void matmul_shared_general(const float *A, const float *B, float *C,
                                      int n) {
  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  float sum = 0.0f;

  // 向上取整计算需要的块数
  int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; t++) {
    // 加载时检查边界
    int a_col = t * TILE_SIZE + tx;
    int b_row = t * TILE_SIZE + ty;

    sA[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
    sB[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += sA[ty][k] * sB[k][tx];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
}

// ============================================================================
// CPU参考实现
// ============================================================================

void matmul_cpu(const float *A, const float *B, float *C, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

// ============================================================================
// 验证结果
// ============================================================================

bool verify_result(const float *C, const float *C_ref, int n, float eps = 1e-3) {
  for (int i = 0; i < n * n; i++) {
    if (std::fabs(C[i] - C_ref[i]) > eps) {
      printf("Mismatch at index %d: got %.4f, expected %.4f\n", i, C[i],
             C_ref[i]);
      return false;
    }
  }
  return true;
}

// ============================================================================
// 性能测试函数
// ============================================================================

template <void (*Kernel)(const float *, const float *, float *, int)>
void benchmark_matmul(const float *d_A, const float *d_B, float *d_C, int n,
                      int iterations, float *time_ms) {
  dim3 blockSize(TILE_SIZE, TILE_SIZE);
  dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // 预热
  Kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 计时
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; i++) {
    Kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(time_ms, start, stop));
  *time_ms /= iterations;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
  printf("========================================\n");
  printf("共享内存优化矩阵乘法示例\n");
  printf("========================================\n\n");

  int n = N;
  size_t bytes = n * n * sizeof(float);

  printf("矩阵大小: %d x %d\n", n, n);
  printf("分块大小: %d x %d\n", TILE_SIZE, TILE_SIZE);
  printf("共享内存使用: %zu 字节/block\n\n",
         2 * TILE_SIZE * TILE_SIZE * sizeof(float));

  // 分配主机内存
  float *h_A = (float *)malloc(bytes);
  float *h_B = (float *)malloc(bytes);
  float *h_C = (float *)malloc(bytes);
  float *h_C_ref = (float *)malloc(bytes);

  // 初始化矩阵（使用小随机值避免浮点溢出）
  srand(42);
  for (int i = 0; i < n * n; i++) {
    h_A[i] = (float)(rand() % 100) / 100.0f;
    h_B[i] = (float)(rand() % 100) / 100.0f;
  }

  // 分配设备内存
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));

  // 拷贝数据到设备
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  int iterations = 10;

  // ---------- 版本1: 朴素实现 ----------
  printf("--- 版本1: 朴素矩阵乘法 ---\n");
  printf("特点: 每个元素从全局内存读取N次\n");

  float time_naive;
  benchmark_matmul<matmul_naive>(d_A, d_B, d_C, n, iterations, &time_naive);

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // 复制作为参考
  memcpy(h_C_ref, h_C, bytes);

  float gflops_naive = 2.0 * n * n * n / time_naive / 1e6;
  printf("执行时间: %.3f ms\n", time_naive);
  printf("性能: %.2f GFLOPS\n\n", gflops_naive);

  // ---------- 版本2: 共享内存分块 ----------
  printf("--- 版本2: 共享内存分块矩阵乘法 ---\n");
  printf("特点: 数据在共享内存中复用\n");

  float time_shared;
  benchmark_matmul<matmul_shared>(d_A, d_B, d_C, n, iterations, &time_shared);

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  bool correct_shared = verify_result(h_C, h_C_ref, n);
  float gflops_shared = 2.0 * n * n * n / time_shared / 1e6;
  printf("正确性: %s\n", correct_shared ? "通过" : "失败");
  printf("执行时间: %.3f ms\n", time_shared);
  printf("性能: %.2f GFLOPS\n", gflops_shared);
  printf("加速比: %.2fx\n\n", time_naive / time_shared);

  // ---------- 版本3: Bank Conflict优化 ----------
  printf("--- 版本3: Bank Conflict 优化版本 ---\n");
  printf("特点: 使用填充避免Bank Conflict\n");

  float time_opt;
  benchmark_matmul<matmul_shared_optimized>(d_A, d_B, d_C, n, iterations, &time_opt);

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  bool correct_opt = verify_result(h_C, h_C_ref, n);
  float gflops_opt = 2.0 * n * n * n / time_opt / 1e6;
  printf("正确性: %s\n", correct_opt ? "通过" : "失败");
  printf("执行时间: %.3f ms\n", time_opt);
  printf("性能: %.2f GFLOPS\n", gflops_opt);
  printf("加速比 (vs naive): %.2fx\n", time_naive / time_opt);
  printf("加速比 (vs shared): %.2fx\n\n", time_shared / time_opt);

  // ---------- 性能分析 ----------
  printf("--- 性能分析 ---\n");
  printf("访存分析:\n");
  printf("  朴素版本: 每个线程读取 2N 个全局内存元素\n");
  printf("  共享内存版本: 每个线程读取 2N/TILE_SIZE 个全局内存元素\n");
  printf("  全局内存访问减少: TILE_SIZE = %d 倍\n\n", TILE_SIZE);

  printf("理论性能上限:\n");
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("  设备: %s\n", prop.name);
  printf("  内存总线宽度: %d bits\n", prop.memoryBusWidth);
  printf("  理论峰值: 请使用 ncu 分析实际性能瓶颈\n");

  // 清理
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  printf("\n========================================\n");
  printf("示例完成\n");
  printf("========================================\n");

  return 0;
}
