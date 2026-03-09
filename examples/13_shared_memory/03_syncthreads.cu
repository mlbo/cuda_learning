/**
 * 03_syncthreads.cu
 * __syncthreads() 同步详解示例
 *
 * 本示例演示：
 * 1. __syncthreads() 的基本用法
 * 2. 不正确使用导致的问题
 * 3. 其他同步原语
 * 4. 内存屏障
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
// 示例1: 正确使用 __syncthreads()
// ============================================================================

// 正确示例：数据加载后同步，确保所有数据就绪
__global__ void correct_syncthreads(float *input, float *output, int n) {
  __shared__ float s_data[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 阶段1: 加载数据
  if (idx < n) {
    s_data[tid] = input[idx];
  }

  // 必须同步！确保所有线程完成写入
  __syncthreads();

  // 阶段2: 使用共享内存数据
  // 现在可以安全地读取其他线程写入的数据
  if (tid > 0 && idx < n) {
    output[idx] = s_data[tid] + s_data[tid - 1];
  } else if (idx < n) {
    output[idx] = s_data[tid];
  }
}

// ============================================================================
// 示例2: 错误使用 - 缺少同步
// ============================================================================

// 错误示例：缺少同步，可能导致读取到未初始化的数据
__global__ void missing_syncthreads(float *input, float *output, int n) {
  __shared__ float s_data[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    s_data[tid] = input[idx];
  }

  // 缺少 __syncthreads()！
  // 其他线程可能还没有完成写入

  if (tid > 0 && idx < n) {
    // 这里读取 s_data[tid - 1] 可能得到旧值
    output[idx] = s_data[tid] + s_data[tid - 1];
  } else if (idx < n) {
    output[idx] = s_data[tid];
  }
}

// ============================================================================
// 示例3: 错误使用 - 条件分支中的同步
// ============================================================================

// 危险示例：条件分支中的同步可能导致死锁或未定义行为
// 注意：这个示例可能不会真正死锁（取决于GPU行为），但是不安全
__global__ void conditional_syncthreads_bad(float *data, int n) {
  __shared__ float s_data[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < 128) {
    s_data[tid] = data[idx];
    // 危险！只有一半线程执行同步
    __syncthreads(); // 这会导致未定义行为
  }

  if (idx < n) {
    data[idx] = s_data[tid];
  }
}

// 正确做法：同步放在条件分支外
__global__ void conditional_syncthreads_good(float *data, int n) {
  __shared__ float s_data[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 所有线程都参与条件判断
  if (tid < 128) {
    s_data[tid] = data[idx];
  }

  // 同步在条件分支外，所有线程都执行
  __syncthreads();

  if (idx < n && tid < 128) {
    data[idx] = s_data[tid] * 2.0f;
  }
}

// ============================================================================
// 示例4: 多阶段同步
// ============================================================================

// 多阶段计算需要多次同步
__global__ void multi_stage_syncthreads(float *input, float *output, int n) {
  __shared__ float s_data[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 阶段1: 加载
  if (idx < n) {
    s_data[tid] = input[idx];
  }
  __syncthreads();

  // 阶段2: 部分和计算
  if (tid < 128) {
    s_data[tid] += s_data[tid + 128];
  }
  __syncthreads(); // 再次同步

  // 阶段3: 继续规约
  if (tid < 64) {
    s_data[tid] += s_data[tid + 64];
  }
  __syncthreads();

  // 最终结果
  if (tid == 0) {
    output[blockIdx.x] = s_data[0] + s_data[32] + s_data[16] + s_data[8] +
                         s_data[4] + s_data[2] + s_data[1];
  }
}

// ============================================================================
// 示例5: 其他同步原语
// ============================================================================

// __syncthreads_count: 返回满足条件的线程数量
__global__ void syncthreads_count_demo(int *input, int *count_output, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int condition = 0;
  if (idx < n) {
    condition = (input[idx] > 500);
  }

  // 统计满足条件的线程数
  int count = __syncthreads_count(condition);

  if (tid == 0) {
    count_output[blockIdx.x] = count;
  }
}

// __syncthreads_and: 所有线程条件都为真时返回非零
__global__ void syncthreads_and_demo(int *input, int *and_output, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int condition = 1;
  if (idx < n) {
    condition = (input[idx] >= 0);
  }

  // 所有线程条件都为真才返回非零
  int result = __syncthreads_and(condition);

  if (tid == 0) {
    and_output[blockIdx.x] = result;
  }
}

// __syncthreads_or: 至少一个线程条件为真时返回非零
__global__ void syncthreads_or_demo(int *input, int *or_output, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int condition = 0;
  if (idx < n) {
    condition = (input[idx] > 1000);
  }

  // 至少一个线程条件为真就返回非零
  int result = __syncthreads_or(condition);

  if (tid == 0) {
    or_output[blockIdx.x] = result;
  }
}

// ============================================================================
// 示例6: Warp 级同步
// ============================================================================

// __syncwarp: 只同步一个 warp 内的线程
__global__ void syncwarp_demo(float *input, float *output, int n) {
  __shared__ float s_data[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane = tid % 32;

  // 每个warp独立处理数据
  if (idx < n) {
    s_data[tid] = input[idx];
  }

  // 只同步当前 warp
  __syncwarp();

  // warp 内协作计算
  if (idx < n) {
    // 使用 warp shuffle 替代共享内存访问
    float val = s_data[tid];
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) {
      output[blockIdx.x * 8 + warp_id] = val;
    }
  }
}

// ============================================================================
// 示例7: 内存屏障
// ============================================================================

// __threadfence_block: 确保块内内存操作对其他线程可见
__global__ void threadfence_demo(float *data, int *flag, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    data[idx] = idx * 2.0f;

    // 确保上面的写入对块内其他线程可见
    __threadfence_block();

    // 设置标志表示数据已就绪
    if (tid == 0) {
      flag[blockIdx.x] = 1;
    }
  }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
  printf("========================================\n");
  printf("__syncthreads() 同步详解示例\n");
  printf("========================================\n\n");

  int N = 1024;
  size_t bytes = N * sizeof(float);
  size_t int_bytes = N * sizeof(int);

  // 分配主机内存
  float *h_input = (float *)malloc(bytes);
  float *h_output = (float *)malloc(bytes);
  int *h_int_input = (int *)malloc(int_bytes);
  int *h_int_output = (int *)malloc(int_bytes);

  // 初始化
  for (int i = 0; i < N; i++) {
    h_input[i] = (float)i;
    h_int_input[i] = i;
  }

  // 分配设备内存
  float *d_input, *d_output;
  int *d_int_input, *d_int_output, *d_flag;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMalloc(&d_int_input, int_bytes));
  CUDA_CHECK(cudaMalloc(&d_int_output, int_bytes));
  CUDA_CHECK(cudaMalloc(&d_flag, int_bytes));

  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_int_input, h_int_input, int_bytes, cudaMemcpyHostToDevice));

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // ---------- 测试1: 正确同步 ----------
  printf("--- 测试1: 正确使用 __syncthreads() ---\n");

  correct_syncthreads<<<gridSize, blockSize>>>(d_input, d_output, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

  printf("输入: [");
  for (int i = 0; i < 5; i++)
    printf("%.0f ", h_input[i]);
  printf("...]\n");
  printf("输出(相邻元素和): [");
  for (int i = 0; i < 5; i++)
    printf("%.0f ", h_output[i]);
  printf("...]\n");
  printf("期望: output[i] = input[i] + input[i-1]\n\n");

  // ---------- 测试2: 缺少同步 ----------
  printf("--- 测试2: 缺少同步的问题 ---\n");

  printf("缺少 __syncthreads() 时，线程可能读取到未初始化的数据\n");
  printf("结果可能是错误的或不一致的\n\n");

  // ---------- 测试3: 多阶段同步 ----------
  printf("--- 测试3: 多阶段规约 ---\n");

  multi_stage_syncthreads<<<gridSize, blockSize>>>(d_input, d_output, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_output, d_output, gridSize * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float total = 0;
  for (int i = 0; i < gridSize; i++) {
    total += h_output[i];
  }
  float expected = (N - 1) * N / 2.0f; // 0+1+2+...+(N-1)
  printf("规约结果: %.0f\n", total);
  printf("期望结果: %.0f\n", expected);
  printf("验证: %s\n\n",
         std::fabs(total - expected) < 1e-3 ? "通过" : "失败");

  // ---------- 测试4: 同步计数 ----------
  printf("--- 测试4: __syncthreads_count ---\n");

  syncthreads_count_demo<<<gridSize, blockSize>>>(d_int_input, d_int_output, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_int_output, d_int_output, gridSize * sizeof(int),
                        cudaMemcpyDeviceToHost));

  int total_count = 0;
  for (int i = 0; i < gridSize; i++) {
    total_count += h_int_output[i];
  }
  printf("统计 input[i] > 500 的元素数量: %d\n", total_count);
  printf("期望数量: %d\n\n", N - 501);

  // ---------- 测试5: __syncthreads_and/or ----------
  printf("--- 测试5: __syncthreads_and/or ---\n");

  syncthreads_and_demo<<<gridSize, blockSize>>>(d_int_input, d_int_output, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_int_output, d_int_output, gridSize * sizeof(int),
                        cudaMemcpyDeviceToHost));

  int and_result = h_int_output[0];
  printf("__syncthreads_and (检查所有元素 >= 0): %s\n\n",
         and_result ? "是" : "否");

  // ---------- 测试6: Warp同步 ----------
  printf("--- 测试6: __syncwarp 和 Warp 级规约 ---\n");

  syncwarp_demo<<<gridSize, blockSize>>>(d_input, d_output, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("每个 block 有 %d 个 warp\n", blockSize / 32);
  printf("使用 __syncwarp() 进行 warp 内同步\n\n");

  // ---------- 性能对比 ----------
  printf("--- 性能对比 ---\n");

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  int iterations = 1000;
  float ms_correct, ms_missing;

  // 正确同步版本
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; i++) {
    correct_syncthreads<<<gridSize, blockSize>>>(d_input, d_output, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms_correct, start, stop));

  // 缺少同步版本（结果可能错误，仅对比性能）
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; i++) {
    missing_syncthreads<<<gridSize, blockSize>>>(d_input, d_output, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms_missing, start, stop));

  printf("相邻元素求和 (%d 次迭代):\n", iterations);
  printf("  正确同步: %.3f ms\n", ms_correct);
  printf("  缺少同步: %.3f ms\n", ms_missing);
  printf("  注意: 缺少同步版本可能更快但结果错误！\n");

  // ---------- 同步最佳实践 ----------
  printf("\n--- __syncthreads() 使用规则 ---\n");
  printf("1. 必须被块内所有线程执行到\n");
  printf("2. 不要在可能发散的条件分支中使用\n");
  printf("3. 在写入共享内存后、读取前使用\n");
  printf("4. 多阶段计算需要多次同步\n");
  printf("5. 使用 __syncwarp() 进行 warp 内同步\n");
  printf("6. 使用 __threadfence_* 确保内存可见性\n");

  // 清理
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_int_input));
  CUDA_CHECK(cudaFree(d_int_output));
  CUDA_CHECK(cudaFree(d_flag));
  free(h_input);
  free(h_output);
  free(h_int_input);
  free(h_int_output);

  printf("\n========================================\n");
  printf("示例完成\n");
  printf("========================================\n");

  return 0;
}
