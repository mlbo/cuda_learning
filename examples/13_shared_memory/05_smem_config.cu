/**
 * 05_smem_config.cu
 * 共享内存配置与查询示例
 *
 * 本示例演示：
 * 1. 查询设备共享内存属性
 * 2. 配置共享内存 Bank 大小
 * 3. 配置共享内存与 L1 缓存划分
 * 4. 占用率计算与优化
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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
// 演示不同共享内存配置的核函数
// ============================================================================

#define TILE_SIZE 32

// 使用大量共享内存的核函数
__global__ void smem_intensive(float *input, float *output, int n) {
  // 使用较大的共享内存
  __shared__ float s_data[TILE_SIZE][TILE_SIZE];
  __shared__ float s_temp[TILE_SIZE][TILE_SIZE];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 简单的数据处理
  if (idx < n) {
    s_data[tid / TILE_SIZE][tid % TILE_SIZE] = input[idx];
  }
  __syncthreads();

  // 一些计算
  if (idx < n) {
    s_temp[tid / TILE_SIZE][tid % TILE_SIZE] =
        s_data[tid / TILE_SIZE][tid % TILE_SIZE] * 2.0f;
  }
  __syncthreads();

  if (idx < n) {
    output[idx] = s_temp[tid / TILE_SIZE][tid % TILE_SIZE];
  }
}

// 动态共享内存版本
extern __shared__ float dynamic_smem[];

__global__ void dynamic_smem_kernel(float *input, float *output, int n,
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
// 查询设备属性
// ============================================================================

void print_device_properties() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  printf("========================================\n");
  printf("设备属性查询\n");
  printf("========================================\n\n");

  printf("设备名称: %s\n", prop.name);
  printf("计算能力: %d.%d\n", prop.major, prop.minor);
  printf("\n");

  printf("--- 共享内存属性 ---\n");
  printf("每个线程块最大共享内存: %zu 字节 (%.1f KB)\n", prop.sharedMemPerBlock,
         prop.sharedMemPerBlock / 1024.0);
  printf("每个 SM 最大共享内存: %zu 字节 (%.1f KB)\n",
         prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor / 1024.0);
  printf("每个线程块最大共享内存 (多处理器): %zu 字节\n",
         prop.sharedMemPerMultiprocessor);
  printf("\n");

  printf("--- 线程和块属性 ---\n");
  printf("每个块最大线程数: %d\n", prop.maxThreadsPerBlock);
  printf("每个 SM 最大线程数: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("Warp 大小: %d\n", prop.warpSize);
  printf("每个 SM 最大线程块数: %d\n", prop.maxBlocksPerMultiProcessor);
  printf("\n");

  printf("--- 寄存器属性 ---\n");
  printf("每个块最大寄存器数: %d\n", prop.regsPerBlock);
  printf("每个 SM 最大寄存器数: %d\n", prop.regsPerMultiprocessor);
  printf("\n");

  printf("--- 内存属性 ---\n");
  printf("全局内存总线宽度: %d bits\n", prop.memoryBusWidth);
  printf("L2 缓存大小: %d 字节\n", prop.l2CacheSize);
  printf("\n");
}

// ============================================================================
// Bank 配置演示
// ============================================================================

void demo_bank_config() {
  printf("========================================\n");
  printf("Bank 配置演示\n");
  printf("========================================\n\n");

  // 查询当前 Bank 配置
  cudaSharedMemConfig config;
  CUDA_CHECK(cudaDeviceGetSharedMemConfig(&config));

  printf("当前 Bank 配置: ");
  switch (config) {
  case cudaSharedMemBankSizeDefault:
    printf("默认 (4 字节)\n");
    break;
  case cudaSharedMemBankSizeFourByte:
    printf("4 字节\n");
    break;
  case cudaSharedMemBankSizeEightByte:
    printf("8 字节\n");
    break;
  default:
    printf("未知\n");
  }
  printf("\n");

  // 设置 Bank 大小
  printf("设置 Bank 大小为 8 字节...\n");
  CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  CUDA_CHECK(cudaDeviceGetSharedMemConfig(&config));
  printf("新的 Bank 配置: ");
  switch (config) {
  case cudaSharedMemBankSizeEightByte:
    printf("8 字节\n");
    break;
  default:
    printf("其他\n");
  }
  printf("\n");

  // 恢复默认
  CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));

  printf("Bank 配置说明:\n");
  printf("- 4 字节: 适用于单精度浮点和整数操作\n");
  printf("- 8 字节: 适用于双精度浮点操作，减少 Bank Conflict\n");
  printf("\n");
}

// ============================================================================
// Cache 配置演示
// ============================================================================

void demo_cache_config() {
  printf("========================================\n");
  printf("共享内存与 L1 缓存划分演示\n");
  printf("========================================\n\n");

  // 查询当前 Cache 配置
  cudaFuncCache cacheConfig;
  CUDA_CHECK(cudaDeviceGetCacheConfig(&cacheConfig));

  printf("当前 Cache 配置: ");
  switch (cacheConfig) {
  case cudaFuncCachePreferNone:
    printf("无偏好\n");
    break;
  case cudaFuncCachePreferShared:
    printf("偏好共享内存\n");
    break;
  case cudaFuncCachePreferL1:
    printf("偏好 L1 缓存\n");
    break;
  case cudaFuncCachePreferEqual:
    printf("均衡划分\n");
    break;
  default:
    printf("未知\n");
  }
  printf("\n");

  // 设置全局 Cache 配置
  printf("\n可用的配置选项:\n");
  printf("1. cudaFuncCachePreferShared - 更多共享内存, 更少 L1\n");
  printf("2. cudaFuncCachePreferL1 - 更多 L1, 更少共享内存\n");
  printf("3. cudaFuncCachePreferEqual - 均衡划分\n");
  printf("\n");

  // 设置针对特定核函数的配置
  printf("设置 smem_intensive 核函数偏好共享内存...\n");
  CUDA_CHECK(cudaFuncSetCacheConfig(smem_intensive, cudaFuncCachePreferShared));

  printf("\nCache 配置建议:\n");
  printf("- 大量使用共享内存: 偏好共享内存\n");
  printf("- 大量使用全局内存(非共享): 偏好 L1\n");
  printf("- 不确定: 使用均衡或无偏好\n");
  printf("\n");
}

// ============================================================================
// 占用率计算演示
// ============================================================================

void demo_occupancy() {
  printf("========================================\n");
  printf("占用率计算演示\n");
  printf("========================================\n\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  int blockSize = 256;
  int smemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(float); // 静态共享内存

  // 计算最大活跃块数
  int maxBlocks;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocks, smem_intensive, blockSize, 0));

  int maxThreads = maxBlocks * blockSize;
  float occupancy = (float)maxThreads / prop.maxThreadsPerMultiProcessor * 100;

  printf("核函数配置:\n");
  printf("  线程块大小: %d\n", blockSize);
  printf("  静态共享内存: %d 字节\n", smemSize);
  printf("\n");

  printf("占用率分析:\n");
  printf("  每个 SM 最大活跃块数: %d\n", maxBlocks);
  printf("  每个 SM 最大活跃线程数: %d\n", maxThreads);
  printf("  理论占用率: %.1f%%\n", occupancy);
  printf("\n");

  // 尝试不同块大小
  printf("不同线程块大小的占用率:\n");
  printf("%-15s %-15s %-15s\n", "块大小", "最大块数", "占用率");

  int blockSizes[] = {32, 64, 128, 192, 256, 512, 1024};
  int numSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);

  for (int i = 0; i < numSizes; i++) {
    int bs = blockSizes[i];
    int mb;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, smem_intensive,
                                                             bs, 0));
    float occ = (float)(mb * bs) / prop.maxThreadsPerMultiProcessor * 100;
    printf("%-15d %-15d %-15.1f%%\n", bs, mb, occ);
  }
  printf("\n");

  // 使用动态共享内存的情况
  printf("使用动态共享内存时的占用率:\n");
  printf("%-15s %-15s %-15s %-15s\n", "共享内存(KB)", "块大小", "最大块数", "占用率");

  int smemSizes[] = {0, 8192, 16384, 32768, 49152};
  int numSmemSizes = sizeof(smemSizes) / sizeof(smemSizes[0]);

  for (int i = 0; i < numSmemSizes; i++) {
    int smem = smemSizes[i];
    int mb;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &mb, dynamic_smem_kernel, 256, smem));
    float occ = (float)(mb * 256) / prop.maxThreadsPerMultiProcessor * 100;
    printf("%-15.1f %-15d %-15d %-15.1f%%\n", smem / 1024.0, 256, mb, occ);
  }
  printf("\n");

  printf("占用率优化建议:\n");
  printf("1. 目标占用率通常在 50%% 以上\n");
  printf("2. 太大的块大小可能限制并发块数\n");
  printf("3. 共享内存使用量会直接影响占用率\n");
  printf("4. 寄存器使用量也会影响占用率\n");
  printf("\n");
}

// ============================================================================
// Shared Memory Carveout 演示
// ============================================================================

void demo_carveout() {
  printf("========================================\n");
  printf("Shared Memory Carveout 演示\n");
  printf("========================================\n\n");

  printf("Carveout 控制共享内存和 L1 缓存的物理划分比例\n\n");

  // 设置 carveout 比例
  printf("设置 smem_intensive 核函数的 carveout:\n");

  int carveout_values[] = {25, 50, 75, 100};
  for (int i = 0; i < 4; i++) {
    int c = carveout_values[i];
    CUDA_CHECK(cudaFuncSetAttribute(
        smem_intensive, cudaFuncAttributePreferredSharedMemoryCarveout, c));
    printf("  Carveout = %d%%: %d%% 给共享内存, %d%% 给 L1\n", c, c, 100 - c);
  }
  printf("\n");

  printf("Carveout 使用建议:\n");
  printf("- 共享内存密集型核函数: 使用较高 carveout (75-100%%)\n");
  printf("- L1 缓存依赖型核函数: 使用较低 carveout (25-50%%)\n");
  printf("- 不确定: 使用默认值\n");
  printf("\n");
}

// ============================================================================
// 实际性能测试
// ============================================================================

void benchmark_configs() {
  printf("========================================\n");
  printf("不同配置的性能对比\n");
  printf("========================================\n\n");

  int N = 1024 * 1024;
  size_t bytes = N * sizeof(float);

  float *h_input = (float *)malloc(bytes);
  float *h_output = (float *)malloc(bytes);

  for (int i = 0; i < N; i++) {
    h_input[i] = (float)i;
  }

  float *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;
  int iterations = 100;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  float ms;

  // 测试不同 Cache 配置
  cudaFuncCache configs[] = {cudaFuncCachePreferShared, cudaFuncCachePreferL1,
                             cudaFuncCachePreferEqual};
  const char *config_names[] = {"PreferShared", "PreferL1", "PreferEqual"};

  printf("Cache 配置对比:\n");
  for (int i = 0; i < 3; i++) {
    CUDA_CHECK(cudaFuncSetCacheConfig(smem_intensive, configs[i]));

    // 预热
    smem_intensive<<<gridSize, blockSize>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    CUDA_CHECK(cudaEventRecord(start));
    for (int j = 0; j < iterations; j++) {
      smem_intensive<<<gridSize, blockSize>>>(d_input, d_output, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("  %-15s: %.3f ms\n", config_names[i], ms / iterations);
  }
  printf("\n");

  // 清理
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  free(h_input);
  free(h_output);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
  print_device_properties();
  demo_bank_config();
  demo_cache_config();
  demo_occupancy();
  demo_carveout();
  benchmark_configs();

  printf("========================================\n");
  printf("示例完成\n");
  printf("========================================\n");

  return 0;
}