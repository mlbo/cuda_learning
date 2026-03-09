/**
 * 第二十一章示例：异步数据拷贝 (memcpy_async)
 *
 * 本示例演示CUDA 11.0引入的异步拷贝API
 * 直接从全局内存到共享内存的异步传输
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// 需要CUDA 11.0+
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

#define TILE_SIZE 32

// 同步版本：传统方式加载
__global__ void sync_load_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int n) {
    __shared__ float smem[TILE_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 同步加载：GMEM -> 寄存器 -> SMEM
    if (idx < n) {
        smem[tid] = input[idx];
    }
    __syncthreads();

    // 计算
    if (idx < n) {
        output[idx] = smem[tid] * 2.0f + 1.0f;
    }
}

// 使用cooperative_groups::memcpy_async
__global__ void cg_memcpy_async_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int n) {
    cg::thread_block block = cg::this_thread_block();
    __shared__ float smem[TILE_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 异步拷贝：GMEM -> SMEM 直接
    size_t copy_size = min((int)TILE_SIZE, n - blockIdx.x * blockDim.x);
    copy_size *= sizeof(float);

    if (copy_size > 0) {
        cg::memcpy_async(block, smem, input + blockIdx.x * blockDim.x, copy_size);
        cg::wait(block);  // 等待拷贝完成

        // 计算
        if (idx < n) {
            output[idx] = smem[tid] * 2.0f + 1.0f;
        }
    }
}

// 使用barrier的memcpy_async
#include <cuda/barrier>

__device__ void barrier_memcpy_example(const float* input, float* output, int n) {
    cg::thread_block block = cg::this_thread_block();
    __shared__ float smem[256];

    // 异步拷贝并使用cooperative groups同步 (CUDA 13+ API)
    cg::memcpy_async(block, smem, input, sizeof(float) * 256);
    cg::wait(block);

    // 计算逻辑...
}

// 使用memcpy_async的矩阵乘法
__global__ void async_memcpy_gemm(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    cg::thread_block block = cg::this_thread_block();

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // 加载A的Tile
        int a_col = t * TILE_SIZE + tx;
        int a_row_start = blockIdx.y * TILE_SIZE;
        int a_valid = min(TILE_SIZE, M - a_row_start);

        if (a_valid > 0 && t * TILE_SIZE < K) {
            int a_copy_size = min(TILE_SIZE, K - t * TILE_SIZE) * sizeof(float);
            cg::memcpy_async(block, &As[0][0], A + row * K + t * TILE_SIZE,
                           a_copy_size);
        }

        // 加载B的Tile
        int b_row = t * TILE_SIZE + ty;
        int b_col_start = blockIdx.x * TILE_SIZE;
        int b_valid = min(TILE_SIZE, N - b_col_start);

        if (b_valid > 0 && t * TILE_SIZE < K) {
            // B的访问模式需要特殊处理
            for (int i = ty; i < TILE_SIZE; i += blockDim.y) {
                int b_idx = (t * TILE_SIZE + i) * N + col;
                if (t * TILE_SIZE + i < K && col < N) {
                    Bs[i][tx] = B[b_idx];
                }
            }
        }

        // 等待数据加载完成
        cg::wait(block);

        // 计算
        for (int k = 0; k < TILE_SIZE && t * TILE_SIZE + k < K; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        cg::sync(block);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 批量异步拷贝示例
__global__ void batch_memcpy_async(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int n, int batch_size) {
    cg::thread_block block = cg::this_thread_block();
    __shared__ float smem[2][256];  // 双缓冲

    int num_batches = (n + batch_size - 1) / batch_size;
    int current_buf = 0;

    for (int batch = 0; batch < num_batches; batch++) {
        int next_buf = 1 - current_buf;
        int batch_offset = batch * batch_size;
        int copy_size = min(batch_size, n - batch_offset) * sizeof(float);

        // 异步加载当前批次
        if (copy_size > 0) {
            cg::memcpy_async(block, smem[current_buf], input + batch_offset, copy_size);
            cg::wait(block);

            // 计算（可以与下一批的加载重叠）
            int tid = threadIdx.x;
            if (tid < min(batch_size, n - batch_offset)) {
                output[batch_offset + tid] = smem[current_buf][tid] * 2.0f;
            }
        }

        current_buf = next_buf;
        cg::sync(block);
    }
}

// 性能测试
void run_benchmark(int n) {
    float *d_input, *d_output;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // 初始化
    cudaMemset(d_input, 1, n * sizeof(float));

    int blockSize = TILE_SIZE;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warmup = 5;
    int iterations = 100;

    // 预热同步版本
    for (int i = 0; i < warmup; i++) {
        sync_load_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
    }
    cudaDeviceSynchronize();

    // 测试同步版本
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        sync_load_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float sync_time;
    cudaEventElapsedTime(&sync_time, start, stop);

    // 预热异步版本
    for (int i = 0; i < warmup; i++) {
        cg_memcpy_async_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
    }
    cudaDeviceSynchronize();

    // 测试异步版本
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cg_memcpy_async_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float async_time;
    cudaEventElapsedTime(&async_time, start, stop);

    printf("\n=== memcpy_async性能测试 (n = %d) ===\n", n);
    printf("同步加载平均时间:  %.3f ms\n", sync_time / iterations);
    printf("异步加载平均时间:  %.3f ms\n", async_time / iterations);
    printf("异步加速比:        %.2fx\n", sync_time / async_time);

    // 计算带宽
    double bytes_moved = 2.0 * n * sizeof(float) * iterations;  // 读+写
    double bandwidth_sync = bytes_moved / (sync_time * 1e-3) / 1e9;
    double bandwidth_async = bytes_moved / (async_time * 1e-3) / 1e9;

    printf("同步带宽:          %.2f GB/s\n", bandwidth_sync);
    printf("异步带宽:          %.2f GB/s\n", bandwidth_async);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 功能测试
void test_correctness() {
    const int N = 256;
    float h_input[N], h_output[N], h_expected[N];
    float *d_input, *d_output;

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
        h_expected[i] = h_input[i] * 2.0f + 1.0f;
    }

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 测试cooperative_groups版本
    cg_memcpy_async_kernel<<<1, TILE_SIZE>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_output[i] - h_expected[i]) > 1e-3) {
            printf("错误 at %d: expected %.1f, got %.1f\n",
                   i, h_expected[i], h_output[i]);
            correct = false;
            break;
        }
    }
    printf("memcpy_async功能测试: %s\n", correct ? "通过" : "失败");

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    printf("========================================\n");
    printf("  memcpy_async演示 - 第二十一章\n");
    printf("========================================\n");

    // 检查CUDA版本
    int driverVersion;
    cudaDriverGetVersion(&driverVersion);
    printf("CUDA Driver版本: %d\n", driverVersion);

    if (driverVersion < 11000) {
        printf("警告: memcpy_async需要CUDA 11.0+\n");
        return 0;
    }

    test_correctness();

    run_benchmark(1024);
    run_benchmark(4096);
    run_benchmark(16384);

    printf("\n提示: 在SM 80+ (A100)上，memcpy_async有硬件加速\n");
    printf("  可以显著降低GMEM到SMEM的延迟\n");
    printf("\n");

    return 0;
}
