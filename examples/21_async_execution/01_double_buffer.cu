/**
 * 第二十一章示例：双缓冲技术
 *
 * 本示例演示如何使用双缓冲实现计算与访存的重叠
 * 双缓冲使用两个共享内存缓冲区，交替进行加载和计算
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// 朴素版本（无延迟隐藏）
__global__ void naive_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              int n, int tile_count) {
    __shared__ float smem[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < tile_count; t++) {
        // 加载到共享内存
        int global_row = row;
        int global_col = t * TILE_SIZE + tx;

        if (global_row < n && global_col < n) {
            smem[ty][tx] = input[global_row * n + global_col];
        } else {
            smem[ty][tx] = 0.0f;
        }
        __syncthreads();

        // 计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += smem[ty][k];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        output[row * n + col] = sum;
    }
}

// 双缓冲版本
__global__ void double_buffer_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int n, int tile_count) {
    // 双缓冲：两个共享内存块
    __shared__ float smem[2][TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int load_buf = 0;
    int compute_buf = 1;

    // 预加载第一个Tile
    int global_col = 0 * TILE_SIZE + tx;
    if (row < n && global_col < n) {
        smem[load_buf][ty][tx] = input[row * n + global_col];
    } else {
        smem[load_buf][ty][tx] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < tile_count; t++) {
        // 交换缓冲区角色
        compute_buf = load_buf;
        load_buf = 1 - load_buf;

        // 预加载下一个Tile（如果存在）
        if (t + 1 < tile_count) {
            global_col = (t + 1) * TILE_SIZE + tx;
            if (row < n && global_col < n) {
                smem[load_buf][ty][tx] = input[row * n + global_col];
            } else {
                smem[load_buf][ty][tx] = 0.0f;
            }
        }

        // 计算当前Tile（与下一次加载重叠）
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += smem[compute_buf][ty][k];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        output[row * n + col] = sum;
    }
}

// 双缓冲GEMM版本
__global__ void double_buffer_gemm(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    // 双缓冲
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // 预加载第一个Tile
    int a_col = 0 * TILE_SIZE + tx;
    int b_row = 0 * TILE_SIZE + ty;

    if (row < M && a_col < K) {
        As[0][ty][tx] = A[row * K + a_col];
    } else {
        As[0][ty][tx] = 0.0f;
    }

    if (b_row < K && col < N) {
        Bs[0][ty][tx] = B[b_row * N + col];
    } else {
        Bs[0][ty][tx] = 0.0f;
    }
    __syncthreads();

    int stage = 0;

    // 主循环
    for (int t = 0; t < num_tiles; t++) {
        int next_stage = 1 - stage;

        // 预加载下一个Tile
        if (t + 1 < num_tiles) {
            a_col = (t + 1) * TILE_SIZE + tx;
            b_row = (t + 1) * TILE_SIZE + ty;

            if (row < M && a_col < K) {
                As[next_stage][ty][tx] = A[row * K + a_col];
            } else {
                As[next_stage][ty][tx] = 0.0f;
            }

            if (b_row < K && col < N) {
                Bs[next_stage][ty][tx] = B[b_row * N + col];
            } else {
                Bs[next_stage][ty][tx] = 0.0f;
            }
        }

        // 计算当前Tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[stage][ty][k] * Bs[stage][k][tx];
        }

        __syncthreads();
        stage = next_stage;
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 性能测试函数
void run_benchmark(int size) {
    float *d_input, *d_output, *d_A, *d_B, *d_C;

    size_t bytes = size * size * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 初始化数据
    cudaMemset(d_input, 1, bytes);
    cudaMemset(d_output, 0, bytes);
    cudaMemset(d_A, 1, bytes);
    cudaMemset(d_B, 1, bytes);
    cudaMemset(d_C, 0, bytes);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((size + TILE_SIZE - 1) / TILE_SIZE,
                 (size + TILE_SIZE - 1) / TILE_SIZE);
    int tile_count = (size + TILE_SIZE - 1) / TILE_SIZE;

    // CUDA事件计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warmup = 5;
    int repeat = 20;

    // 预热
    for (int i = 0; i < warmup; i++) {
        naive_kernel<<<gridDim, blockDim>>>(d_input, d_output, size, tile_count);
    }
    cudaDeviceSynchronize();

    // 测试朴素版本
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        naive_kernel<<<gridDim, blockDim>>>(d_input, d_output, size, tile_count);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naive_time;
    cudaEventElapsedTime(&naive_time, start, stop);

    // 预热双缓冲版本
    for (int i = 0; i < warmup; i++) {
        double_buffer_kernel<<<gridDim, blockDim>>>(d_input, d_output, size, tile_count);
    }
    cudaDeviceSynchronize();

    // 测试双缓冲版本
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        double_buffer_kernel<<<gridDim, blockDim>>>(d_input, d_output, size, tile_count);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float db_time;
    cudaEventElapsedTime(&db_time, start, stop);

    // 测试GEMM版本
    int K = size;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // 预热
    for (int i = 0; i < warmup; i++) {
        double_buffer_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, size, size, K);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        double_buffer_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, size, size, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gemm_time;
    cudaEventElapsedTime(&gemm_time, start, stop);

    printf("\n=== 双缓冲性能测试 (size = %d) ===\n", size);
    printf("朴素版本平均时间:    %.3f ms\n", naive_time / repeat);
    printf("双缓冲版本平均时间:  %.3f ms\n", db_time / repeat);
    printf("双缓冲GEMM平均时间:  %.3f ms\n", gemm_time / repeat);
    printf("双缓冲加速比:        %.2fx\n", naive_time / db_time);

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 简单功能测试
void test_correctness() {
    const int N = 64;
    float h_A[N * N], h_B[N * N], h_C[N * N];
    float *d_A, *d_B, *d_C;

    // 初始化数据
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(N / TILE_SIZE, N / TILE_SIZE);

    double_buffer_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, N, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果 (C = A * B, 其中 A和B都是1，所以 C[i][j] = N)
    bool correct = true;
    for (int i = 0; i < N * N; i++) {
        if (std::fabs(h_C[i] - N) > 1e-3) {
            printf("Error at index %d: expected %d, got %.1f\n", i, N, h_C[i]);
            correct = false;
            break;
        }
    }

    printf("功能测试: %s\n", correct ? "通过" : "失败");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    printf("========================================\n");
    printf("  双缓冲技术演示 - 第二十一章\n");
    printf("========================================\n");

    // 功能测试
    test_correctness();

    // 性能测试
    run_benchmark(256);
    run_benchmark(512);
    run_benchmark(1024);

    printf("\n提示: 使用 ncu 分析性能差异:\n");
    printf("  ncu --set full ./01_double_buffer\n");
    printf("\n");

    return 0;
}
