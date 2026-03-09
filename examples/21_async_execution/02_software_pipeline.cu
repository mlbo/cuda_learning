/**
 * 第二十一章示例：软件流水线
 *
 * 本示例演示软件流水线技术和Warp特化
 * 将任务分成多个阶段并行执行，实现更高的吞吐量
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define NUM_STAGES 3  // 3级流水线

// 朴素版本
__global__ void naive_pipeline(const float* __restrict__ input,
                                float* __restrict__ output,
                                int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int iter = 0; iter < iterations; iter++) {
        for (int i = idx; i < n; i += stride) {
            // 阶段1: 加载
            float val = input[i];

            // 阶段2: 计算
            val = val * 2.0f + 1.0f;

            // 阶段3: 存储
            output[i] = val;
        }
    }
}

// 软件流水线版本（使用共享内存）
__global__ void software_pipeline(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int n, int batch_size) {
    __shared__ float buffer[NUM_STAGES][TILE_SIZE * TILE_SIZE];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int batch_count = (n + batch_size - 1) / batch_size;

    // 流水线寄存器
    float stage_data[NUM_STAGES];
    int stage_valid[NUM_STAGES] = {0};

    for (int batch = 0; batch < batch_count + NUM_STAGES - 1; batch++) {
        // 阶段1: 加载数据到Stage 0
        if (batch < batch_count) {
            int load_idx = batch * batch_size + tid;
            if (load_idx < n) {
                stage_data[0] = input[load_idx];
                stage_valid[0] = 1;
            } else {
                stage_valid[0] = 0;
            }
        }

        // 阶段2: 处理Stage 1的数据
        if (batch >= 1 && batch < batch_count + 1 && stage_valid[1]) {
            stage_data[1] = stage_data[1] * 2.0f + 1.0f;
        }

        // 阶段3: 写回Stage 2的数据
        if (batch >= 2 && batch < batch_count + 2 && stage_valid[2]) {
            int store_idx = (batch - 2) * batch_size + tid;
            if (store_idx < n) {
                output[store_idx] = stage_data[2];
            }
        }

        // 流水线传递：数据向后移动
        for (int s = NUM_STAGES - 1; s > 0; s--) {
            stage_data[s] = stage_data[s - 1];
            stage_valid[s] = stage_valid[s - 1];
        }
    }
}

// Warp特化版本
__global__ void warp_specialization(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int n, int batch_size) {
    __shared__ float smem_buffer[1024];
    __shared__ volatile int producer_ready;
    __shared__ volatile int consumer_done;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int is_producer = (warp_id < 4);  // 前4个warp作为生产者
    int is_consumer = !is_producer;    // 后4个warp作为消费者

    int batch_count = (n + batch_size - 1) / batch_size;

    if (threadIdx.x == 0) {
        producer_ready = 0;
        consumer_done = 1;  // 初始状态：消费者已完成
    }
    __syncthreads();

    for (int batch = 0; batch < batch_count; batch++) {
        if (is_producer) {
            // 生产者：等待消费者完成上一批
            if (lane_id == 0) {
                while (consumer_done == 0);
            }
            __syncwarp();

            // 加载数据到共享内存
            int load_base = batch * batch_size;
            for (int i = lane_id; i < batch_size; i += 32) {
                int idx = load_base + i;
                if (idx < n) {
                    smem_buffer[i] = input[idx];
                }
            }

            // 通知消费者数据就绪
            __syncwarp();
            if (warp_id == 0 && lane_id == 0) {
                producer_ready = 1;
                consumer_done = 0;
            }
        } else {
            // 消费者：等待生产者数据就绪
            if (lane_id == 0) {
                while (producer_ready == 0);
            }
            __syncwarp();

            // 从共享内存读取并计算
            int consumer_warp = warp_id - 4;  // 0-3
            int items_per_warp = batch_size / 4;
            int start = consumer_warp * items_per_warp;

            for (int i = start + lane_id; i < start + items_per_warp; i += 32) {
                int idx = batch * batch_size + i;
                if (idx < n) {
                    float val = smem_buffer[i];
                    val = val * 2.0f + 1.0f;
                    output[idx] = val;
                }
            }

            // 通知生产者消费者完成
            __syncwarp();
            if (warp_id == 4 && lane_id == 0) {
                producer_ready = 0;
                consumer_done = 1;
            }
        }
    }
}

// 使用Cooperative Groups的软件流水线
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void cg_pipeline(const float* __restrict__ input,
                            float* __restrict__ output,
                            int n, int batch_size) {
    cg::thread_block block = cg::this_thread_block();

    __shared__ float stage0[256];  // 加载阶段
    __shared__ float stage1[256];  // 计算阶段
    __shared__ float stage2[256];  // 存储阶段

    int batch_count = (n + batch_size - 1) / batch_size;

    // 初始化：加载第一批
    if (threadIdx.x < batch_size) {
        int idx = threadIdx.x;
        if (idx < n) {
            stage0[threadIdx.x] = input[idx];
        }
    }
    cg::sync(block);

    for (int batch = 0; batch < batch_count + 2; batch++) {
        // 阶段流动
        // Stage 1 -> Stage 2 (存储)
        if (batch >= 2 && threadIdx.x < batch_size) {
            int store_idx = (batch - 2) * batch_size + threadIdx.x;
            if (store_idx < n) {
                output[store_idx] = stage2[threadIdx.x];
            }
        }

        // Stage 0 -> Stage 1 (计算)
        if (batch >= 1 && threadIdx.x < batch_size) {
            stage1[threadIdx.x] = stage0[threadIdx.x] * 2.0f + 1.0f;
        }

        // 加载新的到Stage 0
        if (batch < batch_count && threadIdx.x < batch_size) {
            int load_idx = (batch + 1) * batch_size + threadIdx.x;
            if (load_idx < n) {
                stage0[threadIdx.x] = input[load_idx];
            }
        }

        // Stage流动：stage1 -> stage2, stage0 -> stage1
        cg::sync(block);

        // 复制数据
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            stage2[i] = stage1[i];
            stage1[i] = stage0[i];
        }

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
    cudaMemset(d_output, 0, n * sizeof(float));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    int iterations = 100;
    int batch_size = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warmup = 5;

    // 预热
    for (int i = 0; i < warmup; i++) {
        naive_pipeline<<<numBlocks, blockSize>>>(d_input, d_output, n, 1);
    }
    cudaDeviceSynchronize();

    // 朴素版本
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        naive_pipeline<<<numBlocks, blockSize>>>(d_input, d_output, n, 1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naive_time;
    cudaEventElapsedTime(&naive_time, start, stop);

    // 预热软件流水线
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(4);
    for (int i = 0; i < warmup; i++) {
        software_pipeline<<<gridDim, blockDim>>>(d_input, d_output, n, batch_size);
    }
    cudaDeviceSynchronize();

    // 软件流水线版本
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        software_pipeline<<<gridDim, blockDim>>>(d_input, d_output, n, batch_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float pipeline_time;
    cudaEventElapsedTime(&pipeline_time, start, stop);

    // Warp特化版本
    for (int i = 0; i < warmup; i++) {
        warp_specialization<<<1, 256>>>(d_input, d_output, n, batch_size);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        warp_specialization<<<1, 256>>>(d_input, d_output, n, batch_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float warp_time;
    cudaEventElapsedTime(&warp_time, start, stop);

    printf("\n=== 软件流水线性能测试 (n = %d) ===\n", n);
    printf("朴素版本平均时间:      %.3f ms\n", naive_time / iterations);
    printf("软件流水线平均时间:    %.3f ms\n", pipeline_time / iterations);
    printf("Warp特化平均时间:      %.3f ms\n", warp_time / iterations);
    printf("软件流水线加速比:      %.2fx\n", naive_time / pipeline_time);
    printf("Warp特化加速比:        %.2fx\n", naive_time / warp_time);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 功能测试
void test_correctness() {
    const int N = 1024;
    float h_input[N], h_output[N], h_expected[N];
    float *d_input, *d_output;

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
        h_expected[i] = h_input[i] * 2.0f + 1.0f;
    }

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(1);

    software_pipeline<<<gridDim, blockDim>>>(d_input, d_output, N, 256);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_output[i] - h_expected[i]) > 1e-3) {
            printf("软件流水线错误 at %d: expected %.1f, got %.1f\n",
                   i, h_expected[i], h_output[i]);
            correct = false;
            break;
        }
    }
    printf("软件流水线功能测试: %s\n", correct ? "通过" : "失败");

    // 测试Warp特化
    cudaMemset(d_output, 0, N * sizeof(float));
    warp_specialization<<<1, 256>>>(d_input, d_output, N, 256);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    correct = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_output[i] - h_expected[i]) > 1e-3) {
            printf("Warp特化错误 at %d: expected %.1f, got %.1f\n",
                   i, h_expected[i], h_output[i]);
            correct = false;
            break;
        }
    }
    printf("Warp特化功能测试: %s\n", correct ? "通过" : "失败");

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    printf("========================================\n");
    printf("  软件流水线演示 - 第二十一章\n");
    printf("========================================\n");

    test_correctness();

    run_benchmark(1024);
    run_benchmark(4096);
    run_benchmark(16384);

    printf("\n提示: 流水线技术在以下场景效果最好:\n");
    printf("  1. 多个阶段时间相近\n");
    printf("  2. 数据量足够大\n");
    printf("  3. 有足够的并行资源\n");
    printf("\n");

    return 0;
}
