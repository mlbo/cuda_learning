/**
 * 第二十一章示例：Pipeline Primitives
 *
 * 本示例演示CUDA Pipeline Primitives C风格接口
 * 提供更底层的异步操作控制
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Pipeline primitives头文件
#include <cuda_pipeline.h>

#define TILE_SIZE 32

// 同步版本
__global__ void sync_version(const float* __restrict__ input,
                              float* __restrict__ output,
                              int n, int batch_size) {
    __shared__ float smem[256];

    int batch_count = (n + batch_size - 1) / batch_size;

    for (int batch = 0; batch < batch_count; batch++) {
        int batch_offset = batch * batch_size;
        int tid = threadIdx.x;

        // 同步加载
        if (tid < batch_size && batch_offset + tid < n) {
            smem[tid] = input[batch_offset + tid];
        }
        __syncthreads();

        // 计算
        if (tid < batch_size && batch_offset + tid < n) {
            output[batch_offset + tid] = smem[tid] * 2.0f + 1.0f;
        }
        __syncthreads();
    }
}

// Pipeline primitives版本（单缓冲）
__global__ void pipeline_single_buffer(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int n, int batch_size) {
    __shared__ float smem[256];

    int batch_count = (n + batch_size - 1) / batch_size;
    int tid = threadIdx.x;

    for (int batch = 0; batch < batch_count; batch++) {
        int batch_offset = batch * batch_size;

        // 异步加载
        if (batch_offset + tid < n && tid < batch_size) {
            __pipeline_memcpy_async(
                &smem[tid],
                &input[batch_offset + tid],
                sizeof(float)
            );
        }
        __pipeline_commit();

        // 等待完成
        __pipeline_wait_prior(0);

        // 计算
        if (batch_offset + tid < n && tid < batch_size) {
            output[batch_offset + tid] = smem[tid] * 2.0f + 1.0f;
        }
    }
}

// Pipeline primitives版本（双缓冲）
__global__ void pipeline_double_buffer(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int n, int batch_size) {
    // 双缓冲共享内存
    __shared__ float smem[2][256];

    int batch_count = (n + batch_size - 1) / batch_size;
    int tid = threadIdx.x;
    int stage = 0;

    // 预加载第一个批次
    if (tid < batch_size) {
        __pipeline_memcpy_async(
            &smem[stage][tid],
            &input[tid],
            sizeof(float)
        );
    }
    __pipeline_commit();

    for (int batch = 0; batch < batch_count; batch++) {
        int next_stage = 1 - stage;
        int batch_offset = batch * batch_size;
        int next_batch_offset = (batch + 1) * batch_size;

        // 预加载下一个批次（如果存在）
        if (batch + 1 < batch_count && tid < batch_size && next_batch_offset + tid < n) {
            __pipeline_memcpy_async(
                &smem[next_stage][tid],
                &input[next_batch_offset + tid],
                sizeof(float)
            );
        }

        if (batch + 1 < batch_count) {
            __pipeline_commit();
        }

        // 等待当前批次数据
        __pipeline_wait_prior(0);

        // 计算（与下一批加载重叠）
        if (batch_offset + tid < n && tid < batch_size) {
            output[batch_offset + tid] = smem[stage][tid] * 2.0f + 1.0f;
        }

        stage = next_stage;
    }
}

// 多阶段Pipeline
__global__ void pipeline_multi_stage(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int n, int batch_size) {
    // 三缓冲实现3阶段流水线
    __shared__ float smem[3][256];

    int batch_count = (n + batch_size - 1) / batch_size;
    int tid = threadIdx.x;

    // 预加载前两个批次
    for (int i = 0; i < 2 && i < batch_count; i++) {
        int offset = i * batch_size;
        if (offset + tid < n && tid < batch_size) {
            __pipeline_memcpy_async(&smem[i][tid], &input[offset + tid], sizeof(float));
        }
        __pipeline_commit();
    }

    int stage = 0;

    for (int batch = 0; batch < batch_count; batch++) {
        int load_stage = (batch + 2) % 3;
        int next_batch_offset = (batch + 2) * batch_size;

        // 加载第batch+2批次的数据
        if (batch + 2 < batch_count && tid < batch_size && next_batch_offset + tid < n) {
            __pipeline_memcpy_async(
                &smem[load_stage][tid],
                &input[next_batch_offset + tid],
                sizeof(float)
            );
        }

        if (batch + 2 < batch_count) {
            __pipeline_commit();
        }

        // 等待当前批次
        __pipeline_wait_prior(0);

        // 计算
        int batch_offset = batch * batch_size;
        if (batch_offset + tid < n && tid < batch_size) {
            output[batch_offset + tid] = smem[stage][tid] * 2.0f + 1.0f;
        }

        stage = (stage + 1) % 3;
    }
}

// 使用对齐的memcpy_async
__global__ void aligned_memcpy_async(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int n) {
    __shared__ float smem[256];

    // 使用aligned_size_t确保对齐
    // 要求：地址和大小都对齐到指定边界

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 假设数据已经对齐到16字节
    if (idx < n) {
        // 单个元素的异步拷贝
        __pipeline_memcpy_async(&smem[tid], &input[idx], sizeof(float));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);

    if (idx < n) {
        output[idx] = smem[tid] * 2.0f;
    }
}

// 性能测试
void run_benchmark(int n) {
    float *d_input, *d_output;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemset(d_input, 1, n * sizeof(float));

    int batch_size = 256;
    int blockSize = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warmup = 5;
    int iterations = 100;

    // 预热同步版本
    for (int i = 0; i < warmup; i++) {
        sync_version<<<1, blockSize>>>(d_input, d_output, n, batch_size);
    }
    cudaDeviceSynchronize();

    // 测试同步版本
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        sync_version<<<1, blockSize>>>(d_input, d_output, n, batch_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float sync_time;
    cudaEventElapsedTime(&sync_time, start, stop);

    // 预热单缓冲pipeline
    for (int i = 0; i < warmup; i++) {
        pipeline_single_buffer<<<1, blockSize>>>(d_input, d_output, n, batch_size);
    }
    cudaDeviceSynchronize();

    // 测试单缓冲pipeline
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        pipeline_single_buffer<<<1, blockSize>>>(d_input, d_output, n, batch_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float single_time;
    cudaEventElapsedTime(&single_time, start, stop);

    // 预热双缓冲pipeline
    for (int i = 0; i < warmup; i++) {
        pipeline_double_buffer<<<1, blockSize>>>(d_input, d_output, n, batch_size);
    }
    cudaDeviceSynchronize();

    // 测试双缓冲pipeline
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        pipeline_double_buffer<<<1, blockSize>>>(d_input, d_output, n, batch_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float double_time;
    cudaEventElapsedTime(&double_time, start, stop);

    // 预热多阶段pipeline
    for (int i = 0; i < warmup; i++) {
        pipeline_multi_stage<<<1, blockSize>>>(d_input, d_output, n, batch_size);
    }
    cudaDeviceSynchronize();

    // 测试多阶段pipeline
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        pipeline_multi_stage<<<1, blockSize>>>(d_input, d_output, n, batch_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float multi_time;
    cudaEventElapsedTime(&multi_time, start, stop);

    printf("\n=== Pipeline Primitives性能测试 (n = %d) ===\n", n);
    printf("同步版本平均时间:     %.3f ms\n", sync_time / iterations);
    printf("单缓冲Pipeline平均:   %.3f ms\n", single_time / iterations);
    printf("双缓冲Pipeline平均:   %.3f ms\n", double_time / iterations);
    printf("多阶段Pipeline平均:   %.3f ms\n", multi_time / iterations);
    printf("双缓冲加速比:         %.2fx\n", sync_time / double_time);

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

    // 测试双缓冲版本
    pipeline_double_buffer<<<1, 256>>>(d_input, d_output, N, 256);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_output[i] - h_expected[i]) > 1e-3) {
            printf("双缓冲Pipeline错误 at %d: expected %.1f, got %.1f\n",
                   i, h_expected[i], h_output[i]);
            correct = false;
            break;
        }
    }
    printf("双缓冲Pipeline功能测试: %s\n", correct ? "通过" : "失败");

    // 测试多阶段版本
    cudaMemset(d_output, 0, N * sizeof(float));
    pipeline_multi_stage<<<1, 256>>>(d_input, d_output, N, 256);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    correct = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_output[i] - h_expected[i]) > 1e-3) {
            printf("多阶段Pipeline错误 at %d: expected %.1f, got %.1f\n",
                   i, h_expected[i], h_output[i]);
            correct = false;
            break;
        }
    }
    printf("多阶段Pipeline功能测试: %s\n", correct ? "通过" : "失败");

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    printf("========================================\n");
    printf("  Pipeline Primitives演示 - 第二十一章\n");
    printf("========================================\n");

    // 检查设备能力
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("设备: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    if (prop.major < 7) {
        printf("警告: Pipeline primitives建议SM 7.0+\n");
    }

    test_correctness();

    run_benchmark(1024);
    run_benchmark(4096);
    run_benchmark(16384);

    printf("\nPipeline Primitives API总结:\n");
    printf("  __pipeline_memcpy_async(dst, src, size) - 异步拷贝\n");
    printf("  __pipeline_commit()                     - 提交操作\n");
    printf("  __pipeline_wait_prior(N)                - 等待完成\n");
    printf("\n优化建议:\n");
    printf("  1. 使用对齐的内存地址和大小\n");
    printf("  2. 避免warp分化\n");
    printf("  3. 选择合适的缓冲区数量\n");
    printf("\n");

    return 0;
}
