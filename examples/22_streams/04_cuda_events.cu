/**
 * 第二十二章示例：CUDA事件
 *
 * 本示例演示CUDA事件的使用
 * 包括精确计时、流间同步、事件查询等
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 计算核函数
__global__ void compute_kernel(float* data, int n, int work) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < work; i++) {
            val = val * 1.0001f + 0.0001f;
        }
        data[idx] = val;
    }
}

// 1. 基本事件计时
void basic_timing_example() {
    printf("\n=== 1. 基本事件计时 ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 0, n * sizeof(float)));

    // 创建事件
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 记录开始事件
    CHECK_CUDA(cudaEventRecord(start));

    // 执行核函数
    compute_kernel<<<numBlocks, blockSize>>>(d_data, n, 1000);

    // 记录结束事件
    CHECK_CUDA(cudaEventRecord(stop));

    // 等待事件完成
    CHECK_CUDA(cudaEventSynchronize(stop));

    // 计算时间
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("  核函数执行时间: %.3f ms\n", milliseconds);

    // 计算吞吐量
    double gflops = (double)n * 1000 * 2 / (milliseconds * 1e6);
    printf("  估算GFLOPS: %.2f\n", gflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
}

// 2. 多段代码计时
void multi_segment_timing() {
    printf("\n=== 2. 多段代码计时 ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    // 创建多个事件
    cudaEvent_t events[4];
    for (int i = 0; i < 4; i++) {
        CHECK_CUDA(cudaEventCreate(&events[i]));
    }

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 记录各阶段事件
    CHECK_CUDA(cudaEventRecord(events[0]));
    CHECK_CUDA(cudaMemset(d_data, 1, n * sizeof(float)));

    CHECK_CUDA(cudaEventRecord(events[1]));
    compute_kernel<<<numBlocks, blockSize>>>(d_data, n, 500);

    CHECK_CUDA(cudaEventRecord(events[2]));
    compute_kernel<<<numBlocks, blockSize>>>(d_data, n, 1000);

    CHECK_CUDA(cudaEventRecord(events[3]));
    CHECK_CUDA(cudaEventSynchronize(events[3]));

    // 计算各段时间
    float times[3];
    for (int i = 0; i < 3; i++) {
        CHECK_CUDA(cudaEventElapsedTime(&times[i], events[i], events[i+1]));
    }

    printf("  Memset时间:    %.3f ms\n", times[0]);
    printf("  Kernel1时间:   %.3f ms\n", times[1]);
    printf("  Kernel2时间:   %.3f ms\n", times[2]);

    float total;
    CHECK_CUDA(cudaEventElapsedTime(&total, events[0], events[3]));
    printf("  总时间:        %.3f ms\n", total);

    for (int i = 0; i < 4; i++) {
        CHECK_CUDA(cudaEventDestroy(events[i]));
    }
    CHECK_CUDA(cudaFree(d_data));
}

// 3. 使用事件进行流间同步
void event_stream_sync() {
    printf("\n=== 3. 事件流间同步 ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    // 创建两个流
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    // 创建事件
    cudaEvent_t event;
    CHECK_CUDA(cudaEventCreate(&event));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Stream 1: 生产者
    CHECK_CUDA(cudaEventRecord(start, stream1));
    compute_kernel<<<numBlocks, blockSize, 0, stream1>>>(d_data, n, 1000);
    CHECK_CUDA(cudaEventRecord(event, stream1));  // 标记完成

    // Stream 2: 消费者，等待 Stream 1
    CHECK_CUDA(cudaStreamWaitEvent(stream2, event, 0));
    compute_kernel<<<numBlocks, blockSize, 0, stream2>>>(d_data, n, 500);
    CHECK_CUDA(cudaEventRecord(stop, stream2));

    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("  流间同步执行时间: %.3f ms\n", ms);
    printf("  Stream 2 正确等待 Stream 1 完成\n");

    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaEventDestroy(event));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
}

// 4. 事件查询（非阻塞）
void event_query_example() {
    printf("\n=== 4. 事件查询 ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t event;
    CHECK_CUDA(cudaEventCreate(&event));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 执行核函数并记录事件
    compute_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 5000);
    CHECK_CUDA(cudaEventRecord(event, stream));

    printf("  使用 cudaEventQuery 非阻塞检查...\n");

    int poll_count = 0;
    cudaError_t result;

    while ((result = cudaEventQuery(event)) == cudaErrorNotReady) {
        poll_count++;
        if (poll_count % 100000 == 0) {
            printf("  事件未就绪，轮询次数: %d\n", poll_count);
        }
    }

    if (result == cudaSuccess) {
        printf("  事件已触发，轮询了 %d 次\n", poll_count);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(event));
    CHECK_CUDA(cudaFree(d_data));
}

// 5. 事件属性和标志
void event_flags_example() {
    printf("\n=== 5. 事件标志 ===\n");

    // 默认事件
    cudaEvent_t default_event;
    CHECK_CUDA(cudaEventCreate(&default_event));
    printf("  创建默认事件\n");

    // 禁用计时的事件（开销更小）
    cudaEvent_t no_timing_event;
    CHECK_CUDA(cudaEventCreateWithFlags(&no_timing_event, cudaEventDisableTiming));
    printf("  创建禁用计时事件 (cudaEventDisableTiming)\n");

    // 支持IPC的事件（需要特定设备支持）
    cudaEvent_t ipc_event;
    cudaError_t ipc_result = cudaEventCreateWithFlags(&ipc_event, cudaEventInterprocess);
    if (ipc_result == cudaSuccess) {
        printf("  创建支持IPC事件 (cudaEventInterprocess)\n");
        CHECK_CUDA(cudaEventDestroy(ipc_event));
    } else {
        printf("  IPC事件不支持 (需要特定设备配置，跳过)\n");
    }

    // 性能对比
    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    const int iterations = 100;

    // 测试默认事件
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        compute_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 10);
        CHECK_CUDA(cudaEventRecord(default_event, stream));
        CHECK_CUDA(cudaEventSynchronize(default_event));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float default_time;
    CHECK_CUDA(cudaEventElapsedTime(&default_time, start, stop));

    // 测试禁用计时事件
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        compute_kernel<<<numBlocks, blockSize, 0, stream>>>(d_data, n, 10);
        CHECK_CUDA(cudaEventRecord(no_timing_event, stream));
        CHECK_CUDA(cudaEventSynchronize(no_timing_event));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float no_timing_time;
    CHECK_CUDA(cudaEventElapsedTime(&no_timing_time, start, stop));

    printf("  默认事件开销:     %.3f ms (%d 次)\n", default_time / iterations, iterations);
    printf("  禁用计时开销:     %.3f ms (%d 次)\n", no_timing_time / iterations, iterations);
    printf("  仅用于同步时，禁用计时事件开销更小\n");

    CHECK_CUDA(cudaEventDestroy(default_event));
    CHECK_CUDA(cudaEventDestroy(no_timing_event));
    // ipc_event 已经在上面处理过了，不需要再次销毁
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_data));
}

// 6. 多流计时
void multi_stream_timing() {
    printf("\n=== 6. 多流计时 ===\n");

    const int n = 256 * 1024;
    const int num_streams = 4;
    int chunk_size = n / num_streams;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t streams[num_streams];
    cudaEvent_t start_events[num_streams];
    cudaEvent_t stop_events[num_streams];

    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        CHECK_CUDA(cudaEventCreate(&start_events[i]));
        CHECK_CUDA(cudaEventCreate(&stop_events[i]));
    }

    int blockSize = 256;
    int numBlocks = (chunk_size + blockSize - 1) / blockSize;

    // 在各流中记录并执行
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA(cudaEventRecord(start_events[i], streams[i]));
        compute_kernel<<<numBlocks, blockSize, 0, streams[i]>>>(
            d_data + i * chunk_size, chunk_size, 1000);
        CHECK_CUDA(cudaEventRecord(stop_events[i], streams[i]));
    }

    // 同步所有流
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA(cudaEventSynchronize(stop_events[i]));
    }

    // 报告各流时间
    printf("  各流执行时间:\n");
    for (int i = 0; i < num_streams; i++) {
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_events[i], stop_events[i]));
        printf("    Stream %d: %.3f ms\n", i, ms);
    }

    // 清理
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        CHECK_CUDA(cudaEventDestroy(start_events[i]));
        CHECK_CUDA(cudaEventDestroy(stop_events[i]));
    }
    CHECK_CUDA(cudaFree(d_data));
}

// 计时辅助类
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
    }

    float elapsed_ms() {
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

void timer_class_example() {
    printf("\n=== 7. CudaTimer辅助类 ===\n");

    const int n = 1024 * 1024;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));

    CudaTimer timer;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    timer.start();
    compute_kernel<<<numBlocks, blockSize>>>(d_data, n, 1000);
    timer.stop();

    printf("  使用CudaTimer: %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_data));
}

int main() {
    printf("========================================\n");
    printf("  CUDA事件演示 - 第二十二章\n");
    printf("========================================\n");

    basic_timing_example();
    multi_segment_timing();
    event_stream_sync();
    event_query_example();
    event_flags_example();
    multi_stream_timing();
    timer_class_example();

    printf("\n========================================\n");
    printf("事件API总结:\n");
    printf("  cudaEventCreate:       创建事件\n");
    printf("  cudaEventDestroy:      销毁事件\n");
    printf("  cudaEventRecord:       记录事件\n");
    printf("  cudaEventSynchronize:  同步事件\n");
    printf("  cudaEventQuery:        查询状态\n");
    printf("  cudaEventElapsedTime:  计算时间\n");
    printf("========================================\n\n");

    return 0;
}