/**
 * @file 02_cudnn_conv.cu
 * @brief cuDNN卷积操作示例
 *
 * 本示例展示：
 * 1. cuDNN卷积前向传播
 * 2. 算法自动选择
 * 3. 工作空间管理
 * 4. 性能测量
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// 打印张量维度
// ============================================================================
void print_tensor_desc(const char* name, cudnnTensorDescriptor_t desc) {
    int n, c, h, w, nStride, cStride, hStride, wStride;
    cudnnDataType_t dataType;
    CUDNN_CHECK(cudnnGetTensor4dDescriptor(desc, &dataType, &n, &c, &h, &w,
                                           &nStride, &cStride, &hStride, &wStride));
    printf("%s: N=%d, C=%d, H=%d, W=%d\n", name, n, c, h, w);
}

// ============================================================================
// cuDNN卷积示例
// ============================================================================
void cudnn_conv_demo() {
    printf("=== cuDNN卷积示例 ===\n\n");

    // 创建cuDNN句柄
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // 卷积参数
    int N = 32;       // batch size
    int inC = 64;     // 输入通道数
    int inH = 28;     // 输入高度
    int inW = 28;     // 输入宽度
    int outC = 128;   // 输出通道数
    int kH = 3;       // 卷积核高度
    int kW = 3;       // 卷积核宽度
    int padH = 1;     // padding
    int padW = 1;
    int strideH = 1;  // stride
    int strideW = 1;
    int dilH = 1;     // dilation
    int dilW = 1;

    printf("输入张量: N=%d, C=%d, H=%d, W=%d\n", N, inC, inH, inW);
    printf("卷积核: %d x %d x %d x %d\n", outC, inC, kH, kW);
    printf("Padding: %d x %d, Stride: %d x %d\n", padH, padW, strideH, strideW);

    // 计算输出维度
    int outH = (inH + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
    int outW = (inW + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;

    printf("输出张量: N=%d, C=%d, H=%d, W=%d\n\n", N, outC, outH, outW);

    // 创建输入张量描述符
    cudnnTensorDescriptor_t inputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, N, inC, inH, inW));

    // 创建卷积核描述符
    cudnnFilterDescriptor_t filterDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW, outC, inC, kH, kW));

    // 创建卷积描述符
    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, padH, padW,
                                                strideH, strideW, dilH, dilW,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // 创建输出张量描述符
    cudnnTensorDescriptor_t outputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, N, outC, outH, outW));

    // 分配设备内存
    float *d_input, *d_filter, *d_output;
    size_t inputSize = N * inC * inH * inW * sizeof(float);
    size_t filterSize = outC * inC * kH * kW * sizeof(float);
    size_t outputSize = N * outC * outH * outW * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMalloc(&d_filter, filterSize));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));

    // 初始化数据
    float* h_input = (float*)malloc(inputSize);
    float* h_filter = (float*)malloc(filterSize);
    for (size_t i = 0; i < inputSize / sizeof(float); i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }
    for (size_t i = 0; i < filterSize / sizeof(float); i++) {
        h_filter[i] = (float)(i % 50) / 50.0f;
    }

    CUDA_CHECK(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice));

    // 选择卷积算法
    int requestedAlgoCount = 10;
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t algoPerf[10];

    printf("=== 查找最优算法 ===\n");
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc,
                                                     convDesc, outputDesc,
                                                     requestedAlgoCount,
                                                     &returnedAlgoCount,
                                                     algoPerf));

    printf("找到 %d 个算法:\n", returnedAlgoCount);
    for (int i = 0; i < returnedAlgoCount; i++) {
        printf("  算法 %d: %s, 时间=%.3f ms, 内存=%zu bytes, 确定性=%d\n",
               i, cudnnGetConvolutionForwardAlgorithmString(algoPerf[i].algo),
               algoPerf[i].time, algoPerf[i].memory, algoPerf[i].determinism);
    }

    // 选择最快的算法
    cudnnConvolutionFwdAlgo_t algo = algoPerf[0].algo;
    size_t workspaceSize = algoPerf[0].memory;

    printf("\n选择算法: %s, 工作空间: %zu bytes\n",
           cudnnGetConvolutionForwardAlgorithmString(algo), workspaceSize);

    // 分配工作空间
    void* d_workspace = nullptr;
    if (workspaceSize > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));
    }

    // 执行卷积
    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
                                        inputDesc, d_input,
                                        filterDesc, d_filter,
                                        convDesc, algo,
                                        d_workspace, workspaceSize,
                                        &beta,
                                        outputDesc, d_output));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    int repeat = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
                                            inputDesc, d_input,
                                            filterDesc, d_filter,
                                            convDesc, algo,
                                            d_workspace, workspaceSize,
                                            &beta,
                                            outputDesc, d_output));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= repeat;

    // 计算性能
    double flops = 2.0 * N * outC * outH * outW * inC * kH * kW;
    double gflops = flops / (ms * 1e6);

    printf("\n=== 性能结果 ===\n");
    printf("执行时间: %.4f ms\n", ms);
    printf("性能: %.2f GFLOPS\n", gflops);

    // 测试不同算法
    printf("\n=== 不同算法性能对比 ===\n");
    for (int i = 0; i < (returnedAlgoCount < 5 ? returnedAlgoCount : 5); i++) {
        cudnnConvolutionFwdAlgo_t testAlgo = algoPerf[i].algo;
        size_t testWorkspace = algoPerf[i].memory;

        void* testWorkspacePtr = nullptr;
        if (testWorkspace > 0) {
            CUDA_CHECK(cudaMalloc(&testWorkspacePtr, testWorkspace));
        }

        // 预热
        CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
                                            inputDesc, d_input,
                                            filterDesc, d_filter,
                                            convDesc, testAlgo,
                                            testWorkspacePtr, testWorkspace,
                                            &beta, outputDesc, d_output));
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计时
        CUDA_CHECK(cudaEventRecord(start));
        for (int j = 0; j < repeat; j++) {
            CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
                                                inputDesc, d_input,
                                                filterDesc, d_filter,
                                                convDesc, testAlgo,
                                                testWorkspacePtr, testWorkspace,
                                                &beta, outputDesc, d_output));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float testMs;
        CUDA_CHECK(cudaEventElapsedTime(&testMs, start, stop));
        testMs /= repeat;

        printf("算法 %s: %.4f ms (%.2f GFLOPS)\n",
               cudnnGetConvolutionForwardAlgorithmString(testAlgo),
               testMs, flops / (testMs * 1e6));

        if (testWorkspacePtr) {
            CUDA_CHECK(cudaFree(testWorkspacePtr));
        }
    }

    // 清理
    free(h_input);
    free(h_filter);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output));
    if (d_workspace) {
        CUDA_CHECK(cudaFree(d_workspace));
    }

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CHECK(cudnnDestroy(cudnn));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Pooling示例
// ============================================================================
void cudnn_pool_demo() {
    printf("\n=== cuDNN Pooling示例 ===\n\n");

    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    int N = 32, C = 64, H = 28, W = 28;
    int poolH = 2, poolW = 2;
    int strideH = 2, strideW = 2;
    int padH = 0, padW = 0;

    int outH = (H + 2 * padH - poolH) / strideH + 1;
    int outW = (W + 2 * padW - poolW) / strideW + 1;

    printf("输入: N=%d, C=%d, H=%d, W=%d\n", N, C, H, W);
    printf("Pool: %dx%d, Stride: %dx%d\n", poolH, poolW, strideH, strideW);
    printf("输出: N=%d, C=%d, H=%d, W=%d\n\n", N, C, outH, outW);

    // 创建描述符
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, N, C, H, W));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, N, C, outH, outW));

    cudnnPoolingDescriptor_t poolDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            poolH, poolW, padH, padW,
                                            strideH, strideW));

    // 分配内存
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * C * outH * outW * sizeof(float)));

    // 执行
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(cudnn, poolDesc, &alpha,
                                    inputDesc, d_input,
                                    &beta, outputDesc, d_output));

    printf("Pooling操作完成\n");

    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolDesc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

// ============================================================================
// Activation示例
// ============================================================================
void cudnn_activation_demo() {
    printf("\n=== cuDNN Activation示例 ===\n\n");

    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    int N = 32, C = 64, H = 28, W = 28;

    // 创建描述符
    cudnnTensorDescriptor_t tensorDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensorDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, N, C, H, W));

    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU,
                                             CUDNN_NOT_PROPAGATE_NAN, 0.0));

    // 分配内存
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * C * H * W * sizeof(float)));

    // 执行ReLU
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnActivationForward(cudnn, actDesc, &alpha,
                                       tensorDesc, d_in,
                                       &beta, tensorDesc, d_out));

    printf("ReLU激活完成\n");

    // 清理
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensorDesc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("===== cuDNN示例程序 =====\n\n");

    // 检查cuDNN版本
    size_t version;
    cudnnGetProperty(MAJOR_VERSION, &version);
    printf("cuDNN版本: %zu\n\n", version);

    // 运行示例
    cudnn_conv_demo();
    cudnn_pool_demo();
    cudnn_activation_demo();

    printf("\n=== 提示 ===\n");
    printf("使用cudnnFindConvolutionForwardAlgorithm选择最优算法\n");
    printf("工作空间大小影响算法选择，更大的工作空间可能获得更好性能\n");
    printf("FP16/BF16精度可以利用Tensor Core加速\n");

    return 0;
}