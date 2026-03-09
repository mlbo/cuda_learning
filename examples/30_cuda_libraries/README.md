# 第三十章示例：CUDA官方库实战

本目录包含CUDA官方库（cuBLAS、cuDNN、CUB、CUTLASS）的使用示例。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_cublas_gemm.cu` | cuBLAS GEMM使用示例 |
| `02_cudnn_conv.cu` | cuDNN卷积操作示例 |
| `03_cub_reduce.cu` | CUB并行原语示例 |
| `04_cutlass_gemm.cu` | CUTLASS GEMM示例 |

## 依赖要求

- CUDA Toolkit 11.0+
- cuDNN 8.0+ (用于cuDNN示例)
- CUTLASS 3.0+ (用于CUTLASS示例)

## 编译运行

### 使用Makefile

```bash
make all        # 编译所有示例
make run        # 运行所有示例
make clean      # 清理编译产物
```

### 使用CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make -j
```

## 各示例说明

### 01_cublas_gemm.cu

展示cuBLAS的基本使用：
- SGEMM（单精度GEMM）
- HGEMM（半精度GEMM）
- cuBLASLt轻量级API
- 性能对比

### 02_cudnn_conv.cu

展示cuDNN的卷积操作：
- 卷积前向传播
- 算法自动选择
- 工作空间管理
- 性能分析

### 03_cub_reduce.cu

展示CUB并行原语：
- Device级Reduce
- Device级Sort
- Device级Scan
- Block级操作

### 04_cutlass_gemm.cu

展示CUTLASS GEMM：
- 基本GEMM配置
- FP16 Tensor Core GEMM
- 自定义GEMM参数

## 学习建议

1. 先运行基本示例，理解API使用
2. 对比不同库实现相同功能的性能
3. 阅读官方文档了解更多API
4. 尝试在自己的项目中应用这些库