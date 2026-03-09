# 第十九章示例：Tensor Core编程

本目录包含第十九章配套的示例代码，演示Tensor Core的使用方法。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_tensor_core_basics.cu` | Tensor Core基础概念与检查 |
| `02_wmma_gemm.cu` | WMMA API实现矩阵乘法 |
| `03_mixed_precision.cu` | 混合精度计算示例 |

## 编译运行

```bash
# 编译
make all

# 运行
./01_tensor_core_basics
./02_wmma_gemm
./03_mixed_precision
```

## 注意事项

1. Tensor Core需要Compute Capability 7.0+ (Volta架构起)
2. 编译时需要指定正确的架构：`-arch=sm_70` 或更高
3. 混合精度计算需要注意精度损失

## Tensor Core支持

| 架构 | Tensor Core类型 |
|------|-----------------|
| Volta (SM70) | FP16/FP32 |
| Turing (SM75) | FP16/FP32, INT8/INT32 |
| Ampere (SM80) | FP16/FP32, BF16/FP32, TF32, INT8/INT32 |
| Hopper (SM90) | FP8, FP16, BF16, TF32, INT8 |