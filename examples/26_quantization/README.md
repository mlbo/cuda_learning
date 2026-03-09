# 第二十六章示例：低精度与量化

本目录包含第二十六章配套的示例代码，演示低精度计算和量化技术。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_fp16_basics.cu` | FP16半精度浮点基础 |
| `02_bf16_tf32.cu` | BF16和TF32使用 |
| `03_int8_quantize.cu` | INT8量化基础 |
| `04_quantized_gemm.cu` | 量化GEMM实现 |

## 编译运行

```bash
# 编译所有示例
make all

# 运行单个示例
./01_fp16_basics
./02_bf16_tf32
./03_int8_quantize
./04_quantized_gemm

# 使用ncu分析性能
ncu --set full -o fp16_profile ./01_fp16_basics
ncu-ui fp16_profile.ncu-rep
```

## 学习要点

1. **FP16基础**：理解FP16表示和使用方法
2. **向量化操作**：掌握__half2 SIMD操作
3. **INT8量化**：理解量化原理和实现
4. **Tensor Core**：学习混合精度矩阵加速

## 硬件要求

- **FP16**: Kepler及更新架构 (sm_70+性能最佳)
- **BF16**: Ampere及更新架构 (sm_80+)
- **TF32**: Ampere及更新架构 (sm_80+)
- **INT8 Tensor Core**: Volta及更新架构 (sm_70+)
- **FP8**: Hopper及更新架构 (sm_90+)

## 精度类型对比

```
类型    位宽   数值范围              精度
FP32    32    ±3.4e38              ~7位小数
FP16    16    ±65504               ~3位小数
BF16    16    ±3.4e38              ~2位小数
TF32    19    ±3.4e38              ~3位小数
INT8    8     -128 ~ 127           整数
FP8     8     ~±448(E4M3)/~±57344(E5M2)
```

## 性能对比

| 精度 | 存储大小 | 带宽需求 | Tensor Core性能 |
|------|----------|----------|-----------------|
| FP32 | 4 bytes | 100% | 基准 |
| FP16 | 2 bytes | 50% | ~2-8x |
| BF16 | 2 bytes | 50% | ~2-8x |
| INT8 | 1 byte | 25% | ~4-16x |

## 关键API

### FP16
```cpp
#include <cuda_fp16.h>
__half h = __float2half(1.0f);
float f = __half2float(h);
__half2 h2 = __floats2half2_rn(1.0f, 2.0f);
```

### BF16
```cpp
#include <cuda_bf16.h>
__nv_bfloat16 bf = __float2bfloat16(1.0f);
float f = __bfloat162float(bf);
```

### Tensor Core WMMA
```cpp
#include <mma.h>
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::load_matrix_sync(a_frag, ptr, stride);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

## 量化公式

**对称量化**:
```
Q = round(x / scale)
x' = Q * scale
scale = max(|x|) / 127
```

**非对称量化**:
```
Q = round(x / scale) + zero_point
x' = (Q - zero_point) * scale
scale = (max - min) / 255
zero_point = round(-min / scale)
```

## 代码结构

```
examples/26_quantization/
├── README.md               # 本文件
├── Makefile                # 编译脚本
├── CMakeLists.txt          # CMake配置
├── 01_fp16_basics.cu       # FP16基础示例
├── 02_bf16_tf32.cu         # BF16/TF32示例
├── 03_int8_quantize.cu     # INT8量化示例
└── 04_quantized_gemm.cu    # 量化GEMM示例
```