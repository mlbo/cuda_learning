# 第十八章示例：GEMM分块优化

本目录包含第十八章配套的示例代码，演示GEMM的高级分块优化技术。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_smem_caching.cu` | 共享内存缓存优化 |
| `02_1d_blocktiling.cu` | 1D Block Tiling优化 |
| `03_2d_blocktiling.cu` | 2D Block Tiling优化 |
| `04_warptiling.cu` | Warp Tiling优化 |

## 编译运行

```bash
# 编译所有示例
make all

# 运行单个示例
./01_smem_caching
./02_1d_blocktiling
./03_2d_blocktiling
./04_warptiling

# 使用 ncu 分析性能
ncu ./04_warptiling
```

## 学习要点

1. **SMEM Caching**：利用共享内存缓存Tile数据
2. **1D Block Tiling**：在M方向扩展每个线程的计算量
3. **2D Block Tiling**：同时在M和N方向扩展
4. **Warp Tiling**：层次化分块，充分利用GPU结构

## 性能对比

| 方法 | 关键优化 | 相对性能 |
|------|----------|----------|
| SMEM Caching | 共享内存缓存 | 基准 |
| 1D Block Tiling | M方向扩展 | ~1.37x |
| 2D Block Tiling | M+N方向扩展 | ~1.04x |
| Warp Tiling | 层次化分块 | ~1.19x |

## 与cuBLAS对比

我们的Warp Tiling实现可达到cuBLAS的约70-82%性能。