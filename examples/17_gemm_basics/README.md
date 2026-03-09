# 第十七章示例：GEMM优化入门

本目录包含第十七章配套的示例代码，演示GEMM的基础实现与优化。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_naive_gemm.cu` | 朴素GEMM实现 - 展示访存问题 |
| `02_coalesced_gemm.cu` | 内存合并优化的GEMM |
| `03_tiled_gemm.cu` | 分块GEMM实现 |

## 编译运行

```bash
# 编译所有示例
make all

# 运行单个示例
./01_naive_gemm
./02_coalesced_gemm
./03_tiled_gemm

# 使用 ncu 分析性能
ncu ./01_naive_gemm
ncu ./02_coalesced_gemm
ncu ./03_tiled_gemm
```

## 学习要点

1. **朴素实现的问题**：理解B矩阵访问的内存合并问题
2. **内存合并优化**：如何通过调整访存模式提高性能
3. **分块技术**：利用共享内存减少全局内存访问
4. **数据复用**：理解矩阵乘法中的数据复用模式

## 性能对比

| 方法 | 关键优化 | 相对性能 |
|------|----------|----------|
| 朴素实现 | 无 | 1x |
| 内存合并 | 访存模式优化 | ~12x |
| 分块GEMM | 共享内存缓存 | ~15x |

## NCU分析要点

```bash
# 分析内存合并情况
ncu --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    ./01_naive_gemm

# 分析共享内存使用
ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared.op_ld_sum,\
    l1tex__data_pipe_lsu_wavefronts_mem_shared.op_st_sum \
    ./03_tiled_gemm
```