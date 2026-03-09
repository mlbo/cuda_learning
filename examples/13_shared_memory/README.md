# 第13章：共享内存深入

本章示例演示共享内存的基本概念、使用方法和优化技巧。

## 示例列表

| 文件 | 描述 |
|------|------|
| `01_smem_basics.cu` | 共享内存基础：声明、访问、基本用途 |
| `02_static_vs_dynamic.cu` | 静态与动态共享内存对比 |
| `03_syncthreads.cu` | `__syncthreads()` 同步详解 |
| `04_matrix_mul_smem.cu` | 使用共享内存优化矩阵乘法 |
| `05_smem_config.cu` | 共享内存配置与查询 |

## 编译运行

### 使用 Makefile

```bash
# 编译所有示例
make all

# 运行单个示例
./01_smem_basics

# 运行所有示例
make run

# 清理
make clean
```

### 使用 CMake

```bash
# 在项目根目录
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make 13_01_smem_basics 13_02_static_vs_dynamic ...
```

## 学习要点

1. **共享内存基础**
   - `__shared__` 关键字声明
   - 块内线程共享数据
   - 比全局内存快 20-30 倍

2. **静态 vs 动态**
   - 静态：编译时确定大小
   - 动态：运行时确定，核函数启动参数指定

3. **同步机制**
   - `__syncthreads()` 块内屏障同步
   - 使用注意事项和常见陷阱

4. **矩阵乘法优化**
   - 分块策略
   - 减少全局内存访问
   - 数据复用

5. **配置与调优**
   - Bank 配置
   - 共享内存与 L1 缓存划分
   - 占用率计算

## 性能分析

使用 Nsight Compute 分析共享内存性能：

```bash
# 分析 Bank Conflict
ncu --set full --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./04_matrix_mul_smem

# 分析共享内存吞吐量
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum ./04_matrix_mul_smem
```

## 参考链接

- [CUDA C++ Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUDA Best Practices Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
