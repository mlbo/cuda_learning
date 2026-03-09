# 第十五章示例：Bank Conflict 优化

本目录包含第十五章配套的示例代码，演示共享内存 Bank Conflict 的检测与优化方法。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_bank_structure.cu` | Bank 结构演示 - 展示 32 个 Bank 的工作原理 |
| `02_bank_conflict_demo.cu` | Bank Conflict 演示 - 树状规约中的冲突分析 |
| `03_padding_solution.cu` | Padding 解决方案 - 通过内存填充消除冲突 |
| `04_xor_swizzling.cu` | XOR Swizzling 解决方案 - 索引重映射技术 |
| `05_matrix_transpose.cu` | 矩阵转置优化案例 - 综合对比各种方法 |

## 编译运行

```bash
# 使用 Makefile 编译
make all

# 使用 CMake 编译
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make

# 运行单个示例
./01_bank_structure
./02_bank_conflict_demo

# 运行所有示例
make run
```

## NCU 性能分析

```bash
# 分析 Bank Conflict 相关指标
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    ./02_bank_conflict_demo

# 完整分析
ncu --set full -o bank_conflict_report ./05_matrix_transpose

# 对比优化前后的 Bank Conflict
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    ./03_padding_solution
```

## 学习要点

### 1. Bank 结构基础
- 共享内存有 32 个 Bank
- 每个 Bank 宽度 4 字节
- Bank ID = (地址 / 4) % 32
- Warp 内 32 线程对应 32 个 Bank

### 2. Bank Conflict 类型
- **N-way 冲突**：N 个线程同时访问同一 Bank 的不同地址
- **广播**：多线程访问同一地址，无冲突
- **跨步访问**：stride 为 2 的幂时可能产生冲突

### 3. 解决方案对比

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 算法优化 | 调整访问模式 | 无额外开销 | 可能需要大改 |
| Padding | 内存填充 | 简单直接 | 增加内存使用 |
| XOR Swizzling | 索引重映射 | 无内存开销 | 理解成本高 |

### 4. 性能影响

| 冲突程度 | 性能损失 |
|---------|---------|
| 无冲突 | 基准 |
| 2-way 冲突 | 2 倍延迟 |
| 4-way 冲突 | 4 倍延迟 |
| 32-way 冲突 | 32 倍延迟 |

## 示例详解

### 01_bank_structure.cu
演示共享内存 Bank 的基本结构：
- 计算 Bank ID 的方法
- 连续访问与 Bank 映射的关系
- 验证无冲突访问模式

### 02_bank_conflict_demo.cu
展示 Bank Conflict 产生过程：
- 跨步访问产生的冲突
- 树状规约中的 Bank Conflict 分析
- 使用 NCU 检测冲突

### 03_padding_solution.cu
Padding 解决方案示例：
- 矩阵转置中的 Bank Conflict
- 通过填充消除冲突
- 内存开销分析

### 04_xor_swizzling.cu
XOR Swizzling 技术：
- XOR 运算的索引重映射
- 无额外内存开销
- GEMM 中的应用

### 05_matrix_transpose.cu
综合优化案例：
- 对比 Naive、Padding、XOR Swizzling 三种方法
- 性能测试与 NCU 分析
- 最佳实践总结

## 常见问题

**Q: 为什么是 32 个 Bank？**
A: 与 Warp 大小（32 线程）匹配，设计目的是让一个 Warp 可以并行访问不同 Bank。

**Q: 广播和多播有什么区别？**
A: 广播是多线程读取同一地址；多播是多线程读取同一 Bank 的同一缓存行内的不同地址（如同一 128B 内）。

**Q: Padding 应该填充多少？**
A: 通常填充 1 即可（如 tile[32][33]），但具体取决于访问模式。

**Q: XOR Swizzling 在什么场景最有效？**
A: 在有规则的 2D 访问模式（如矩阵乘法、卷积）中效果最好。
