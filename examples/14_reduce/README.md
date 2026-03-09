# 第十四章示例：规约算法优化

本目录包含第十四章配套的示例代码，演示多种规约算法优化技术。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_naive_reduce.cu` | 朴素规约实现 - 展示性能问题 |
| `02_tree_reduce.cu` | 树状规约（共享内存） |
| `03_warp_shuffle_reduce.cu` | Warp Shuffle 规约 |
| `04_two_pass_reduce.cu` | 两阶段规约 |
| `05_cooperative_groups_reduce.cu` | Cooperative Groups 规约 |
| `06_performance_comparison.cu` | 性能对比测试 |

## 编译运行

```bash
# 编译所有示例
make all

# 运行单个示例
./01_naive_reduce
./02_tree_reduce

# 运行性能对比
./06_performance_comparison

# 使用 ncu 分析性能
ncu ./06_performance_comparison
```

## 学习要点

1. **朴素规约的问题**：理解原子操作导致的串行化
2. **树状规约原理**：分治思想在规约中的应用
3. **Warp Shuffle**：高效的 Warp 内数据交换
4. **Bank Conflict**：共享内存访问模式优化
5. **Two-Pass 规约**：避免原子操作的两阶段方法
6. **Cooperative Groups**：跨 Block 同步的新特性

## 性能对比

| 方法 | 原子操作次数 | 相对性能 | 适用场景 |
|------|--------------|----------|----------|
| 朴素原子规约 | N | 最慢 | 不推荐 |
| Warp 规约 | N/32 | 较快 | 小数据 |
| 树状规约 | GridSize | 快 | 中等数据 |
| Shuffle + SM | GridSize | 更快 | 中等数据 |
| Two-Pass | 0 | 很快 | 大数据 |
| Cooperative Groups | 0 | 最快 | 大数据 |