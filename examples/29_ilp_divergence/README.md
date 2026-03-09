# 第二十九章示例：ILP与Warp Divergence

本目录包含ILP（指令级并行）和Warp Divergence相关的示例代码。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_ilp_basics.cu` | ILP基础概念演示 |
| `02_dual_issue.cu` | Dual Issue双发射示例 |
| `03_warp_divergence.cu` | Warp Divergence问题演示 |
| `04_branch_optimization.cu` | 分支优化策略示例 |

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
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # 根据你的GPU设置架构
make -j
```

## 性能分析

使用Nsight Compute分析：

```bash
# 分析ILP效果
ncu --set full --metrics smsp__inst_issued.sum,smsp__inst_executed.sum ./01_ilp_basics

# 分析Divergence
ncu --set full --metrics smsp__sass_branch_targets.sum ./03_warp_divergence
```

## 学习建议

1. 先运行基础示例，理解概念
2. 使用ncu分析性能差异
3. 对比优化前后的性能数据
4. 尝试修改参数观察效果变化