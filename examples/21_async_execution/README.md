# 第二十一章示例：异步执行与延迟隐藏

本目录包含第二十一章配套的示例代码，演示延迟隐藏技术和异步执行优化。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_double_buffer.cu` | 双缓冲技术实现 |
| `02_software_pipeline.cu` | 软件流水线实现 |
| `03_memcpy_async.cu` | 异步拷贝API使用 |
| `04_pipeline_primitives.cu` | Pipeline Primitives接口 |

## 编译运行

```bash
# 编译所有示例
make all

# 运行单个示例
./01_double_buffer
./02_software_pipeline
./03_memcpy_async
./04_pipeline_primitives

# 使用 ncu 分析性能
ncu --set full ./01_double_buffer
ncu --set full ./03_memcpy_async
```

## 学习要点

1. **双缓冲技术**：理解如何使用两个缓冲区实现计算与访存重叠
2. **软件流水线**：掌握多阶段流水线的调度方式
3. **异步拷贝**：学习memcpy_async API的使用方法
4. **Pipeline**：理解cuda::pipeline的管理机制

## 性能对比

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| 双缓冲 | 简单直接 | 计算与访存时间相近 |
| 软件流水线 | 更灵活 | 多阶段处理 |
| memcpy_async | CUDA 11+ | 单次异步加载 |
| Pipeline | 功能完整 | 复杂异步场景 |

## NCU分析要点

```bash
# 分析Stall原因
ncu --metrics smsp__warp_issue_stall_long_scoreboard_per_warp_active.pct \
    ./01_double_buffer

# 分析内存吞吐
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    ./03_memcpy_async

# 比较同步与异步版本
ncu --set full -o sync_report ./sync_version
ncu --set full -o async_report ./async_version
```

## 硬件要求

- Compute Capability 7.0+ (基本异步功能)
- Compute Capability 8.0+ (硬件加速异步拷贝)

## 代码结构

```
examples/21_async_execution/
├── README.md              # 本文件
├── Makefile               # 编译脚本
├── CMakeLists.txt         # CMake配置
├── 01_double_buffer.cu    # 双缓冲示例
├── 02_software_pipeline.cu # 软件流水线示例
├── 03_memcpy_async.cu     # 异步拷贝示例
└── 04_pipeline_primitives.cu # Pipeline Primitives示例
```