# 第 16 章：Cooperative Groups 示例

本目录包含 Cooperative Groups API 的示例代码，涵盖从基础的块内同步到跨块同步的各种用法。

## 文件列表

| 文件 | 描述 | 计算能力要求 |
|------|------|--------------|
| `01_thread_block.cu` | `thread_block` 基础用法 | CC 5.0+ |
| `02_warp_group.cu` | `thread_block_tile` 和 warp 级操作 | CC 5.0+ |
| `03_grid_sync.cu` | 跨块同步 `grid.sync()` | CC 6.0+ |
| `04_reduce_with_cg.cu` | 使用 CG 的高性能规约实现 | CC 6.0+ |
| `05_launch_cooperative.cu` | `cudaLaunchCooperativeKernel` 使用 | CC 6.0+ |
| `06_thread_block_cluster.cu` | Thread Block Cluster 示例 | CC 9.0+ |

## 编译方法

### 使用 Makefile

```bash
make all          # 编译所有示例
make 01_thread_block  # 编译单个示例
make clean        # 清理编译产物
```

### 使用 CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # 根据你的 GPU 设置架构
make -j
```

## 运行要求

- CUDA 9.0 或更高版本
- 计算能力 5.0+ 的 GPU（基础示例）
- 计算能力 6.0+ 的 GPU（grid.sync 示例）
- 计算能力 9.0+ 的 GPU（Cluster 示例）

## 平台限制

Grid 同步功能需要：
- Linux 平台（无 MPS 或 CC 7.0+ 有 MPS）
- 最新 Windows 版本
- TCC 模式

## 学习顺序

1. 先运行 `01_thread_block.cu` 理解基本概念
2. 运行 `02_warp_group.cu` 学习 warp 级操作
3. 运行 `03_grid_sync.cu` 理解跨块同步
4. 运行 `04_reduce_with_cg.cu` 看完整的应用示例
5. 运行 `05_launch_cooperative.cu` 学习正确的启动方式
6. 如果有 H100+ GPU，运行 `06_thread_block_cluster.cu`

## 参考资料

- [CUDA Programming Guide - Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- [CUDA C++ API Reference - Cooperative Groups](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__COOPERATIVE__GROUPS.html)