# CUDA 学习教程

> 从零开始学习 CUDA 并行编程，适合完全新手到高级优化

## 项目简介

本项目是一套完整的 CUDA 学习教程，包含：

- **30 章教程文档**：循序渐进，从基础入门到高级优化
- **30 个示例代码**：每章配套可运行代码，详细中文注释
- **CMake 编译支持**：一键编译所有示例
- **辅助脚本**：环境检查、编译、快速运行

## 官方文档来源

本教程内容参考并整合了以下 NVIDIA 官方文档：

| 文档名称 | 版本 | 链接 | 本地路径 |
|----------|------|------|----------|
| **CUDA C++ Programming Guide** | 12.2.1 | [官方链接](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-c-programming-guide/) | [本地文档](./lecture_slides/cuda_12_2_1_programming_guide/) |
| CUDA Best Practices Guide | 12.2.1 | [官方链接](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-c-best-practices-guide/) | - |
| CUDA Runtime API | 12.2.1 | [官方链接](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-runtime-api/) | - |

> **说明**：本地文档 `lecture_slides/cuda_12_2_1_programming_guide/` 包含完整的 CUDA 12.2.1 编程指南及配套图片，可供离线查阅。

## 目录结构

```
cuda_learning/
├── tutorials/                    # 教程文档 (30章)
├── examples/                     # 示例代码 (与章节对应)
│   ├── 01_gpu_basics/           # GPU基础
│   ├── 02_cuda_intro/           # CUDA入门
│   ├── ...
│   └── 30_cuda_libraries/       # CUDA官方库
├── lecture_slides/              # 课程讲义
├── scripts/                      # 辅助脚本
└── README.md
```

## 快速开始

### 1. 环境要求

- NVIDIA GPU（Compute Capability 5.3+）
- CUDA Toolkit 11.0+
- CMake 3.18+
- Linux 操作系统

### 2. 编译示例

```bash
# 使用脚本编译
./scripts/compile_all.sh

# 或手动编译
cd examples
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # 根据你的GPU选择架构
make -j
```

### 3. 运行示例

```bash
# 运行指定章节
./scripts/run_example.sh 12    # 第12章：原子操作

# 运行所有示例
./scripts/run_example.sh --all
```

## 章节内容

### 第一部分：基础入门 (1-7章)

| 章节 | 教程 | 学习目标 |
|:----:|------|----------|
| 01 | [什么是GPU并行计算](tutorials/01_什么是GPU并行计算.md) | 理解并行计算概念，CPU vs GPU 区别 |
| 02 | [CUDA是什么](tutorials/02_CUDA是什么.md) | 了解 CUDA 架构，Host/Device 概念 |
| 03 | [GPU硬件架构入门](tutorials/03_GPU硬件架构入门.md) | 认识 SM、CUDA Core、Warp |
| 04 | [线程层级结构](tutorials/04_线程层级结构.md) | 掌握 Grid/Block/Thread 组织 |
| 05 | [第一个CUDA程序](tutorials/05_第一个CUDA程序.md) | 编写并运行第一个程序 |
| 06 | [内存管理基础](tutorials/06_内存管理基础.md) | 学会 cudaMalloc/cudaMemcpy |
| 07 | [核函数深入](tutorials/07_核函数深入.md) | 理解 __global__、<< <>>> |

### 第二部分：进阶优化 (8-11章)

| 章节 | 教程 | 学习目标 |
|:----:|------|----------|
| 08 | [性能分析入门](tutorials/08_性能分析入门.md) | 使用 nsys/ncu 分析性能 |
| 09 | [内存访问优化](tutorials/09_内存访问优化.md) | 合并访问、向量化访存 |
| 10 | [精度与性能](tutorials/10_精度与性能.md) | FP16 编程与优化 |
| 11 | [Roofline模型](tutorials/11_Roofline模型.md) | 分析性能瓶颈 |

### 第三部分：内存与同步机制 (12-16章)

| 章节 | 教程 | 学习目标 |
|:----:|------|----------|
| 12 | [原子操作与竞争条件](tutorials/12_原子操作与竞争条件.md) | 理解竞争条件，掌握原子操作 |
| 13 | [共享内存深入](tutorials/13_共享内存深入.md) | 共享内存原理与优化 |
| 14 | [规约算法优化](tutorials/14_规约算法优化.md) | 树状规约、Warp Shuffle |
| 15 | [Bank Conflict优化](tutorials/15_Bank_Conflict优化.md) | Bank冲突检测与解决 |
| 16 | [Cooperative Groups](tutorials/16_Cooperative_Groups.md) | 跨块同步与协作编程 |

### 第四部分：核心算子实现 (17-20章)

| 章节 | 教程 | 学习目标 |
|:----:|------|----------|
| 17 | [GEMM优化入门](tutorials/17_GEMM优化入门.md) | 矩阵乘法Naive到分块 |
| 18 | [GEMM分块优化](tutorials/18_GEMM分块优化.md) | 1D/2D Blocktiling、Warptiling |
| 19 | [Tensor Core编程](tutorials/19_Tensor_Core编程.md) | WMMA API、混合精度GEMM |
| 20 | [卷积算子实现](tutorials/20_卷积算子实现.md) | 直接卷积、im2col优化 |

### 第五部分：系统级优化 (21-25章)

| 章节 | 教程 | 学习目标 |
|:----:|------|----------|
| 21 | [异步执行与延迟隐藏](tutorials/21_异步执行与延迟隐藏.md) | 双缓冲、软件流水线 |
| 22 | [CUDA流与并发](tutorials/22_CUDA流与并发.md) | 多流、Event、并发内核 |
| 23 | [数据传输优化](tutorials/23_数据传输优化.md) | Pinned Memory、Unified Memory |
| 24 | [CUDA Graph](tutorials/24_CUDA_Graph.md) | 图捕获与执行优化 |
| 25 | [多GPU编程](tutorials/25_多GPU编程.md) | P2P传输、NCCL、AllReduce |

### 第六部分：工业级调优 (26-30章)

| 章节 | 教程 | 学习目标 |
|:----:|------|----------|
| 26 | [低精度与量化](tutorials/26_低精度与量化.md) | FP16/BF16/INT8量化 |
| 27 | [PTX与底层优化](tutorials/27_PTX与底层优化.md) | PTX汇编、内联PTX |
| 28 | [微指令级调优](tutorials/28_微指令级调优.md) | 循环展开、编译器选项 |
| 29 | [ILP与Warp Divergence](tutorials/29_ILP与Warp_Divergence.md) | 指令级并行、分支优化 |
| 30 | [CUDA官方库实战](tutorials/30_CUDA官方库实战.md) | cuBLAS、cuDNN、CUB、CUTLASS |

## 学习路线

```
┌─────────────────────────────────────────────────────────────┐
│  第一部分：基础入门 (1-7章)                                  │
│  GPU概念 → CUDA入门 → 硬件架构 → 线程层级 →                 │
│  第一个程序 → 内存管理 → 核函数深入                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  第二部分：进阶优化 (8-11章)                                 │
│  性能分析 → 内存访问优化 → 精度与性能 → Roofline模型         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  第三部分：内存与同步机制 (12-16章)                          │
│  原子操作 → 共享内存 → 规约优化 → Bank Conflict → CG         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  第四部分：核心算子实现 (17-20章)                            │
│  GEMM入门 → GEMM分块优化 → Tensor Core → 卷积算子            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  第五部分：系统级优化 (21-25章)                              │
│  异步执行 → 多流 → 数据传输 → CUDA Graph → 多GPU             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  第六部分：工业级调优 (26-30章)                              │
│  低精度量化 → PTX优化 → 微指令调优 → ILP/分支 → 官方库       │
└─────────────────────────────────────────────────────────────┘
```

## GPU 架构参考

| 架构 | GPU 示例 | Compute Capability | CMake 参数 |
|------|----------|-------------------|------------|
| Volta | V100 | 7.0 | `-DCMAKE_CUDA_ARCHITECTURES=70` |
| Turing | RTX 2080, T4 | 7.5 | `-DCMAKE_CUDA_ARCHITECTURES=75` |
| Ampere | A100 | 8.0 | `-DCMAKE_CUDA_ARCHITECTURES=80` |
| Ampere | RTX 3090, A40 | 8.6 | `-DCMAKE_CUDA_ARCHITECTURES=86` |
| Ada Lovelace | RTX 4090 | 8.9 | `-DCMAKE_CUDA_ARCHITECTURES=89` |
| Hopper | H100 | 9.0 | `-DCMAKE_CUDA_ARCHITECTURES=90` |

## 教程特点

- **从零开始**：假设读者完全不了解 GPU 编程
- **图解优先**：使用 Mermaid 图解释所有抽象概念
- **代码说话**：每行代码都有详细中文注释
- **即学即练**：每个概念都配可运行代码
- **循序渐进**：每章只引入 2-3 个新概念
- **结合官方文档**：每章标注对应的CUDA官方文档章节

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA FP16 API](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html)

## 课程来源

本教程基于 InfiniTensor 大模型与人工智能系统训练营 2025 冬季课程内容整理。

---

*祝学习愉快！*