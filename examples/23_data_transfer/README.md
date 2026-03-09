# 第二十三章示例：数据传输优化

本目录包含第二十三章配套的示例代码，演示CUDA数据传输的各种优化技术。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_pinned_memory.cu` | 锁页内存使用 |
| `02_async_transfer.cu` | 异步数据传输 |
| `03_zero_copy.cu` | 零拷贝内存 |
| `04_unified_memory.cu` | 统一内存 |
| `05_prefetch_advise.cu` | 预取与建议优化 |

## 编译运行

```bash
# 编译所有示例
make all

# 运行单个示例
./01_pinned_memory
./02_async_transfer
./03_zero_copy
./04_unified_memory
./05_prefetch_advise

# 使用 nsys 分析数据传输
nsys profile --stats=true -o report ./02_async_transfer
nsys-ui report.nsys-rep
```

## 学习要点

1. **锁页内存**：理解分页内存和锁页内存的区别
2. **异步传输**：掌握cudaMemcpyAsync的使用和流中执行
3. **零拷贝**：学习GPU直接访问主机内存的方式
4. **统一内存**：理解CPU/GPU共享内存池的概念
5. **预取与建议**：掌握优化统一内存性能的技术

## 数据传输方式对比

| 方式 | API | 特点 | 适用场景 |
|------|-----|------|----------|
| 同步传输 | cudaMemcpy | 简单，阻塞 | 简单场景 |
| 异步传输 | cudaMemcpyAsync | 非阻塞，需锁页内存 | 高性能 |
| 零拷贝 | cudaHostAllocMapped | 无显式传输 | 小数据频繁访问 |
| 统一内存 | cudaMallocManaged | 自动迁移 | 快速开发 |
| 统一内存优化 | + Prefetch/Advise | 接近最优性能 | 最佳实践 |

## 性能优化建议

1. 使用锁页内存加速数据传输
2. 使用异步传输实现计算与传输重叠
3. 小数据场景考虑零拷贝
4. 统一内存配合预取和建议获得最佳性能

## Nsys分析

```bash
# 分析数据传输
nsys profile --trace=cuda,nvtx -o transfer_analysis ./02_async_transfer

# 查看报告
nsys-ui transfer_analysis.nsys-rep
```

时间线视图可以看到：
- H2D/D2H传输时间
- 传输与计算的重叠
- Page Fault事件（统一内存）

## 代码结构

```
examples/23_data_transfer/
├── README.md                # 本文件
├── Makefile                 # 编译脚本
├── CMakeLists.txt           # CMake配置
├── 01_pinned_memory.cu      # 锁页内存示例
├── 02_async_transfer.cu     # 异步传输示例
├── 03_zero_copy.cu          # 零拷贝示例
├── 04_unified_memory.cu     # 统一内存示例
└── 05_prefetch_advise.cu    # 预取与建议示例
```