# 第二十二章示例：CUDA流与并发

本目录包含第二十二章配套的示例代码，演示CUDA流的使用和并发编程技术。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_stream_basics.cu` | CUDA流基础操作 |
| `02_multi_stream.cu` | 多流并发执行 |
| `03_stream_sync.cu` | 流同步机制 |
| `04_cuda_events.cu` | CUDA事件使用 |
| `05_concurrent_kernels.cu` | 并发核函数执行 |

## 编译运行

```bash
# 编译所有示例
make all

# 运行单个示例
./01_stream_basics
./02_multi_stream
./03_stream_sync
./04_cuda_events
./05_concurrent_kernels

# 使用 nsys 分析流并发
nsys profile --stats=true -o report ./02_multi_stream
nsys-ui report.nsys-rep
```

## 学习要点

1. **流基础**：理解默认流和命名流的区别
2. **多流并发**：学习如何使用多流实现并发执行
3. **流同步**：掌握各种同步方式的适用场景
4. **CUDA事件**：学会使用事件进行精确计时和流间同步
5. **并发核函数**：理解核函数并发的条件和限制

## 流编程最佳实践

| 实践 | 说明 |
|------|------|
| 使用非阻塞流 | 避免与默认流的隐式同步 |
| 合理划分数据 | 平衡各流的负载 |
| 最小化同步 | 减少同步点提高并发度 |
| 使用锁页内存 | 加速异步传输 |

## Nsys分析

```bash
# 生成时间线报告
nsys profile --trace=cuda,nvtx -o timeline ./02_multi_stream

# 查看报告
nsys-ui timeline.nsys-rep
```

时间线视图可以看到：
- 各流的并发执行情况
- H2D/D2H传输与核函数的重叠
- 流之间的依赖关系

## 代码结构

```
examples/22_streams/
├── README.md              # 本文件
├── Makefile               # 编译脚本
├── CMakeLists.txt         # CMake配置
├── 01_stream_basics.cu    # 流基础示例
├── 02_multi_stream.cu     # 多流示例
├── 03_stream_sync.cu      # 同步示例
├── 04_cuda_events.cu      # 事件示例
└── 05_concurrent_kernels.cu # 并发核函数示例
```