# 第二十四章示例：CUDA Graph

本目录包含第二十四章配套的示例代码，演示CUDA Graph的使用和优化技术。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_graph_basics.cu` | CUDA Graph基础操作 |
| `02_stream_capture.cu` | 流捕获技术 |
| `03_graph_instantiate.cu` | 图实例化与执行 |
| `04_graph_optimize.cu` | 图优化技术 |

## 编译运行

```bash
# 在项目根目录
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make 24_01_graph_basics 24_02_stream_capture 24_03_graph_instantiate 24_04_graph_optimize

# 运行单个示例
./24_01_graph_basics
./24_02_stream_capture
./24_03_graph_instantiate
./24_04_graph_optimize

# 使用 nsys 分析Graph性能
nsys profile --stats=true -o report ./24_04_graph_optimize
nsys-ui report.nsys-rep
```

## 学习要点

1. **图基础**：理解CUDA Graph的执行流程
2. **流捕获**：掌握不同捕获方式和模式
3. **图实例化**：学习图的实例化和执行
4. **性能优化**：理解Graph带来的性能提升

## CUDA Graph执行流程

```
捕获(Capture) -> 创建图(Create) -> 实例化(Instantiate) -> 执行(Launch)
                        ↑                                            |
                        |____________________________________________|
                                      可多次重复执行
```

## 性能对比

| 核函数数量 | 无Graph | 有Graph | 提升 |
|------------|---------|---------|------|
| 1 | 基准 | ~5% | 小 |
| 10 | 基准 | ~20-30% | 中等 |
| 100 | 基准 | ~50-70% | 显著 |

## 关键API

| 函数 | 描述 |
|------|------|
| `cudaStreamBeginCapture()` | 开始流捕获 |
| `cudaStreamEndCapture()` | 结束流捕获 |
| `cudaGraphInstantiate()` | 实例化图 |
| `cudaGraphLaunch()` | 执行图 |
| `cudaGraphExecDestroy()` | 销毁图实例 |
| `cudaGraphDestroy()` | 销毁图 |

## Nsys分析

```bash
# 分析CUDA Graph性能
nsys profile --trace=cuda,nvtx -o graph_trace ./24_04_graph_optimize

# 查看报告
nsys-ui graph_trace.nsys-rep
```

时间线视图可以看到：
- 图启动开销
- 节点执行时间
- 节点间的空泡(bubble)
- 并行执行情况

## 适用场景

**适合**：
- 重复执行的相同操作序列
- 多个小核函数
- 深度学习推理
- 需要减少CPU开销

**不适合**：
- 操作序列每次不同
- 单次执行
- 需要频繁重建图

## 代码结构

```
examples/24_cuda_graph/
├── README.md                # 本文件
├── Makefile                 # 编译脚本
├── CMakeLists.txt           # CMake配置
├── 01_graph_basics.cu       # 图基础示例
├── 02_stream_capture.cu     # 流捕获示例
├── 03_graph_instantiate.cu  # 图实例化示例
└── 04_graph_optimize.cu     # 图优化示例
```
