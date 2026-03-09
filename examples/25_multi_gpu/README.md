# 第二十五章示例：多GPU编程

本目录包含第二十五章配套的示例代码，演示多GPU编程的核心技术。

## 文件说明

| 文件 | 描述 |
|------|------|
| `01_device_selection.cu` | 设备枚举与选择 |
| `02_p2p_transfer.cu` | P2P点对点传输 |
| `03_nccl_basics.cu` | NCCL基础使用 |
| `04_allreduce.cu` | NCCL AllReduce操作 |

## 编译运行

```bash
# 编译所有示例
make all

# 运行单个示例
./01_device_selection
./02_p2p_transfer
./03_nccl_basics
./04_allreduce

# 使用nsys分析多GPU性能
nsys profile --trace=cuda,nvtx -o multi_gpu_trace ./04_allreduce
nsys-ui multi_gpu_trace.nsys-rep
```

## 学习要点

1. **设备管理**：学会枚举和选择GPU设备
2. **P2P传输**：理解点对点传输原理和使用方法
3. **NCCL基础**：掌握NCCL库的基本使用
4. **集合通信**：理解AllReduce等集合操作

## 系统要求

- CUDA 11.0+
- NCCL库（通常随CUDA Toolkit安装）
- 多GPU系统（部分示例需要）

## 查看GPU拓扑

```bash
# 查看GPU连接拓扑
nvidia-smi topo -m

# 查看P2P能力
nvidia-smi nvlink --status
```

## 关键API

### 设备管理
| 函数 | 描述 |
|------|------|
| `cudaGetDeviceCount()` | 获取设备数量 |
| `cudaSetDevice()` | 选择设备 |
| `cudaGetDeviceProperties()` | 获取设备属性 |

### P2P传输
| 函数 | 描述 |
|------|------|
| `cudaDeviceCanAccessPeer()` | 检查P2P能力 |
| `cudaDeviceEnablePeerAccess()` | 启用P2P |
| `cudaMemcpyPeer()` | P2P传输 |

### NCCL
| 函数 | 描述 |
|------|------|
| `ncclCommInitAll()` | 初始化通信组 |
| `ncclAllReduce()` | AllReduce操作 |
| `ncclSend()/ncclRecv()` | 点对点通信 |

## 多GPU执行模型

```
主机线程                GPU 0                GPU 1
    |                    |                    |
    |-- cudaSetDevice(0) |                    |
    |   kernel<<<...>>> --|                    |
    |                    |-- kernel执行 ------|
    |                    |                    |
    |-- cudaSetDevice(1) |                    |
    |   kernel<<<...>>> --|                    |
    |                    |                    |-- kernel执行
    |                    |                    |
    |-- 同步等待         |                    |
```

## NCCL通信操作

```
AllReduce示意:
GPU0: [1,2] ----\
GPU1: [3,4] ----+---> Reduce: [4,6] --+---> Broadcast:
GPU2: [5,6] ----/                     |      GPU0: [4,6]
                                      +---> GPU1: [4,6]
                                      |      GPU2: [4,6]
                                      +---> GPU3: [4,6]
```

## 性能优化提示

1. **通信与计算重叠**：使用不同的CUDA流
2. **批量通信**：减少通信次数
3. **拓扑感知**：优先使用NVLink连接的GPU对
4. **梯度压缩**：减少传输数据量

## 代码结构

```
examples/25_multi_gpu/
├── README.md              # 本文件
├── Makefile               # 编译脚本
├── CMakeLists.txt         # CMake配置
├── 01_device_selection.cu # 设备选择示例
├── 02_p2p_transfer.cu     # P2P传输示例
├── 03_nccl_basics.cu      # NCCL基础示例
└── 04_allreduce.cu        # AllReduce示例
```