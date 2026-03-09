# 第二十七章：PTX与底层优化 - 示例代码

本目录包含PTX底层优化的示例代码，展示如何在CUDA中使用内联PTX汇编进行性能优化。

## 文件说明

| 文件 | 说明 |
|------|------|
| `01_ptx_basics.cu` | PTX基础语法演示 |
| `02_inline_ptx.cu` | 内联PTX汇编使用示例 |
| `03_ptx_optimization.cu` | PTX性能优化案例 |
| `04_inspect_ptx.cu` | 查看和分析PTX代码 |

## 编译方法

### 使用Makefile

```bash
make all        # 编译所有示例
make run        # 运行所有示例
make clean      # 清理编译产物
```

### 使用CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make -j
```

### 单独编译

```bash
nvcc -O3 -arch=sm_80 01_ptx_basics.cu -o 01_ptx_basics
```

## 运行示例

```bash
# 运行PTX基础示例
./01_ptx_basics

# 运行内联PTX示例
./02_inline_ptx

# 运行优化示例
./03_ptx_optimization

# 查看PTX代码
./04_inspect_ptx
```

## 学习要点

1. **PTX指令格式**：了解PTX的基本语法和指令格式
2. **内联汇编**：掌握在CUDA代码中使用PTX汇编的方法
3. **SIMD指令**：学习PTX的SIMD视频指令优化
4. **性能分析**：学会查看和分析生成的PTX代码

## 相关章节

- [第二十七章：PTX与底层优化](../../tutorials/27_PTX与底层优化.md)
- [第二十八章：微指令级调优](../../tutorials/28_微指令级调优.md)