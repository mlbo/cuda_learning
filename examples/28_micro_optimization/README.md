# 第二十八章：微指令级调优 - 示例代码

本目录包含微指令级调优的示例代码，展示编译器指令、限定符和编译选项的使用方法。

## 文件说明

| 文件 | 说明 |
|------|------|
| `01_pragma_unroll.cu` | 循环展开优化示例 |
| `02_restrict_qualifier.cu` | __restrict__限定符使用示例 |
| `03_compiler_options.cu` | 编译器选项调优示例 |
| `04_alignment.cu` | 内存对齐优化示例 |
| `05_ldg_const.cu` | __ldg与常量内存使用示例 |

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
# 基本编译
nvcc -O3 -arch=sm_80 01_pragma_unroll.cu -o 01_pragma_unroll

# 使用快速数学
nvcc -O3 -arch=sm_80 -use_fast_math 03_compiler_options.cu -o 03_fast_math

# 限制寄存器数量
nvcc -O3 -arch=sm_80 -maxrregcount=64 03_compiler_options.cu -o 03_limited_regs

# 查看编译详情
nvcc -O3 -arch=sm_80 -Xptxas -v 03_compiler_options.cu -o 03_verbose
```

## 运行示例

```bash
# 运行循环展开示例
./01_pragma_unroll

# 运行__restrict__示例
./02_restrict_qualifier

# 运行编译器选项示例
./03_compiler_options

# 运行内存对齐示例
./04_alignment

# 运行__ldg与常量内存示例
./05_ldg_const
```

## 学习要点

1. **#pragma unroll**：理解循环展开的收益与代价
2. **__restrict__**：消除指针别名，提升编译器优化能力
3. **编译选项**：掌握常用编译选项及其作用
4. **内存对齐**：理解对齐对性能的影响
5. **只读缓存**：合理使用__ldg和常量内存

## 相关章节

- [第二十七章：PTX与底层优化](../../tutorials/27_PTX与底层优化.md)
- [第二十八章：微指令级调优](../../tutorials/28_微指令级调优.md)