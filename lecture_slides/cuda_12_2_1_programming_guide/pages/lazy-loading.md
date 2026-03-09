# Lazy Loading — CUDA C Programming Guide

- Source: https://docs.nvidia.com/cuda/archive/12.2.1/cuda-c-programming-guide/lazy-loading.html
- Exported At (UTC): 2026-03-08T14:51:24.147046+00:00
- Local Images In This Page: 0

# Lazy Loading

## What is Lazy Loading?

Lazy Loading delays loading of CUDA modules and kernels from program initalization closer to kernels execution.
If a program does not use every single kernel it has included, then some kernels will be loaded unneccesarily.
This is very common, especially if you include any libraries.
Most of the time, programs only use a small amount of kernels from libraries they include.

Thanks to Lazy Loading, programs are able to only load kernels they are actually going to use, saving time on initialization.
This reduces memory overhead, both on GPU memory and host memory

Lazy Loading is enabled by setting the `CUDA_MODULE_LOADING` environment variable to `LAZY`.

Firstly, CUDA Runtime will no longer load all modules during program initialization, with the exception of modules containing managed variables.
Each module will be loaded on first usage of a variable or a kernel from that module.
This optimization is only relevant to CUDA Runtime users, CUDA Driver users are unaffected. This optimization shipped in CUDA 11.8.

Secondly, loading a module (`cuModuleLoad*()` family of functions) will not be loading kernels immediately,
instead it will delay loading of a kernel until `cuModuleGetFunction()` is called.
There are certain exceptions here, some kernels have to be loaded during `cuModuleLoad*()`,
such as kernels of which pointers are stored in global variables.
This optimization is relevant to both CUDA Runtime and CUDA Driver users.
CUDA Runtime will only call `cuModuleGetFunction()` when a kernel is used/referenced for the first time.
This optimization shipped in CUDA 11.7.

Both of these optimizations are designed to be invisible to the user, assuming CUDA Programming Model is followed.

## Lazy Loading version support

Lazy Loading is a CUDA Runtime and CUDA Driver feature. Upgrades to both might be necessary to utilize the feature.

### Driver

Lazy Loading requires R515+ user-mode library, but it supports Forward Compatibility, meaning it can run on top of older kernel mode drivers.

Without R515+ user-mode library, Lazy Loading is not available in any shape or form, even if toolkit version is 11.7+.

### Toolkit

Lazy Loading was introduced in CUDA 11.7, and received a significant upgrade in CUDA 11.8.

If your application uses CUDA Runtime, then in order to see benefits from Lazy Loading your application must use 11.7+ CUDA Runtime.

As CUDA Runtime is usually linked statically into programs and libraries,
this means that you have to recompile your program with CUDA 11.7+ toolkit and use CUDA 11.7+ libraries.

Otherwise you will not see the benefits of Lazy Loading, even if your driver version supports it.

If only some of your libraries are 11.7+, you will only see benefits of Lazy Loading in those libraries.
Other libraries will still load everything eagerly.

### Compiler

Lazy Loading does not require any compiler support. Both SASS and PTX compiled with pre-11.7 compilers can be loaded with Lazy Loading enabled,
and will see full benefits of the feature. However, 11.7+ CUDA Runtime is still required, as described above.

## Triggering loading of kernels in lazy mode

Loading kernels and variables happens automatically, without any need for explicit loading.
Simply launching a kernel or referencing a variable or a kernel will automatically load relevant modules and kernels.

However, if for any reason you wish to load a kernel without executing it or modifying it in any way, we recommend the following.

### CUDA Driver API

Loading of kernels happens during `cuModuleGetFunction()` call.
This call is necessary even without Lazy Loading, as it is the only way to obtain a kernel handle.

However, you can also use this API to control with finer granularity when kernels are loaded.

### CUDA Runtime API

CUDA Runtime API manages module management automatically,
so we recommend simply using `cudaFuncGetAttributes()` to reference the kernel.

This will ensure that the kernel is loaded without changing the state.

## Querying whether Lazy Loading is turned on

In order to check whether user enabled Lazy Loading, `CUresult cuModuleGetLoadingMode ( CUmoduleLoadingMode* mode )` can be used.

It’s important to note that CUDA must be initialized before running this function. Sample usage can be seen in the snippet below.

```
#include "cuda.h"
#include "assert.h"
#include "iostream"

int main() {
        CUmoduleLoadingMode mode;

        assert(CUDA_SUCCESS == cuInit(0));
        assert(CUDA_SUCCESS == cuModuleGetLoadingMode(&mode));

        std::cout << "CUDA Module Loading Mode is " << ((mode == CU_MODULE_LAZY_LOADING) ? "lazy" : "eager") << std::endl;

        return 0;
}
```

## Possible issues when adopting lazy loading

Lazy Loading is designed so that it should not require any modifications to applications to use it.
That said, there are some caveats, especially when applications are not fully compliant with CUDA Programming Model.

### Concurrent execution

Loading kernels might require context synchronization.
Some programs incorrectly treat the possibility of concurrent execution of kernels as a guarantee.
In such cases, if program assumes that two kernels will be able to execute concurrently,
and one of the kernels will not return without the other kernel executing, there is a possibility of a deadlock.

If kernel A will be spinning in an infinite loop until kernel B is executing.
In such case launching kernel B will trigger lazy loading of kernel B. If this loading will require context synchronization,
then we have a deadlock: kernel A is waiting for kernel B, but loading kernel B is stuck waiting for kernel A to finish to synchronize the context.

Such program is an anti-pattern, but if for any reason you want to keep it you can do the following:

* preload all kernels that you hope to execute concurrently prior to launching them

### Allocators

Lazy Loading delays loading code from initialization phase of the program closer to execution phase.
Loading code onto the GPU requires memory allocation.

If your application tries to allocate the entire VRAM on startup, e.g. to use it for its own allocator,
then it might turn out that there will be no more memory left to load the kernels.
This is despite the fact that overall Lazy Loading frees up more memory for the user.
CUDA will need to allocate some memory to load each kernel, which usually happens at first launch time of each kernel.
If your application allocator greedily allocated everything, CUDA will fail to allocate memory.

Possible solutions:

* use cudaMallocAsync() instead of an allocator that allocates the entire VRAM on startup
* add some buffer to compensate for the delayed loading of kernels
* preload all kernels that will be used in the program before trying to initialize your allocator

### Autotuning

Some applications launch several kernels implementing the same functionality to determine which one is the fastest.
While it is overall advisable to run at least one warmup iteration, it becomes especially important with Lazy Loading.
After all, including time taken to load the kernel will skew your results.

Possible solutions:

* do at least one warmup interaction prior to measurement
* preload the benchmarked kernel prior to launching it
