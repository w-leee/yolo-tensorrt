# 🛠️ 底层编译与部署排错手册 (Troubleshooting Guide)

> **适用环境**：Ubuntu 26.04 (LTS) / GCC 15.2 / CUDA 12.6 / TensorRT 10.7 / CMake 4.2

在将纯 C++ TensorRT 推理工程接入 CUDA 预处理时，由于系统底层工具链极新（Ubuntu 26.04），与相对稳定的深度学习生态（CUDA 12.6）存在版本代沟，触发了以下三个极其经典的底层踩坑记录。在此作系统化复盘与归档。

---

## 🚨 问题一：GCC 版本过新导致 nvcc 拒绝编译

### 🔍 现象 (Symptom)

执行 `cmake ..` 时立即报错：

```text
error: #error -- unsupported GNU version! gcc versions later than 13 are not supported!
The nvcc flag '-allow-unsupported-compiler' can be used to override this version check.
```

### 🧠 根本原因 (Root Cause)

Ubuntu 26.04 默认的 GCC 版本为 15.2，而 CUDA 12.6 的主机编译器支持上限为 GCC 13。nvcc 在编译时检测到过高的 GCC 版本，直接拒绝工作。

### 💡 解决方案 (Solution)

**不要**强行使用 `-allow-unsupported-compiler` 标志，这可能导致不可预料的运行时崩溃。正确做法是降级并指定主机编译器：

1. 安装兼容版本的 GCC：

    ```bash
    sudo apt install gcc-12 g++-12 -y
    ```

2. 在 `CMakeLists.txt` 中，于 `project(...)` 之前强制指定 CUDA 主机编译器：

    ```cmake
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12")
    ```

---

## 🚨 问题二：glibc 版本过新导致数学函数声明冲突

### 🔍 现象 (Symptom)

编译 `.cu` 文件时，报大量头文件异常声明冲突：

```text
/usr/include/x86_64-linux-gnu/bits/mathcalls.h(83): error: exception specification is incompatible with that of previous function "cospi"
/usr/include/x86_64-linux-gnu/bits/mathcalls.h(206): error: exception specification is incompatible with that of previous function "rsqrt"
```

### 🧠 根本原因 (Root Cause)

Ubuntu 26.04 的 glibc（2.43）在 C++17 下，为 `rsqrt`、`cospi` 等数学函数添加了 `noexcept(true)` 异常规范。而 CUDA 12.6 自带的 `math_functions.h` 中，这些函数未加此规范，导致两边的声明在 nvcc 混合编译时发生冲突。

### 💡 解决方案 (Solution)

注入自定义的兼容性头文件，清除 glibc 的异常规范宏 `__THROW`：

1. 在 `cpp/include/` 下新建 `glibc_c23_math_compat.h`：

    ```c
    #ifndef GLIBC_C23_MATH_COMPAT_H
    #define GLIBC_C23_MATH_COMPAT_H

    #if defined(__cplusplus) && defined(__linux__)
    #include <features.h>
    #include <sys/cdefs.h>
    #ifdef __GLIBC__
    #undef __THROW
    #define __THROW
    #endif
    #endif

    #endif
    ```

2. 在 `CMakeLists.txt` 中强制 nvcc 的宿主编译器包含该文件：

    ```cmake
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-include,${CMAKE_CURRENT_SOURCE_DIR}/include/glibc_c23_math_compat.h")
    ```

---

## 🚨 问题三：C++ 标准库撕裂 (GLIBCXX / CXXABI 链接错误)

### 🔍 现象 (Symptom)

编译全部通过，在最后的 Link 阶段报大量未定义引用（undefined reference）：

```text
/usr/lib/x86_64-linux-gnu/libgdcmMSFF.so.3.0: undefined reference to `std::ios_base_library_init()@GLIBCXX_3.4.32'
/usr/lib/x86_64-linux-gnu/libopencv_core.so.4.10.0: undefined reference to `__cxa_call_terminate@CXXABI_1.3.15'
```

### 🧠 根本原因 (Root Cause)

整个工程的主程序（`main.cpp` 等）默认由 GCC 15 编译，依赖较新的 C++ 标准库符号（`GLIBCXX_3.4.32`）；而 CUDA 主机端强制指定了 GCC 12 编译，链接器最终使用了 GCC 12 的旧版 `libstdc++` 进行缝合，导致找不到 GCC 15 生成的符号。

### 💡 解决方案 (Solution)

**保持编译器版本绝对统一。** 既然 CUDA 12.6 必须用 GCC 12，那就让整个 C++ 工程也全部使用 GCC 12：

在 `CMakeLists.txt` 中指定全局 C++ 编译器：

```cmake
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
```

> **注**：如果必须保留 GCC 15 编译主程序，可尝试在 `CMAKE_EXE_LINKER_FLAGS` 中增加 `-static-libstdc++ -static-libgcc` 静态链接标志。
