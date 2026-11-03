/*
=========================================================
【兼容性补丁】glibc_c23_math_compat.h
【作用】Ubuntu 26.04 自带的 glibc 2.43 在 C++17 下为 rsqrt 等数学函数添加了 noexcept 规范。
       这与 CUDA 12.6 自带头文件的声明不符，导致 nvcc 编译报错。
       此头文件通过重定义 __THROW 宏，清除该限制，实现版本兼容。
=========================================================
*/
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