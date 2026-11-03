cmake_minimum_required(VERSION 3.20)

/* 
【工程配置】统一工具链版本，解决 Ubuntu 26.04 (GCC 15) 与 CUDA 12.6 (最高支持 GCC 13) 的冲突。
*/
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12")

/* 
【兼容性补丁】强制 nvcc 包含兼容头文件，解决新版 glibc 数学函数的 noexcept 声明与 CUDA 头文件冲突的问题。
*/
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-include,${CMAKE_CURRENT_SOURCE_DIR}/include/glibc_c23_math_compat.h")

/* 
【启用 CUDA】项目同时启用 C++ (CXX) 和 CUDA 语言支持。
*/
project(yolo_trt_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

/*
【依赖路径】配置 TensorRT 和 CUDA 的本地路径。
*/
set(TENSORRT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/TensorRT-10.7.0.23)
set(CUDA_ROOT /usr/local/cuda-12.6)

include_directories(
    ${TENSORRT_ROOT}/include
    ${CUDA_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(
    ${TENSORRT_ROOT}/lib
    ${CUDA_ROOT}/lib64
)

find_package(OpenCV REQUIRED)

/* 
【链接优化】静态链接 libstdc++ 和 libgcc，防止不同 GCC 版本编译出的符号在链接时发生冲突。
*/
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")

/*
【编译目标】将纯 C++ 代码与 CUDA 核心代码混合编译，生成最终可执行文件。
*/
add_executable(yolo_trt_demo
    src/main.cpp
    src/trt_engine.cpp
    src/preprocess.cu
)

target_link_libraries(yolo_trt_demo
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
    ${OpenCV_LIBS}
)

target_compile_options(yolo_trt_demo PRIVATE -Wall -O2)