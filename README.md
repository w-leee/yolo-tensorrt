# 🚀 YOLOv8 TensorRT 工业缺陷检测 C++ 部署端 

本项目是一个轻量级、无 Python 依赖的 C++ TensorRT 推理工程，专为工业级混凝土裂缝检测设计。通过解耦算法训练与工程部署，实现了极致的内存管理与端到端推理加速。

## 📊 性能基准 (Benchmark)

> **硬件平台**：RTX 3050 Laptop (4GB) / WSL2 Ubuntu / CUDA 12.6 / TensorRT 10.7 / OpenCV 4.10 / GCC 15.2

| 阶段 | 耗时 (ms) | 吞吐量 (FPS) | 说明 |
| :--- | :--- | :--- | :--- |
| PyTorch (Python) | 14.60 ms | 68 | 原始模型纯推理 |
| TRT FP16 (Python) | 3.72 ms | 269 | 纯推理 (不含前后处理) |
| TRT INT8 (Python) | 3.53 ms | 221 | 纯推理 (小图调度开销大) |
| TRT FP16 (C++ 端到端) | 3.73 ms | 267 | 图像解码、预处理、推理、NMS后处理 |


## 📂 项目结构

```text
yolo-trt-deploy/
├── cpp/                  # C++ 核心推理工程
│   ├── include/          # 头文件 (trt_engine.h)
│   ├── src/              # 源码 (main.cpp, trt_engine.cpp)
│   └── CMakeLists.txt    # CMake 构建脚本
├── trt_python/           # 模型量化与引擎编译脚本 (ONNX -> FP16/INT8)
├── weights/engines/      # 存放模型文件 (.onnx / .engine)
├── test_images/          # 测试样例图片
├── third_party/          # 本地 TensorRT SDK 依赖库
└── README.md