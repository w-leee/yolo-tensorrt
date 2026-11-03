# 🚀 YOLOv8 TensorRT 工业缺陷检测 C++ 部署端 

本项目是一个轻量级、无 Python 依赖的 C++ TensorRT 推理工程，专为工业级混凝土裂缝检测设计。通过解耦算法训练与工程部署，实现了极致的内存管理与端到端推理加速。

## 📊 性能基准 (Benchmark)

> **硬件平台**：RTX 3050 Laptop (4GB) / WSL2 Ubuntu / CUDA 12.6 / TensorRT 10.7 / OpenCV 4.10 / GCC 15.2

| 阶段 | 耗时 (ms) | 吞吐量 (FPS) | 说明 |
| :--- | :--- | :--- | :--- |
| PyTorch (Python) | 14.60 ms | 68 | 原始模型纯推理 |
| TRT FP16 (Python) | 3.72 ms | 269 | 纯推理 (不含前后处理) |
| TRT FP16 (C++ 纯CPU预处理) | 4.02 ms | 248 | 图像解码、CPU预处理、推理、NMS |
| **TRT FP16 (C++ + CUDA预处理)** | **3.23 ms** | **308** | **CUDA Kernel 实现 HWC转CHW，极致加速** |


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
```
## 🗺️ 后续优化路线 (Roadmap)
- [ ] **零拷贝前处理**：引入 `cv::cuda::GpuMat` 彻底消除 CPU 到 GPU 的显存拷贝开销。
- [ ] **多线程流水线**：实现生产者-消费者模型，支持多路视频流并发推理。
- [ ] **量化部署**：完善 INT8 校准流程，在精度允许场景下进一步突破吞吐上限。
