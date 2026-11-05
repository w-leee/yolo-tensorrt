# YOLOv8 TensorRT 工业缺陷检测 C++ 部署端

本项目是通过解耦算法训练与工程部署，实现了极致的内存管理与端到端推理加速。

## 性能基准 (Benchmark)

| 阶段 | 耗时 (ms) | 吞吐量 (FPS) | 说明 |
| :--- | :--- | :--- | :--- |
| PyTorch (Python) | 14.60 ms | 68 | 原始模型纯推理 |
| TRT FP16 (Python) | 3.72 ms | 269 | 纯推理 (不含前后处理) |
| TRT FP16 (C++ 纯CPU预处理) | 4.02 ms | 248 | 图像解码、CPU预处理、推理、NMS |
| **TRT FP16 (C++ + CUDA预处理)** | **3.23 ms** | **308** | **CUDA Kernel 实现 HWC转CHW，极致加速** |

## 项目结构

```text
yolo-trt-deploy/
├── cpp/                  # C++ 核心推理工程
│   ├── include/          # 头文件
│   ├── src/              # 源码
│   └── CMakeLists.txt    # 构建脚本
├── trt_python/           # 模型量化与引擎编译脚本
├── weights/engines/      # 模型文件存放目录
├── test_images/          # 测试样例图片
├── third_party/          # 本地 TensorRT SDK 依赖库
└── README.md