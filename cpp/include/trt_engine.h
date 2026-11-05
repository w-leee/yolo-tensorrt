#pragma once
/*
=========================================================
【AI 部署三件套：三大核心头文件解析】

在 C++ AI 部署项目中，这三个头文件就是支撑整个推理流水线的“三根擎天柱”：

1. <opencv2/opencv.hpp> - 图像处理总管 (CPU 侧)
   作用：负责解析图片、处理图像矩阵。
   项目中：使用 cv::imread 读图，cv::resize 缩放，cv::cvtColor 转色，以及最后画框保存。
   比喻：它是 CPU 端的“图像切菜板”，把原始 JPEG 变成可操作的数字矩阵。

2. <cuda_runtime.h> - 显卡内存与通讯总管 (CPU-GPU 桥梁)
   作用：负责向显卡申请显存、主机与设备间的数据拷贝、启动核函数。
   项目中：使用 cudaMalloc 分配显存，cudaMemcpyAsync 传输数据，<<< >>> 启动 GPU 并行计算。
   比喻：它是连接主板和显卡的“物流公司”，负责把数据精准、高速地搬运到显卡的仓库里。

3. <NvInfer.h> - TensorRT 核心推理库 (GPU 侧)
   作用：负责加载引擎，在显卡上执行极限并行的矩阵乘法。
   项目中：使用 createInferRuntime 反序列化，enqueueV3 执行推理。
   比喻：它是安装在显卡里的“超算中心”，只要数据到位，它就能以极低延迟完成 AI 核心运算。

【全局协作流程】
OpenCV 读图切菜 -> CUDA 搬运上车 -> TensorRT 核心狂算 -> CUDA 拉回结果 -> OpenCV 画框收尾。
=========================================================
*/
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/* 检测框结构体 */
struct Detection
{
    float x1, y1, x2, y2;
    float conf;
    int class_id;
};

class TRTEngine
{
  public:
    TRTEngine(const std::string& engine_path);
    ~TRTEngine();

    std::vector<Detection> infer(const cv::Mat& img, float conf_thresh = 0.25f, float iou_thresh = 0.45f);

  private:
    void preprocess(const cv::Mat& img, float* gpu_input);
    std::vector<Detection> postprocess(const float* output, int img_w, int img_h, float conf_thresh, float iou_thresh);
    static float iou(const Detection& a, const Detection& b);

    /* TensorRT 核心对象（生命周期由析构函数接管） */
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    /* 显存指针 */
    void* gpu_input_ = nullptr;     /* 存放经过归一化、CHW 排布的 float 张量 (TensorRT 输入) */
    void* gpu_output_ = nullptr;    /* 存放 TensorRT 推理输出的 float 张量 */
    void* gpu_rgb_input_ = nullptr; /* 存放从 CPU 传过来的原始 RGB 数据 (HWC, uchar) */
    cudaStream_t stream_ = nullptr; /* CUDA 异步流，用于隐藏 CPU-GPU 传输延迟 */

    /* 模型固定参数 */
    int input_w_ = 416;
    int input_h_ = 416;
    int num_anchors_ = 3549;
    int num_classes_ = 1;
    int output_size_ = 0;
};