#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

/* 检测框结构体 */
struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int class_id;
};

class TRTEngine {
public:
    TRTEngine(const std::string& engine_path);
    ~TRTEngine();

    std::vector<Detection> infer(const cv::Mat& img,
                                  float conf_thresh = 0.25f,
                                  float iou_thresh  = 0.45f);

private:
    void preprocess(const cv::Mat& img, float* gpu_input);
    std::vector<Detection> postprocess(const float* output, int img_w, int img_h,
                                        float conf_thresh, float iou_thresh);
    static float iou(const Detection& a, const Detection& b);

    /* TensorRT 核心对象（生命周期由析构函数接管） */
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    /* 显存指针 */
    void* gpu_input_ = nullptr;      /* 存放经过归一化、CHW 排布的 float 张量 (TensorRT 输入) */
    void* gpu_output_ = nullptr;     /* 存放 TensorRT 推理输出的 float 张量 */
    void* gpu_rgb_input_ = nullptr;  /* 存放从 CPU 传过来的原始 RGB 数据 (HWC, uchar) */
    cudaStream_t stream_ = nullptr;  /* CUDA 异步流，用于隐藏 CPU-GPU 传输延迟 */

    /* 模型固定参数 */
    int input_w_ = 416;
    int input_h_ = 416;
    int num_anchors_ = 3549;
    int num_classes_ = 1;
    int output_size_ = 0;
};