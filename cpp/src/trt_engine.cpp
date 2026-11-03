#include "trt_engine.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

/* 
【CUDA 函数声明】声明由 preprocess.cu 提供的 GPU 启动函数。
*/
void launch_preprocess_kernel(const unsigned char* d_src, float* d_dst, int total_pixels);

/* TensorRT 日志记录器 */
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

static Logger gLogger;

/* 
【构造函数】加载 Engine、分配显存、创建 CUDA 流。
*/
TRTEngine::TRTEngine(const std::string& engine_path) {
    /* 读取引擎二进制文件 */
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("无法打开 engine: " + engine_path);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    /* 反序列化引擎并创建执行上下文 */
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!engine_) {
        throw std::runtime_error("反序列化 engine 失败");
    }
    context_ = engine_->createExecutionContext();

    /* 设置输入 Shape (动态 Shape 必须先设置) */
    nvinfer1::Dims4 input_dims{1, 3, input_h_, input_w_};
    context_->setInputShape("images", input_dims);

    /* 
    【显存分配】
    gpu_rgb_input_: 用于暂存原图 (3 * H * W 字节)
    gpu_input_: 用于存放 TensorRT 最终需要的浮点张量 (1 * 3 * H * W 个 float)
    */
    cudaMalloc(&gpu_rgb_input_, 3 * input_h_ * input_w_ * sizeof(unsigned char));
    cudaMalloc(&gpu_input_, 1 * 3 * input_h_ * input_w_ * sizeof(float));

    /* 获取输出 Shape 并分配输出显存 */
    auto out_dims = context_->getTensorShape("output0");
    std::cout << "[DEBUG] 输出维度数: " << out_dims.nbDims << std::endl;
    output_size_ = 1;
    for (int i = 0; i < out_dims.nbDims; i++) {
        std::cout << "[DEBUG] 维度 " << i << ": " << out_dims.d[i] << std::endl;
        output_size_ *= out_dims.d[i];
    }

    /* 兜底逻辑：如果动态获取失败，硬编码 */
    if (output_size_ <= 0) {
        output_size_ = 5 * num_anchors_;
        std::cout << "[WARN] 从 context 拿不到输出 shape，硬编码为 " << output_size_ << std::endl;
    }

    cudaMalloc(&gpu_output_, output_size_ * sizeof(float));

    /* 创建异步 CUDA 流 */
    cudaStreamCreate(&stream_);

    std::cout << "[TRTEngine] 加载成功，输入 "
              << input_w_ << "x" << input_h_
              << "，输出元素数 " << output_size_ << std::endl;
}

/* 
【析构函数】逆序释放资源，防止显存泄漏。
*/
TRTEngine::~TRTEngine() {
    if (stream_) cudaStreamDestroy(stream_);
    if (gpu_rgb_input_) cudaFree(gpu_rgb_input_);
    if (gpu_input_) cudaFree(gpu_input_);
    if (gpu_output_) cudaFree(gpu_output_);
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
}

/* 
【预处理】将 CPU 上的图像转换为 TensorRT 需要的 GPU 张量。
流程：CPU 缩放 -> CPU 转 RGB -> 拷贝到显存 -> GPU 核函数并行转 CHW + 归一化。
*/
void TRTEngine::preprocess(const cv::Mat& img, float* gpu_input) {
    cv::Mat resized, rgb;
    cv::resize(img, resized, cv::Size(input_w_, input_h_));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    /* 保证内存连续，便于 CUDA 直接拷贝 */
    if (!rgb.isContinuous()) rgb = rgb.clone();

    /* 
    【数据拷贝】将轻量的原图数据拷贝到 gpu_rgb_input_ 显存中。
    这一步的数据量较小（几百 KB），远比拷贝处理后的 float 张量要快。
    */
    cudaMemcpyAsync(gpu_rgb_input_, rgb.data, 
                    input_w_ * input_h_ * 3 * sizeof(unsigned char), 
                    cudaMemcpyHostToDevice, stream_);

    /* 
    【GPU 核函数】启动 CUDA 线程，并行完成 HWC 到 CHW 的转换，并做归一化。
    结果直接写入 TensorRT 的输入显存 gpu_input_ 中，避免额外的拷贝。
    */
    int total_pixels = input_w_ * input_h_;
    launch_preprocess_kernel((unsigned char*)gpu_rgb_input_, gpu_input, total_pixels);
}

/* 
【IoU 计算】用于 NMS 的后处理。
*/
float TRTEngine::iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

/* 
【后处理】对模型输出进行解码（DFL 映射）和 NMS 过滤。
*/
std::vector<Detection> TRTEngine::postprocess(const float* output,
                                                int img_w, int img_h,
                                                float conf_thresh,
                                                float iou_thresh) {
    std::vector<Detection> detections;

    for (int i = 0; i < num_anchors_; i++) {
        float conf = output[4 * num_anchors_ + i];
        if (conf < conf_thresh) continue;

        float cx = output[0 * num_anchors_ + i];
        float cy = output[1 * num_anchors_ + i];
        float w  = output[2 * num_anchors_ + i];
        float h  = output[3 * num_anchors_ + i];

        Detection d;
        d.x1 = (cx - w / 2) * img_w / input_w_;
        d.y1 = (cy - h / 2) * img_h / input_h_;
        d.x2 = (cx + w / 2) * img_w / input_w_;
        d.y2 = (cy + h / 2) * img_h / input_h_;
        d.conf = conf;
        d.class_id = 0;
        detections.push_back(d);
    }

    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });

    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (!suppressed[j] && iou(detections[i], detections[j]) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

/* 
【推理主函数】串联预处理、引擎推理和后处理。
*/
std::vector<Detection> TRTEngine::infer(const cv::Mat& img,
                                          float conf_thresh,
                                          float iou_thresh) {
    preprocess(img, static_cast<float*>(gpu_input_));

    nvinfer1::Dims4 input_dims{1, 3, input_h_, input_w_};
    context_->setInputShape("images", input_dims);

    context_->setTensorAddress("images", gpu_input_);
    context_->setTensorAddress("output0", gpu_output_);

    /* 异步推理 */
    context_->enqueueV3(stream_);

    /* 异步拷贝回结果，并同步等待 */
    std::vector<float> output_data(output_size_);
    cudaMemcpyAsync(output_data.data(), gpu_output_,
                    output_size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    return postprocess(output_data.data(), img.cols, img.rows,
                       conf_thresh, iou_thresh);
}