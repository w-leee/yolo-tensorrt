#include "trt_engine.h"
#include <chrono>
#include <iostream>
#include <algorithm>

/* 
命令行测试指令
./ 代表“当前”目录（你现在的所在地）
../ 代表“上一级”目录（退回上一级）
../../ 代表“上两级”目录（连退两级）

./yolo_trt ../../weights/engines/best_fp16.engine ../../test_images/input_test.jpg output_test.jpg
*/

/*
char：一个字符（比如 'A'）。
char*：一个字符串（比如 "hello"），它是一个指针，指向一串字符的起始地址。
char**：一个指向 char* 的指针。也就是“指向字符串的指针的指针”
*/
int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "用法: " << argv[0] << " <engine> <image> [output]\n";
        return -1;
    }

    std::string engine_path = argv[1];
    std::string image_path = argv[2];
    std::string output_path = (argc >= 4) ? argv[3] : "result.jpg";

    // 加载 engine
    TRTEngine engine(engine_path);

/*
=========================================================
【OpenCV 核心类 cv::Mat 图像属性与方法查阅手册】

在 AI 部署中，我们主要关注数据的内存布局、尺寸以及如何传输给 GPU。

1. 基础属性查询：
   - img.rows      : 图像高度（行数）
   - img.cols      : 图像宽度（列数）
   - img.channels(): 通道数（3 为彩色，1 为灰度）
   - img.empty()   : 判断图像是否读取失败或为空
   - img.type()    : 返回数据类型（如 CV_8UC3 代表 8位无符号3通道）

2. 底层内存操作（CUDA 极速传输核心）：
   - img.data          : 返回底层连续像素的一维裸指针 (uchar*)，用于 cudaMemcpy
   - img.isContinuous(): 判断内存是否绝对连续（如果不连续，需 clone 后传 GPU）
   - img.clone()       : 深拷贝，申请独立新内存
   - img.copyTo(dst)   : 将图像深拷贝给目标 Mat

3. 像素访问与修改：
   - img.at<cv::Vec3b>(row, col): 安全访问指定坐标的 BGR 像素（极慢，慎用）
   - img.ptr<uchar>(row)        : 获取指定行首地址的指针（遍历极快）
   - img.setTo(color)           : 将整图或 ROI 区域填充为特定颜色

4. 图像变换：
   - img.convertTo(dst, CV_32F, 1.0/255.0): 转换数据类型并做归一化
   - img.reshape(channels, rows)          : 改变矩阵维度（不拷贝数据）
   注：OpenCV 的 Resize、CvtColor 等操作是全局函数，写为 cv::resize(img, dst, ...)。

5. 常用几何结构类：
   - cv::Point(x, y) : 定义二维坐标点
   - cv::Scalar(B, G, R): 定义颜色（画图时常用）
=========================================================
*/
    
    // 读图
    cv::Mat img = cv::imread(image_path);
    if (img.empty())
    {
        std::cerr << "读不到图片: " << image_path << std::endl;
        return -1;
    }

    // ★ Warmup 10 次
    std::cout << "[Warmup] 10 次..." << std::endl;
    for (int i = 0; i < 10; i++)
    {
        engine.infer(img, 0.25f, 0.45f);
    }

/*
=========================================================
【Benchmark 性能压测模块：100 次极限测速】

逻辑流程：
1. 准备一个小本本 (times 数组)，用来记录每次推理所花费的毫秒数。
2. 准备一个纸箱 (detections 数组)，用来接收推理吐出来的检测框。
3. 开启 100 次循环，每次循环里：按下秒表 -> 执行推理 -> 停下秒表 -> 把耗时写进小本本。

关键语法：
- std::chrono::high_resolution_clock::now(): 获取高精度时间点。
- std::chrono::duration<double, std::milli>: 将时间差转换为 double 类型的毫秒。
- .count(): 提取 duration 对象中的纯数字。
=========================================================
*/

    // ★ 测速 100 次
    std::cout << "[Benchmark] 100 次..." << std::endl;
    int runs = 100;
    std::vector<double> times;
    std::vector<Detection> detections;

    for (int i = 0; i < runs; i++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        detections = engine.infer(img, 0.25f, 0.45f);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // 统计
    double sum = 0, tmin = 1e9, tmax = 0;
    for (double t : times)
    {
        sum += t;
        if (t < tmin)
            tmin = t;
        if (t > tmax)
            tmax = t;
    }
    double avg = sum / runs;

    std::cout << "\n========== 结果 ==========\n";
    std::cout << "平均: " << avg << " ms\n";
    std::cout << "最小: " << tmin << " ms\n";
    std::cout << "最大: " << tmax << " ms\n";
    std::cout << "FPS:  " << 1000.0 / avg << "\n";
    std::cout << "检测到 " << detections.size() << " 个目标\n";

/*
=========================================================
【结果可视化与持久化】
遍历检测结果，在原图上绘制边界框和置信度标签，最后保存至本地。
=========================================================
*/
    
    // 画框
    for (const auto& d : detections)
    {
/* 
[1] 绘制检测框
- cv::rectangle(图像, 左上角, 右下角, 颜色(BGR), 线条粗细)
- (0, 255, 0) 代表纯绿色，线宽为 2 像素。
*/
        cv::rectangle(img, cv::Point(d.x1, d.y1), cv::Point(d.x2, d.y2), cv::Scalar(0, 255, 0), 2);

/* 
[2] 生成标签文本
- std::to_string(d.conf): 将浮点数置信度转为字符串。
- .substr(0, 4): 截取前 4 个字符（如 "0.95"），防止文本过长遮挡画面。
*/
        std::string label = "positive " + std::to_string(d.conf).substr(0, 4);

/* 
[3] 计算文本绘制坐标 (防越界处理)
- 如果检测框紧贴图片边缘，文本可能会画到画面外。
- 使用 std::max 确保 x 坐标至少为 10，y 坐标至少为 25。
*/
        int text_x = std::max(static_cast<int>(d.x1), 10); 
        int text_y = std::max(static_cast<int>(d.y1), 25); 

/* 
[4] 绘制文本背景板 (提升可读性)
- 如果背景颜色与文字颜色相近，字会看不清。
- 在文字位置先画一个黑色实心矩形打底。
- 参数 -1 代表实心填充 (FILLED)，宽度 130，高度 20。
*/
        cv::rectangle(img, cv::Point(text_x, text_y - 20), cv::Point(text_x + 130, text_y), cv::Scalar(0, 0, 0), -1);

/* 
[5] 绘制文字
- cv::putText(图像, 文本, 坐标, 字体, 缩放大小, 颜色(BGR), 线宽)
- 使用亮黄色 (0, 255, 255) 配合黑色背景，对比度极高。
- y 坐标向上偏移 5 像素，让文字垂直居中于背景板。
*/
        cv::putText(img, label, cv::Point(text_x, text_y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }

/* 
=========================================================
【保存图像】
将处理后的 cv::Mat 写入本地硬盘，并在终端打印路径。
=========================================================
*/
    cv::imwrite(output_path, img);
    std::cout << "结果保存到: " << output_path << std::endl;
    
    return 0;
}