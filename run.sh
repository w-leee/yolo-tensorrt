#!/bin/bash

# 进入 build 目录
cd cpp/build

# 编译
make -j$(nproc)

# 运行程序
./yolo_trt \
    ../../weights/engines/best_fp16.engine \
    ../../test_images/input_test.jpg \
    output_test.jpg