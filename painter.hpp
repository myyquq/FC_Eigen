//
// Created by myyquq on 2023/4/3.
//

#ifndef MNIST_PAINTER_HPP
#define MNIST_PAINTER_HPP

#include "config.h"
#include "opencv2/opencv.hpp"
#include <functional>

namespace Painter {

    float BRUSH_WIDTH_SCALE = 0.05;
    int WIDTH = 784;
    int HEIGHT = 784;

    cv::Mat canvas;  // 画布
    cv::Scalar brushColor;  // 画笔颜色
    std::function<void()> call;

    // 鼠标事件回调函数
    void onMouse(int event, int x, int y, int flags, void* userdata) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            cv::circle(canvas, cv::Point(x, y), WIDTH * BRUSH_WIDTH_SCALE, brushColor, -1);  // 在画布上绘制黑色点
        } else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
            cv::circle(canvas, cv::Point(x, y), WIDTH * BRUSH_WIDTH_SCALE, brushColor, -1);  // 在画布上绘制黑色点
        } else if (event == cv::EVENT_LBUTTONUP) {
            call();
        }
    }

    // 清空画布函数
    void clearCanvas() {
        canvas = cv::Scalar(255, 255, 255);  // 清空画布
    }

    void draw(int width, int height, Matrix &data, const std::function<void()> &notify) {
        WIDTH = width;
        HEIGHT = height;

        // 创建一个白色的画布
        canvas = cv::Mat(WIDTH, HEIGHT, CV_8UC3, cv::Scalar(255, 255, 255));

        // 设置窗口名称
        cv::String windowName = "绘图窗口";

        // 设置画笔颜色为黑色
        brushColor = cv::Scalar(0, 0, 0);

        // 创建窗口
        cv::namedWindow(windowName);

        // 设置鼠标事件回调函数
        cv::setMouseCallback(windowName, onMouse);

        call = [&data, &notify]() {
            cv::Mat canvasGray;
            cv::cvtColor(canvas, canvasGray, cv::COLOR_BGR2GRAY);
            cv::Mat canvasArray = canvasGray > 0;
            data = Matrix::Zero(WIDTH, HEIGHT);
            for (int i = 0; i < WIDTH; i++) {
                for (int j = 0; j < HEIGHT; j++) {
                    data(i, j) = 255. - canvasArray.at<uchar>(i, j);
                }
            }
            notify();
        };

        while (true) {
            // 显示画布
            cv::imshow(windowName, canvas);

            // 等待按键事件，ESC键退出
            char key = cv::waitKey(1);
            if (key == 27) {
                break;
            } else if (key == 'c' || key == 'C') {
                // 清空画布并重置笔数
                clearCanvas();
            }
        }
    }

}


#endif //MNIST_PAINTER_HPP