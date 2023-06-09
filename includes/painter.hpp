﻿//
// Created by myyquq on 2023/4/3.
//

#ifndef MNIST_PAINTER_HPP
#define MNIST_PAINTER_HPP

#include <opencv2/opencv.hpp>
#include <functional>
#include "config.h"

namespace Painter {

    float BRUSH_WIDTH_SCALE = 0.04;
    int WIDTH = 784;
    int HEIGHT = 784;

    cv::Mat canvas;
    cv::Scalar brushColor;

    void clearCanvas() {
        canvas = cv::Scalar(255, 255, 255);
    }

    std::function<void()> call;

    void onMouse(int event, int x, int y, int flags, void* userdata) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            cv::circle(canvas, cv::Point(x, y), WIDTH * BRUSH_WIDTH_SCALE, brushColor, -1);
        } else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
            cv::circle(canvas, cv::Point(x, y), WIDTH * BRUSH_WIDTH_SCALE, brushColor, -1);
        } else if (event == cv::EVENT_LBUTTONUP) {
            call();
        }
    }

    void draw(int height, int width, const std::function<void(Matrix)> &notify) {
        HEIGHT = height;
        WIDTH = width;

        call = [&notify]() {
            cv::Mat canvasGray;
            cv::cvtColor(canvas, canvasGray, cv::COLOR_BGR2GRAY);
            cv::Mat canvasArray = canvasGray > 0;
            Matrix data = Matrix::Zero(HEIGHT, WIDTH);
            for (int i = 0; i < HEIGHT; i++) {
                for (int j = 0; j < WIDTH; j++) {
                    data(i, j) = 255.f - canvasArray.at<uchar>(i, j);
                }
            }
            notify(data);
        };

        canvas = cv::Mat(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::String windowName = "Canvas";
        brushColor = cv::Scalar(0, 0, 0);
        cv::namedWindow(windowName);
        cv::setMouseCallback(windowName, onMouse);

        while (true) {
            cv::imshow(windowName, canvas);

            int key = cv::waitKey(1);
            if (key == 27) {
                break;
            } else if (key == 'c' || key == 'C') {
                clearCanvas();
            }
        }
    }
}


#endif //MNIST_PAINTER_HPP