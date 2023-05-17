//
// Created by myyquq on 2023/4/3.
//
#ifndef MNIST_CONFIG_H
#define MNIST_CONFIG_H

#include "Eigen/Dense"
#include "fmt/core.h"
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <iomanip>
#include <cmath>
#include <random>
#include <stdexcept>
#include <chrono>


using          value_type       = float;      /// 矩阵底层数据结构
constexpr auto EPS              = 1e-6;       /// 精度
constexpr int  PROGRESS_BAR_LEN = 35;   /// 进度条长度
using          Shape            = std::vector<int>;
using std::string;
using std::cout;
using std::endl;
using std::cerr;
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;



#endif //MNIST_CONFIG_H
