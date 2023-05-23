//
// Created by myyquq on 2023/4/4.
//

#ifndef MNIST_UTILS_HPP
#define MNIST_UTILS_HPP

#include "config.h"

namespace Utils {

    inline string to_string(const Shape &shape) {
        string s = "(";
        for (auto i: shape)
            s += (i == -1 ? "None" : std::to_string(i)) + ", ";
        s.pop_back();
        s.pop_back();
        s += ")";
        return s;
    }

    inline string progress_bar(int current, int total, int bar_length = 35) {
        string current_str = std::to_string(current);
        string total_str = std::to_string(total);
        int pos = bar_length * current / total;
        if (pos == bar_length - 1) {
            return string(total_str.size() - current_str.size(), ' ') + current_str + "/" + total_str + " [" +
                   string(bar_length - 1, '=') + ">]";
        } else if (pos == bar_length) {
            return string(total_str.size() - current_str.size(), ' ') + current_str + "/" + total_str + " [" +
                   string(bar_length, '=') + "]";
        }
        return string(total_str.size() - current_str.size(), ' ') + current_str + "/" + total_str + " [" +
                string(pos, '=') + ">" + string(bar_length - pos - 1, '.') + "]";
    }

    inline string time_format(double ms) {
        if (ms < 1000) return fmt::format("{:.2f}ms", ms);
        else if (ms < 60 * 1000) {
            int s = ms / 1000;
            ms -= s * 1000;
            return fmt::format("{}s {:.0f}ms", s, ms);
        } else if (ms < 60 * 60 * 1000) {
            int m = ms / 1000 / 60;
            ms -= m * 1000 * 60;
            int s = ms / 1000;
            ms -= s * 1000;
            return fmt::format("{}m {}s {:.0f}ms", m, s, ms);
        } else {
            int h = ms / 1000 / 60 / 60;
            ms -= h * 1000 * 60 * 60;
            int m = ms / 1000 / 60;
            ms -= m * 1000 * 60;
            int s = ms / 1000;
            ms -= s * 1000;
            return fmt::format("{}h {}m {}s {:.0f}ms", h, m, s, ms);
        }
    }

    inline Matrix argmax(const Matrix &x, int axis) {
        if (axis == 0) {
            Matrix result(x.cols(), 1);
            for (int i = 0; i < x.cols(); ++i) {
                value_type max = x.col(i).maxCoeff();
                for (int j = 0; j < x.rows(); ++j) {
                    if (::abs(x(j, i) - max) < EPS) {
                        result(i) = j;
                        break;
                    }
                }
            }
            return result;
        } else {
            Matrix result(x.rows(), 1);
            for (int i = 0; i < x.rows(); ++i) {
                value_type max = x.row(i).maxCoeff();
                for (int j = 0; j < x.cols(); ++j) {
                    if (::abs(x(i, j) - max) < EPS) {
                        result(i) = j;
                        break;
                    }
                }
            }
            return result;
        }
    }

    inline value_type accuracy(const Matrix &pred, const Matrix &ground_truth, bool onehot = true) {
        if (onehot) {
            Matrix pred_label = argmax(pred, 1);
            Matrix true_label = argmax(ground_truth, 1);
            return (abs(pred_label.array() - true_label.array()) < EPS).cast<value_type>().sum() / pred_label.size();
        } else {
            return (abs(pred.array() - ground_truth.array()) < EPS).cast<value_type>().sum() / pred.size();
        }
    }

    Matrix to_categorical(const Matrix &int_labels, int num_classes) {
        Matrix result(int_labels.rows(), num_classes);
        for (int i = 0; i < int_labels.rows(); ++i) {
            result.row(i).setZero();
            result(i, int(int_labels(i))) = 1;
        }
        return result;
    }

    Matrix resize(const Matrix &m, int h, int w) {
        Matrix result(h, w);
        for (int i = 0; i < h; ++i) {
            int x = i * m.rows() / h;
            for (int j = 0; j < w; ++j) {
                int y = j * m.cols() / w;
                result(i, j) = m(x, y);
            }
        }
        return result;
    }

    Matrix reshape(const Matrix &m, std::pair<int, int> shape) {
        auto &[h, w] = shape;
        if (h * w != m.size()) {
            throw std::runtime_error(fmt::format("reshape: shape ({}, {}) cannot match size {}", h, w, m.size()));
        }
        Matrix result(h, w);
        for (int i = 0; i < h * w; ++i) {
            result(i) = m(i / m.cols(), i % m.cols());  /// col-major
        }
        return result;
    }

}

#endif //MNIST_UTILS_HPP
