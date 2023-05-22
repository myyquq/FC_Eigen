//
// Created by myyquq on 2023/4/3.
//

#ifndef MNIST_ACTIVATION_HPP
#define MNIST_ACTIVATION_HPP

#include <queue>
#include "config.h"
#include "Eigen/Core"

namespace Activation {
    using Eigen::Dynamic;
    using Eigen::AutoOrder;
//    using Tensor = Eigen::Tensor<value_type, Dynamic>;

    class Activation {
    public:
        virtual inline Matrix forward(const Matrix &x) = 0;
        virtual inline Matrix backward(const Matrix &x) = 0;

//        virtual Tensor forward(const Tensor &x) = 0;
//        virtual Tensor backward(const Tensor &x) = 0;
    };

    class ReLU : public Activation {
    public:
        inline Matrix forward(const Matrix &x) override {
            output = x.cwiseMax(0.);
            return output;
        }

        inline Matrix backward(const Matrix &x) override {
            return x.cwiseProduct(output.cwiseMax(0.));
        }

    private:
        Matrix output;
    };

    class Sigmoid : public Activation {
    public:
        inline Matrix forward(const Matrix &x) override {
            output = 1 / (1 + (-x).array().exp());
            return output;
        }

        inline Matrix backward(const Matrix &x) override {
            return (output.array() - output.array().square()) * x.array();
        }

    private:
        Matrix output;
    };

    class Softmax : public Activation {
    public:
        inline Matrix forward(const Matrix &x) override {
            Vector max = x.rowwise().maxCoeff();
            output = (x - max.replicate(1, x.cols())).array().exp();
            Vector sum = output.rowwise().sum();
            for (int i = 0; i < output.rows(); ++i) {
                output.row(i) /= sum(i);
            }
            return output;
        }

        inline Matrix backward(const Matrix &x) override {
            Matrix ret = Matrix::Zero(x.rows(), x.cols());
            for (int i = 0; i < x.rows(); ++i) {
                Matrix output_diag = output.row(i).asDiagonal();
                Matrix output_diag_minus_output_transpose = output_diag - output.row(i).transpose() * output.row(i);
                for (int j = 0; j < x.cols(); ++j) {
                    ret(i, j) += (output_diag_minus_output_transpose.row(j).array() * x.row(i).array()).sum();
                }
            }
            return ret;
        }

    private:
        Matrix output;
    };

    class Tanh : public Activation {
    public:
        inline Matrix forward(const Matrix &x) override {
            output = x.array().tanh();
            return output;
        }

        inline Matrix backward(const Matrix &x) override {
            return (1 - output.array().square()) * x.array();
        }
    private:
        Matrix output;
    };

    class Linear : public Activation {
    public:
        inline Matrix forward(const Matrix &x) override {
            return x;
        }

        inline Matrix backward(const Matrix &x) override {
            return x;
        }
    };
}

#endif //MNIST_ACTIVATION_HPP
