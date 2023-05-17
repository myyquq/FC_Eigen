//
// Created by myyquq on 2023/4/4.
//

#ifndef MNIST_LOSS_HPP
#define MNIST_LOSS_HPP

#include "config.h"

namespace Loss {
    class Loss {
    public:
        static value_type loss(const Matrix &x, const Matrix &y) {
            throw std::runtime_error("Loss function not implemented");
        }
        virtual value_type forward(const Matrix &x, const Matrix &y) = 0;   /// same as loss()
        virtual Matrix backward(const Matrix &x, const Matrix &y) = 0;
    };

//    class CategoricalCrossEntropy: public Loss {
//    public:
//        CategoricalCrossEntropy() = default;
//
//        static value_type loss(const Matrix &y_pred, const Matrix &y_true) {
//            return (y_pred - y_true).squaredNorm() / y_pred.rows();
//        }
//
//        value_type forward(const Matrix &y_pred, const Matrix &y_true) override {
//            return (y_pred - y_true).squaredNorm() / y_pred.rows();
//        }
//
//        Matrix backward(const Matrix &y_pred, const Matrix &y_true) override {
//            return (y_pred - y_true) / y_pred.rows();
//        }
//    };

    class MeanSquaredError: public Loss {
    public:
        MeanSquaredError() = default;

        static value_type loss(const Matrix &y_pred, const Matrix &y_true) {
            return (y_pred - y_true).squaredNorm() / y_pred.rows();
        }

        value_type forward(const Matrix &y_pred, const Matrix &y_true) override {
            return (y_pred - y_true).squaredNorm() / y_pred.rows();
        }

        Matrix backward(const Matrix &y_pred, const Matrix &y_true) override {
            return (y_pred - y_true) / y_pred.rows();
        }
    };
}

#endif //MNIST_LOSS_HPP
