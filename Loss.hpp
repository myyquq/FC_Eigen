//
// Created by myyquq on 2023/4/4.
//

#ifndef MNIST_LOSS_HPP
#define MNIST_LOSS_HPP

#include "config.h"

namespace Loss {
    class Loss {
    public:
        virtual inline value_type forward(const Matrix &y_pred, const Matrix &y_true) = 0;

        virtual inline Matrix backward(const Matrix &y_pred, const Matrix &y_true) {
            return (y_pred - y_true) / y_pred.rows();
        };
    };

    class MeanAbsoluteError : public Loss {
    public:
        MeanAbsoluteError() = default;

        static value_type loss(const Matrix &y_pred, const Matrix &y_true) {
            return (y_pred - y_true).cwiseAbs().sum() / y_pred.rows();
        }

        inline value_type forward(const Matrix &y_pred, const Matrix &y_true) override {
            return (y_pred - y_true).cwiseAbs().sum() / y_pred.rows();
        }
    };

    class MeanSquaredError : public Loss {
    public:
        MeanSquaredError() = default;

        static value_type loss(const Matrix &y_pred, const Matrix &y_true) {
            return (y_pred - y_true).squaredNorm() / y_pred.rows();
        }

        inline value_type forward(const Matrix &y_pred, const Matrix &y_true) override {
            return (y_pred - y_true).squaredNorm() / y_pred.rows();
        }
    };

    class BinaryCrossEntropy : public Loss {
    public:
        BinaryCrossEntropy() = default;

        /// yTrue×log(y)+(1-yTrue)×log(1-y)
        static value_type loss(const Matrix &y_pred, const Matrix &y_true) {
            Matrix clipped_y_pred = y_pred.cwiseMax(1e-7).cwiseMin(1 - 1e-7);
            return -(y_true.array() * clipped_y_pred.array().log() +
                     (1 - y_true.array()) * (1 - clipped_y_pred.array()).log()).sum() / y_pred.rows();
        }

        inline value_type forward(const Matrix &y_pred, const Matrix &y_true) override {
            Matrix clipped_y_pred = y_pred.cwiseMax(1e-7).cwiseMin(1 - 1e-7);
            return -(y_true.array() * clipped_y_pred.array().log() +
                     (1 - y_true.array()) * (1 - clipped_y_pred.array()).log()).sum() / y_pred.rows();
        }
    };
}

#endif //MNIST_LOSS_HPP
