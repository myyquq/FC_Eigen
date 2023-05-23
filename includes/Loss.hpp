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
    };

    class MeanAbsoluteError : public Loss {
    public:
        MeanAbsoluteError() = default;

        /**
         * @brief Calculate the mean absolute error between y_pred and y_true.
         * @param y_pred: the predicted value
         * @param y_true: the true value
         * */
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

        /**
         * @brief Calculate the mean squared error between y_pred and y_true.
         * @param y_pred: the predicted value
         * @param y_true: the true value
         * */
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

        /**
         * @brief Calculate the binary cross entropy between y_pred and y_true.
         * @param y_pred: the predicted value
         * @param y_true: the true value
         * */
        static value_type loss(const Matrix &y_pred, const Matrix &y_true) {
            Matrix clipped_y_pred = y_pred.cwiseMax(EPSILON).cwiseMin(1 - EPSILON);
            return -(y_true.array() * clipped_y_pred.array().log() +
                     (1 - y_true.array()) * (1 - clipped_y_pred.array()).log()).sum() / y_pred.rows();
        }

        inline value_type forward(const Matrix &y_pred, const Matrix &y_true) override {
            Matrix clipped_y_pred = y_pred.cwiseMax(1e-7).cwiseMin(1 - 1e-7);
            return -(y_true.array() * clipped_y_pred.array().log() +
                     (1 - y_true.array()) * (1 - clipped_y_pred.array()).log()).sum() / y_pred.rows();
        }

        static constexpr value_type EPSILON = 1e-7;
    };
}

#endif //MNIST_LOSS_HPP
