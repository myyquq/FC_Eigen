//
// Created by myyquq on 2023/4/3.
//

#ifndef MNIST_LAYER_HPP
#define MNIST_LAYER_HPP

#include <utility>
#include <random>

#include "config.h"
#include "Activation.hpp"
#include "Model.hpp"

namespace Layer {
    using Eigen::Dynamic;
    using Eigen::AutoOrder;


    class Layer {
    public:
        Layer(string name = "Layer") : name_(std::move(name)) {}
        virtual Matrix forward(const Matrix &x, bool train) = 0;
        virtual Matrix backward(const Matrix &x) = 0;
        virtual std::vector<int> input_shape() = 0;
        virtual std::vector<int> output_shape() = 0;
        virtual int64_t parameters() = 0;

        string name() {
            return name_;
        }

//    protected:
        string name_;
        Activation::Activation *activation_;
        Shape input_shape_;
    };

    class Input : public Layer {
        static int count;
    public:
        Input(const Shape &shape) : Layer("Input" + std::to_string(count++)) {
            if (shape.size() == 1)
                input_shape_ = {-1, shape[0]};
            else if (shape.size() == 2)
                input_shape_ = shape;
            else
                throw std::invalid_argument("Input shape must be 1D or 2D");
        }

        Matrix forward(const Matrix &x, bool train) override {
            return x;
        }

        Matrix backward(const Matrix &x) override {
            return {};
        }

        std::vector<int> input_shape() override {
            return input_shape_;
        }

        std::vector<int> output_shape() override {
            return input_shape_;
        }

        int64_t parameters() override {
            return 0;
        }
    };

//    class Flatten: public Layer {   /// currently only support 2D input, deprecated
//        static int count;
//    public:
//        Flatten(): Layer("Flatten" + std::to_string(count++)) {}
//        Matrix forward(const Matrix &x) override {
//            return x.reshaped<Eigen::AutoOrder>(1, x.rows() * x.cols());
//        }
//        Matrix backward(const Matrix &x) override {
//            return x.reshaped(Eigen::NoChange, input_shape_[0]);
//        }
//        std::vector<int> input_shape() override {
//            return input_shape_;
//        }
//        std::vector<int> output_shape() override {
//            return output_shape_;
//        }
//        int64_t parameters() override {
//            return 0;
//        }
//    private:
//        Shape output_shape_;
//    };

    class Dense : public Layer {
        static int count;
    public:
        Dense(int units, string activation, value_type learning_rate = 0.) : Layer("Dense" + std::to_string(count++)) {
            units_ = units;
            learning_rate_ = learning_rate;
            output_shape_ = {-1, units_};
            std::transform(activation.begin(), activation.end(), activation.begin(), ::tolower);
            if (activation == "relu") {
                activation_ = new Activation::ReLU();
            } else if (activation == "sigmoid") {
                activation_ = new Activation::Sigmoid();
            } else if (activation == "softmax") {
                activation_ = new Activation::Softmax();
            } else if (activation == "linear") {
                activation_ = new Activation::Linear();
            } else {
                throw std::invalid_argument("Invalid activation function");
            }
        }

        void build(const Shape &shape) {
            weights_ = Matrix::Random(shape[1], units_);
            biases_ = Matrix::Random(1, units_);
            delta_ = Matrix::Zero(shape[1], units_);
            initialized = true;
        }

        Matrix forward(const Matrix &x, bool train) override {
            if (!initialized) {
                input_shape_ = {-1, int(x.cols())};    /// -1 means unknown batch_size
                build(input_shape_);
            }
            input_ = x;
            output_ = input_ * weights_ + biases_.replicate(x.rows(), 1);
            return activation_->forward(output_);
        }

        Matrix backward(const Matrix &x) override {
            delta_ = activation_->backward(x);
            return delta_ * weights_.transpose();
        }

        std::vector<int> input_shape() override {
            return input_shape_;
        }

        std::vector<int> output_shape() override {
            return output_shape_;
        }

        int64_t parameters() override {
            return input_shape_[1] * units_ + units_;
        }

    public:
        int units_;
        bool initialized = false;
        value_type learning_rate_ = 0.;
        Shape output_shape_;
        Matrix weights_, biases_;
        Matrix input_, output_, delta_;
    };

    class Dropout : public Layer {
        static int count;
    public:
        Dropout(value_type rate) : Layer("Dropout" + std::to_string(count++)), rate_(rate) {
            if (rate_ < 0 || rate_ > 1)
                throw std::invalid_argument("Dropout rate must be in [0, 1]");
        }

        Matrix forward(const Matrix &x, bool train = true) override {
            if (train) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::bernoulli_distribution dist(1 - rate_);
                mask_ = Matrix::Zero(x.rows(), x.cols()).unaryExpr([&](value_type _) -> value_type {
                    return dist(gen);
                });
                return x.cwiseProduct(mask_);
            } else {
                return x * (1 - rate_);
            }
        }

        Matrix backward(const Matrix &x) override {
            return x.cwiseProduct(mask_);
        }

        std::vector<int> input_shape() override {
            return input_shape_;
        }

        std::vector<int> output_shape() override {
            return input_shape_;
        }

        int64_t parameters() override {
            return 0;
        }

//    private:
        value_type rate_;
        Matrix mask_;
    };


    int Input::count = 1;
//    int Flatten::count = 1;
    int Dense::count = 1;
    int Dropout::count = 1;
}

#endif //MNIST_LAYER_HPP
