//
// Created by myyquq on 2023/4/3.
//

#ifndef MNIST_MODEL_HPP
#define MNIST_MODEL_HPP

#include <ranges>

#include "config.h"
#include "Layer.hpp"
#include "Activation.hpp"
#include "Optimizer.hpp"
#include "Utils.hpp"
#include "Loss.hpp"


namespace Model {
    using Eigen::Dynamic;
    using Eigen::AutoOrder;
//    using Tensor = Eigen::Tensor<value_type, Dynamic>;

/// TODO:
    class Model {
    public:
        Model() {}
        virtual void summary() = 0;
        virtual void compile(Optimizer::Optimizer *optimizer, Loss::Loss *loss) = 0;

    private:
        virtual Matrix forward(const Matrix &x) = 0;
        virtual void backward(const Matrix &x) = 0;
        virtual void update() = 0;
        virtual Matrix predict(const Matrix &x) = 0;

    protected:
    };


    class Sequential : public Model {
    public:
        Sequential() {}

        template<typename T>
        void add(T *layer) {
            if (!layers_.empty()) {
                layer->input_shape_ = layers_.back()->output_shape();
            }
            layers_.push_back(layer);
        }

        Matrix forward(const Matrix &x) override {
            Matrix y = x;
            for (auto layer: layers_) {
                y = layer->forward(y, true);
            }
            return y;
        }

        void backward(const Matrix &x) override {
            Matrix y = x;
            for (auto &layer: std::ranges::reverse_view(layers_)) {
                y = layer->backward(y);
            }
        }

        void update() override {
            for (auto layer: layers_) {
                optimizer_->update(layer);
            }
        }

        void summary() override {
            fmt::print("Model: Sequential\n");
            fmt::print("_________________________________________________________________\n");
            fmt::print("{:<14} {:<18} {:<18} {:<14}\n", "Layer", "Input Shape", "Output Shape", "Param #");
            fmt::print("=================================================================\n");
            int64_t params = 0;
            for (auto layer: layers_) {
                fmt::print("{:<14} {:<18} {:<18} {:<14}\n",
                           layer->name(),
                           Utils::to_string(layer->input_shape()),
                           Utils::to_string(layer->output_shape()),
                           std::to_string(layer->parameters())
                );
                params += layer->parameters();
            }
            fmt::print("=================================================================\n");
            fmt::print("Total params: {}\n", params);
        }

        void compile(Optimizer::Optimizer *optimizer, Loss::Loss *loss) override {
            optimizer_ = optimizer;
            loss_ = loss;
        }

        void fit(const Matrix &x, const Matrix &y, int epochs, int batch_size) {
            int n = int(x.rows());
            if (batch_size > n)
                throw std::runtime_error("batch_size must be less than or equal to the number of samples");
            int n_batch = n / batch_size;
            for (int epoch = 0; epoch < epochs; ++epoch) {
                fmt::print("Epoch {}/{}\n", epoch + 1, epochs);
                value_type total_loss = 0, total_accuracy = 0;
                auto start = std::chrono::system_clock::now();
                for (int i = 0; i < n_batch; ++i) {
                    Matrix x_batch      = x.block(i * batch_size, 0, batch_size, x.cols());
                    Matrix y_batch      = y.block(i * batch_size, 0, batch_size, y.cols());
                    Matrix y_pred       = forward(x_batch);
                    value_type accuracy = Utils::accuracy(y_pred, y_batch);
                    value_type loss     = loss_->forward(y_pred, y_batch);
                    total_loss          += loss;
                    total_accuracy      += accuracy;
                    backward(loss_->backward(y_pred, y_batch));
                    update();
                    auto now = std::chrono::system_clock::now();
                    double elapsed = double(std::chrono::duration_cast<std::chrono::microseconds>(now - start).count());
                    fmt::print("{} - ETA: {} - loss: {:.6f} - accuracy: {:.6f}{}\r",
                               Utils::progress_bar(i + 1, n_batch, PROGRESS_BAR_LEN),
                               Utils::time_format(elapsed / 1000 / (i + 1) * (n_batch - i - 1)),
                               loss,
                               accuracy,
                               string(20, ' ')
                    );
                }
                auto stop = std::chrono::system_clock::now();
                double elapsed = double(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
                fmt::print("{} - {}/step - loss: {:.6f} - accuracy: {:.6f}{}\n",
                           Utils::progress_bar(n_batch, n_batch, PROGRESS_BAR_LEN),
                           Utils::time_format(elapsed / 1000),
                           total_loss     / value_type(n_batch),
                           total_accuracy / value_type(n_batch),
                           string(20, ' ')
                );
            }
        }

        Matrix predict(const Matrix &x) override {
            Matrix y = x;
            for (auto layer: layers_) {
                y = layer->forward(y, false);
            }
            return y;
        }

        int64_t parameters() {
            int64_t n = 0;
            for (auto layer: layers_) {
                n += layer->parameters();
            }
            return n;
        }

    private:
        std::vector<Layer::Layer *> layers_;
        Optimizer::Optimizer *optimizer_;
        Loss::Loss *loss_;
    };
}

#endif //MNIST_MODEL_HPP
