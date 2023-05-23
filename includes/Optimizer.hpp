//
// Created by myyquq on 2023/4/3.
//

#ifndef MNIST_OPTIMIZER_HPP
#define MNIST_OPTIMIZER_HPP

#include "config.h"
#include "Activation.hpp"
#include "Layer.hpp"

namespace Optimizer {
    using Eigen::Dynamic;
    using Eigen::AutoOrder;

    class Optimizer {
    public:
        virtual void update(Layer::Layer *layer) = 0;
    };

    class SGD : public Optimizer {
    public:
        /**
         * @brief Stochastic Gradient Descent optimizer.
         * @param lr: learning rate
         * */
        SGD(double lr = 0.01) : lr_(lr) {}

        void update(Layer::Layer *layer) override {
            if (layer->name().starts_with("Dense")) {
                auto *dense = dynamic_cast<Layer::Dense *>(layer);
                value_type lr = (dense->learning_rate_ == 0) ? lr_ : dense->learning_rate_;
                dense->weights_ -= lr * (dense->input_.transpose() * dense->delta_);
                dense->biases_ -= lr * (dense->delta_.colwise().sum());
                dense->delta_.setZero();
            }
        }

    private:
        double lr_;
    };

}

#endif //MNIST_OPTIMIZER_HPP
