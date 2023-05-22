#include <iostream>
#include "Eigen/Dense"
#include "Model.hpp"
#include "dataloader.hpp"
#include "Utils.hpp"
#include "painter.hpp"

using Layer::Dense;
using Layer::Input;
using Layer::Dropout;


int main() {
    Eigen::initParallel();
    Eigen::setNbThreads(4);
    auto [train_data, train_label, test_data, test_label] =
            dataloader::load_mnist("../dataset",true); /// onehot labels
    train_data /= 255.;   /// preprocessing
    test_data /= 255.;

    fmt::print("\n");
    auto model = Model::Sequential();
    model.add(new Input({784}));
    model.add(new Dense(256, "relu", 2e-1));
    model.add(new Dropout(0.5));
    model.add(new Dense(10, "softmax", 5e-2));
    model.compile(new Optimizer::SGD(), new Loss::MeanSquaredError());
    model.summary();

    fmt::print("\n");
    model.fit(train_data, train_label, 150, 32);

    auto test_pred = model.predict(test_data);
    fmt::print("\ntest accuracy: {:.2f}%\n", Utils::accuracy(test_pred, test_label) * 100.);
    fmt::print("test loss: {:.6f}\n", Loss::MeanSquaredError::loss(test_pred, test_label));

    fmt::print("\nPress C to clear the canvas, press Esc to exit.\n");

    int height = 28 * 28, width = 28 * 28;
    Matrix canvas = Matrix::Zero(height, width);
    Painter::draw(height, width, canvas, [&]() {
        Matrix resized = Utils::resize(canvas, 28, 28) / 255.;
        resized = Utils::reshape(resized, {1, 28 * 28});
        Matrix pred = model.predict(resized);
        int pred_label = int(Utils::argmax(pred, 1)(0));
        float pred_prob = pred(0, pred_label);
        fmt::print("Prediction: {}, probability: {:.2f}%{}\r", pred_label, pred_prob, string(20, ' '));
    });


    /**
     * Test:
     *      large batch size, slow convergence:
     *          Dense 128 0.5 -> sigmoid -> Dropout 0.2 -> Dense 64 0.2 -> sigmoid -> Dropout 0.2 -> Dense 10 0.1-> softmax, 250 epochs, 1024 batch size
     *                accuracy: approximately 91%
     *
     *          Dense 128 0.5 -> sigmoid -> Dropout 0.5 -> Dense 10 0.2-> softmax, 250 epochs, 1024 batch size
     *                accuracy: approximately 92%
     *
     *          Dense 128 0.5 -> relu -> Dropout 0.2 -> Dense 32 0.2 -> relu -> Dropout 0.2 -> Dense 10 0.2-> softmax, 150 epochs, 1024 batch size
     *                accuracy: approximately 93%
     *
     *
     *          Dense 128 0.5 -> relu -> Dropout 0.5 -> Dense 10 0.2-> softmax, 250 epochs, 1024 batch size
     *                accuracy: approximately 94%
     *
     *
     *      small batch size, fast convergence, but easily overfit:
     *          Dense 128 0.5 -> sigmoid -> Dropout 0.5 -> Dense 10 0.2-> softmax, 100 epochs, 32 batch size
     *                accuracy: approximately 96%
     *
     *          Dense 128 0.5 -> relu -> Dropout 0.5 -> Dense 10 0.2-> softmax, 100 epochs, 32 batch size
     *                accuracy: approximately 97%
     *
     * */

    return 0;
}
