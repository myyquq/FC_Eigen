#include <iostream>
#include "Eigen/Dense"
#include "Model.hpp"
#include "dataloader.hpp"
#include "Utils.hpp"

using Layer::Dense;
using Layer::Input;
using Layer::Dropout;

int main() {
    auto [train_data, train_label, test_data, test_label] =
            dataloader::load_mnist("../dataset",true); /// onehot labels
    train_data /= 255.;   /// preprocessing
    test_data /= 255.;

    fmt::print("\n");
    auto model = Model::Sequential();
    model.add(new Input({784}));
    model.add(new Dense(128, "relu", 2e-1));
    model.add(new Dropout(0.5));
    model.add(new Dense(10, "softmax", 5e-2));
    model.compile(new Optimizer::SGD(), new Loss::MeanSquaredError());
    model.summary();

    fmt::print("\n");
    model.fit(train_data, train_label, 100, 32);

    auto test_pred = model.predict(test_data);
    fmt::print("\ntest accuracy: {:.4f}%\n", Utils::accuracy(test_pred, test_label) * 100.);
    fmt::print("test loss: {:.6f}\n", Loss::MeanSquaredError::loss(test_pred, test_label));
    auto test_pred_label = Utils::argmax(test_pred, 1);
    auto test_label_label = Utils::argmax(test_label, 1);
    fmt::print("First 50 test samples:\n");
    for (int i = 0; i < 50; ++i) {
        fmt::print("label: {}, pred: {}, {}\n",
                   test_label_label(i),
                   test_pred_label(i),
                   (test_pred_label(i) == test_label_label(i) ? "correct" : "WRONG")
        );
    }


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