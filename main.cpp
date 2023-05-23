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
    Eigen::setNbThreads(2);
    auto [train_data, train_label, test_data, test_label] =
            dataloader::load_mnist("../dataset",true); /// onehot labels
    train_data /= 255.;   /// preprocessing
    test_data /= 255.;

    fmt::print("\n");
    auto model = Model::Sequential();
    model.add(new Input(/*shape=*/{784}));
    model.add(new Dense(/*units=*/256, /*activation=*/"relu", /*learning_rate=*/2e-1));
    model.add(new Dropout(/*rate=*/0.5));
    model.add(new Dense(/*units=*/10, /*activation=*/"softmax", /*learning_rate=*/5e-2));
    model.compile(new Optimizer::SGD(), new Loss::MeanSquaredError());
    model.summary();

    fmt::print("\n");
    model.fit(train_data, train_label, /*epochs=*/150, /*batch_size=*/32);

    auto test_pred = model.predict(test_data);
    fmt::print("\ntest accuracy: {:.2f}%\n", Utils::accuracy(test_pred, test_label) * 100.);
    fmt::print("test loss: {:.6f}\n", Loss::MeanSquaredError::loss(test_pred, test_label));

    fmt::print("\nPress C to clear the canvas, press Esc to exit.\n");

    int height = 28 * 28, width = 28 * 28;
    Painter::draw(height, width, [&](Matrix canvas) {
        Matrix resized = Utils::resize(canvas, 28, 28) / 255.;
        resized = Utils::reshape(resized, {1, 28 * 28});
        Matrix pred = model.predict(resized);
        int pred_label = int(Utils::argmax(pred, 1)(0));
        float pred_prob = pred(0, pred_label);
        fmt::print("Prediction: {}, probability: {:.2f}%{}\r", pred_label, pred_prob * 100., string(20, ' '));
    });

    return 0;
}
