//
// Created by myyquq on 2023/4/2.
//

#ifndef MNIST_DATALOADER_HPP
#define MNIST_DATALOADER_HPP

#include "config.h"
#include "fmt/core.h"
#include "Utils.hpp"
#include <utility>
#include <fstream>
#include <string>

namespace dataloader {
    using std::tuple;
    using std::string;

    // [train_data, train_label, test_data, test_label]
    /**
     * @brief Load MNIST dataset from csv files.
     * @param path: the path of the dataset.
     * @param one_hot: whether to use one-hot encoding for labels.
     * @return a tuple of four matrices: train_data, train_label, test_data, test_label.
     * */
    tuple<Matrix, Matrix, Matrix, Matrix> load_mnist(const string &path, bool one_hot = true) {
        const int train_data_size = 60000;
        const int test_data_size = 10000;
        const int data_size = 28 * 28;
        const int label_size = 10;
        const int header = 1;
        Matrix train_data = Matrix::Constant(train_data_size, data_size, 0);
        Matrix test_data = Matrix::Constant(test_data_size, data_size, 0);
        Matrix train_label, test_label;
        if (one_hot) {
            train_label = Matrix::Constant(train_data_size, label_size, 0);
            test_label = Matrix::Constant(test_data_size, label_size, 0);
        } else {
            train_label = Matrix::Constant(train_data_size, 1, 0);
            test_label = Matrix::Constant(test_data_size, 1, 0);
        }
        std::ifstream train_file(path + "/mnist_train.csv");
        std::ifstream test_file(path + "/mnist_test.csv");
        if (!train_file.is_open()) {
            fmt::print("Error: Cannot open file: {}.\n", path + "/mnist_train.csv");
            exit(1);
        }
        if (!test_file.is_open()) {
            fmt::print("Error: Cannot open file: {}.\n", path + "/mnist_test.csv");
            exit(1);
        }
        string cell;
        for (int i = 0; i < header; ++i) {
            std::getline(train_file, cell);
            std::getline(test_file, cell);
        }
        fmt::print("Loading MNIST training data: \n");
        for (int i = 0; i < train_data_size; ++i) {
            if (i % 100 == 99) {
                fmt::print(
                        "{}{}\r",
                        Utils::progress_bar(i + 1, train_data_size, PROGRESS_BAR_LEN),
                        string(20, ' ')
                );
            }
            std::getline(train_file, cell, ',');
            int label = std::stoll(cell);
            if (one_hot) {
                train_label(i, label) = 1;
            } else {
                train_label(i, 0) = label;  // inplicit conversion from int to float
            }
            for (int64_t j = 0; j < data_size - 1; ++j) {
                std::getline(train_file, cell, ',');
                train_data(i, j) = std::stof(cell);
            }
            std::getline(train_file, cell);
            train_data(i, data_size - 1) = std::stof(cell);
        }
        fmt::print("\nLoading MNIST testing data: \n");
        for (int i = 0; i < test_data_size; ++i) {
            if (i % 100 == 99) {
                fmt::print(
                        "{}{}\r",
                        Utils::progress_bar(i + 1, test_data_size, PROGRESS_BAR_LEN),
                        string(20, ' ')
                );
            }
            std::getline(test_file, cell, ',');
            int64_t label = std::stoll(cell);
            if (one_hot) {
                test_label(i, label) = 1;
            } else {
                test_label(i, 0) = label;
            }
            for (int64_t j = 0; j < data_size - 1; ++j) {
                std::getline(test_file, cell, ',');
                test_data(i, j) = std::stof(cell);
            }
            std::getline(test_file, cell);
            test_data(i, data_size - 1) = std::stof(cell);
        }
        fmt::print("\nMNIST dataset loaded successfully.\n");
        return {train_data, train_label, test_data, test_label};
    }
}

#endif //MNIST_DATALOADER_HPP
