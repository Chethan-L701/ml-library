#pragma once

#include <ml/matrix/matrix2d.hpp>
#include <ml/nn/nn.hpp>

#include <concepts>
#include <cstddef>
#include <functional>
#include <print>
#include <stdexcept>
#include <vector>

namespace ml {
namespace nn {
template <typename T>
    requires std::floating_point<T> || std::integral<T>
NeuralNetwork<T>::NeuralNetwork(std::vector<size_t> arch, std::function<T()> fill, std::function<T(T)> activation)
    : arch(arch), activation(activation) {
    for (size_t i{}; i < arch.size(); ++i) {
        (*this).activations.push_back(matrix::Matrix2D<T>(1, arch[i]).fill(0.0f));
        if (i != arch.size() - 1)
            (*this).layers.push_back(DenseLayer<T>(arch[i], arch[i + 1], fill));
    }
}

template <typename T>
    requires std::floating_point<T> || std::integral<T>
void NeuralNetwork<T>::print() {
    for (size_t i{}; i < arch.size(); ++i) {
        if (i == 0) {
            std::print("input({}) -> ", arch[i]);
        } else if (i == arch.size() - 1) {
            std::print("output({})", arch[i]);
        } else {
            std::print("layer{}({}) -> ", i, arch[i]);
        }
    }
    std::println();
}

template <typename T>
    requires std::floating_point<T> || std::integral<T>
void NeuralNetwork<T>::print_all() {
    std::println("activations [] : [ ");
    for (size_t i{}; i < activations.size(); ++i) {
        std::println(" activation {}: [", i);
        activations[i].print();
        std::println("  ]");
    }
    std::println("]");
    std::println("------------------------------------------------------------");
    for (size_t i{}; i < layers.size(); ++i) {
        std::println("weight {}: [", i);
        layers[i].weight.print();
        std::println("]");

        std::println("bias   {}: [", i);
        layers[i].bias.print();
        std::println("]");
        std::println("------------------------------------------------------------");
    }
}

template <typename T>
    requires std::floating_point<T> || std::integral<T>
void NeuralNetwork<T>::forward(const matrix::Matrix2D<T> &input) {
    activations[0] = input;
    for (size_t i{0}; i < layers.size(); i++) {
        auto &layer = layers[i];
        activations[i + 1] = matrix::add(matrix::multiply(activations[i], layer.weight), layer.bias).apply(activation);
    }
}

template <typename T>
    requires std::floating_point<T> || std::integral<T>
T NeuralNetwork<T>::cost(matrix::Matrix2D<T> &input, matrix::Matrix2D<T> &output) {
    T c{};

    auto [irows, _] = input.size();
    auto [_, ocols] = output.size();

    for (size_t i{}; i < irows; i++) {
        auto x = input.row(i);
        forward(x);

        auto _y = output.row(i);                      // actual output
        auto y = activations[activations.size() - 1]; // predicted output
        auto diff = matrix::subtract(y, _y);

        for (size_t j{}; j < ocols; ++j) {
            c += (diff[0, j] * diff[0, j]);
        }
    }
    return c / irows;
}

template <typename T>
    requires std::floating_point<T> || std::integral<T>
std::vector<DenseLayer<T>> NeuralNetwork<T>::finite_difference(matrix::Matrix2D<T> &input, matrix::Matrix2D<T> &output,
                                                               float eps) {

    std::vector<DenseLayer<T>> diff;
    T c = cost(input, output);

    for (DenseLayer<T> &layer : layers) {
        auto [wrows, wcols] = layer.weight.size();
        auto [_, bcols] = layer.bias.size();

        DenseLayer<T> d(layer.weight.size());
        for (size_t i{}; i < wrows; ++i) {
            for (size_t j{}; j < wcols; ++j) {
                T temp = layer.weight[i, j];
                layer.weight[i, j] += eps;
                d.weight[i, j] = (cost(input, output) - c) / eps;
                layer.weight[i, j] = temp;
            }
        }
        for (size_t i{}; i < bcols; ++i) {
            T temp = layer.bias[0, i];
            layer.bias[0, i] += eps;
            d.bias[0, i] = (cost(input, output) - c) / eps;
            layer.bias[0, i] = temp;
        }
        diff.push_back(std::move(d));
    }

    return diff;
}

template <typename T>
    requires std::floating_point<T> || std::integral<T>
void NeuralNetwork<T>::learn(matrix::Matrix2D<T> &input, matrix::Matrix2D<T> &output, float eps, float rate) {
    std::vector<DenseLayer<T>> fd = finite_difference(input, output, eps);
    auto apply_rate = [rate](T e) { return e * rate; };
    for (size_t i{}; i < fd.size(); ++i) {
        DenseLayer<T> d = DenseLayer<T>{fd[i].weight.apply(apply_rate), fd[i].bias.apply(apply_rate)};
        layers[i] = layers[i] - d;
    }
}

template <typename T>
    requires std::floating_point<T> || std::integral<T>
void NeuralNetwork<T>::train(matrix::Matrix2D<T> &input, matrix::Matrix2D<T> &output, size_t cycles, float eps,
                             float rate) {
    if (input.size().first != output.size().first) {
        throw std::runtime_error("ERROR : number of rows in the training input and output data should be same");
    }

    if (activations[0].size() != input.row(0).size()) {
        throw std::runtime_error("ERROR : the dimension of input matrix does not match the dimesion of input layer");
    }

    if (activations[activations.size() - 1].size() != output.row(0).size()) {
        throw std::runtime_error("ERROR : the dimension of output matrix does not match the dimesion of input layer");
    }

    for (size_t i{}; i < cycles; ++i) {
        learn(input, output, eps, rate);
    }
}

template <typename T>
    requires std::floating_point<T> || std::integral<T>
matrix::Matrix2D<T> NeuralNetwork<T>::predict(matrix::Matrix2D<T> &input) {
    auto [orows, _] = input.size();
    auto [_, ocols] = activations[activations.size() - 1].size();
    matrix::Matrix2D<T> result(orows, ocols);
    for (size_t i{}; i < orows; i++) {
        auto x = input.row(i);
        forward(x);
        auto y = activations[activations.size() - 1]; // predicted output
        for (size_t j{}; j < ocols; ++j) {
            result[i, j] = y[0, j];
        }
    }
    return result;
};
} // namespace nn
} // namespace ml
