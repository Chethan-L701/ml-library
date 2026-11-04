#pragma once

#include <ml/matrix/matrix2d.hpp>

#include <concepts>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace ml {
namespace nn {

template <typename T>
    requires std::floating_point<T> || std::integral<T>
struct DenseLayer {
    matrix::Matrix2D<T> weight;
    matrix::Matrix2D<T> bias;

    DenseLayer(size_t rows, size_t cols, std::function<T()> fill = []() { return 0; }) {
        weight = matrix::Matrix2D<T>(rows, cols).fill(fill);
        bias = matrix::Matrix2D<T>(1, cols).fill(fill);
    }
    DenseLayer(std::pair<size_t, size_t> dim, std::function<T()> fill = []() { return 0; }) {
        weight = matrix::Matrix2D<T>(dim.first, dim.second).fill(fill);
        bias = matrix::Matrix2D<T>(1, dim.second).fill(fill);
    }

    DenseLayer(const matrix::Matrix2D<T> &weight, const matrix::Matrix2D<T> &bias) : weight(weight), bias(bias) {}

    DenseLayer(const DenseLayer &) = default;
    DenseLayer(DenseLayer &&) = default;
    DenseLayer &operator=(const DenseLayer &) = default;
    DenseLayer &operator=(DenseLayer &&) = default;

    DenseLayer operator-(const DenseLayer<T> &l) const {
        return DenseLayer<T>{matrix::subtract((*this).weight, l.weight), matrix::subtract((*this).bias, l.bias)};
    };
    DenseLayer operator+(const DenseLayer<T> &l) const {
        return DenseLayer<T>{matrix::add((*this).weight, l.weight), matrix::add((*this).bias, l.bias)};
    };
};

template <typename T>
    requires std::floating_point<T> || std::integral<T>
class NeuralNetwork {
    std::vector<size_t> arch;
    std::vector<ml::matrix::Matrix2D<T>> activations;
    std::vector<DenseLayer<T>> layers;

    std::function<T(T)> activation;

  public:
    NeuralNetwork(std::vector<size_t>, std::function<T()>, std::function<T(T)>);
    void train(matrix::Matrix2D<T> &, matrix::Matrix2D<T> &, size_t cycles = 10000, float eps = 1e-3,
               float rate = 1e-3);

    matrix::Matrix2D<T> predict(matrix::Matrix2D<T> &input);
    T cost(matrix::Matrix2D<T> &, matrix::Matrix2D<T> &);

    NeuralNetwork(const NeuralNetwork &) = default;
    NeuralNetwork(NeuralNetwork &&) = default;
    NeuralNetwork &operator=(const NeuralNetwork &) = default;
    NeuralNetwork &operator=(NeuralNetwork &&) = default;
    ~NeuralNetwork() = default;

    void print();
    void print_all();

  private:
    std::vector<DenseLayer<T>> finite_difference(matrix::Matrix2D<T> &, matrix::Matrix2D<T> &, float);
    void forward(const matrix::Matrix2D<T> &);
    void learn(matrix::Matrix2D<T> &, matrix::Matrix2D<T> &, float eps, float rate);
};
} // namespace nn
} // namespace ml

#include <ml/nn/nn.tpp>
