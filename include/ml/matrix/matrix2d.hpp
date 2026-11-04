#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace ml {
namespace matrix {
template <typename T>
    requires std::integral<T> || std::floating_point<T>
class Matrix2D {
    struct {
        size_t rows;
        size_t cols;
        std::vector<T> elements;
    } m;

  public:
    Matrix2D(size_t, size_t);
    Matrix2D(std::vector<std::vector<T>>);

    Matrix2D(const Matrix2D &other) = default;                // copy constructor
    Matrix2D &operator=(const Matrix2D<T> &) = default;       // copy assignment
    Matrix2D(Matrix2D &&other) noexcept = default;            // move constructor
    Matrix2D &operator=(Matrix2D &&other) noexcept = default; // move assignment
    Matrix2D() = default;

    T &operator[](size_t, size_t);
    const T &operator[](size_t, size_t) const;
    bool operator==(const Matrix2D<T>&);

    Matrix2D row(size_t r);
    Matrix2D column(size_t c);
    std::pair<Matrix2D<T>, Matrix2D<T>> split_col(size_t c) const;
    std::pair<Matrix2D<T>, Matrix2D<T>> split_row(size_t r) const;

    Matrix2D &apply(std::function<T(T)>);
    Matrix2D &fill(std::function<T()>);
    Matrix2D &fill(T);

    Matrix2D &transpose();
    Matrix2D &add(const Matrix2D<T> &);
    Matrix2D &subtract(const Matrix2D<T> &);
    Matrix2D &multiply(const Matrix2D<T> &);

    std::pair<size_t, size_t> size() const;
    void print() const;
};

template <typename T> Matrix2D<T> identity(size_t size);
template <typename T> Matrix2D<T> add(const Matrix2D<T> &a, const Matrix2D<T> &b);
template <typename T> Matrix2D<T> subtract(const Matrix2D<T> &a, const Matrix2D<T> &b);
template <typename T> Matrix2D<T> multiply(const Matrix2D<T> &a, const Matrix2D<T> &b);
template <typename T> Matrix2D<T> transpose(const Matrix2D<T> &t);

}; // namespace matrix

} // namespace ml

#include <ml/matrix/matrix2d.tpp>
