#pragma once

#include <ml/matrix/matrix2d.hpp>

#include <cstddef>
#include <functional>
#include <print>
#include <stdexcept>
#include <vector>

namespace ml {

namespace matrix {
template <typename T>
    requires std::integral<T> || std::floating_point<T>
Matrix2D<T>::Matrix2D(size_t rows, size_t cols) {
    m = {.rows = rows, .cols = cols};
    m.elements.resize(rows * cols);
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
Matrix2D<T>::Matrix2D(std::vector<std::vector<T>> mat) {
    m.rows = mat.size();
    m.cols = mat[0].size();
    for (std::vector<T> &v : mat) {
        if (m.cols != v.size()) {
            throw std::runtime_error("ERROR: The rows are of variable sizes, all the "
                                     "rows must have same number of elements");
        }
    }

    m.elements.resize(m.rows * m.cols);

    for (size_t i{}; i < m.rows; i++) {
        for (size_t j{}; j < m.cols; ++j) {
            (*this)[i, j] = mat[i][j];
        }
    }
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
T &Matrix2D<T>::operator[](size_t r, size_t c) {
    return m.elements[r * m.cols + c];
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
const T &Matrix2D<T>::operator[](size_t r, size_t c) const {
    return m.elements[r * m.cols + c];
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
bool Matrix2D<T>::operator==(const Matrix2D<T> &other) {
    if ((*this).size() != other.size()) {
        return false;
    }
    for (size_t i{}; i < m.elements.size(); ++i) {
        if (m.elements[i] != other.m.elements[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
Matrix2D<T> Matrix2D<T>::row(size_t r) {
    if (r > m.rows) {
        throw std::runtime_error("ERROR : Exceed the number the number of rows in the matrix.");
    }

    std::vector<std::vector<T>> vr;
    vr.push_back(std::vector<T>(m.elements.begin() + (r * m.cols), m.elements.begin() + ((r * m.cols) + m.cols)));
    Matrix2D<T> row(vr);
    return row;
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
Matrix2D<T> Matrix2D<T>::column(size_t c) {
    if (c > m.cols) {
        throw std::runtime_error("ERROR : Exceed the number the number of columns in the matrix.");
    }

    Matrix2D<T> column(m.cols, 1);
    for (size_t i{}; i < m.rows; ++i) {
        column[i, 0] = (*this)[i, c];
    }
    return column;
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
std::pair<Matrix2D<T>, Matrix2D<T>> Matrix2D<T>::split_col(size_t c) const {
    if (c > m.cols) {
        throw std::runtime_error("ERROR : Exceeded the number of columns in the matrix.");
    }

    Matrix2D<T> p(m.rows, c);
    Matrix2D<T> q(m.rows, (m.cols - c));

    auto [_, pcols] = p.size();
    auto [_, qcols] = q.size();

    for (size_t i{}; i < m.rows; ++i) {
        for (size_t j{}; j < pcols; ++j) {
            p[i, j] = (*this)[i, j];
        }
        for (size_t j{}; j < qcols; ++j) {
            q[i, j] = (*this)[i, pcols + j];
        }
    }
    return {p, q};
}
template <typename T>
    requires std::integral<T> || std::floating_point<T>
std::pair<Matrix2D<T>, Matrix2D<T>> Matrix2D<T>::split_row(size_t r) const {
    if (r > m.rows) {
        throw std::runtime_error("ERROR : Exceeded the number of row in the matrix.");
    }

    Matrix2D<T> p(r, m.cols);
    Matrix2D<T> q(m.rows - r, m.cols);

    auto [prows, _] = p.size();
    auto [qrows, _] = q.size();

    for (size_t j{}; j < m.rows; ++j) {
        for (size_t i{}; i < prows; ++i) {
            p[i, j] = (*this)[i, j];
        }
        for (size_t i{}; i < qrows; ++i) {
            q[i, j] = (*this)[i + prows, j];
        }
    }
    return {p, q};
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
Matrix2D<T> &Matrix2D<T>::fill(T val) {
    this->fill([val]() -> T { return val; });
    return (*this);
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
Matrix2D<T> &Matrix2D<T>::fill(std::function<T()> f) {
    for (size_t i{}; i < m.rows; ++i) {
        for (size_t j{}; j < m.cols; ++j) {
            (*this)[i, j] = f();
        }
    }
    return (*this);
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
Matrix2D<T> &Matrix2D<T>::apply(std::function<T(T)> f) {
    for (auto &i : m.elements) {
        i = f(i);
    }
    return (*this);
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
std::pair<size_t, size_t> Matrix2D<T>::size() const {
    return {m.rows, m.cols};
}

template <typename T>
    requires std::integral<T> || std::floating_point<T>
Matrix2D<T> &Matrix2D<T>::transpose() {
    (*this) = std::move(matrix::transpose(*this));
    return (*this);
};

template <typename T>
    requires std::integral<T> || std::floating_point<T>
Matrix2D<T> &Matrix2D<T>::add(const Matrix2D<T> &b) {
    (*this) = std::move(matrix::add(*this, b));
    return (*this);
};

template <typename T>
    requires std::integral<T> || std::floating_point<T>
Matrix2D<T> &Matrix2D<T>::subtract(const Matrix2D<T> &b) {
    (*this) = std::move(matrix::subtract(*this, b));
    return (*this);
};

template <typename T>
    requires std::integral<T> || std::floating_point<T>
Matrix2D<T> &Matrix2D<T>::multiply(const Matrix2D<T> &b) {
    (*this) = std::move(matrix::multiply(*this, b));
    return (*this);
};

template <typename T>
    requires std::integral<T> || std::floating_point<T>
void Matrix2D<T>::print() const {
    for (int i{}; i < m.rows; ++i) {
        for (int j{}; j < m.cols; ++j) {
            std::print("{:12}", (*this)[i, j]);
        }
        std::println();
    }
}

template <typename T> Matrix2D<T> identity(size_t size) {
    Matrix2D<T> m(size, size);

    for (size_t i{}; i < size; ++i) {
        for (size_t j{}; j < size; ++j) {
            if (i == j) {
                m[i, j] = 1;
            }
        }
    }

    return m;
}

template <typename T> Matrix2D<T> add(const Matrix2D<T> &a, const Matrix2D<T> &b) {
    if (a.size() == b.size()) {
        auto [rows, cols] = a.size();
        Matrix2D<T> sum(rows, cols);

        for (size_t i{}; i < rows; ++i) {
            for (size_t j{}; j < cols; ++j) {
                sum[i, j] = a[i, j] + b[i, j];
            }
        }

        return sum;
    } else {
        throw std::runtime_error("ERROR : Both matrices should be of the same dimension.");
    }
}

template <typename T> Matrix2D<T> subtract(const Matrix2D<T> &a, const Matrix2D<T> &b) {
    if (a.size() == b.size()) {
        auto [rows, cols] = a.size();
        Matrix2D<T> diff(rows, cols);

        for (size_t i{}; i < rows; ++i) {
            for (size_t j{}; j < cols; ++j) {
                diff[i, j] = a[i, j] - b[i, j];
            }
        }

        return diff;
    } else {
        throw std::runtime_error("ERROR : Both matrices should be of the same dimension.");
    }
}

template <typename T> Matrix2D<T> multiply(const Matrix2D<T> &a, const Matrix2D<T> &b) {
    auto [a_rows, a_cols] = a.size();
    auto [b_rows, b_cols] = b.size();

    if (a_cols == b_rows) {
        Matrix2D<T> product(a_rows, b_cols);

        for (size_t i{}; i < a_rows; ++i) {
            for (size_t j{}; j < b_cols; ++j) {
                product[i, j] = 0;
                for (size_t k{}; k < a_cols; k++) {
                    product[i, j] += a[i, k] * b[k, j];
                }
            }
        }

        return product;
    } else {
        throw std::runtime_error("ERROR: the number of colums in the first matrix should be same as the "
                                 "number of rows in the second matrix.");
    }
}

template <typename T> Matrix2D<T> transpose(const Matrix2D<T> &t) {
    auto [rows, cols] = t.size();

    Matrix2D<T> tp(cols, rows);

    for (size_t i{}; i < cols; ++i) {
        for (size_t j{}; j < rows; ++j) {
            tp[i, j] = t[j, i];
        }
    }

    return tp;
}
} // namespace matrix
} // namespace ml
