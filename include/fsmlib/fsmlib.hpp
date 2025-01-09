/// @file fsmlib.hpp
/// @brief Defines fixed-size vector and matrix types for linear algebra operations.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <cstddef>

#include "fsmlib/traits.hpp"

#define COLUMN_MAJOR

/// @brief Namespace for the fixed-size matrix and vector library.
namespace fsmlib
{

/// @brief Abstract base class for fixed-size vectors.
/// @tparam T The type of the vector elements.
/// @tparam N The number of elements in the vector.
template <typename T, std::size_t N>
class VectorBase : public fsmlib::traits::valid_container_t {
public:
    /// @brief Type of the elements in the vector.
    using value_type = T;
    /// @brief Type of the size of the vector.
    using size_type  = std::integral_constant<std::size_t, N>;

    /// @brief Virtual destructor.
    virtual ~VectorBase() = default;

    /// @brief Returns the size of the vector.
    /// @return Size of the vector.
    virtual constexpr std::size_t size() const noexcept
    {
        return N;
    }

    /// @brief Access an element by index (const version).
    /// @param index Index of the element to access.
    /// @return const reference to the element.
    virtual const T &at(std::size_t index) const = 0;

    /// @brief Access an element by index (non-const version).
    /// @param index Index of the element to access.
    /// @return Reference to the element.
    virtual T &at(std::size_t index) = 0;

    /// @brief Access an element by index (const version).
    /// @param index Index of the element to access.
    /// @return const reference to the element.
    virtual const T &operator[](std::size_t index) const = 0;

    /// @brief Access an element by index (non-const version).
    /// @param index Index of the element to access.
    /// @return Reference to the element.
    virtual T &operator[](std::size_t index) = 0;

    /// @brief Provides access to the underlying data (const version).
    /// @return Pointer to the underlying data.
    virtual const T *data() const noexcept = 0;

    /// @brief Provides access to the underlying data (non-const version).
    /// @return Pointer to the underlying data.
    virtual T *data() noexcept = 0;

    /// @brief Begin iterator (non-const version).
    /// @return Pointer to the beginning of the vector.
    virtual T *begin() noexcept = 0;

    /// @brief End iterator (non-const version).
    /// @return Pointer to the end of the vector.
    virtual T *end() noexcept = 0;

    /// @brief Begin iterator (const version).
    /// @return Const pointer to the beginning of the vector.
    virtual const T *begin() const noexcept = 0;

    /// @brief End iterator (const version).
    /// @return Const pointer to the end of the vector.
    virtual const T *end() const noexcept = 0;

protected:
    /// @brief Checks if the index is within bounds.
    /// @param index index to check.
    virtual void check_bounds(std::size_t index) const = 0;
};

/// @brief Concrete class for a fixed-size vector.
/// @tparam T The type of the vector elements.
/// @tparam N The number of elements in the vector.
template <typename T, std::size_t N>
class Vector : public VectorBase<T, N> {
public:
    /// @brief Default constructor.
    constexpr Vector() : data_{}
    {
    }

    /// @brief Copy constructor.
    Vector(const fsmlib::VectorBase<T, N> &other) : data_{}
    {
        if (this != &other) {
            std::copy(other.data(), other.data() + N, data_);
        }
    }

    /// @brief Copy constructor.
    Vector(const fsmlib::VectorBase<const T, N> &other) : data_{}
    {
        if (this != &other) {
            std::copy(other.data(), other.data() + N, data_);
        }
    }

    /// @brief Constructor from an initializer list.
    /// @param init Initializer list of elements.
    constexpr Vector(std::initializer_list<T> init) : data_{}
    {
        if (init.size() != N) {
            throw std::out_of_range("Initializer list size does not match vector size");
        }
        std::size_t i = 0;
        for (const auto &value : init) {
            data_[i++] = value;
        }
    }

    constexpr const T &at(std::size_t index) const override
    {
        check_bounds(index);
        return data_[index];
    }

    constexpr T &at(std::size_t index) override
    {
        check_bounds(index);
        return data_[index];
    }

    constexpr const T &operator[](std::size_t index) const override
    {
        check_bounds(index);
        return data_[index];
    }

    constexpr T &operator[](std::size_t index) override
    {
        check_bounds(index);
        return data_[index];
    }

    constexpr const T *data() const noexcept override
    {
        return data_;
    }

    constexpr T *data() noexcept override
    {
        return data_;
    }

    constexpr T *begin() noexcept override
    {
        return data_;
    }

    constexpr T *end() noexcept override
    {
        return data_ + N;
    }

    constexpr const T *begin() const noexcept override
    {
        return data_;
    }

    constexpr const T *end() const noexcept override
    {
        return data_ + N;
    }

    /// @brief Copy assignment operator.
    /// @param other The vector to copy from.
    /// @return Reference to the assigned vector.
    Vector &operator=(const fsmlib::VectorBase<T, N> &other)
    {
        if (this != &other) {
            std::copy(other.data(), other.data() + N, data_);
        }
        return *this;
    }

private:
    T data_[N]; ///< Internal storage for the vector elements.

    void check_bounds(std::size_t index) const override
    {
        if (index >= N) {
            throw std::out_of_range("Vector index out of bounds");
        }
    }
};

/// @brief Base class for fixed-size matrices.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
template <typename T, std::size_t Rows, std::size_t Cols = Rows>
class MatrixBase : public fsmlib::traits::valid_container_t {
public:
    /// @brief Type of the elements in the matrix.
    using value_type = T;
    /// @brief Type of the size of the matrix.
    using size_type  = std::pair<std::size_t, std::size_t>;

    /// @brief Virtual destructor.
    virtual ~MatrixBase() = default;

    /// @brief Returns the size of the matrix.
    /// @return Pair containing the number of rows and columns.
    virtual constexpr std::size_t size() const noexcept
    {
        return Rows * Cols;
    }

    /// @brief Returns the number of rows in the matrix.
    /// @return Number of rows.
    constexpr std::size_t rows() const noexcept
    {
        return Rows;
    }

    /// @brief Returns the number of columns in the matrix.
    /// @return Number of columns.
    constexpr std::size_t cols() const noexcept
    {
        return Cols;
    }

    /// @brief Access an element by row and column (const version).
    /// @param row Row index of the element.
    /// @param col Column index of the element.
    /// @return const reference to the element.
    virtual const T &at(std::size_t row, std::size_t col) const = 0;

    /// @brief Access an element by row and column (non-const version).
    /// @param row Row index of the element.
    /// @param col Column index of the element.
    /// @return Reference to the element.
    virtual T &at(std::size_t row, std::size_t col) = 0;

    /// @brief Access an element by linear index (const version).
    /// @param index Linear index of the element.
    /// @return const reference to the element.
    virtual const T &at(std::size_t index) const = 0;

    /// @brief Access an element by linear index (non-const version).
    /// @param index Linear index of the element.
    /// @return Reference to the element.
    virtual T &at(std::size_t index) = 0;

    /// @brief Access an element by row and column (const version).
    /// @param row Row index of the element.
    /// @param col Column index of the element.
    /// @return const reference to the element.
    virtual const T &operator()(std::size_t row, std::size_t col) const = 0;

    /// @brief Access an element by row and column (non-const version).
    /// @param row Row index of the element.
    /// @param col Column index of the element.
    /// @return Reference to the element.
    virtual T &operator()(std::size_t row, std::size_t col) = 0;

    /// @brief Access an element by linear index (const version).
    /// @param index Linear index of the element.
    /// @return const reference to the element.
    virtual const T &operator[](std::size_t index) const = 0;

    /// @brief Access an element by linear index (non-const version).
    /// @param index Linear index of the element.
    /// @return Reference to the element.
    virtual T &operator[](std::size_t index) = 0;

    /// @brief Provides access to the underlying data (const version).
    /// @return Pointer to the underlying data.
    virtual const T *data() const noexcept = 0;

    /// @brief Provides access to the underlying data (non-const version).
    /// @return Pointer to the underlying data.
    virtual T *data() noexcept = 0;

protected:
    /// @brief Converts row and column indices to a linear index.
    /// @param row the row index.
    /// @param col the column index.
    /// @return the linear index.
    virtual std::size_t indices_to_linear_index(std::size_t row, std::size_t col) const = 0;

    /// @brief Checks if the row and column indices are within bounds.
    /// @param row Row index to check.
    /// @param col Column index to check.
    virtual void check_bounds(std::size_t row, std::size_t col) const = 0;

    /// @brief Checks if the linear index is within bounds.
    /// @param index Linear index to check.
    virtual void check_bounds(std::size_t index) const = 0;
};

/// @brief Fixed-size matrix class in column-major order.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
template <typename T, std::size_t Rows, std::size_t Cols = Rows>
class Matrix : public MatrixBase<T, Rows, Cols> {
public:
    /// @brief Default constructor.
    constexpr Matrix() : data_{}
    {
    }

    /// @brief Copy constructor.
    Matrix(const MatrixBase<T, Rows, Cols> &other) : data_{}
    {
        if (this != &other) {
            std::copy(other.data(), other.data() + (Rows * Cols), data_);
        }
    }

    /// @brief Copy constructor.
    Matrix(const MatrixBase<const T, Rows, Cols> &other) : data_{}
    {
        std::copy(other.data(), other.data() + (Rows * Cols), data_);
    }

    /// @brief Constructor from an initializer list of initializer lists.
    /// @param init Initializer list of rows, each being an initializer list of elements.
    constexpr Matrix(std::initializer_list<std::initializer_list<T>> init) : data_{}
    {
        if (init.size() != Rows) {
            throw std::out_of_range("Initializer list row count does not match matrix row count");
        }
#ifdef COLUMN_MAJOR
        std::size_t row = 0;
        for (const auto &row_init : init) {
            if (row_init.size() != Cols) {
                throw std::out_of_range("Initializer list column count does not match matrix column count");
            }
            std::size_t col = 0;
            for (const auto &value : row_init) {
                at(row, col++) = value;
            }
            ++row;
        }
#else
        std::size_t col = 0;
        for (const auto &col_init : init) {
            if (col_init.size() != Rows) {
                throw std::out_of_range("Initializer list row count does not match matrix row count");
            }
            std::size_t row = 0;
            for (const auto &value : col_init) {
                at(row++, col) = value;
            }
            ++col;
        }
#endif
    }

    constexpr const T &at(std::size_t row, std::size_t col) const override
    {
        check_bounds(row, col);
        return data_[this->indices_to_linear_index(row, col)];
    }

    constexpr T &at(std::size_t row, std::size_t col) override
    {
        check_bounds(row, col);
        return data_[this->indices_to_linear_index(row, col)];
    }

    constexpr const T &at(std::size_t index) const override
    {
        check_bounds(index);
        return data_[index];
    }

    constexpr T &at(std::size_t index) override
    {
        check_bounds(index);
        return data_[index];
    }

    constexpr const T &operator()(std::size_t row, std::size_t col) const override
    {
        check_bounds(row, col);
        return data_[this->indices_to_linear_index(row, col)];
    }

    constexpr T &operator()(std::size_t row, std::size_t col) override
    {
        check_bounds(row, col);
        return data_[this->indices_to_linear_index(row, col)];
    }

    constexpr const T &operator[](std::size_t index) const override
    {
        check_bounds(index);
        return data_[index];
    }

    constexpr T &operator[](std::size_t index) override
    {
        check_bounds(index);
        return data_[index];
    }

    constexpr const T *data() const noexcept override
    {
        return data_;
    }

    constexpr T *data() noexcept override
    {
        return data_;
    }

    /// @brief Copy assignment operator.
    /// @param other The matrix to copy from.
    /// @return Reference to the assigned matrix.
    Matrix &operator=(const MatrixBase<T, Rows, Cols> &other)
    {
        if (this != &other) {
            std::copy(other.data(), other.data() + (Rows * Cols), data_);
        }
        return *this;
    }

private:
    T data_[Rows * Cols]; ///< Internal storage in column-major order.

    std::size_t indices_to_linear_index(std::size_t row, std::size_t col) const override
    {
#ifdef COLUMN_MAJOR
        return col * Rows + row;
#else
        return row * Cols + col;
#endif
    }

    void check_bounds(std::size_t row, std::size_t col) const override
    {
        if (row >= Rows || col >= Cols) {
            throw std::out_of_range("Matrix indices out of bounds");
        }
    }

    void check_bounds(std::size_t index) const override
    {
        if (index >= Rows * Cols) {
            throw std::out_of_range("Matrix linear index out of bounds");
        }
    }
};

/// @brief Converts a Vector to a single-column or single-row Matrix based on the template parameter.
/// @tparam T The type of the vector elements.
/// @tparam N The number of elements in the vector.
/// @tparam IsColumn If true, the output is a column vector; if false, a row vector.
/// @param vec The input vector.
/// @return A Matrix with one column (if IsColumn is true) or one row (if IsColumn is false).
template <bool IsColumn, typename T, std::size_t N>
[[nodiscard]] constexpr auto to_matrix(const fsmlib::VectorBase<T, N> &vec)
{
    if constexpr (IsColumn) {
        // Create a column matrix
        fsmlib::Matrix<T, N, 1> mat;
        for (std::size_t i = 0; i < N; ++i) {
            mat[i][0] = vec[i];
        }
        return mat;
    } else {
        // Create a row matrix
        fsmlib::Matrix<T, 1, N> mat;
        for (std::size_t i = 0; i < N; ++i) {
            mat[0][i] = vec[i];
        }
        return mat;
    }
}

/// @brief Converts a Matrix with a single column to a Vector.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the matrix (size of the vector).
/// @param mat The input matrix with one column.
/// @return A Vector with size equal to the number of rows.
template <typename T, std::size_t Rows>
constexpr auto to_vector(const fsmlib::MatrixBase<T, Rows, 1> &mat)
{
    fsmlib::Vector<T, Rows> vec;
    for (std::size_t i = 0; i < Rows; ++i) {
        vec[i] = mat[i][0];
    }
    return vec;
}

/// @brief Converts a Matrix with a single row to a Vector.
/// @tparam T The type of the matrix elements.
/// @tparam Cols The number of columns in the matrix (size of the vector).
/// @param mat The input matrix with one row.
/// @return A Vector with size equal to the number of columns.
template <typename T, std::size_t Cols>
constexpr auto to_vector(const fsmlib::MatrixBase<T, 1, Cols> &mat)
{
    fsmlib::Vector<T, Cols> vec;
    for (std::size_t i = 0; i < Cols; ++i) {
        vec[i] = mat[0][i];
    }
    return vec;
}

/// @brief Horizontally stacks two matrices.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in both matrices.
/// @tparam Cols1 The number of columns in the first matrix.
/// @tparam Cols2 The number of columns in the second matrix.
/// @param A The first matrix (Rows x Cols1).
/// @param B The second matrix (Rows x Cols2).
/// @return A matrix of size Rows x (Cols1 + Cols2) containing A and B stacked horizontally.
template <typename T, std::size_t Rows, std::size_t Cols1, std::size_t Cols2>
constexpr Matrix<T, Rows, Cols1 + Cols2> hstack(const MatrixBase<T, Rows, Cols1> &A,
                                                const MatrixBase<T, Rows, Cols2> &B)
{
    Matrix<T, Rows, Cols1 + Cols2> result = {};

    // Copy the elements of A
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols1; ++j) {
            result(i, j) = A(i, j);
        }
    }

    // Copy the elements of B
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols2; ++j) {
            result(i, Cols1 + j) = B(i, j);
        }
    }

    return result;
}

/// @brief Vertically stacks two matrices.
/// @tparam T The type of the matrix elements.
/// @tparam Cols The number of columns in both matrices.
/// @tparam Rows1 The number of rows in the first matrix.
/// @tparam Rows2 The number of rows in the second matrix.
/// @param A The first matrix (Rows1 x Cols).
/// @param B The second matrix (Rows2 x Cols).
/// @return A matrix of size (Rows1 + Rows2) x Cols containing A and B stacked vertically.
template <typename T, std::size_t Rows1, std::size_t Rows2, std::size_t Cols>
constexpr Matrix<T, Rows1 + Rows2, Cols> vstack(const MatrixBase<T, Rows1, Cols> &A,
                                                const MatrixBase<T, Rows2, Cols> &B)
{
    Matrix<T, Rows1 + Rows2, Cols> result = {};

    // Copy the elements of A
    for (std::size_t i = 0; i < Rows1; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            result(i, j) = A(i, j);
        }
    }

    // Copy the elements of B
    for (std::size_t i = 0; i < Rows2; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            result(Rows1 + i, j) = B(i, j);
        }
    }

    return result;
}

/// @brief Creates a matrix filled with zeros.
/// @tparam T The type of the elements in the matrix.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @return A matrix of size Rows x Cols filled with zeros.
template <typename T, std::size_t Rows, std::size_t Cols>
constexpr fsmlib::Matrix<T, Rows, Cols> zeros()
{
    fsmlib::Matrix<T, Rows, Cols> result = {};
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            result(i, j) = static_cast<T>(0);
        }
    }
    return result;
}

/// @brief Creates a matrix filled with ones.
/// @tparam T The type of the elements in the matrix.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @return A matrix of size Rows x Cols filled with ones.
template <typename T, std::size_t Rows, std::size_t Cols>
constexpr fsmlib::Matrix<T, Rows, Cols> ones()
{
    fsmlib::Matrix<T, Rows, Cols> result = {};
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            result(i, j) = static_cast<T>(1);
        }
    }
    return result;
}

/// @brief Creates a vector filled with zeros.
/// @tparam T The type of the elements in the vector.
/// @tparam N The number of elements in the vector.
/// @return A vector of size N filled with zeros.
template <typename T, std::size_t N>
constexpr fsmlib::Vector<T, N> zeros()
{
    fsmlib::Vector<T, N> result = {};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<T>(0);
    }
    return result;
}

/// @brief Creates a vector filled with ones.
/// @tparam T The type of the elements in the vector.
/// @tparam N The number of elements in the vector.
/// @return A vector of size N filled with ones.
template <typename T, std::size_t N>
constexpr fsmlib::Vector<T, N> ones()
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Create a vector to store the ones.
    fsmlib::Vector<data_type_t, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<T>(1);
    }
    return result;
}

/// @brief Extracts a column from the matrix.
/// @tparam T The type of the matrix elements.
/// @tparam N1 The number of rows in the matrix.
/// @tparam N2 The number of columns in the matrix.
/// @param mat The input matrix.
/// @param col_index The index of the column to extract.
/// @return A vector containing the elements of the specified column.
/// @throws std::out_of_range If the column index is out of bounds.
template <typename T, std::size_t N1, std::size_t N2>
constexpr fsmlib::Vector<T, N1> column(const fsmlib::MatrixBase<T, N1, N2> &mat, std::size_t col_index)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Check if the column index is out of bounds.
    if (col_index >= N2) {
        throw std::out_of_range("column: Column index out of bounds");
    }
    // Create a vector to store the column elements.
    fsmlib::Vector<data_type_t, N1> result{};
    // Copy the elements of the column.
    for (std::size_t i = 0; i < N1; ++i) {
        result[i] = mat(i, col_index);
    }
    return result;
}

/// @brief Extracts a row from the matrix.
/// @tparam T The type of the matrix elements.
/// @tparam N1 The number of rows in the matrix.
/// @tparam N2 The number of columns in the matrix.
/// @param mat The input matrix.
/// @param row_index The index of the row to extract.
/// @return A vector containing the elements of the specified row.
/// @throws std::out_of_range If the row index is out of bounds.
template <typename T, std::size_t N1, std::size_t N2>
constexpr fsmlib::Vector<T, N2> row(const fsmlib::MatrixBase<T, N1, N2> &mat, std::size_t row_index)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Check if the row index is out of bounds.
    if (row_index >= N1) {
        throw std::out_of_range("row: Row index out of bounds");
    }
    // Create a vector to store the row elements.
    fsmlib::Vector<data_type_t, N2> result{};
    // Copy the elements of the row.
    for (std::size_t j = 0; j < N2; ++j) {
        result[j] = mat(row_index, j);
    }
    return result;
}

/// @brief Sorts the indices of a vector based on its values, either in ascending or descending order.
/// @tparam T The type of the vector elements.
/// @tparam N The number of elements in the vector.
/// @tparam Ascending A boolean flag indicating the sort order (true for ascending, false for descending).
/// @param vec The input vector whose indices are to be sorted.
/// @returns A vector of indices sorted based on the values of the input vector.
template <bool Ascending, typename T, std::size_t N>
[[nodiscard]] constexpr inline auto sort_indices(const fsmlib::VectorBase<T, N> &vec)
{
    fsmlib::Vector<std::size_t, N> indices = {};
    for (std::size_t i = 0; i < N; ++i) {
        indices[i] = i;
    }

    // Sort indices based on the values in the vector
    for (std::size_t i = 0; i < N - 1; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            bool condition = Ascending ? (vec[indices[i]] > vec[indices[j]]) : (vec[indices[i]] < vec[indices[j]]);
            if (condition) {
                std::swap(indices[i], indices[j]);
            }
        }
    }

    return indices;
}

/// @brief Reorders the elements of a vector based on the provided indices.
/// @tparam T The type of the vector elements.
/// @tparam N The number of elements in the vector.
/// @param vec The input vector to be reordered.
/// @param indices A vector of indices specifying the new order of elements.
/// @returns A reordered vector with elements arranged according to the provided indices.
/// @details This function creates a new vector with elements of the input vector
///          rearranged based on the `indices` vector.
/// @note The `indices` vector must have the same size as the input vector.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto reorder(const fsmlib::VectorBase<T, N> &vec,
                                            const fsmlib::VectorBase<std::size_t, N> &indices)
{
    fsmlib::Vector<T, N> result = {};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = vec[indices[i]];
    }
    return result;
}

/// @brief Reorders the rows or columns of a matrix based on the provided indices.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @tparam IsColumnReorder A boolean flag indicating whether to reorder columns (true) or rows (false).
/// @param mat The input matrix to be reordered.
/// @param indices A vector of indices specifying the new order of columns or rows.
/// @returns A matrix with columns or rows rearranged according to the provided indices.
/// @details This function uses a compile-time flag to determine whether to reorder rows or columns.
///          The size of the `indices` vector must match the dimension being reordered.
template <bool IsColumnReorder, typename T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr inline auto
reorder(const fsmlib::MatrixBase<T, Rows, Cols> &mat,
        const fsmlib::VectorBase<std::size_t, IsColumnReorder ? Cols : Rows> &indices)
{
    // Ensure indices size matches the dimension being reordered
    static_assert(IsColumnReorder ? (Cols > 0) : (Rows > 0), "Matrix dimensions must be non-zero.");

    fsmlib::Matrix<T, Rows, Cols> result = {};

    if constexpr (IsColumnReorder) {
        // Reorder columns
        for (std::size_t col = 0; col < Cols; ++col) {
            for (std::size_t row = 0; row < Rows; ++row) {
                result(row, col) = mat(row, indices[col]);
            }
        }
    } else {
        // Reorder rows
        for (std::size_t row = 0; row < Rows; ++row) {
            for (std::size_t col = 0; col < Cols; ++col) {
                result(row, col) = mat(indices[row], col);
            }
        }
    }

    return result;
}

/// @brief Checks if a matrix is symmetric.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the square matrix.
/// @param mat The input matrix.
/// @param tolerance The tolerance for checking symmetry.
/// @return True if the matrix is symmetric; false otherwise.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline bool is_symmetric(const fsmlib::MatrixBase<T, N, N> &mat, T tolerance = 1e-9)
{
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (std::abs(mat(i, j) - mat(j, i)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

/// @brief Creates a diagonal matrix from a vector.
/// @tparam T The type of vector elements.
/// @tparam N The size of the vector.
/// @param vec The input vector.
/// @return A diagonal matrix with the vector elements on the main diagonal.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline Matrix<T, N, N> diag(const fsmlib::VectorBase<T, N> &vec)
{
    Matrix<T, N, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i][i] = vec[i];
    }
    return result;
}

/// @brief Extracts the diagonal of a matrix as a vector.
/// @tparam T The type of matrix elements.
/// @tparam N1 The number of rows in the matrix.
/// @tparam N2 The number of columns in the matrix.
/// @param mat The input matrix.
/// @return A vector containing the diagonal elements of the matrix.
template <typename T, std::size_t N1, std::size_t N2>
[[nodiscard]] constexpr inline Vector<T, (N1 < N2 ? N1 : N2)> diag(const MatrixBase<T, N1, N2> &mat)
{
    constexpr std::size_t min_dim = (N1 < N2 ? N1 : N2);
    Vector<T, min_dim> result{};
    for (std::size_t i = 0; i < min_dim; ++i) {
        result[i] = mat[i][i];
    }
    return result;
}

} // namespace fsmlib
