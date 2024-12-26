/// @file view.hpp
/// @brief Provides views of matrices and vectors.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include "fsmlib/fsmlib.hpp"

namespace fsmlib
{

#include <cstddef>
#include <stdexcept>
#include <type_traits>

/// @brief A view into a subset of a larger vector.
/// @tparam T The type of elements in the vector.
/// @tparam Size The size of the view.
template <typename T, std::size_t Size>
class VectorView : public VectorBase<T, Size> {
public:
    /// @brief Constructor to create a VectorView from a larger vector.
    ///
    /// @param data The pointer to the start of the larger vector.
    /// @param offset The offset into the larger vector where the view begins.
    /// @param total_size The total size of the larger vector.
    VectorView(T *data, std::size_t offset, std::size_t total_size)
        : data_(data), offset_(offset), total_size_(total_size)
    {
        if ((offset + Size) > total_size) {
            throw std::out_of_range("View exceeds the bounds of the underlying vector.");
        }
    }

    /// @brief Access an element by index (const version).
    /// @param index The index of the element to access.
    /// @return const reference to the element.
    constexpr const T &operator[](std::size_t index) const override
    {
        if (index >= Size) {
            throw std::out_of_range("Index out of bounds.");
        }
        return data_[offset_ + index];
    }

    /// @brief Access an element by index (non-const version).
    /// @param index The index of the element to access.
    /// @return Reference to the element.
    constexpr T &operator[](std::size_t index) override
    {
        if (index >= Size) {
            throw std::out_of_range("Index out of bounds.");
        }
        return data_[offset_ + index];
    }

    /// @brief Returns the size of the view.
    /// @return The size of the view.
    constexpr std::size_t size() const noexcept override
    {
        return Size;
    }

    /// @brief Provides access to the underlying data (const version).
    constexpr const T *data() const noexcept override
    {
        return data_;
    }

    /// @brief Provides access to the underlying data (non-const version).
    constexpr T *data() noexcept override
    {
        return data_;
    }

    /// @brief Returns an iterator to the beginning of the view.
    /// @return A pointer to the first element in the view.
    constexpr T *begin() noexcept
    {
        return data_ + offset_;
    }

    /// @brief Returns an iterator to the end of the view.
    /// @return A pointer to one past the last element in the view.
    constexpr T *end() noexcept
    {
        return data_ + offset_ + Size;
    }

    /// @brief Returns a const iterator to the beginning of the view.
    /// @return A pointer to the first element in the view.
    constexpr const T *begin() const noexcept
    {
        return data_ + offset_;
    }

    /// @brief Returns a const iterator to the end of the view.
    /// @return A pointer to one past the last element in the view.
    constexpr const T *end() const noexcept
    {
        return data_ + offset_ + Size;
    }

private:
    T *data_;                ///< Pointer to the start of the larger vector.
    std::size_t offset_;     ///< Offset into the larger vector where the view begins.
    std::size_t total_size_; ///< Offset into the larger vector where the view begins.
};

/// @brief A view into a subset of a larger matrix.
/// @tparam T The type of elements in the matrix.
/// @tparam Rows The number of rows in the view.
/// @tparam Cols The number of columns in the view.
template <typename T, std::size_t Rows, std::size_t Cols>
class MatrixView : public MatrixBase<T, Rows, Cols> {
public:
    /// @brief Constructor to create a MatrixView from a larger matrix.
    ///
    /// @param data The pointer to the start of the larger matrix (row-major order).
    /// @param row_offset The starting row offset for the view.
    /// @param col_offset The starting column offset for the view.
    /// @param total_rows The total number of rows in the larger matrix.
    /// @param total_cols The total number of columns in the larger matrix.
    MatrixView(T *data, std::size_t row_offset, std::size_t col_offset, std::size_t total_rows, std::size_t total_cols)
        : data_(data), row_offset_(row_offset), col_offset_(col_offset), total_rows_(total_rows),
          total_cols_(total_cols)
    {
        if (row_offset + Rows > total_rows || col_offset + Cols > total_cols) {
            throw std::out_of_range("View exceeds the bounds of the underlying matrix.");
        }
    }

    /// @brief Access an element by row and column (const version).
    /// @param row Row index relative to the view.
    /// @param col Column index relative to the view.
    /// @return const reference to the element.
    constexpr const T &at(std::size_t row, std::size_t col) const override
    {
        check_bounds(row, col);
        return data_[(col_offset_ + col) * total_rows_ + (row_offset_ + row)];
    }

    /// @brief Access an element by row and column (non-const version).
    /// @param row Row index relative to the view.
    /// @param col Column index relative to the view.
    /// @return Reference to the element.
    constexpr T &at(std::size_t row, std::size_t col) override
    {
        check_bounds(row, col);
        return data_[(col_offset_ + col) * total_rows_ + (row_offset_ + row)];
    }

    /// @brief Access an element by row and column (const version).
    /// @param row Row index relative to the view.
    /// @param col Column index relative to the view.
    /// @return const reference to the element.
    constexpr const T &operator()(std::size_t row, std::size_t col) const
    {
        check_bounds(row, col);
        return data_[(col_offset_ + col) * total_rows_ + (row_offset_ + row)];
    }

    /// @brief Access an element by row and column (non-const version).
    /// @param row Row index relative to the view.
    /// @param col Column index relative to the view.
    /// @return Reference to the element.
    constexpr T &operator()(std::size_t row, std::size_t col)
    {
        check_bounds(row, col);
        return data_[(col_offset_ + col) * total_rows_ + (row_offset_ + row)];
    }

    /// @brief Access an element by linear index (const version).
    /// @param index Linear index relative to the view.
    /// @return const reference to the element.
    constexpr const T &operator[](std::size_t index) const override
    {
        if (index >= Rows * Cols) {
            throw std::out_of_range("Matrix linear index out of bounds");
        }
        // Linear access directly maps the index to the data with the view offset.
        std::size_t col = index / Rows;
        std::size_t row = index % Rows;
        return data_[(col_offset_ + col) * total_rows_ + (row_offset_ + row)];
    }

    /// @brief Access an element by linear index (non-const version).
    /// @param index Linear index relative to the view.
    /// @return Reference to the element.
    constexpr T &operator[](std::size_t index) override
    {
        if (index >= Rows * Cols) {
            throw std::out_of_range("Matrix linear index out of bounds");
        }
        // Linear access directly maps the index to the data with the view offset.
        std::size_t col = index / Rows;
        std::size_t row = index % Rows;
        return data_[(col_offset_ + col) * total_rows_ + (row_offset_ + row)];
    }

    /// @brief Returns the size of the view as a pair (rows, columns).
    /// @return A pair representing the number of rows and columns in the view.
    constexpr typename std::size_t size() const noexcept override
    {
        return Rows * Cols;
    }

    /// @brief Provides access to the underlying data (const version).
    constexpr const T *data() const noexcept override
    {
        return data_;
    }

    /// @brief Provides access to the underlying data (non-const version).
    constexpr T *data() noexcept override
    {
        return data_;
    }

private:
    T *data_;                ///< Pointer to the start of the larger matrix (row-major order).
    std::size_t row_offset_; ///< Row offset for the view.
    std::size_t col_offset_; ///< Column offset for the view.
    std::size_t total_rows_; ///< Total rows in the larger matrix.
    std::size_t total_cols_; ///< Total columns in the larger matrix.

    /// @brief Checks if the given row and column indices are within bounds.
    /// @param row Row index relative to the view.
    /// @param col Column index relative to the view.
    void check_bounds(std::size_t row, std::size_t col) const
    {
        if (row >= Rows || col >= Cols) {
            throw std::out_of_range("Index out of bounds.");
        }
    }
};

/// @brief Helper function to create a VectorView from a Vector (non-const version).
/// @tparam T The type of the vector elements.
/// @tparam ViewSize The size of the view.
/// @tparam N The size of the vector.
/// @param vector The vector to create the view from.
/// @param offset The offset from which the view begins.
/// @return A VectorView object spanning the specified range of the vector.
template <typename T, std::size_t ViewSize, std::size_t N>
constexpr auto view(Vector<T, N> &vector, std::size_t offset)
{
    static_assert(ViewSize <= N, "View size exceeds the bounds of the vector");
    if (offset + ViewSize > N) {
        throw std::out_of_range("View range exceeds the bounds of the vector");
    }
    return VectorView<T, ViewSize>(vector.data(), offset, N);
}

/// @brief Helper function to create a VectorView from a Vector (const version).
/// @tparam T The type of the vector elements.
/// @tparam ViewSize The size of the view.
/// @tparam N The size of the vector.
/// @param vector The vector to create the view from.
/// @param offset The offset from which the view begins.
/// @return A VectorView object spanning the specified range of the vector.
template <typename T, std::size_t ViewSize, std::size_t N>
constexpr auto view(const Vector<T, N> &vector, std::size_t offset)
{
    static_assert(ViewSize <= N, "View size exceeds the bounds of the vector");
    if (offset + ViewSize > N) {
        throw std::out_of_range("View range exceeds the bounds of the vector");
    }
    return VectorView<const T, ViewSize>(vector.data(), offset, N);
}

/// @brief Helper function to create a MatrixView from a Matrix (non-const version).
/// @tparam T The type of the matrix elements.
/// @tparam ViewRows The number of rows in the view.
/// @tparam ViewCols The number of columns in the view.
/// @tparam Rows The total number of rows in the matrix.
/// @tparam Cols The total number of columns in the matrix.
/// @param matrix The matrix to create the view from.
/// @param row_offset The starting row offset for the view.
/// @param col_offset The starting column offset for the view.
/// @return A MatrixView object spanning the specified range of the matrix.
template <typename T, std::size_t ViewRows, std::size_t ViewCols, std::size_t Rows, std::size_t Cols>
constexpr auto view(Matrix<T, Rows, Cols> &matrix, std::size_t row_offset, std::size_t col_offset)
{
    static_assert(ViewRows <= Rows, "View rows exceed the bounds of the matrix");
    static_assert(ViewCols <= Cols, "View columns exceed the bounds of the matrix");
    if (row_offset + ViewRows > Rows || col_offset + ViewCols > Cols) {
        throw std::out_of_range("View range exceeds the bounds of the matrix");
    }
    return MatrixView<T, ViewRows, ViewCols>(matrix.data(), row_offset, col_offset, Rows, Cols);
}

/// @brief Helper function to create a MatrixView from a Matrix (const version).
/// @tparam T The type of the matrix elements.
/// @tparam ViewRows The number of rows in the view.
/// @tparam ViewCols The number of columns in the view.
/// @tparam Rows The total number of rows in the matrix.
/// @tparam Cols The total number of columns in the matrix.
/// @param matrix The matrix to create the view from.
/// @param row_offset The starting row offset for the view.
/// @param col_offset The starting column offset for the view.
/// @return A MatrixView object spanning the specified range of the matrix.
template <typename T, std::size_t ViewRows, std::size_t ViewCols, std::size_t Rows, std::size_t Cols>
constexpr auto view(const Matrix<T, Rows, Cols> &matrix, std::size_t row_offset, std::size_t col_offset)
{
    static_assert(ViewRows <= Rows, "View rows exceed the bounds of the matrix");
    static_assert(ViewCols <= Cols, "View columns exceed the bounds of the matrix");
    if (row_offset + ViewRows > Rows || col_offset + ViewCols > Cols) {
        throw std::out_of_range("View range exceeds the bounds of the matrix");
    }
    return MatrixView<const T, ViewRows, ViewCols>(matrix.data(), row_offset, col_offset, Rows, Cols);
}

} // namespace fsmlib
