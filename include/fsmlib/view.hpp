/// @file view.hpp
/// @brief Provides views of matrices and vectors.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include <type_traits>
#include <stdexcept>
#include <cstddef>

#include "fsmlib/fsmlib.hpp"

namespace fsmlib
{

/// @brief A view into a subset of a larger vector.
/// @tparam T The type of elements in the vector.
/// @tparam N The size of the view.
template <typename T, std::size_t N>
class VectorView : public VectorBase<T, N> {
public:
    /// @brief Constructor to create a VectorView from a larger vector.
    ///
    /// @param data The pointer to the start of the larger vector.
    /// @param offset The offset into the larger vector where the view begins.
    /// @param total_size The total size of the larger vector.
    VectorView(T *data, std::size_t offset, std::size_t total_size)
        : data_(data), offset_(offset), total_size_(total_size)
    {
        if ((offset + N) > total_size) {
            throw std::out_of_range("View exceeds the bounds of the underlying vector.");
        }
    }

    constexpr const T &at(std::size_t index) const override
    {
        check_bounds(index);
        return data_[offset_ + index];
    }

    constexpr T &at(std::size_t index) override
    {
        check_bounds(index);
        return data_[offset_ + index];
    }

    constexpr const T &operator[](std::size_t index) const override
    {
        check_bounds(index);
        return data_[offset_ + index];
    }

    constexpr T &operator[](std::size_t index) override
    {
        check_bounds(index);
        return data_[offset_ + index];
    }

    constexpr std::size_t size() const noexcept override
    {
        return N;
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
        return data_ + offset_;
    }

    constexpr T *end() noexcept override
    {
        return data_ + offset_ + N;
    }

    constexpr const T *begin() const noexcept override
    {
        return data_ + offset_;
    }

    constexpr const T *end() const noexcept override
    {
        return data_ + offset_ + N;
    }

    /// @brief Copy assignment operator.
    /// @param other The vector view to copy from.
    /// @return A reference to the current vector view.
    VectorView &operator=(const VectorBase<T, N> &other)
    {
        if (this != &other) {
            for (std::size_t i = 0; i < N; ++i) {
                at(i) = other.at(i);
            }
        }
        return *this;
    }

private:
    T *data_;                ///< Pointer to the start of the larger vector.
    std::size_t offset_;     ///< Offset into the larger vector where the view begins.
    std::size_t total_size_; ///< Offset into the larger vector where the view begins.

    void check_bounds(std::size_t index) const override
    {
        if (index >= N) {
            throw std::out_of_range("Vector index out of bounds");
        }
    }
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

    constexpr const T &at(std::size_t row, std::size_t col) const override
    {
        check_bounds(row, col);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr T &at(std::size_t row, std::size_t col) override
    {
        check_bounds(row, col);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr const T &at(std::size_t index) const override
    {
        check_bounds(index);
        auto [row, col] = map_indices(index);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr T &at(std::size_t index) override
    {
        check_bounds(index);
        auto [row, col] = map_indices(index);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr const T &operator()(std::size_t row, std::size_t col) const override
    {
        check_bounds(row, col);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr T &operator()(std::size_t row, std::size_t col) override
    {
        check_bounds(row, col);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr const T &operator[](std::size_t index) const override
    {
        check_bounds(index);
        auto [row, col] = map_indices(index);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr T &operator[](std::size_t index) override
    {
        check_bounds(index);
        auto [row, col] = map_indices(index);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr const T *data() const noexcept override
    {
        return data_;
    }

    constexpr T *data() noexcept override
    {
        return data_;
    }

    /// @brief Begin iterator (non-const version).
    /// @param other The matrix view to copy from.
    /// @return Reference to the assigned matrix view.
    MatrixView &operator=(const MatrixBase<T, Rows, Cols> &other)
    {
        if (this != &other) {
            // Copy elements from the other matrix to this view.
            for (std::size_t row = 0; row < Rows; ++row) {
                for (std::size_t col = 0; col < Cols; ++col) {
                    at(row, col) = other.at(row, col);
                }
            }
        }
        return *this;
    }

private:
    T *data_;                ///< Pointer to the start of the larger matrix (row-major order).
    std::size_t row_offset_; ///< Row offset for the view.
    std::size_t col_offset_; ///< Column offset for the view.
    std::size_t total_rows_; ///< Total rows in the larger matrix.
    std::size_t total_cols_; ///< Total columns in the larger matrix.

    /// @brief Maps a linear index to row and column indices in the view.
    /// @param index Linear index relative to the view.
    /// @return A pair of indices {row, column}.
    constexpr std::pair<std::size_t, std::size_t> map_indices(std::size_t index) const
    {
        if (index >= Rows * Cols) {
            throw std::out_of_range("Matrix linear index out of bounds");
        }
#ifdef COLUMN_MAJOR
        std::size_t col = index / Rows;
        std::size_t row = index % Rows;
#else
        std::size_t row = index / Cols;
        std::size_t col = index % Cols;
#endif
        return { row, col };
    }

    std::size_t indices_to_linear_index(std::size_t row, std::size_t col) const override
    {
#ifdef COLUMN_MAJOR
        return (col_offset_ + col) * total_rows_ + (row_offset_ + row);
#else
        return (row_offset_ + row) * total_cols_ + (col_offset_ + col);
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

/// @brief A view into a cofactor (submatrix) of a larger matrix.
/// @tparam T The type of elements in the matrix.
/// @tparam Rows The number of rows in the original matrix.
/// @tparam Cols The number of columns in the original matrix.
template <typename T, std::size_t Rows, std::size_t Cols>
class CofactorView : public MatrixBase<T, Rows - 1, Cols - 1> {
public:
    /// @brief Constructor to create a CofactorView from a larger matrix.
    ///
    /// @param data The pointer to the start of the larger matrix (row-major order).
    /// @param row_to_skip The row to exclude from the view.
    /// @param col_to_skip The column to exclude from the view.
    /// @param total_rows The total number of rows in the larger matrix.
    /// @param total_cols The total number of columns in the larger matrix.
    CofactorView(T *data,
                 std::size_t row_to_skip,
                 std::size_t col_to_skip,
                 std::size_t total_rows,
                 std::size_t total_cols)
        : data_(data), row_to_skip_(row_to_skip), col_to_skip_(col_to_skip), total_rows_(total_rows),
          total_cols_(total_cols)
    {
        if (row_to_skip >= Rows || col_to_skip >= Cols) {
            throw std::out_of_range("CofactorView row or column to skip is out of bounds.");
        }
    }

    /// @brief Change the row to be skipped.
    /// @param new_row_to_skip The new row index to exclude from the view.
    void set_row_to_skip(std::size_t new_row_to_skip)
    {
        if (new_row_to_skip >= Rows) {
            throw std::out_of_range("Row to skip is out of range.");
        }
        row_to_skip_ = new_row_to_skip;
    }

    /// @brief Change the column to be skipped.
    /// @param new_col_to_skip The new column index to exclude from the view.
    void set_col_to_skip(std::size_t new_col_to_skip)
    {
        if (new_col_to_skip >= Cols) {
            throw std::out_of_range("Column to skip is out of range.");
        }
        col_to_skip_ = new_col_to_skip;
    }

    constexpr const T &at(std::size_t row, std::size_t col) const override
    {
        check_bounds(row, col);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr T &at(std::size_t row, std::size_t col) override
    {
        check_bounds(row, col);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr const T &at(std::size_t index) const override
    {
        check_bounds(index);
        auto [row, col] = map_indices(index);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr T &at(std::size_t index) override
    {
        check_bounds(index);
        auto [row, col] = map_indices(index);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr const T &operator()(std::size_t row, std::size_t col) const override
    {
        check_bounds(row, col);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr T &operator()(std::size_t row, std::size_t col) override
    {
        check_bounds(row, col);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr const T &operator[](std::size_t index) const override
    {
        check_bounds(index);
        auto [row, col] = map_indices(index);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr T &operator[](std::size_t index) override
    {
        check_bounds(index);
        auto [row, col] = map_indices(index);
        return data_[indices_to_linear_index(row, col)];
    }

    constexpr const T *data() const noexcept override
    {
        return data_;
    }

    constexpr T *data() noexcept override
    {
        return data_;
    }

private:
    T *data_;                 ///< Pointer to the start of the larger matrix (row-major order).
    std::size_t row_to_skip_; ///< Row to exclude from the view.
    std::size_t col_to_skip_; ///< Column to exclude from the view.
    std::size_t total_rows_;  ///< Total rows in the larger matrix.
    std::size_t total_cols_;  ///< Total columns in the larger matrix.

    /// @brief Maps a linear index to row and column indices in the view.
    /// @param index Linear index relative to the view.
    /// @return A pair of indices {row, column}.
    constexpr std::pair<std::size_t, std::size_t> map_indices(std::size_t index) const
    {
#ifdef COLUMN_MAJOR
        std::size_t col = index / (Rows - 1);
        std::size_t row = index % (Rows - 1);
#else
        std::size_t row = index / (Cols - 1);
        std::size_t col = index % (Cols - 1);
#endif
        // Adjust indices to account for the skipped row and column
        if (row >= row_to_skip_) {
            row += 1;
        }
        if (col >= col_to_skip_) {
            col += 1;
        }
        return { row, col };
    }

    std::size_t indices_to_linear_index(std::size_t row, std::size_t col) const override
    {
#ifdef COLUMN_MAJOR
        // Column-major order: index increases by rows first
        return (col >= col_to_skip_ ? col + 1 : col) * total_rows_ + (row >= row_to_skip_ ? row + 1 : row);
#else
        // Row-major order: index increases by columns first
        return (row >= row_to_skip_ ? row + 1 : row) * total_cols_ + (col >= col_to_skip_ ? col + 1 : col);
#endif
    }

    void check_bounds(std::size_t row, std::size_t col) const override
    {
        if (row >= Rows - 1 || col >= Cols - 1) {
            throw std::out_of_range("CofactorView indices out of bounds.");
        }
    }

    void check_bounds(std::size_t index) const override
    {
        if (index >= (Rows - 1) * (Cols - 1)) {
            throw std::out_of_range("Matrix linear index out of bounds");
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
constexpr auto view(VectorBase<T, N> &vector, std::size_t offset)
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
constexpr auto view(const VectorBase<T, N> &vector, std::size_t offset)
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
constexpr auto view(MatrixBase<T, Rows, Cols> &matrix, std::size_t row_offset, std::size_t col_offset)
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
constexpr auto view(const MatrixBase<T, Rows, Cols> &matrix, std::size_t row_offset, std::size_t col_offset)
{
    static_assert(ViewRows <= Rows, "View rows exceed the bounds of the matrix");
    static_assert(ViewCols <= Cols, "View columns exceed the bounds of the matrix");
    if (row_offset + ViewRows > Rows || col_offset + ViewCols > Cols) {
        throw std::out_of_range("View range exceeds the bounds of the matrix");
    }
    return MatrixView<const T, ViewRows, ViewCols>(matrix.data(), row_offset, col_offset, Rows, Cols);
}

/// @brief Helper function to create a CofactorView from a Matrix (non-const version).
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the original matrix.
/// @tparam Cols The number of columns in the original matrix.
/// @param matrix The matrix to create the cofactor view from.
/// @param row_to_skip The row index to remove.
/// @param col_to_skip The column index to remove.
/// @return A CofactorView object spanning the specified range of the matrix.
template <typename T, std::size_t Rows, std::size_t Cols>
constexpr auto cofactor_view(MatrixBase<T, Rows, Cols> &matrix, std::size_t row_to_skip, std::size_t col_to_skip)
{
    if (row_to_skip >= Rows || col_to_skip >= Cols) {
        throw std::out_of_range("Row or column index out of range for CofactorView.");
    }
    return CofactorView<T, Rows, Cols>(matrix.data(), row_to_skip, col_to_skip, Rows, Cols);
}

/// @brief Helper function to create a CofactorView from a Matrix (non-const version).
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the original matrix.
/// @tparam Cols The number of columns in the original matrix.
/// @param matrix The matrix to create the cofactor view from.
/// @param row_to_skip The row index to remove.
/// @param col_to_skip The column index to remove.
/// @return A CofactorView object spanning the specified range of the matrix.
template <typename T, std::size_t Rows, std::size_t Cols>
constexpr auto cofactor_view(const MatrixBase<T, Rows, Cols> &matrix, std::size_t row_to_skip, std::size_t col_to_skip)
{
    if (row_to_skip >= Rows || col_to_skip >= Cols) {
        throw std::out_of_range("Row or column index out of range for CofactorView.");
    }
    return CofactorView<const T, Rows, Cols>(matrix.data(), row_to_skip, col_to_skip, Rows, Cols);
}

} // namespace fsmlib
