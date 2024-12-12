/// @file fsmlib.hpp
/// @brief Defines fixed-size vector and matrix types for linear algebra operations.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>

namespace fsmlib
{

/// @brief Marker class for valid containers.
class valid_container_t {};

/// @brief Alias for a fixed-size vector.
/// @tparam T The type of the vector elements.
/// @tparam N The number of elements in the vector.
template <typename T, std::size_t N>
class Vector : public valid_container_t {
private:
    T data[N]; ///< Internal storage for the vector elements.

public:
    using value_type = T;
    using size_type  = std::integral_constant<std::size_t, N>;

    /// @brief Default constructor.
    constexpr Vector()
        : data{}
    {
    }

    /// @brief Constructor from an initializer list.
    constexpr Vector(std::initializer_list<T> init)
    {
        if (init.size() != N) {
            throw std::out_of_range("Initializer list size does not match vector size");
        }
        std::copy(init.begin(), init.end(), data);
    }

    /// @brief Access an element by index (const version).
    constexpr const T &operator[](std::size_t index) const
    {
        return data[index];
    }

    /// @brief Access an element by index (non-const version).
    constexpr T &operator[](std::size_t index)
    {
        return data[index];
    }

    /// @brief Returns the size of the vector.
    constexpr std::size_t size() const noexcept
    {
        return N;
    }

    /// @brief Begin iterator.
    constexpr T *begin() noexcept
    {
        return data;
    }

    /// @brief End iterator.
    constexpr T *end() noexcept
    {
        return data + N;
    }

    /// @brief Const begin iterator.
    constexpr const T *begin() const noexcept
    {
        return data;
    }

    /// @brief Const end iterator.
    constexpr const T *end() const noexcept
    {
        return data + N;
    }
};

/// @brief Alias for a fixed-size matrix.
/// @tparam T The type of the matrix elements.
/// @tparam N1 The number of rows in the matrix.
/// @tparam N2 The number of columns in the matrix (default is N1).
template <class T, std::size_t N1, std::size_t N2 = N1>
using Matrix = Vector<Vector<T, N2>, N1>;

/// @brief Trait to check if a type is a valid container.
/// @tparam T The type to check.
template <typename T>
struct is_valid_container : std::is_base_of<valid_container_t, T> {};

/// @brief Helper variable template for is_valid_container.
/// @tparam T The type to check.
template <typename T>
inline constexpr bool is_valid_container_v = is_valid_container<T>::value;

/// @brief Trait to determine the fixed size of a container.
/// @tparam Vec The container type.
template <typename Vec, typename = void>
struct fixed_size;

/// @brief Specialization of fixed_size for containers with a size_type.
/// @tparam Vec The container type.
template <typename Vec>
struct fixed_size<Vec, std::void_t<typename Vec::size_type>> {
    /// @brief The fixed size of the container.
    static constexpr std::size_t value = Vec::size_type::value;
};

/// @brief Helper variable template for fixed_size.
/// @tparam Vec The container type.
template <typename Vec>
inline constexpr std::size_t fixed_size_v = fixed_size<Vec>::value;

/// @brief Trait to trigger a static assertion for invalid types.
/// @tparam ... Args The types to evaluate.
template <typename...>
struct always_false : std::false_type {};

/// @brief Helper variable template for always_false.
/// @tparam ... Args The types to evaluate.
template <typename... Args>
inline constexpr bool always_false_v = always_false<Args...>::value;

/// @brief A non-owning view of a contiguous block of memory with a specified range.
/// @tparam T The type of the elements.
/// @tparam M The number of elements in the view.
template <typename T, std::size_t M, bool IsConst = false>
class View : public valid_container_t {
private:
    using PointerType = std::conditional_t<IsConst, const T *, T *>;
    PointerType data;   ///< Pointer to the start of the underlying memory.
    std::size_t offset; ///< Offset from the start of the underlying memory.

public:
    using value_type = T;
    using size_type  = std::integral_constant<std::size_t, M>;

    /// @brief Constructor for creating a view from a Vector.
    /// @param vector The Vector from which the view is created.
    /// @param offset The offset within the Vector where the view starts (default is 0).
    constexpr View(PointerType data, std::size_t offset = 0)
        : data(data), offset(offset)
    {
        // Nothing to do.
    }

    /// @brief Access an element by index (const version).
    /// @param index The index within the view.
    /// @return Reference to the element at the specified index.
    constexpr const T &operator[](std::size_t index) const
    {
        if (index >= M) {
            throw std::out_of_range("Index out of range for the view");
        }
        return data[offset + index];
    }

    /// @brief Access an element by index (non-const version).
    /// @param index The index within the view.
    /// @return Reference to the element at the specified index.
    template <bool Enable = !IsConst, typename = std::enable_if_t<Enable>>
    constexpr T &operator[](std::size_t index)
    {
        if (index >= M) {
            throw std::out_of_range("Index out of range for the view");
        }
        return data[offset + index];
    }

    /// @brief Returns the size of the view.
    /// @return The number of elements in the view.
    constexpr std::size_t size() const noexcept
    {
        return M;
    }

    /// @brief Begin iterator.
    /// @return Pointer to the first element in the view.
    template <bool Enable = !IsConst, typename = std::enable_if_t<Enable>>
    constexpr T *begin() noexcept
    {
        return data + offset;
    }

    /// @brief End iterator.
    /// @return Pointer to one past the last element in the view.
    template <bool Enable = !IsConst, typename = std::enable_if_t<Enable>>
    constexpr T *end() noexcept
    {
        return data + offset + M;
    }

    /// @brief Const begin iterator.
    /// @return Pointer to the first element in the view.
    constexpr const T *begin() const noexcept
    {
        return data + offset;
    }

    /// @brief Const end iterator.
    /// @return Pointer to one past the last element in the view.
    constexpr const T *end() const noexcept
    {
        return data + offset + M;
    }
};

/// @brief A non-owning view of a subpart of a matrix.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the view.
/// @tparam Cols The number of columns in the view.
template <typename T, std::size_t Rows, std::size_t Cols, bool IsConst = false>
class MatrixView : public valid_container_t {
private:
    using PointerType = std::conditional_t<IsConst, const T *, T *>;
    PointerType data;       ///< Pointer to the start of the underlying memory.
    std::size_t stride;     ///< Number of elements per row in the original matrix.
    std::size_t row_offset; ///< Starting row offset in the matrix.
    std::size_t col_offset; ///< Starting column offset in the matrix.

public:
    using value_type = T;
    using size_type  = std::pair<std::integral_constant<std::size_t, Rows>, std::integral_constant<std::size_t, Cols>>;

    /// @brief Constructor for creating a matrix view.
    /// @param data Pointer to the start of the original matrix data.
    /// @param stride The number of elements per row in the original matrix.
    /// @param row_offset The starting row offset.
    /// @param col_offset The starting column offset.
    constexpr MatrixView(PointerType data, std::size_t stride, std::size_t row_offset = 0, std::size_t col_offset = 0)
        : data(data), stride(stride), row_offset(row_offset), col_offset(col_offset)
    {
        // Nothing to do.
    }

    /// @brief Access a row in the view.
    /// @param row The row index within the view.
    /// @return A `View` object representing the row.
    constexpr auto operator[](std::size_t row) const
    {
        if (row >= Rows) {
            throw std::out_of_range("Row index out of range for MatrixView");
        }
        return View<T, Cols, true>(data + (row_offset + row) * stride + col_offset);
    }

    /// @brief Access an element by index (non-const version).
    /// @param index The index within the view.
    /// @return Reference to the element at the specified index.
    template <bool Enable = !IsConst, typename = std::enable_if_t<Enable>>
    constexpr auto operator[](std::size_t row)
    {
        if (row >= Rows) {
            throw std::out_of_range("Row index out of range for MatrixView");
        }
        return View<T, Cols, false>(data + (row_offset + row) * stride + col_offset);
    }

    /// @brief Returns the size of the view.
    /// @return The number of elements in the view.
    constexpr std::size_t size() const noexcept
    {
        return Rows;
    }
};

/// @brief Helper function to create a view from a Vector.
/// @tparam T The type of the vector elements.
/// @tparam M The size of the view.
/// @tparam N The size of the vector.
/// @param vector The vector to create the view from.
/// @param offset The starting offset for the view.
/// @return A View object spanning the specified range of the vector.
template <typename T, std::size_t M, std::size_t N>
constexpr auto view(Vector<T, N> &vector, std::size_t offset = 0)
{
    if ((offset + M) > N) {
        throw std::out_of_range("View range exceeds the bounds of the vector");
    }
    return View<T, M, false>(vector.begin(), offset);
}

/// @brief Helper function to create a view from a Vector.
/// @tparam T The type of the vector elements.
/// @tparam M The size of the view.
/// @tparam N The size of the vector.
/// @param vector The vector to create the view from.
/// @param offset The starting offset for the view.
/// @return A View object spanning the specified range of the vector.
template <typename T, std::size_t M, std::size_t N>
constexpr auto view(const Vector<T, N> &vector, std::size_t offset = 0)
{
    if ((offset + M) > N) {
        throw std::out_of_range("View range exceeds the bounds of the vector");
    }
    return View<T, M, true>(vector.begin(), offset);
}

/// @brief Helper function to create a view from a Matrix (non-const version).
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the view.
/// @tparam Cols The number of columns in the view.
/// @tparam MatrixRows The total number of rows in the matrix.
/// @tparam MatrixCols The total number of columns in the matrix.
/// @param matrix The matrix to create the view from.
/// @param row_offset The starting row offset for the view.
/// @param col_offset The starting column offset for the view.
/// @return A MatrixView object spanning the specified range of the matrix.
template <typename T, std::size_t Rows, std::size_t Cols, std::size_t MatrixRows, std::size_t MatrixCols>
constexpr auto view(Matrix<T, MatrixRows, MatrixCols> &matrix, std::size_t row_offset = 0, std::size_t col_offset = 0)
{
    if ((row_offset + Rows) > MatrixRows || (col_offset + Cols) > MatrixCols) {
        throw std::out_of_range("MatrixView range exceeds the bounds of the matrix");
    }
    return MatrixView<T, Rows, Cols, false>(matrix[0].begin(), MatrixCols, row_offset, col_offset);
}

/// @brief Helper function to create a view from a Matrix (const version).
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the view.
/// @tparam Cols The number of columns in the view.
/// @tparam MatrixRows The total number of rows in the matrix.
/// @tparam MatrixCols The total number of columns in the matrix.
/// @param matrix The matrix to create the view from.
/// @param row_offset The starting row offset for the view.
/// @param col_offset The starting column offset for the view.
/// @return A MatrixView object spanning the specified range of the matrix.
template <typename T, std::size_t Rows, std::size_t Cols, std::size_t MatrixRows, std::size_t MatrixCols>
constexpr auto view(const Matrix<T, MatrixRows, MatrixCols> &matrix, std::size_t row_offset = 0, std::size_t col_offset = 0)
{
    if ((row_offset + Rows) > MatrixRows || (col_offset + Cols) > MatrixCols) {
        throw std::out_of_range("MatrixView range exceeds the bounds of the matrix");
    }
    return MatrixView<T, Rows, Cols, true>(matrix[0].begin(), MatrixCols, row_offset, col_offset);
}

/// @brief Converts a Vector to a single-column or single-row Matrix based on the template parameter.
/// @tparam T The type of the vector elements.
/// @tparam N The number of elements in the vector.
/// @tparam IsColumn If true, the output is a column vector; if false, a row vector.
/// @param vec The input vector.
/// @return A Matrix with one column (if IsColumn is true) or one row (if IsColumn is false).
template <bool IsColumn, typename T, std::size_t N>
constexpr auto to_matrix(const fsmlib::Vector<T, N> &vec)
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
constexpr auto to_vector(const fsmlib::Matrix<T, Rows, 1> &mat)
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
constexpr auto to_vector(const fsmlib::Matrix<T, 1, Cols> &mat)
{
    fsmlib::Vector<T, Cols> vec;
    for (std::size_t i = 0; i < Cols; ++i) {
        vec[i] = mat[0][i];
    }
    return vec;
}

} // namespace fsmlib
