/// @file math.hpp
/// @brief Provides basic algebra operations.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include <cmath>

#include "fsmlib/core.hpp"

namespace fsmlib
{

/// @brief Namespace for supporting math operations.
namespace details
{

/// @brief Applies an element-wise operation on two vectors and returns the result.
///
/// @tparam T1 The type of elements in the left-hand side vector.
/// @tparam T2 The type of elements in the right-hand side vector.
/// @tparam N The number of elements in the vectors.
/// @tparam Operation The type of the operation to apply.
/// @param lhs The left-hand side vector.
/// @param rhs The right-hand side vector.
/// @param op The binary operation to apply element-wise.
/// @return A new vector containing the result of applying the operation element-wise.
template <typename T1, typename T2, std::size_t N, typename Operation>
[[nodiscard]]
constexpr inline auto
apply_elementwise(const fsmlib::VectorBase<T1, N> &lhs, const fsmlib::VectorBase<T2, N> &rhs, Operation op)
{
    fsmlib::Vector<decltype(op(lhs[0], rhs[0])), N> result;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = op(lhs[i], rhs[i]);
    }
    return result;
}

/// @brief Applies an element-wise operation on two vectors, modifying the left-hand side in place.
///
/// @tparam T1 The type of elements in the left-hand side vector.
/// @tparam T2 The type of elements in the right-hand side vector.
/// @tparam N The number of elements in the vectors.
/// @tparam Operation The type of the operation to apply. It must accept two arguments of types `T1` and `T2`.
/// @param lhs The left-hand side vector to modify in place. The results of the operation are stored here.
/// @param rhs The right-hand side vector, whose elements are used as input for the operation. This vector is not modified.
/// @param op The binary operation to apply element-wise. It must accept arguments of types `T1` and `T2`.
template <typename T1, typename T2, std::size_t N, typename Operation>
constexpr inline void
apply_elementwise(fsmlib::VectorBase<T1, N> &lhs, const fsmlib::VectorBase<T2, N> &rhs, Operation op)
{
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] = op(lhs[i], rhs[i]);
    }
}

/// @brief Applies an element-wise operation between a vector and a scalar and returns the result.
///
/// @tparam T1 The type of elements in the vector.
/// @tparam T2 The type of the scalar (must be arithmetic).
/// @tparam N The number of elements in the vector.
/// @tparam Operation The type of the operation to apply.
/// @param lhs The vector.
/// @param rhs The scalar.
/// @param op The binary operation to apply element-wise.
/// @return A new vector containing the result of applying the operation element-wise.
template <
    typename T1,
    typename T2,
    std::size_t N,
    typename Operation,
    typename = std::enable_if_t<std::is_arithmetic_v<T2>>>
[[nodiscard]]
constexpr inline auto apply_elementwise(const fsmlib::VectorBase<T1, N> &lhs, const T2 &rhs, Operation op)
{
    fsmlib::Vector<decltype(op(lhs[0], rhs)), N> result;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = op(lhs[i], rhs);
    }
    return result;
}

/// @brief Applies an element-wise operation between a vector and a scalar, modifying the vector in place.
///
/// @tparam T1 The type of elements in the vector.
/// @tparam T2 The type of the scalar (must be arithmetic).
/// @tparam N The number of elements in the vector.
/// @tparam Operation The type of the operation to apply. It must accept arguments of types `T1` and `T2`.
/// @param lhs The vector to modify in place. The results of the operation are stored here.
/// @param rhs The scalar value used as the second operand in the operation.
/// @param op The binary operation to apply element-wise. It must accept a vector element and the scalar as arguments.
template <
    typename T1,
    typename T2,
    std::size_t N,
    typename Operation,
    typename = std::enable_if_t<std::is_arithmetic_v<T2>>>
constexpr inline void apply_elementwise(fsmlib::VectorBase<T1, N> &lhs, const T2 &rhs, Operation op)
{
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] = op(lhs[i], rhs);
    }
}

/// @brief Applies an element-wise operation between a scalar and a vector and returns the result.
///
/// @tparam T1 The type of the scalar (must be arithmetic).
/// @tparam T2 The type of elements in the vector.
/// @tparam N The number of elements in the vector.
/// @tparam Operation The type of the operation to apply.
/// @param lhs The scalar.
/// @param rhs The vector.
/// @param op The binary operation to apply element-wise.
/// @return A new vector containing the result of applying the operation element-wise.
template <
    typename T1,
    typename T2,
    std::size_t N,
    typename Operation,
    typename = std::enable_if_t<std::is_arithmetic_v<T1>>>
[[nodiscard]]
constexpr inline auto apply_elementwise(const T1 &lhs, const fsmlib::VectorBase<T2, N> &rhs, Operation op)
{
    fsmlib::Vector<decltype(op(lhs, rhs[0])), N> result;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = op(lhs, rhs[i]);
    }
    return result;
}

/// @brief Applies an element-wise operation on two matrices and returns the result.
///
/// @tparam T1 The type of elements in the left-hand side matrix.
/// @tparam T2 The type of elements in the right-hand side matrix.
/// @tparam Rows The number of rows in the matrices.
/// @tparam Cols The number of columns in the matrices.
/// @tparam Operation The type of the operation to apply.
/// @param lhs The left-hand side matrix.
/// @param rhs The right-hand side matrix.
/// @param op The binary operation to apply element-wise.
/// @return A new matrix containing the result of applying the operation element-wise.
template <typename T1, typename T2, std::size_t Rows, std::size_t Cols, typename Operation>
[[nodiscard]]
constexpr inline auto apply_elementwise(
    const fsmlib::MatrixBase<T1, Rows, Cols> &lhs,
    const fsmlib::MatrixBase<T2, Rows, Cols> &rhs,
    Operation op)
{
    fsmlib::Matrix<decltype(op(lhs.at(0, 0), rhs.at(0, 0))), Rows, Cols> result;
    for (std::size_t i = 0; i < (Rows * Cols); ++i) {
        result[i] = op(lhs[i], rhs[i]);
    }
    return result;
}

/// @brief Applies an element-wise binary operation on two matrices, modifying the left-hand side in place.
///
/// @tparam T1 The type of elements in the left-hand side matrix.
/// @tparam T2 The type of elements in the right-hand side matrix.
/// @tparam Rows The number of rows in the matrices.
/// @tparam Cols The number of columns in the matrices.
/// @tparam Operation The type of the binary operation to apply. It must be callable with two arguments of types `T1` and `T2`.
/// @param lhs The left-hand side matrix to modify in place. The results of the operation will be stored here.
/// @param rhs The right-hand side matrix, whose elements are used as input for the operation. This matrix is not modified.
/// @param op The binary operation to apply element-wise. The operation must accept two arguments of types `T1` and `T2`.
template <typename T1, typename T2, std::size_t Rows, std::size_t Cols, typename Operation>
constexpr inline void
apply_elementwise(fsmlib::MatrixBase<T1, Rows, Cols> &lhs, const fsmlib::MatrixBase<T2, Rows, Cols> &rhs, Operation op)
{
    for (std::size_t i = 0; i < (Rows * Cols); ++i) {
        lhs[i] = op(lhs[i], rhs[i]);
    }
}

/// @brief Applies an element-wise operation between a matrix and a scalar and returns the result.
///
/// @tparam T1 The type of elements in the matrix.
/// @tparam T2 The type of the scalar (must be arithmetic).
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @tparam Operation The type of the operation to apply.
/// @param lhs The matrix.
/// @param rhs The scalar.
/// @param op The binary operation to apply element-wise.
/// @return A new matrix containing the result of applying the operation element-wise.
template <
    typename T1,
    typename T2,
    std::size_t Rows,
    std::size_t Cols,
    typename Operation,
    typename = std::enable_if_t<std::is_arithmetic_v<T2>>>
[[nodiscard]]
constexpr inline auto apply_elementwise(const fsmlib::MatrixBase<T1, Rows, Cols> &lhs, const T2 &rhs, Operation op)
{
    fsmlib::Matrix<decltype(op(lhs.at(0, 0), rhs)), Rows, Cols> result;
    for (std::size_t i = 0; i < (Rows * Cols); ++i) {
        result[i] = op(lhs[i], rhs);
    }
    return result;
}

/// @brief Applies an element-wise operation between a matrix and a scalar, modifying the matrix in place.
///
/// @tparam T1 The type of elements in the matrix.
/// @tparam T2 The type of the scalar (must be arithmetic).
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @tparam Operation The type of the operation to apply. It must accept arguments of types `T1` and `T2`.
/// @param lhs The matrix to modify in place. The results of the operation are stored here.
/// @param rhs The scalar value used as the second operand in the operation.
/// @param op The binary operation to apply element-wise. It must accept a matrix element and the scalar as arguments.
template <
    typename T1,
    typename T2,
    std::size_t Rows,
    std::size_t Cols,
    typename Operation,
    typename = std::enable_if_t<std::is_arithmetic_v<T2>>>
constexpr inline void apply_elementwise(fsmlib::MatrixBase<T1, Rows, Cols> &lhs, const T2 &rhs, Operation op)
{
    for (std::size_t i = 0; i < (Rows * Cols); ++i) {
        lhs[i] = op(lhs[i], rhs);
    }
}

/// @brief Applies an element-wise operation between a scalar and a matrix and returns the result.
///
/// @tparam T1 The type of the scalar (must be arithmetic).
/// @tparam T2 The type of elements in the matrix.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @tparam Operation The type of the operation to apply.
/// @param lhs The scalar.
/// @param rhs The matrix.
/// @param op The binary operation to apply element-wise.
/// @return A new matrix containing the result of applying the operation element-wise.
template <
    typename T1,
    typename T2,
    std::size_t Rows,
    std::size_t Cols,
    typename Operation,
    typename = std::enable_if_t<std::is_arithmetic_v<T1>>>
[[nodiscard]]
constexpr inline auto apply_elementwise(const T1 &lhs, const fsmlib::MatrixBase<T2, Rows, Cols> &rhs, Operation op)
{
    fsmlib::Matrix<decltype(op(lhs, rhs.at(0, 0))), Rows, Cols> result;
    for (std::size_t i = 0; i < (Rows * Cols); ++i) {
        result[i] = op(lhs, rhs[i]);
    }
    return result;
}

/// @brief Applies an element-wise operation between a scalar and a matrix, modifying the matrix in place.
///
/// @tparam T1 The type of the scalar (must be arithmetic).
/// @tparam T2 The type of elements in the matrix.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @tparam Operation The type of the operation to apply. It must accept arguments of types `T1` and `T2`.
/// @param lhs The scalar value used as the first operand in the operation.
/// @param rhs The matrix to modify in place. The results of the operation are stored here.
/// @param op The binary operation to apply element-wise. It must accept a scalar and a matrix element as arguments.
template <
    typename T1,
    typename T2,
    std::size_t Rows,
    std::size_t Cols,
    typename Operation,
    typename = std::enable_if_t<std::is_arithmetic_v<T1>>>
constexpr inline void apply_elementwise(T1 lhs, fsmlib::MatrixBase<T2, Rows, Cols> &rhs, Operation op)
{
    for (std::size_t i = 0; i < (Rows * Cols); ++i) {
        rhs[i] = op(lhs, rhs[i]);
    }
}

/// @brief Applies a unary operation on a vector and returns the result.
///
/// @tparam T The type of elements in the vector.
/// @tparam N The number of elements in the vector.
/// @tparam Operation The type of the unary operation to apply.
/// @param vec The vector.
/// @param op The unary operation to apply element-wise.
/// @return A new vector containing the result of applying the operation element-wise.
template <typename T, std::size_t N, typename Operation>
[[nodiscard]]
constexpr inline auto apply_elementwise(const fsmlib::VectorBase<T, N> &vec, Operation op)
{
    fsmlib::Vector<decltype(op(vec[0])), N> result;
    for (std::size_t i = 0; i < vec.size(); ++i) {
        result[i] = op(vec[i]);
    }
    return result;
}

/// @brief Applies a unary operation on a vector, modifying it in place.
///
/// @tparam T The type of elements in the vector.
/// @tparam N The number of elements in the vector.
/// @tparam Operation The type of the unary operation to apply. It must accept a single argument of type `T`.
/// @param vec The vector to modify in place. The operation is applied to each element.
/// @param op The unary operation to apply element-wise.
template <typename T, std::size_t N, typename Operation>
constexpr inline void apply_elementwise(fsmlib::VectorBase<T, N> &vec, Operation op)
{
    for (std::size_t i = 0; i < vec.size(); ++i) {
        vec[i] = op(vec[i]);
    }
}

/// @brief Applies a unary operation on a matrix and returns the result.
///
/// @tparam T The type of elements in the matrix.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @tparam Operation The type of the unary operation to apply.
/// @param mat The matrix.
/// @param op The unary operation to apply element-wise.
/// @return A new matrix containing the result of applying the operation element-wise.
template <typename T, std::size_t Rows, std::size_t Cols, typename Operation>
[[nodiscard]]
constexpr inline auto apply_elementwise(const fsmlib::MatrixBase<T, Rows, Cols> &mat, Operation op)
{
    fsmlib::Matrix<decltype(op(mat.at(0, 0))), Rows, Cols> result;
    for (std::size_t i = 0; i < (Rows * Cols); ++i) {
        result[i] = op(mat[i]);
    }
    return result;
}

/// @brief Applies a unary operation on a matrix, modifying it in place.
///
/// @tparam T The type of elements in the matrix.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @tparam Operation The type of the unary operation to apply, such as a lambda or function object.
/// @param mat The matrix to modify in place. The operation is applied element-wise to this matrix.
/// @param op The unary operation to apply to each element. It must be callable with a single argument of type `T`.
template <typename T, std::size_t Rows, std::size_t Cols, typename Operation>
constexpr inline void apply_elementwise(fsmlib::MatrixBase<T, Rows, Cols> &mat, Operation op)
{
    for (std::size_t i = 0; i < (Rows * Cols); ++i) {
        mat[i] = op(mat[i]);
    }
}

/// @brief Checks if two floating-point values are approximately equal.
/// @tparam T1 The type of the first value.
/// @tparam T2 The type of the second value.
/// @param a The first value.
/// @param b The second value.
/// @param tolerance The tolerance for comparison (default: 1e-09).
/// @returns True if the values are approximately equal, false otherwise.
template <typename T1, typename T2>
inline bool approximately_equal(T1 a, T2 b, double tolerance = 1e-09)
{
    return std::fabs(a - b) <= tolerance * std::fmax(std::fabs(a), std::fabs(b));
}

/// @brief Checks if the first floating-point value is approximately less than or equal to the second.
/// @tparam T1 The type of the first value.
/// @tparam T2 The type of the second value.
/// @param a The first value.
/// @param b The second value.
/// @returns True if the first value is less than or approximately equal to the second, false otherwise.
template <typename T1, typename T2>
inline bool approximately_lesser_than_equal(T1 a, T2 b)
{
    return (a < b) || (fsmlib::details::approximately_equal(a, b));
}

/// @brief Checks if the first floating-point value is approximately greater than or equal to the second.
/// @tparam T1 The type of the first value.
/// @tparam T2 The type of the second value.
/// @param a The first value.
/// @param b The second value.
/// @returns True if the first value is greater than or approximately equal to the second, false otherwise.
template <typename T1, typename T2>
inline bool approximately_greater_than_equal(T1 a, T2 b)
{
    return (a > b) || (fsmlib::details::approximately_equal(a, b));
}

} // namespace details

/// @brief Creates an identity matrix with a given value along the diagonal.
/// @tparam T The type of the matrix elements.
/// @tparam N1 The number of rows in the matrix.
/// @tparam N2 The number of columns in the matrix (defaults to N1 for square matrices).
/// @param value The value to place on the diagonal (default is 1).
/// @returns An identity matrix with the specified diagonal value.
template <class T, std::size_t N1, std::size_t N2 = N1>
[[nodiscard]]
constexpr inline auto eye(T value = 1)
{
    constexpr std::size_t cnt = N1 < N2 ? N1 : N2;
    fsmlib::Matrix<T, N1, N2> r{};
    for (std::size_t i = 0; i < cnt; ++i) {
        r(i, i) = value;
    }
    return r;
}

/// @brief Computes the inner (dot) product of two vectors.
/// @tparam T The type of the first vector elements.
/// @tparam U The type of the second vector elements.
/// @tparam N The number of elements in the vectors.
/// @param l The first vector.
/// @param r The second vector.
/// @returns The inner product of the two vectors.
template <class T, class U, std::size_t N>
[[nodiscard]]
constexpr inline auto inner_product(const fsmlib::VectorBase<T, N> &l, const fsmlib::VectorBase<U, N> &r)
{
    decltype(l[0] * r[0]) ret{};
    for (std::size_t i = 0; i < N; ++i) {
        ret += l[i] * r[i];
    }
    return ret;
}

/// @brief Computes the dot product (inner product) of two vectors.
/// @tparam T The type of the first vector elements.
/// @tparam U The type of the second vector elements.
/// @tparam N The number of elements in the vectors.
/// @param l The first vector.
/// @param r The second vector.
/// @returns The inner product of the two vectors.
template <class T, class U, std::size_t N>
[[nodiscard]]
constexpr inline auto dot(const fsmlib::VectorBase<T, N> &l, const fsmlib::VectorBase<U, N> &r)
{
    return fsmlib::inner_product(l, r);
}

/// @brief Computes the outer product of two vectors.
/// @tparam T The type of the first vector elements.
/// @tparam U The type of the second vector elements.
/// @tparam M The number of elements in the first vector.
/// @tparam N The number of elements in the second vector.
/// @param lv The first vector.
/// @param rv The second vector.
/// @returns A matrix representing the outer product of the two vectors.
template <class T, class U, std::size_t M, std::size_t N>
[[nodiscard]]
constexpr inline auto outer_product(const fsmlib::VectorBase<T, M> &lv, const fsmlib::VectorBase<U, N> &rv)
{
    fsmlib::Matrix<decltype(lv[0] * rv[0]), M, N> ret{};
    for (std::size_t r = 0; r < M; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            ret(r, c) = lv[r] * rv[c];
        }
    }
    return ret;
}

/// @brief Multiplies a square matrix with a column vector.
/// @tparam T The type of the matrix and vector elements.
/// @tparam N1 The number of rows in the matrix.
/// @tparam N2 The number of columns in the matrix (must match the vector size).
/// @param mat The matrix to multiply.
/// @param vec The vector to multiply.
/// @returns A vector representing the result of the multiplication.
template <typename T1, typename T2, std::size_t N1, std::size_t N2>
[[nodiscard]]
constexpr inline auto multiply(const fsmlib::MatrixBase<T1, N1, N2> &mat, const fsmlib::VectorBase<T2, N2> &vec)
{
    using result_type_t                      = std::remove_const_t<std::common_type_t<T1, T2>>;
    // Prepare the result.Z
    fsmlib::Vector<result_type_t, N1> result = {};
    for (std::size_t i = 0; i < N1; ++i) {
        for (std::size_t j = 0; j < N2; ++j) {
            result[i] += mat(i, j) * vec[j];
        }
    }
    return result;
}

/// @brief Multiplies two matrices.
/// @tparam T1 The type of the first matrix elements.
/// @tparam T2 The type of the second matrix elements.
/// @tparam N1 The number of rows in the first matrix.
/// @tparam N2 The number of columns in the first matrix (and rows in the second matrix).
/// @tparam N3 The number of columns in the second matrix.
/// @param A The first matrix.
/// @param B The second matrix.
/// @returns The resulting matrix after multiplication.
template <typename T1, typename T2, std::size_t N1, std::size_t N2, std::size_t N3>
[[nodiscard]]
constexpr inline auto multiply(const fsmlib::MatrixBase<T1, N1, N2> &A, const fsmlib::MatrixBase<T2, N2, N3> &B)
{
    // Promotes types for compatibility.
    using result_type_t                          = std::remove_const_t<std::common_type_t<T1, T2>>;
    // Prepare the result.
    fsmlib::Matrix<result_type_t, N1, N3> result = {};
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N3; ++c) {
            result(r, c) = 0;
            for (std::size_t k = 0; k < N2; ++k) {
                result(r, c) += static_cast<result_type_t>(A(r, k)) * static_cast<result_type_t>(B(k, c));
            }
        }
    }

    return result;
}

/// @brief Checks if a predicate is true for any element of a vector.
/// @tparam T The type of the elements in the vector.
/// @tparam F The predicate function.
/// @tparam N The size of the vector.
/// @param container The vector to check.
/// @param pred The predicate to apply.
/// @returns True if the predicate is true for any element, false otherwise.
template <class T, std::size_t N, class F>
[[nodiscard]]
constexpr inline auto any(const fsmlib::VectorBase<T, N> &container, F pred)
{
    for (std::size_t i = 0; i < N; ++i) {
        if (pred(container[i])) {
            return true;
        }
    }
    return false;
}

/// @brief Checks if any element in a boolean vector is true.
/// @tparam T The type of the elements in the vector (must be implicitly convertible to bool).
/// @tparam N The size of the vector.
/// @param container The vector to check.
/// @returns True if any element evaluates to true, false otherwise.
template <class T, std::size_t N>
[[nodiscard]]
constexpr inline auto any(const fsmlib::VectorBase<T, N> &container)
{
    for (std::size_t i = 0; i < N; ++i) {
        if (container[i]) {
            return true;
        }
    }
    return false;
}

/// @brief Checks if a predicate is true for any element of a matrix.
/// @tparam T The type of the elements in the matrix.
/// @tparam F The predicate function.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param container The matrix to check.
/// @param pred The predicate to apply.
/// @returns True if the predicate is true for any element, false otherwise.
template <class T, std::size_t Rows, std::size_t Cols, class F>
[[nodiscard]]
constexpr inline auto any(const fsmlib::MatrixBase<T, Rows, Cols> &container, F pred)
{
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            if (pred(container(i, j))) {
                return true;
            }
        }
    }
    return false;
}

/// @brief Checks if any element in a boolean matrix is true.
/// @tparam T The type of the elements in the matrix (must be implicitly convertible to bool).
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param container The matrix to check.
/// @returns True if any element evaluates to true, false otherwise.
template <class T, std::size_t Rows, std::size_t Cols>
[[nodiscard]]
constexpr inline auto any(const fsmlib::MatrixBase<T, Rows, Cols> &container)
{
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            if (container(i, j)) {
                return true;
            }
        }
    }
    return false;
}

/// @brief Checks if a predicate is true for all elements of a vector.
/// @tparam T The type of the elements in the vector.
/// @tparam F The predicate function.
/// @tparam N The size of the vector.
/// @param container The vector to check.
/// @param pred The predicate to apply.
/// @returns True if the predicate is true for all elements, false otherwise.
template <class T, std::size_t N, class F>
[[nodiscard]]
constexpr inline auto all(const fsmlib::VectorBase<T, N> &container, F pred)
{
    for (std::size_t i = 0; i < N; ++i) {
        if (!pred(container[i])) {
            return false;
        }
    }
    return true;
}

/// @brief Checks if all elements in a boolean vector are true.
/// @tparam T The type of the elements in the vector (must be implicitly convertible to bool).
/// @tparam N The size of the vector.
/// @param container The vector to check.
/// @returns True if all elements evaluate to true, false otherwise.
template <class T, std::size_t N>
[[nodiscard]]
constexpr inline auto all(const fsmlib::VectorBase<T, N> &container)
{
    for (std::size_t i = 0; i < N; ++i) {
        if (!container[i]) {
            return false;
        }
    }
    return true;
}

/// @brief Checks if a predicate is true for all elements of a matrix.
/// @tparam T The type of the elements in the matrix.
/// @tparam F The predicate function.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param container The matrix to check.
/// @param pred The predicate to apply.
/// @returns True if the predicate is true for all elements, false otherwise.
template <class T, std::size_t Rows, std::size_t Cols, class F>
[[nodiscard]]
constexpr inline auto all(const fsmlib::MatrixBase<T, Rows, Cols> &container, F pred)
{
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            if (!pred(container(i, j))) {
                return false;
            }
        }
    }
    return true;
}

/// @brief Checks if all elements in a boolean matrix are true.
/// @tparam T The type of the elements in the matrix (must be implicitly convertible to bool).
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param container The matrix to check.
/// @returns True if all elements evaluate to true, false otherwise.
template <class T, std::size_t Rows, std::size_t Cols>
[[nodiscard]]
constexpr inline auto all(const fsmlib::MatrixBase<T, Rows, Cols> &container)
{
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            if (!container(i, j)) {
                return false;
            }
        }
    }
    return true;
}

/// @brief Swaps the values of two rows in a matrix.
/// @tparam T The type of the matrix elements.
/// @tparam N The number of columns in the matrix.
/// @param matrix The matrix to modify.
/// @param i The index of the first row.
/// @param j The index of the second row.
/// @param start_column The starting column for the swap (default: 0).
/// @param end_column The ending column for the swap (default: all columns).
template <typename T, std::size_t N>
constexpr inline void swap_rows(
    fsmlib::MatrixBase<T, N> &matrix,
    std::size_t i,
    std::size_t j,
    std::size_t start_column = 0,
    std::size_t end_column   = std::numeric_limits<std::size_t>::max())
{
    end_column = std::min(end_column, N);
    for (std::size_t c = start_column; c < end_column; ++c) {
        std::swap(matrix(i, c), matrix(j, c));
    }
}

/// @brief Computes the absolute value of each element in a vector.
/// @tparam T The type of the elements in the vector.
/// @tparam N The size of the vector.
/// @param v The input vector.
/// @returns A vector with the absolute value of each element.
template <class T, std::size_t N>
[[nodiscard]]
constexpr inline auto abs(const fsmlib::VectorBase<T, N> &v)
{
    return fsmlib::details::apply_elementwise(v, [](const auto &x) { return std::abs(x); });
}

/// @brief Computes the absolute value of each element in a matrix.
/// @tparam T The type of the elements in the matrix.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param m The input matrix.
/// @returns A matrix with the absolute value of each element.
template <class T, std::size_t Rows, std::size_t Cols>
[[nodiscard]]
constexpr inline auto abs(const fsmlib::MatrixBase<T, Rows, Cols> &m)
{
    return fsmlib::details::apply_elementwise(m, [](const auto &x) { return std::abs(x); });
}

/// @brief Computes the square root of each element in a vector.
/// @tparam T The type of the elements in the vector.
/// @tparam N The size of the vector.
/// @param v The input vector.
/// @returns A vector with the square root of each element.
template <class T, std::size_t N>
[[nodiscard]]
constexpr inline auto sqrt(const fsmlib::VectorBase<T, N> &v)
{
    return fsmlib::details::apply_elementwise(v, [](const auto &x) { return std::sqrt(x); });
}

/// @brief Computes the square root of each element in a matrix.
/// @tparam T The type of the elements in the matrix.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param m The input matrix.
/// @returns A matrix with the square root of each element.
template <class T, std::size_t Rows, std::size_t Cols>
[[nodiscard]]
constexpr inline auto sqrt(const fsmlib::MatrixBase<T, Rows, Cols> &m)
{
    return fsmlib::details::apply_elementwise(m, [](const auto &x) { return std::sqrt(x); });
}

/// @brief Computes the log of each element in a vector.
/// @tparam T The type of the elements in the vector.
/// @tparam N The size of the vector.
/// @param v The input vector.
/// @returns A vector with the log of each element.
template <class T, std::size_t N>
[[nodiscard]]
constexpr inline auto log(const fsmlib::VectorBase<T, N> &v)
{
    return fsmlib::details::apply_elementwise(v, [](const auto &x) { return std::log(x); });
}

/// @brief Computes the natural logarithm of each element in a matrix.
/// @tparam T The type of the elements in the matrix.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param m The input matrix.
/// @returns A matrix with the natural logarithm of each element.
template <class T, std::size_t Rows, std::size_t Cols>
[[nodiscard]]
constexpr inline auto log(const fsmlib::MatrixBase<T, Rows, Cols> &m)
{
    return fsmlib::details::apply_elementwise(m, [](const auto &x) { return std::log(x); });
}

/// @brief Computes the trace of a square matrix, i.e., the sum of the elements along the main diagonal.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the square matrix (N x N).
/// @param A The input square matrix.
/// @return The sum of the diagonal elements.
template <typename T, std::size_t N>
[[nodiscard]]
constexpr inline auto trace(const fsmlib::MatrixBase<T, N, N> &A)
{
    // Select the right type.
    using data_type_t  = std::remove_const_t<T>;
    // Initialize the result.
    data_type_t result = 0;
    // Sum the diagonal elements.
    for (std::size_t i = 0; i < N; ++i) {
        result += A(i, i);
    }
    return result;
}

} // namespace fsmlib

/// @brief Performs element-wise addition between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of the element-wise addition.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_combination_v<E1, E2>>>
[[nodiscard]]
constexpr inline auto operator+(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a + b; });
}

/// @brief Performs element-wise subtraction between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of the element-wise subtraction.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_combination_v<E1, E2>>>
[[nodiscard]]
constexpr inline auto operator-(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a - b; });
}

/// @brief Performs element-wise multiplication between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of the element-wise multiplication.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_combination_v<E1, E2>>>
[[nodiscard]]
constexpr inline auto operator*(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a * b; });
}

/// @brief Performs element-wise division between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of the element-wise division.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_combination_v<E1, E2>>>
[[nodiscard]]
constexpr inline auto operator/(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a / b; });
}

/// @brief Performs element-wise equality comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise equality comparison.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_combination_v<E1, E2>>>
[[nodiscard]]
constexpr inline auto operator==(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a == b; });
}

/// @brief Performs element-wise inequality comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise inequality comparison.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_combination_v<E1, E2>>>
[[nodiscard]]
constexpr inline auto operator!=(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a != b; });
}

/// @brief Performs element-wise less-than comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise less-than comparison.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_combination_v<E1, E2>>>
[[nodiscard]]
constexpr inline auto operator<(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a < b; });
}

/// @brief Performs element-wise greater-than comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise greater-than comparison.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_combination_v<E1, E2>>>
[[nodiscard]]
constexpr inline auto operator>(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a > b; });
}

/// @brief Performs element-wise less-than-or-equal comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise less-than-or-equal comparison.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_combination_v<E1, E2>>>
[[nodiscard]]
constexpr inline auto operator<=(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a <= b; });
}

/// @brief Performs element-wise greater-than-or-equal comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise greater-than-or-equal comparison.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_combination_v<E1, E2>>>
[[nodiscard]]
constexpr inline auto operator>=(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a >= b; });
}

/// @brief Performs element-wise addition and assignment between two operands.
///
/// @tparam E1 The type of the left-hand side operand (container).
/// @tparam E2 The type of the right-hand side operand (container or scalar).
/// @param lhs The left-hand side operand, modified in place.
/// @param rhs The right-hand side operand (container or scalar).
/// @return The modified left-hand side operand after the addition.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_mutable_combination_v<E1, E2>>>
constexpr inline auto operator+=(E1 &lhs, const E2 &rhs)
{
    fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a + b; });
    return lhs;
}

/// @brief Performs element-wise subtraction and assignment between two operands.
///
/// @tparam E1 The type of the left-hand side operand (container).
/// @tparam E2 The type of the right-hand side operand (container or scalar).
/// @param lhs The left-hand side operand, modified in place.
/// @param rhs The right-hand side operand (container or scalar).
/// @return The modified left-hand side operand after the subtraction.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_mutable_combination_v<E1, E2>>>
constexpr inline auto operator-=(E1 &lhs, const E2 &rhs)
{
    fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a - b; });
    return lhs;
}

/// @brief Performs element-wise multiplication and assignment between two operands.
///
/// @tparam E1 The type of the left-hand side operand (container).
/// @tparam E2 The type of the right-hand side operand (container or scalar).
/// @param lhs The left-hand side operand, modified in place.
/// @param rhs The right-hand side operand (container or scalar).
/// @return The modified left-hand side operand after the multiplication.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_mutable_combination_v<E1, E2>>>
constexpr inline auto operator*=(E1 &lhs, const E2 &rhs)
{
    fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a * b; });
    return lhs;
}

/// @brief Performs element-wise division and assignment between two operands.
///
/// @tparam E1 The type of the left-hand side operand (container).
/// @tparam E2 The type of the right-hand side operand (container or scalar).
/// @param lhs The left-hand side operand, modified in place.
/// @param rhs The right-hand side operand (container or scalar).
/// @return The modified left-hand side operand after the division.
template <typename E1, typename E2, typename = std::enable_if_t<fsmlib::traits::is_valid_mutable_combination_v<E1, E2>>>
constexpr inline auto operator/=(E1 &lhs, const E2 &rhs)
{
    fsmlib::details::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a / b; });
    return lhs;
}
