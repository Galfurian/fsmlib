/// @file math.hpp
/// @brief Provides basic algebra operations.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include "fsmlib/fsmlib.hpp"

#include <cmath>

namespace fsmlib
{

namespace details
{

/// @brief Applies a binary operation element-wise between two vectors.
/// @tparam T The type of the first vector elements.
/// @tparam U The type of the second vector elements.
/// @tparam F The binary operation.
/// @tparam N The indices of the elements to operate on.
/// @param l The first vector.
/// @param r The second vector.
/// @param func The binary operation to apply.
/// @return The resulting vector after applying the operation.
template <class T, class U, class F, std::size_t... N>
[[nodiscard]] constexpr inline auto
apply_binary_helper(
    const fsmlib::Vector<T, sizeof...(N)> &l,
    const fsmlib::Vector<U, sizeof...(N)> &r,
    F func,
    std::integer_sequence<std::size_t, N...>)
{
    using ResultType = fsmlib::Vector<decltype(func(l[0], r[0])), sizeof...(N)>;
    return ResultType{ func(l[N], r[N])... };
}

/// @brief Applies a binary operation element-wise between two vectors (in-place).
/// @tparam T The type of the first vector elements.
/// @tparam U The type of the second vector elements.
/// @tparam F The binary operation.
/// @tparam N The indices of the elements to operate on.
/// @param l The first vector (modified in-place).
/// @param r The second vector.
/// @param func The binary operation to apply.
template <class T, class U, class F, std::size_t... N>
constexpr inline void
apply_binary_helper(
    fsmlib::Vector<T, sizeof...(N)> &l,
    const fsmlib::Vector<U, sizeof...(N)> &r,
    F func,
    std::integer_sequence<std::size_t, N...>)
{
    ((l[N] = func(l[N], r[N])), ...);
}

/// @brief Applies a binary operation between a vector and a scalar.
/// @tparam T The type of the vector elements.
/// @tparam U The type of the scalar.
/// @tparam F The binary operation.
/// @tparam N The indices of the elements to operate on.
/// @param l The vector.
/// @param r The scalar.
/// @param func The binary operation to apply.
/// @param s The sequence of indices.
/// @return The resulting vector after applying the operation.
template <class T, class U, class F, std::size_t... N>
[[nodiscard]] constexpr inline auto
apply_binary_helper(
    const fsmlib::Vector<T, sizeof...(N)> &l,
    const U &r,
    F func,
    std::integer_sequence<std::size_t, N...> s)
{
    using return_type_t = fsmlib::Vector<decltype(func(l[0], r)), s.size()>;
    return return_type_t{ func(l[N], r)... };
}

/// @brief Applies a binary operation between a vector and a scalar (in-place).
/// @tparam T The type of the vector elements.
/// @tparam U The type of the scalar.
/// @tparam F The binary operation.
/// @tparam N The indices of the elements to operate on.
/// @param l The vector (modified in-place).
/// @param r The scalar.
/// @param func The binary operation to apply.
template <class T, class U, class F, std::size_t... N>
constexpr inline void
apply_binary_helper(
    fsmlib::Vector<T, sizeof...(N)> &l,
    const U &r,
    F func,
    std::integer_sequence<std::size_t, N...>)
{
    ((l[N] = func(l[N], r)), ...);
}

/// @brief Applies a binary operation between a scalar and a vector.
/// @tparam T The type of the vector elements.
/// @tparam U The type of the scalar.
/// @tparam F The binary operation.
/// @tparam N The indices of the elements to operate on.
/// @param l The scalar.
/// @param r The vector.
/// @param func The binary operation to apply.
/// @param s The sequence of indices.
/// @return The resulting vector after applying the operation.
template <class T, class U, class F, std::size_t... N>
[[nodiscard]] constexpr inline auto
apply_binary_helper(
    const U &l,
    const fsmlib::Vector<T, sizeof...(N)> &r,
    F func,
    std::integer_sequence<std::size_t, N...> s)
{
    using return_type_t = fsmlib::Vector<decltype(func(l, r[0])), s.size()>;
    return return_type_t{ func(l, r[N])... };
}

/// @brief Applies a unary operation element-wise to a vector.
/// @tparam T The type of the vector elements.
/// @tparam F The unary operation.
/// @tparam N The indices of the elements to operate on.
/// @param v The input vector.
/// @param func The unary operation to apply.
/// @param s The sequence of indices.
/// @return The resulting vector after applying the operation.
template <class T, class F, std::size_t... N>
[[nodiscard]] constexpr inline auto
apply_unary_helper(
    const fsmlib::Vector<T, sizeof...(N)> &v,
    F func,
    std::integer_sequence<std::size_t, N...> s)
{
    using return_type_t = fsmlib::Vector<decltype(func(v[0])), s.size()>;
    return return_type_t{ func(v[N])... };
}

/// @brief Applies a unary operation element-wise to a vector (in-place).
/// @tparam T The type of the vector elements.
/// @tparam F The unary operation.
/// @tparam N The indices of the elements to operate on.
/// @param v The input vector (modified in-place).
/// @param func The unary operation to apply.
/// @param s The sequence of indices.
template <class T, class F, std::size_t... N>
constexpr inline void
apply_unary_helper(
    fsmlib::Vector<T, sizeof...(N)> &v,
    F func,
    std::integer_sequence<std::size_t, N...>)
{
    ((v[N] = func(v[N])), ...);
}

/// @brief Applies a unary operation to all elements of a container.
/// @tparam N The number of elements.
/// @tparam F The unary operation.
/// @tparam T The container type.
/// @param func The unary operation to apply.
/// @param arg The container to operate on.
/// @return The resulting container after applying the operation.
template <std::size_t N, class F, class T>
[[nodiscard]] constexpr inline auto
apply_unary(F func, const T &arg)
{
    return apply_unary_helper(arg, func, std::make_integer_sequence<std::size_t, N>{});
}

/// @brief Applies a unary operation to all elements of a container (in-place).
/// @tparam N The number of elements.
/// @tparam F The unary operation.
/// @tparam T The container type (modified in-place).
/// @param func The unary operation to apply.
/// @param arg The container to operate on (modified in-place).
template <std::size_t N, class F, class T>
constexpr inline void
apply_unary(F func, T &arg)
{
    apply_unary_helper(arg, func, std::make_integer_sequence<std::size_t, N>{});
}

/// @brief Applies a binary operation to all elements of two containers.
/// @tparam N The number of elements.
/// @tparam F The binary operation.
/// @tparam T1 The type of the first container.
/// @tparam T2 The type of the second container.
/// @param func The binary operation to apply.
/// @param arg1 The first container.
/// @param arg2 The second container.
/// @return The resulting container after applying the operation.
template <std::size_t N, class F, class T1, class T2>
[[nodiscard]] constexpr inline auto
apply_binary(F func, const T1 &arg1, const T2 &arg2)
{
    return apply_binary_helper(arg1, arg2, func, std::make_integer_sequence<std::size_t, N>{});
}

/// @brief Applies a binary operation to all elements of two containers (in-place).
/// @tparam N The number of elements.
/// @tparam F The binary operation.
/// @tparam T1 The type of the first container (modified in-place).
/// @tparam T2 The type of the second container.
/// @param func The binary operation to apply.
/// @param arg1 The first container (modified in-place).
/// @param arg2 The second container.
template <std::size_t N, class F, class T1, class T2>
constexpr void apply_binary(F func, T1 &arg1, const T2 &arg2)
{
    apply_binary_helper(arg1, arg2, func, std::make_integer_sequence<std::size_t, N>{});
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
[[nodiscard]] constexpr inline auto
eye(T value = 1)
{
    constexpr std::size_t cnt = N1 < N2 ? N1 : N2;
    fsmlib::Matrix<T, N1, N2> r{};
    for (std::size_t i = 0; i < cnt; ++i) {
        r[i][i] = value;
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
[[nodiscard]] constexpr inline auto
inner_product(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
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
[[nodiscard]] constexpr inline auto
dot(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
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
[[nodiscard]] constexpr inline auto
outer_product(const fsmlib::Vector<T, M> &lv, const fsmlib::Vector<U, N> &rv)
{
    fsmlib::Matrix<decltype(lv[0] * rv[0]), M, N> ret{};
    for (std::size_t r = 0; r < M; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            ret[r][c] = lv[r] * rv[c];
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
[[nodiscard]] constexpr inline auto
multiply(const Matrix<T1, N1, N2> &mat, const Vector<T2, N2> &vec)
{
    using result_type_t = std::remove_const_t<std::common_type_t<T1, T2>>;
    // Prepare the result.Z
    fsmlib::Vector<result_type_t, N1> result = {};
    for (std::size_t i = 0; i < N1; ++i) {
        for (std::size_t j = 0; j < N2; ++j) {
            result[i] += mat[i][j] * vec[j];
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
[[nodiscard]] constexpr inline auto
multiply(const fsmlib::Matrix<T1, N1, N2> &A, const fsmlib::Matrix<T2, N2, N3> &B)
{
    // Promotes types for compatibility.
    using result_type_t = std::remove_const_t<std::common_type_t<T1, T2>>;
    // Prepare the result.
    fsmlib::Matrix<result_type_t, N1, N3> result = {};
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N3; ++c) {
            result[r][c] = 0;
            for (std::size_t k = 0; k < N2; ++k) {
                result[r][c] += static_cast<result_type_t>(A[r][k]) * static_cast<result_type_t>(B[k][c]);
            }
        }
    }

    return result;
}

/// @brief Checks if a predicate is true for all elements of a container.
/// @tparam T The type of the elements in the container.
/// @tparam F The predicate function.
/// @tparam N The size of the container.
/// @param container The container to check.
/// @param pred The predicate to apply.
/// @returns True if the predicate is true for all elements, false otherwise.
template <class T, std::size_t N, class F>
[[nodiscard]] constexpr inline auto
all(const fsmlib::Vector<T, N> &container, F pred)
{
    for (std::size_t i = 0; i < N; ++i) {
        // Base case: Apply the predicate to the boolean value.
        if constexpr (std::is_same_v<T, bool>) {
            if (!pred(container[i])) {
                return false;
            }
        }
        // Recursive case: Check nested vectors.
        else {
            if (!all(container[i], pred)) {
                return false;
            }
        }
    }
    return true;
}

/// @brief Checks if all elements in a boolean vector are true.
/// @tparam N The size of the container.
/// @param container The container to check.
/// @returns True if all elements are true, false otherwise.
template <class T, std::size_t N>
[[nodiscard]] constexpr inline auto
all(const fsmlib::Vector<T, N> &container)
{
    for (std::size_t i = 0; i < N; ++i) {
        // Base case: Check if the current element is true.
        if constexpr (std::is_same_v<T, bool>) {
            if (!container[i]) {
                return false;
            }
        }
        // Recursive case: Check nested vectors.
        else {
            if (!all(container[i])) {
                return false;
            }
        }
    }
    return true;
}

/// @brief Checks if a predicate is true for any element of a container.
/// @tparam T The type of the elements in the container.
/// @tparam F The predicate function.
/// @tparam N The size of the container.
/// @param container The container to check.
/// @param pred The predicate to apply.
/// @returns True if the predicate is true for any element, false otherwise.
template <class T, std::size_t N, class F>
[[nodiscard]] constexpr inline auto
any(const fsmlib::Vector<T, N> &container, F pred)
{
    for (std::size_t i = 0; i < N; ++i) {
        // Base case: Apply the predicate to the boolean value.
        if constexpr (std::is_same_v<T, bool>) {
            if (pred(container[i])) {
                return true;
            }
        }
        // Recursive case: Check nested vectors.
        else {
            if (any(container[i], pred)) {
                return true;
            }
        }
    }
    return false;
}

/// @brief Checks if any element in a boolean vector is true.
/// @tparam N The size of the container.
/// @param container The container to check.
/// @returns True if any element is true, false otherwise.
template <class T, std::size_t N>
[[nodiscard]] constexpr inline auto
any(const fsmlib::Vector<T, N> &container)
{
    for (std::size_t i = 0; i < N; ++i) {
        // Base case: Check if the current element is true.
        if constexpr (std::is_same_v<T, bool>) {
            if (container[i]) {
                return true;
            }
        }
        // Recursive case: Check nested vectors.
        else {
            if (any(container[i])) {
                return true;
            }
        }
    }
    return false;
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
constexpr inline void
swap_rows(fsmlib::Matrix<T, N> &matrix, std::size_t i, std::size_t j, std::size_t start_column = 0, std::size_t end_column = std::numeric_limits<std::size_t>::max())
{
    end_column = std::min(end_column, N);
    for (std::size_t c = start_column; c < end_column; ++c) {
        std::swap(matrix[i][c], matrix[j][c]);
    }
}

/// @brief Computes the absolute value of each element in a vector.
/// @tparam T The type of the elements in the vector.
/// @tparam N The size of the vector.
/// @param v The input vector.
/// @returns A vector with the absolute value of each element.
template <class T, std::size_t N>
[[nodiscard]] constexpr inline auto
abs(const fsmlib::Vector<T, N> &v)
{
    if constexpr (std::is_arithmetic_v<T>) {
        return fsmlib::details::apply_unary<N>([](const T &value) { return std::abs(value); }, v);
    } else {
        return fsmlib::details::apply_unary<N>([](const T &value) { return fsmlib::abs(value); }, v);
    }
}

/// @brief Computes the square root of each element in a vector.
/// @tparam T The type of the elements in the vector.
/// @tparam N The size of the vector.
/// @param v The input vector.
/// @returns A vector with the square root of each element.
template <class T, std::size_t N>
[[nodiscard]] constexpr inline auto
sqrt(const fsmlib::Vector<T, N> &v)
{
    if constexpr (std::is_arithmetic_v<T>) {
        return fsmlib::details::apply_unary<N>([](const T &value) { return std::sqrt(value); }, v);
    } else {
        return fsmlib::details::apply_unary<N>([](const T &value) { return fsmlib::sqrt(value); }, v);
    }
}

/// @brief Computes the log of each element in a vector.
/// @tparam T The type of the elements in the vector.
/// @tparam N The size of the vector.
/// @param v The input vector.
/// @returns A vector with the log of each element.
template <class T, std::size_t N>
[[nodiscard]] constexpr inline auto
log(const fsmlib::Vector<T, N> &v)
{
    if constexpr (std::is_arithmetic_v<T>) {
        return fsmlib::details::apply_unary<N>([](const T &value) { return std::log(value); }, v);
    } else {
        return fsmlib::details::apply_unary<N>([](const T &value) { return fsmlib::log(value); }, v);
    }
}

/// @brief Computes the trace of a square matrix, i.e., the sum of the elements along the main diagonal.
/// @tparam T The type of the matrix elements.
/// @tparam Size The size of the square matrix (Size x Size).
/// @param A The input square matrix.
/// @return The sum of the diagonal elements.
template <typename T, std::size_t Size>
[[nodiscard]] constexpr inline auto
trace(const fsmlib::Matrix<T, Size, Size> &A)
{
    T result = 0;
    // Sum the diagonal elements.
    for (std::size_t i = 0; i < Size; ++i) {
        result += A[i][i];
    }
    return result;
}

template <typename E1,
          typename E2,
          typename Operation,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) || (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>)>>
[[nodiscard]] constexpr inline auto
apply_elementwise(const E1 &lhs, const E2 &rhs, Operation op)
{
    if constexpr (fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) {
        // Both are containers with fixed sizes.
        static_assert(fsmlib::fixed_size_v<E1> == fsmlib::fixed_size_v<E2>,
                      "Sizes of the containers must match for element-wise operation");
        using ResultType = decltype(op(lhs[0], rhs[0]));
        fsmlib::Vector<ResultType, fsmlib::fixed_size_v<E1>> result;

        for (std::size_t i = 0; i < lhs.size(); ++i) {
            result[i] = op(lhs[i], rhs[i]);
        }

        return result;
    } else if constexpr (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) {
        // Left-hand side is a container, and right-hand side is a scalar.
        using ResultType = decltype(op(lhs[0], rhs));
        fsmlib::Vector<ResultType, fsmlib::fixed_size_v<E1>> result;

        for (std::size_t i = 0; i < lhs.size(); ++i) {
            result[i] = op(lhs[i], rhs);
        }

        return result;
    } else if constexpr (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>) {
        // Left-hand side is a scalar, and right-hand side is a container.
        using ResultType = decltype(op(lhs, rhs[0]));
        fsmlib::Vector<ResultType, fsmlib::fixed_size_v<E2>> result;

        for (std::size_t i = 0; i < rhs.size(); ++i) {
            result[i] = op(lhs, rhs[i]);
        }

        return result;
    } else {
        static_assert(fsmlib::always_false<E1, E2>::value, "Invalid types for element-wise operation");
    }
}

template <typename E1,
          typename E2,
          typename Operation,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>)>>
[[nodiscard]] constexpr inline auto
apply_elementwise(E1 &lhs, const E2 &rhs, Operation op)
{
    if constexpr (fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) {
        // Both are containers with fixed sizes.
        static_assert(fsmlib::fixed_size_v<E1> == fsmlib::fixed_size_v<E2>,
                      "Sizes of the containers must match for element-wise operation");
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            lhs[i] = op(lhs[i], rhs[i]); // Assign the result of the operation back to lhs
        }
        return lhs;
    } else if constexpr (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) {
        // lhs is a container, and rhs is a scalar.
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            lhs[i] = op(lhs[i], rhs); // Assign the result of the operation back to lhs
        }
        return lhs;
    } else {
        static_assert(fsmlib::always_false<E1, E2>::value, "Invalid types for element-wise operation");
    }
}

} // namespace fsmlib

/// @brief Performs element-wise addition between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of the element-wise addition.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) || (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>)>>
[[nodiscard]] constexpr inline auto
operator+(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a + b; });
}

/// @brief Performs element-wise subtraction between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of the element-wise subtraction.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) || (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>)>>
[[nodiscard]] constexpr inline auto
operator-(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a - b; });
}

/// @brief Performs element-wise multiplication between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of the element-wise multiplication.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) || (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>)>>
[[nodiscard]] constexpr inline auto
operator*(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a * b; });
}

/// @brief Performs element-wise division between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of the element-wise division.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) || (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>)>>
[[nodiscard]] constexpr inline auto
operator/(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a / b; });
}

/// @brief Performs element-wise equality comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise equality comparison.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) || (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>)>>
[[nodiscard]] constexpr inline auto
operator==(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a == b; });
}

/// @brief Performs element-wise inequality comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise inequality comparison.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) || (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>)>>
[[nodiscard]] constexpr inline auto
operator!=(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a != b; });
}

/// @brief Performs element-wise less-than comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise less-than comparison.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) || (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>)>>
[[nodiscard]] constexpr inline auto
operator<(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a < b; });
}

/// @brief Performs element-wise greater-than comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise greater-than comparison.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) || (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>)>>
[[nodiscard]] constexpr inline auto
operator>(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a > b; });
}

/// @brief Performs element-wise less-than-or-equal comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise less-than-or-equal comparison.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) || (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>)>>
[[nodiscard]] constexpr inline auto
operator<=(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a <= b; });
}

/// @brief Performs element-wise greater-than-or-equal comparison between two operands.
///
/// @tparam E1 The type of the left-hand side operand.
/// @tparam E2 The type of the right-hand side operand.
/// @param lhs The left-hand side operand (container or scalar).
/// @param rhs The right-hand side operand (container or scalar).
/// @return A new fsmlib::Vector containing the result of element-wise greater-than-or-equal comparison.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) || (std::is_arithmetic_v<E1> && fsmlib::is_valid_container_v<E2>)>>
[[nodiscard]] constexpr inline auto
operator>=(const E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a >= b; });
}

/// @brief Performs element-wise addition and assignment between two operands.
///
/// @tparam E1 The type of the left-hand side operand (container).
/// @tparam E2 The type of the right-hand side operand (container or scalar).
/// @param lhs The left-hand side operand, modified in place.
/// @param rhs The right-hand side operand (container or scalar).
/// @return The modified left-hand side operand after the addition.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>)>>
constexpr inline auto
operator+=(E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a + b; });
}

/// @brief Performs element-wise subtraction and assignment between two operands.
///
/// @tparam E1 The type of the left-hand side operand (container).
/// @tparam E2 The type of the right-hand side operand (container or scalar).
/// @param lhs The left-hand side operand, modified in place.
/// @param rhs The right-hand side operand (container or scalar).
/// @return The modified left-hand side operand after the subtraction.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>)>>
constexpr inline auto
operator-=(E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a - b; });
}

/// @brief Performs element-wise multiplication and assignment between two operands.
///
/// @tparam E1 The type of the left-hand side operand (container).
/// @tparam E2 The type of the right-hand side operand (container or scalar).
/// @param lhs The left-hand side operand, modified in place.
/// @param rhs The right-hand side operand (container or scalar).
/// @return The modified left-hand side operand after the multiplication.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>)>>
constexpr inline auto
operator*=(E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a * b; });
}

/// @brief Performs element-wise division and assignment between two operands.
///
/// @tparam E1 The type of the left-hand side operand (container).
/// @tparam E2 The type of the right-hand side operand (container or scalar).
/// @param lhs The left-hand side operand, modified in place.
/// @param rhs The right-hand side operand (container or scalar).
/// @return The modified left-hand side operand after the division.
template <typename E1,
          typename E2,
          typename = std::enable_if_t<(fsmlib::is_valid_container_v<E1> && fsmlib::is_valid_container_v<E2>) || (fsmlib::is_valid_container_v<E1> && std::is_arithmetic_v<E2>)>>
constexpr inline auto
operator/=(E1 &lhs, const E2 &rhs)
{
    return fsmlib::apply_elementwise(lhs, rhs, [](const auto &a, const auto &b) { return a / b; });
}
