/// @file math.hpp
/// @brief Provides basic algebra operations.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include "fsmlib/fsmlib.hpp"

#include <cmath>

/// @brief Vector addition.
/// @param l The left-hand side vector.
/// @param r The right-hand side vector.
/// @return The result of element-wise addition.
template <class T, class U, std::size_t N>
constexpr auto operator+(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r);

/// @brief Scalar addition with a vector.
/// @param l The scalar value.
/// @param r The vector.
/// @return The result of adding the scalar to each element of the vector.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator+(const U &l, const fsmlib::Vector<T, N> &r);

/// @brief Vector addition with a scalar.
/// @param l The vector.
/// @param r The scalar value.
/// @return The result of adding the scalar to each element of the vector.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator+(const fsmlib::Vector<T, N> &l, const U &r);

/// @brief Vector subtraction.
/// @param l The left-hand side vector.
/// @param r The right-hand side vector.
/// @return The result of element-wise subtraction.
template <class T, class U, std::size_t N>
constexpr auto operator-(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r);

/// @brief Scalar subtraction with a vector.
/// @param l The scalar value.
/// @param r The vector.
/// @return The result of subtracting each element of the vector from the scalar.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator-(const U &l, const fsmlib::Vector<T, N> &r);

/// @brief Vector subtraction with a scalar.
/// @param l The vector.
/// @param r The scalar value.
/// @return The result of subtracting the scalar from each element of the vector.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator-(const fsmlib::Vector<T, N> &l, const U &r);

/// @brief Vector multiplication.
/// @param l The left-hand side vector.
/// @param r The right-hand side vector.
/// @return The result of element-wise multiplication.
template <class T, class U, std::size_t N>
constexpr auto operator*(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r);

/// @brief Scalar multiplication with a vector.
/// @param l The scalar value.
/// @param r The vector.
/// @return The result of multiplying the scalar with each element of the vector.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator*(const U &l, const fsmlib::Vector<T, N> &r);

/// @brief Vector multiplication with a scalar.
/// @param l The vector.
/// @param r The scalar value.
/// @return The result of multiplying the scalar with each element of the vector.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator*(const fsmlib::Vector<T, N> &l, const U &r);

/// @brief Vector division.
/// @param l The left-hand side vector.
/// @param r The right-hand side vector.
/// @return The result of element-wise division.
template <class T, class U, std::size_t N>
constexpr auto operator/(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r);

/// @brief Scalar division with a vector.
/// @param l The scalar value.
/// @param r The vector.
/// @return The result of dividing the scalar by each element of the vector.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator/(const U &l, const fsmlib::Vector<T, N> &r);

/// @brief Vector division with a scalar.
/// @param l The vector.
/// @param r The scalar value.
/// @return The result of dividing each element of the vector by the scalar.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator/(const fsmlib::Vector<T, N> &l, const U &r);

/// @brief Equality comparison for vectors.
/// @param l The left-hand side vector.
/// @param r The right-hand side vector.
/// @return True if all elements of the vectors are equal, false otherwise.
template <class T, class U, std::size_t N>
constexpr auto operator==(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r);

/// @brief Scalar equality comparison with a vector.
/// @param l The scalar value.
/// @param r The vector.
/// @return True if the scalar equals all elements of the vector, false otherwise.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator==(const U &l, const fsmlib::Vector<T, N> &r);

/// @brief Vector equality comparison with a scalar.
/// @param l The vector.
/// @param r The scalar value.
/// @return True if the scalar equals all elements of the vector, false otherwise.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator==(const fsmlib::Vector<T, N> &l, const U &r);

/// @brief Inequality comparison for vectors.
/// @param l The left-hand side vector.
/// @param r The right-hand side vector.
/// @return True if at least an element is different, false otherwise.
template <class T, class U, std::size_t N>
constexpr auto operator!=(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r);

/// @brief Scalar inequality comparison with a vector.
/// @param l The scalar value.
/// @param r The vector.
/// @return True if the scalar is al least different from one element of the vector, false otherwise.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator!=(const U &l, const fsmlib::Vector<T, N> &r);

/// @brief Vector inequality comparison with a scalar.
/// @param l The vector.
/// @param r The scalar value.
/// @return True if the scalar is al least different from one element of the vector, false otherwise.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator!=(const fsmlib::Vector<T, N> &l, const U &r);

/// @brief Element-wise addition assignment (Vector += Vector).
/// @tparam T The type of elements in the left-hand vector.
/// @tparam U The type of elements in the right-hand vector.
/// @tparam N The size of the vectors.
/// @param l The left-hand vector (modified in-place).
/// @param r The right-hand vector.
/// @return The modified left-hand vector.
template <class T, class U, std::size_t N>
constexpr auto operator+=(fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r);

/// @brief Element-wise subtraction assignment (Vector -= Vector).
/// @tparam T The type of elements in the left-hand vector.
/// @tparam U The type of elements in the right-hand vector.
/// @tparam N The size of the vectors.
/// @param l The left-hand vector (modified in-place).
/// @param r The right-hand vector.
/// @return The modified left-hand vector.
template <class T, class U, std::size_t N>
constexpr auto operator-=(fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r);

/// @brief Element-wise multiplication assignment (Vector *= Vector).
/// @tparam T The type of elements in the left-hand vector.
/// @tparam U The type of elements in the right-hand vector.
/// @tparam N The size of the vectors.
/// @param l The left-hand vector (modified in-place).
/// @param r The right-hand vector.
/// @return The modified left-hand vector.
template <class T, class U, std::size_t N>
constexpr auto operator*=(fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r);

/// @brief Element-wise division assignment (Vector /= Vector).
/// @tparam T The type of elements in the left-hand vector.
/// @tparam U The type of elements in the right-hand vector.
/// @tparam N The size of the vectors.
/// @param l The left-hand vector (modified in-place).
/// @param r The right-hand vector.
/// @return The modified left-hand vector.
template <class T, class U, std::size_t N>
constexpr auto operator/=(fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r);

/// @brief Scalar addition assignment (Vector += Scalar).
/// @tparam T The type of elements in the vector.
/// @tparam U The type of the scalar.
/// @tparam N The size of the vector.
/// @param l The vector (modified in-place).
/// @param r The scalar value.
/// @return The modified vector.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator+=(fsmlib::Vector<T, N> &l, const U &r);

/// @brief Scalar subtraction assignment (Vector -= Scalar).
/// @tparam T The type of elements in the vector.
/// @tparam U The type of the scalar.
/// @tparam N The size of the vector.
/// @param l The vector (modified in-place).
/// @param r The scalar value.
/// @return The modified vector.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator-=(fsmlib::Vector<T, N> &l, const U &r);

/// @brief Scalar multiplication assignment (Vector *= Scalar).
/// @tparam T The type of elements in the vector.
/// @tparam U The type of the scalar.
/// @tparam N The size of the vector.
/// @param l The vector (modified in-place).
/// @param r The scalar value.
/// @return The modified vector.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator*=(fsmlib::Vector<T, N> &l, const U &r);

/// @brief Scalar division assignment (Vector /= Scalar).
/// @tparam T The type of elements in the vector.
/// @tparam U The type of the scalar.
/// @tparam N The size of the vector.
/// @param l The vector (modified in-place).
/// @param r The scalar value.
/// @return The modified vector.
template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator/=(fsmlib::Vector<T, N> &l, const U &r);

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
constexpr inline auto apply_binary_helper(
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
constexpr inline void apply_binary_helper(
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
constexpr inline auto apply_binary_helper(
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
constexpr inline void apply_binary_helper(
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
constexpr inline auto apply_binary_helper(
    const U &l,
    const fsmlib::Vector<T, sizeof...(N)> &r,
    F func,
    std::integer_sequence<std::size_t, N...> s)
{
    using return_type_t = fsmlib::Vector<decltype(func(l, r[0])), s.size()>;
    return return_type_t{ func(l, r[N])... };
}

/// @brief Applies a unary operation to all elements of a container.
/// @tparam N The number of elements.
/// @tparam F The unary operation.
/// @tparam T The container type.
/// @param func The unary operation to apply.
/// @param arg The container to operate on.
/// @return The resulting container after applying the operation.
template <std::size_t N, class F, class T>
constexpr auto apply_unary(F func, T &&arg)
{
    return apply_unary_helper(arg, func, std::make_integer_sequence<std::size_t, N>{});
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
constexpr auto apply_binary(F func, const T1 &arg1, const T2 &arg2)
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

/// @brief Functor to compute the negation of a value.
struct negate {
    /// @brief Negates the given value.
    /// @tparam T The type of the value.
    /// @param l The value to negate.
    /// @returns The negated value.
    template <class T>
    constexpr auto operator()(const T &l) const noexcept
    {
        return -l;
    }
};

/// @brief Functor to compute the addition of two values.
struct plus {
    /// @brief Adds two values.
    /// @tparam T The type of the first value.
    /// @tparam U The type of the second value.
    /// @param l The first value.
    /// @param r The second value.
    /// @returns The sum of the two values.
    template <class T, class U>
    constexpr auto operator()(const T &l, const U &r) const noexcept
    {
        return l + r;
    }
};

/// @brief Functor to compute the subtraction of two values.
struct minus {
    /// @brief Subtracts the second value from the first value.
    /// @tparam T The type of the first value.
    /// @tparam U The type of the second value.
    /// @param l The first value.
    /// @param r The second value.
    /// @returns The result of the subtraction.
    template <class T, class U>
    constexpr auto operator()(const T &l, const U &r) const noexcept
    {
        return l - r;
    }
};

/// @brief Functor to compute the multiplication of two values.
struct multiplies {
    /// @brief Multiplies two values.
    /// @tparam T The type of the first value.
    /// @tparam U The type of the second value.
    /// @param l The first value.
    /// @param r The second value.
    /// @returns The product of the two values.
    template <class T, class U>
    constexpr auto operator()(const T &l, const U &r) const noexcept
    {
        return l * r;
    }
};

/// @brief Functor to compute the division of two values.
struct divides {
    /// @brief Divides the first value by the second value.
    /// @tparam T The type of the first value.
    /// @tparam U The type of the second value.
    /// @param l The first value (numerator).
    /// @param r The second value (denominator).
    /// @returns The result of the division.
    template <class T, class U>
    constexpr auto operator()(const T &l, const U &r) const noexcept
    {
        return l / r;
    }
};

/// @brief Functor to check equality between two values.
struct equal {
    /// @brief Checks if two values are equal.
    /// @tparam T The type of the first value.
    /// @tparam U The type of the second value.
    /// @param l The first value.
    /// @param r The second value.
    /// @returns True if the two values are equal, false otherwise.
    template <class T, class U>
    constexpr auto operator()(const T &l, const U &r) const noexcept
    {
        return l == r;
    }
};

/// @brief Functor to check inequality between two values.
struct not_equal {
    /// @brief Checks if two values are not equal.
    /// @tparam T The type of the first value.
    /// @tparam U The type of the second value.
    /// @param l The first value.
    /// @param r The second value.
    /// @returns True if the two values are not equal, false otherwise.
    template <class T, class U>
    constexpr auto operator()(const T &l, const U &r) const noexcept
    {
        return l != r;
    }
};

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
constexpr inline auto eye(T value = 1)
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
constexpr inline auto inner_product(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
{
    decltype(l[0] * r[0]) ret{};
    for (std::size_t i = 0; i < N; ++i) {
        ret += l[i] * r[i];
    }
    return ret;
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
constexpr inline auto outer_product(const fsmlib::Vector<T, M> &lv, const fsmlib::Vector<U, N> &rv)
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
template <typename T, std::size_t N1, std::size_t N2>
constexpr inline auto multiply(const Matrix<T, N1, N2> &mat, const Vector<T, N2> &vec)
{
    Vector<T, N1> result = {};
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
constexpr inline auto multiply(const fsmlib::Matrix<T1, N1, N2> &A, const fsmlib::Matrix<T2, N2, N3> &B)
{
    using data_type_t                          = std::remove_const_t<decltype(T1(0) * T2(0))>;
    fsmlib::Matrix<data_type_t, N1, N3> result = {};
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N3; ++c) {
            for (std::size_t k = 0; k < N2; ++k) {
                result[r][c] += A[r][k] * B[k][c];
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
constexpr inline bool all(const fsmlib::Vector<T, N> &container, F pred)
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
constexpr inline bool all(const fsmlib::Vector<T, N> &container)
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
constexpr inline bool any(const fsmlib::Vector<T, N> &container, F pred)
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
constexpr inline bool any(const fsmlib::Vector<T, N> &container)
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
inline void swap_rows(fsmlib::Matrix<T, N> &matrix, std::size_t i, std::size_t j, std::size_t start_column = 0, std::size_t end_column = std::numeric_limits<std::size_t>::max())
{
    end_column = std::min(end_column, N);
    for (std::size_t c = start_column; c < end_column; ++c) {
        std::swap(matrix[i][c], matrix[j][c]);
    }
}

} // namespace fsmlib

template <class T, class U, std::size_t N>
constexpr auto operator+(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::plus{}, l, r);
}

template <class T, class U, std::size_t N, typename>
constexpr auto operator+(const U &l, const fsmlib::Vector<T, N> &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::plus{}, l, r);
}

template <class T, class U, std::size_t N, typename>
constexpr auto operator+(const fsmlib::Vector<T, N> &l, const U &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::plus{}, l, r);
}

template <class T, class U, std::size_t N>
constexpr auto operator-(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::minus{}, l, r);
}

template <class T, class U, std::size_t N, typename>
constexpr auto operator-(const U &l, const fsmlib::Vector<T, N> &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::minus{}, l, r);
}

template <class T, class U, std::size_t N, typename>
constexpr auto operator-(const fsmlib::Vector<T, N> &l, const U &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::minus{}, l, r);
}

template <class T, class U, std::size_t N>
constexpr auto operator*(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::multiplies{}, l, r);
}

template <class T, class U, std::size_t N, typename>
constexpr auto operator*(const U &l, const fsmlib::Vector<T, N> &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::multiplies{}, l, r);
}

template <class T, class U, std::size_t N, typename>
constexpr auto operator*(const fsmlib::Vector<T, N> &l, const U &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::multiplies{}, l, r);
}

template <class T, class U, std::size_t N>
constexpr auto operator/(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::divides{}, l, r);
}

template <class T, class U, std::size_t N, typename>
constexpr auto operator/(const U &l, const fsmlib::Vector<T, N> &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::divides{}, l, r);
}

template <class T, class U, std::size_t N, typename>
constexpr auto operator/(const fsmlib::Vector<T, N> &l, const U &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::divides{}, l, r);
}

template <class T, class U, std::size_t N>
constexpr auto operator==(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::equal{}, l, r);
}

template <class T, class U, std::size_t N>
constexpr auto operator!=(const fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
{
    return fsmlib::details::apply_binary<N>(fsmlib::details::not_equal{}, l, r);
}

// Element-wise addition (Vector += Vector)
template <class T, class U, std::size_t N>
constexpr auto operator+=(fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
{
    fsmlib::details::apply_binary<N>(fsmlib::details::plus{}, l, r);
    return l;
}

// Element-wise subtraction (Vector -= Vector)
template <class T, class U, std::size_t N>
constexpr auto operator-=(fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
{
    fsmlib::details::apply_binary<N>(fsmlib::details::minus{}, l, r);
    return l;
}

// Element-wise multiplication (Vector *= Vector)
template <class T, class U, std::size_t N>
constexpr auto operator*=(fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
{
    fsmlib::details::apply_binary<N>(fsmlib::details::multiplies{}, l, r);
    return l;
}

// Element-wise division (Vector /= Vector)
template <class T, class U, std::size_t N>
constexpr auto operator/=(fsmlib::Vector<T, N> &l, const fsmlib::Vector<U, N> &r)
{
    fsmlib::details::apply_binary<N>(fsmlib::details::divides{}, l, r);
    return l;
}

// Scalar addition (Vector += Scalar)
template <class T, class U, std::size_t N, typename>
constexpr auto operator+=(fsmlib::Vector<T, N> &l, const U &r)
{
    fsmlib::details::apply_binary<N>(fsmlib::details::plus{}, l, r);
    return l;
}

// Scalar subtraction (Vector -= Scalar)
template <class T, class U, std::size_t N, typename>
constexpr auto operator-=(fsmlib::Vector<T, N> &l, const U &r)
{
    fsmlib::details::apply_binary<N>(fsmlib::details::minus{}, l, r);
    return l;
}

// Scalar multiplication (Vector *= Scalar)
template <class T, class U, std::size_t N, typename>
constexpr auto operator*=(fsmlib::Vector<T, N> &l, const U &r)
{
    fsmlib::details::apply_binary<N>(fsmlib::details::multiplies{}, l, r);
    return l;
}

// Scalar division (Vector /= Scalar)
template <class T, class U, std::size_t N, typename>
constexpr auto operator/=(fsmlib::Vector<T, N> &l, const U &r)
{
    fsmlib::details::apply_binary<N>(fsmlib::details::divides{}, l, r);
    return l;
}
