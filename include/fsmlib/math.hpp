
#pragma once

#include "mx/base.hpp"

#include <cmath>

template <class T, class U, std::size_t N>
constexpr auto operator+(const mx::Vector<T, N> &l, const mx::Vector<U, N> &r);

template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator+(const U &l, const mx::Vector<T, N> &r);

template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator+(const mx::Vector<T, N> &l, const U &r);

template <class T, class U, std::size_t N>
constexpr auto operator-(const mx::Vector<T, N> &l, const mx::Vector<U, N> &r);

template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator-(const U &l, const mx::Vector<T, N> &r);

template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator-(const mx::Vector<T, N> &l, const U &r);

template <class T, class U, std::size_t N>
constexpr auto operator*(const mx::Vector<T, N> &l, const mx::Vector<U, N> &r);

template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator*(const U &l, const mx::Vector<T, N> &r);

template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator*(const mx::Vector<T, N> &l, const U &r);

template <class T, class U, std::size_t N>
constexpr auto operator/(const mx::Vector<T, N> &l, const mx::Vector<U, N> &r);

template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator/(const U &l, const mx::Vector<T, N> &r);

template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator/(const mx::Vector<T, N> &l, const U &r);

template <class T, class U, std::size_t N>
constexpr auto operator==(const mx::Vector<T, N> &l, const mx::Vector<U, N> &r);

template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator==(const U &l, const mx::Vector<T, N> &r);

template <class T, class U, std::size_t N, typename = typename std::enable_if_t<std::is_arithmetic_v<U>, U>>
constexpr auto operator==(const mx::Vector<T, N> &l, const U &r);

namespace mx
{

namespace details
{

template <class T, class U, class F, std::size_t... N>
constexpr inline auto apply_binary_helper(
    const mx::Vector<T, sizeof...(N)> &l,
    const mx::Vector<U, sizeof...(N)> &r,
    F func,
    std::integer_sequence<std::size_t, N...>)
{
    using ResultType = mx::Vector<decltype(func(l[0], r[0])), sizeof...(N)>;
    return ResultType{ func(l[N], r[N])... };
}

template <class T, class U, class F, std::size_t N1, std::size_t... N>
constexpr inline auto apply_binary_helper(
    const mx::Matrix<T, N1, sizeof...(N)> &l,
    const mx::Matrix<U, N1, sizeof...(N)> &r,
    F func,
    std::integer_sequence<std::size_t, N1>)
{
    using RowType = decltype(apply_binary_helper(
        l[0], r[0], func, std::make_integer_sequence<std::size_t, sizeof...(N)>{}));
    mx::Matrix<RowType, N1, sizeof...(N)> result;
    for (std::size_t i = 0; i < N1; ++i) {
        result[i] = apply_binary_helper(
            l[i], r[i], func, std::make_integer_sequence<std::size_t, sizeof...(N)>{});
    }
    return result;
}

template <class T, class U, class F, std::size_t... N>
constexpr inline auto apply_binary_helper(
    const U &l,
    const mx::Vector<T, sizeof...(N)> &r,
    F func,
    std::integer_sequence<std::size_t, N...> s)
{
    using return_type_t = mx::Vector<decltype(func(l, r[0])), s.size()>;
    return return_type_t{ func(l, r[N])... };
}

template <class T, class U, class F, std::size_t... N>
constexpr inline auto apply_binary_helper(
    const mx::Vector<T, sizeof...(N)> &l,
    const U &r,
    F func,
    std::integer_sequence<std::size_t, N...> s)
{
    using return_type_t = mx::Vector<decltype(func(l[0], r)), s.size()>;
    return return_type_t{ func(l[N], r)... };
}

template <std::size_t N, class F, class T>
constexpr auto apply_unary(F func, T &&arg)
{
    return apply_unary_helper(arg, func, std::make_integer_sequence<std::size_t, N>{});
}

template <std::size_t N, class F, class T1, class T2>
constexpr auto apply_binary(F func, const T1 &arg1, const T2 &arg2)
{
    return apply_binary_helper(arg1, arg2, func, std::make_integer_sequence<std::size_t, N>{});
}

struct negate {
    template <class T>
    constexpr inline auto operator()(const T &l) const noexcept
    {
        return -l;
    }
};

struct plus {
    template <class T, class U>
    constexpr inline auto operator()(const T &l, const U &r) const noexcept
    {
        return l + r;
    }
};

struct minus {
    template <class T, class U>
    constexpr inline auto operator()(const T &l, const U &r) const noexcept
    {
        return l - r;
    }
};

struct multiplies {
    template <class T, class U>
    constexpr inline auto operator()(const T &l, const U &r) const noexcept
    {
        return l * r;
    }
};

struct divides {
    template <class T, class U>
    constexpr inline auto operator()(const T &l, const U &r) const noexcept
    {
        return l / r;
    }
};

struct equal {
    template <class T, class U>
    constexpr inline auto operator()(const T &l, const U &r) const noexcept
    {
        return l == r;
    }
};

/// @brief Checks if the two floating point values are equal.
/// @param a the first value.
/// @param b the second value.
/// @returns true if they are approximately equal.
/// @returns false otherwise.
template <typename T1, typename T2>
inline bool approximately_equal(T1 a, T2 b, double tolerance = 1e-09)
{
    return std::fabs(a - b) <= tolerance * std::fmax(std::fabs(a), std::fabs(b));
}

/// @brief Checks if the fir floating point value is lesser than or equal to the second one.
/// @param a the first value.
/// @param b the second value.
/// @returns true if (a <= b).
/// @returns false otherwise.
template <typename T1, typename T2>
inline bool approximately_lesser_than_equal(T1 a, T2 b)
{
    return (a < b) || (mx::details::approximately_equal(a, b));
}

/// @brief Checks if the fir floating point value is greater than or equal to the second one.
/// @param a the first value.
/// @param b the second value.
/// @returns true if (a >= b).
/// @returns false otherwise.
template <typename T1, typename T2>
inline bool approximately_greater_than_equal(T1 a, T2 b)
{
    return (a > b) || (mx::details::approximately_equal(a, b));
}

template <class T, std::size_t N1, std::size_t N2 = N1>
constexpr inline auto eye(T value = 1)
{
    constexpr std::size_t cnt = N1 < N2 ? N1 : N2;
    mx::Matrix<T, N1, N2> r{};
    for (std::size_t i = 0; i < cnt; ++i) {
        r[i][i] = value;
    }
    return r;
}

/// @brief Swaps the values of the two rows.
/// @param matrix the matrix.
/// @param i the first row.
/// @param j the second row.
/// @param start_column the starting column.
/// @param end_column the ending column.
template <typename T, std::size_t N>
inline void swap_rows(mx::Matrix<T, N> &matrix, std::size_t i, std::size_t j, std::size_t start_column = 0, std::size_t end_column = std::numeric_limits<std::size_t>::max())
{
    end_column = std::min(end_column, N);
    for (std::size_t c = start_column; c < end_column; ++c) {
        std::swap(matrix[i][c], matrix[j][c]);
    }
}

/// @brief Computes the infinity norm of a matrix, i.e., largest infinity norm among the rows of the matrix.
/// @param A the input matrix.
/// @returns the infinity norm.
template <typename T, std::size_t N1, std::size_t N2 = N1>
constexpr inline auto infinity_norm(const mx::Matrix<T, N1, N2> &A)
{
    std::remove_const_t<T> max{}, accum{};
    for (std::size_t r = 0; r < N1; ++r) {
        accum = 0.;
        for (std::size_t c = 0; c < N2; ++c) {
            accum += std::abs(A[r][c]);
        }
        max = std::max(max, accum);
    }
    return max;
}

/// @brief Scale down matrix A by a power of 2, such that norm(A) < 1.
/// @param A the input matrix.
/// @returns the square root of the sum of all the squares.
template <typename T, std::size_t N1, std::size_t N2 = N1>
constexpr inline auto log2_ceil(const mx::Matrix<T, N1, N2> &A)
{
    std::size_t iterations       = 0;
    std::remove_const_t<T> scale = 1.0;
    const auto norm              = mx::details::infinity_norm(A);
    while ((norm * scale) > 1.0) {
        scale *= 0.5;
        ++iterations;
    }
    return std::make_pair(iterations, scale);
}

/// @brief Computes the square norm of the vector.
/// @param v the vector.
/// @return the square norm of the vector.
template <typename T, std::size_t N>
inline auto square_norm(const mx::Vector<T, N> &v)
{
    std::remove_const_t<T> accum = 0;
    for (std::size_t i = 0; i < N; ++i) {
        accum += v[i] * v[i];
    }
    return std::sqrt(accum);
}

/// @brief The Frobenius norm of a matrix.
/// @param A the input matrix.
/// @returns the norm.
template <typename T, std::size_t N1, std::size_t N2 = N1>
inline auto square_norm(const mx::Matrix<T, N1, N2> &A)
{
    std::remove_const_t<T> accum = 0;
    // Compute the sum of squares of the elements of the given matrix.
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N2; ++c) {
            accum += A[r][c] * A[r][c];
        }
    }
    // Return the square root of the sum of squares.
    return std::sqrt(accum);
}

} // namespace details

template <class T, std::size_t Cols>
[[nodiscard]] constexpr mx::Matrix<T, Cols, 1> to_matrix(const mx::Vector<T, Cols> &v) noexcept
{
    mx::Matrix<T, Cols, 1> ret;
    for (std::size_t i = 0; i < Cols; ++i) {
        ret[i][0] = v[i];
    }
    return ret;
}

/// @brief Matrix transpose.
/// @param m matrix to transpose.
/// @return transposed matrix.
template <class T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr mx::Matrix<T, Cols, Rows> transpose(const mx::Matrix<T, Rows, Cols> m)
{
    mx::Matrix<T, Cols, Rows> ret{};
    for (std::size_t r = 0; r < Rows; ++r) {
        for (std::size_t c = 0; c < Cols; ++c) {
            ret[c][r] = m[r][c];
        }
    }
    return ret;
}

template <class T, class U, std::size_t N>
constexpr inline auto inner_product(const mx::Vector<T, N> &l, const mx::Vector<U, N> &r)
{
    decltype(l[0] * r[0]) ret{};
    for (std::size_t i = 0; i < N; ++i) {
        ret += l[i] * r[i];
    }
    return ret;
}

template <class T, class U, std::size_t M, std::size_t N>
constexpr inline auto outer_product(const mx::Vector<T, M> &lv, const mx::Vector<U, N> &rv)
{
    mx::Matrix<decltype(lv[0] * rv[0]), M, N> ret{};
    for (std::size_t r = 0; r < M; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            ret[r][c] = lv[r] * rv[c];
        }
    }
    return ret;
}

/// @brief Left-Multiplies a (colum) vector with a square matrix.
/// @param mx matrix to multiply vector with.
/// @param vec vector to multiply matrix with.
/// @return result of multiplication.
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

/// @brief Multiplication between two matrices.
/// @param A first matrix.
/// @param B second matrix.
/// @returns the resulting matrix.
template <typename T1, typename T2, std::size_t N1, std::size_t N2, std::size_t N3>
constexpr inline auto multiply(const mx::Matrix<T1, N1, N2> &A, const mx::Matrix<T2, N2, N3> &B)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<decltype(T1(0) * T2(0))>;
    // Declare the output matrix.
    mx::Matrix<data_type_t, N1, N3> result;
    // Perform the operation.
    for (std::size_t r = 0; r < N1; r++) {
        for (std::size_t c = 0; c < N3; c++) {
            for (std::size_t k = 0; k < N2; k++) {
                result[r][c] += A[r][k] * B[k][c];
            }
        }
    }
    return result;
}

} // namespace mx

template <class T, class U, std::size_t N>
constexpr inline auto operator+(const mx::Vector<T, N> &l, const mx::Vector<U, N> &r)
{
    return mx::details::apply_binary<N>(mx::details::plus{}, l, r);
}

template <class T, class U, std::size_t N>
constexpr inline auto operator+(const U &l, const mx::Vector<T, N> &r)
{
    return mx::details::apply_binary<N>(mx::details::plus{}, l, r);
}

template <class T, class U, std::size_t N>
constexpr inline auto operator+(const mx::Vector<T, N> &l, const U &r)
{
    return mx::details::apply_binary<N>(mx::details::plus{}, l, r);
}

template <class T, class U, std::size_t N>
constexpr inline auto operator-(const mx::Vector<T, N> &l, const mx::Vector<U, N> &r)
{
    return mx::details::apply_binary<N>(mx::details::minus{}, l, r);
}
template <class T, class U, std::size_t N>
constexpr inline auto operator-(const U &l, const mx::Vector<T, N> &r)
{
    return mx::details::apply_binary<N>(mx::details::minus{}, l, r);
}
template <class T, class U, std::size_t N>
constexpr inline auto operator-(const mx::Vector<T, N> &l, const U &r)
{
    return mx::details::apply_binary<N>(mx::details::minus{}, l, r);
}

template <class T, class U, std::size_t N>
constexpr inline auto operator*(const mx::Vector<T, N> &l, const mx::Vector<U, N> &r)
{
    return mx::details::apply_binary<N>(mx::details::multiplies{}, l, r);
}
template <class T, class U, std::size_t N>
constexpr inline auto operator*(const U &l, const mx::Vector<T, N> &r)
{
    return mx::details::apply_binary<N>(mx::details::multiplies{}, l, r);
}
template <class T, class U, std::size_t N>
constexpr inline auto operator*(const mx::Vector<T, N> &l, const U &r)
{
    return mx::details::apply_binary<N>(mx::details::multiplies{}, l, r);
}

template <class T, class U, std::size_t N>
constexpr inline auto operator/(const mx::Vector<T, N> &l, const mx::Vector<U, N> &r)
{
    return mx::details::apply_binary<N>(mx::details::divides{}, l, r);
}
template <class T, class U, std::size_t N>
constexpr inline auto operator/(const U &l, const mx::Vector<T, N> &r)
{
    return mx::details::apply_binary<N>(mx::details::divides{}, l, r);
}
template <class T, class U, std::size_t N>
constexpr inline auto operator/(const mx::Vector<T, N> &l, const U &r)
{
    return mx::details::apply_binary<N>(mx::details::divides{}, l, r);
}

template <class T, class U, std::size_t N>
constexpr inline auto operator==(const mx::Vector<T, N> &l, const mx::Vector<U, N> &r)
{
    return mx::details::apply_binary<N>(mx::details::equal{}, l, r);
}
template <class T, class U, std::size_t N>
constexpr inline auto operator==(const U &l, const mx::Vector<T, N> &r)
{
    return mx::details::apply_binary<N>(mx::details::equal{}, l, r);
}
template <class T, class U, std::size_t N>
constexpr inline auto operator==(const mx::Vector<T, N> &l, const U &r)
{
    return mx::details::apply_binary<N>(mx::details::equal{}, l, r);
}
