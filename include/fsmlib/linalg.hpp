/// @file linalg.hpp
/// @brief Provides linear algebra operations and overloads for vector and
/// matrix arithmetic.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include "fsmlib/math.hpp"

namespace fsmlib
{

namespace linalg
{

/// @brief Computes the infinity norm of a matrix.
/// @param A The input matrix.
/// @returns The largest infinity norm among the rows of the matrix.
template <typename T, std::size_t N1, std::size_t N2 = N1>
constexpr inline auto infinity_norm(const fsmlib::Matrix<T, N1, N2> &A)
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

/// @brief Scales down a matrix by a power of 2 such that norm(A) < 1.
/// @param A The input matrix.
/// @returns A pair containing the number of scaling iterations and the scaling factor.
template <typename T, std::size_t N1, std::size_t N2 = N1>
constexpr inline auto log2_ceil(const fsmlib::Matrix<T, N1, N2> &A)
{
    std::size_t iterations       = 0;
    std::remove_const_t<T> scale = 1.0;
    const auto norm              = fsmlib::linalg::infinity_norm(A);
    while ((norm * scale) >= 1.0) {
        scale *= 0.5;
        ++iterations;
    }
    return std::make_pair(iterations, scale);
}

/// @brief Computes the square norm of a vector.
/// @param v The input vector.
/// @returns The square norm of the vector.
template <typename T, std::size_t N>
inline auto square_norm(const fsmlib::Vector<T, N> &v)
{
    std::remove_const_t<T> accum = 0;
    for (std::size_t i = 0; i < N; ++i) {
        accum += v[i] * v[i];
    }
    return std::sqrt(accum);
}

/// @brief Computes the Frobenius norm of a matrix.
/// @param A The input matrix.
/// @returns The Frobenius norm of the matrix.
template <typename T, std::size_t N1, std::size_t N2 = N1>
inline auto square_norm(const fsmlib::Matrix<T, N1, N2> &A)
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

/// @brief Converts a vector to a column matrix.
/// @param v The input vector.
/// @returns The column matrix representation of the vector.
template <class T, std::size_t Cols>
[[nodiscard]] constexpr fsmlib::Matrix<T, Cols, 1> to_matrix(const fsmlib::Vector<T, Cols> &v) noexcept
{
    fsmlib::Matrix<T, Cols, 1> ret;
    for (std::size_t i = 0; i < Cols; ++i) {
        ret[i][0] = v[i];
    }
    return ret;
}

/// @brief Computes the transpose of a matrix.
/// @param m The input matrix.
/// @returns The transposed matrix.
template <class T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr fsmlib::Matrix<T, Cols, Rows> transpose(const fsmlib::Matrix<T, Rows, Cols> m)
{
    fsmlib::Matrix<T, Cols, Rows> ret{};
    for (std::size_t r = 0; r < Rows; ++r) {
        for (std::size_t c = 0; c < Cols; ++c) {
            ret[c][r] = m[r][c];
        }
    }
    return ret;
}

/// @brief Computes the cofactor of a matrix.
/// @param matrix The input matrix.
/// @param p The row to remove.
/// @param q The column to remove.
/// @returns A matrix with row p and column q removed.
template <typename T, std::size_t N>
[[nodiscard]] constexpr auto cofactor(const fsmlib::Matrix<T, N> &matrix, std::size_t p, std::size_t q)
{
    // Create the output matrix.
    fsmlib::Matrix<T, N - 1> output{};
    // Create the indexing variables.
    std::size_t i, j, row, col;
    // Looping for each element of the matrix.
    for (i = 0, j = 0, row = 0, col = 0; row < N; ++row) {
        for (col = 0; col < N; ++col) {
            // Copying only those element which are not in given row and column.
            if ((row != p) && (col != q)) {
                output[i][j++] = matrix[row][col];
                // When the row is filled, increase row index and reset col index.
                if (j == (N - 1)) {
                    j = 0, ++i;
                }
            }
        }
    }
    return output;
}

/// @brief Computes the determinant of a matrix using Gaussian Elimination.
/// @param matrix The input matrix.
/// @returns The determinant of the matrix.
template <typename CT, std::size_t N>
[[nodiscard]] constexpr auto determinant(const fsmlib::Matrix<CT, N> &matrix)
{
    // Fast exit with a 1x1.
    if constexpr (N == 1) {
        return matrix[0][0];
    }
    // Fast exit with a 2x2.
    if constexpr (N == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    // Fast exit with a 3x3.
    if constexpr (N == 3) {
        return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2]) -
               matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
               matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    }
    // Fast exit with a 4x4.
    if constexpr (N == 4) {
        return matrix[0][3] * matrix[1][2] * matrix[2][1] * matrix[3][0] - matrix[0][2] * matrix[1][3] * matrix[2][1] * matrix[3][0] -
               matrix[0][3] * matrix[1][1] * matrix[2][2] * matrix[3][0] + matrix[0][1] * matrix[1][3] * matrix[2][2] * matrix[3][0] +
               matrix[0][2] * matrix[1][1] * matrix[2][3] * matrix[3][0] - matrix[0][1] * matrix[1][2] * matrix[2][3] * matrix[3][0] -
               matrix[0][3] * matrix[1][2] * matrix[2][0] * matrix[3][1] + matrix[0][2] * matrix[1][3] * matrix[2][0] * matrix[3][1] +
               matrix[0][3] * matrix[1][0] * matrix[2][2] * matrix[3][1] - matrix[0][0] * matrix[1][3] * matrix[2][2] * matrix[3][1] -
               matrix[0][2] * matrix[1][0] * matrix[2][3] * matrix[3][1] + matrix[0][0] * matrix[1][2] * matrix[2][3] * matrix[3][1] +
               matrix[0][3] * matrix[1][1] * matrix[2][0] * matrix[3][2] - matrix[0][1] * matrix[1][3] * matrix[2][0] * matrix[3][2] -
               matrix[0][3] * matrix[1][0] * matrix[2][1] * matrix[3][2] + matrix[0][0] * matrix[1][3] * matrix[2][1] * matrix[3][2] +
               matrix[0][1] * matrix[1][0] * matrix[2][3] * matrix[3][2] - matrix[0][0] * matrix[1][1] * matrix[2][3] * matrix[3][2] -
               matrix[0][2] * matrix[1][1] * matrix[2][0] * matrix[3][3] + matrix[0][1] * matrix[1][2] * matrix[2][0] * matrix[3][3] +
               matrix[0][2] * matrix[1][0] * matrix[2][1] * matrix[3][3] - matrix[0][0] * matrix[1][2] * matrix[2][1] * matrix[3][3] -
               matrix[0][1] * matrix[1][0] * matrix[2][2] * matrix[3][3] + matrix[0][0] * matrix[1][1] * matrix[2][2] * matrix[3][3];
    }
    using T = std::remove_const_t<CT>;
    // We need to create a temporary.
    fsmlib::Matrix<T, N> A(matrix);
    // Create the indexing variables.
    std::size_t c, r = 0, k;
    // Initialize the determinant, and create both pivot and ratio variable.
    T det = static_cast<T>(1.), pivot, ratio;
    // We convert the temporary to upper triangular form.
    for (c = 0; c < N; ++c) {
        // If we have a negative value on the diagonal, we need to move it
        // somewhere else.
        if (A[c][c] == 0.) {
            // Right now, I'm trying to find a place below the current
            k = c + 1;
            while ((k < N) && (A[k][c] == 0.)) {
                k++;
            }
            // If we did not find a non-zero value, we have a singular matrix.
            if (k == N) {
                break;
            }
            // Swap the rows.
            fsmlib::swap_rows(A, c, k);
            // Every time we swap rows, we need to change the sign to the
            // determinant.
            det *= -1;
        }
        // Store the pivot.
        pivot = A[c][c];
        for (r = c + 1; r < N; ++r) {
            ratio = A[r][c] / pivot;
            for (k = c; k < N; ++k) {
                A[r][k] -= ratio * A[c][k];
            }
        }
        // Multiply the pivot for the determinant variable.
        det *= pivot;
    }
    return det;
}

/// @brief Computes the adjoint of a matrix.
/// @param matrix The input matrix.
/// @returns The adjoint of the matrix.
template <typename T, std::size_t N>
[[nodiscard]] constexpr auto adjoint(const fsmlib::Matrix<T, N> &matrix)
{
    // Return 1.
    if constexpr (N == 1) {
        return matrix[0][0];
    } else {
        // Prepare the output matrix.
        Matrix<std::remove_const_t<T>, N> adj;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                // Get cofactor of A[i][j]
                auto support = fsmlib::linalg::cofactor(matrix, i, j);
                // Sign of adj[j][i] positive if sum of row and column indexes is
                // even. Interchanging rows and columns to get the transpose of the
                // cofactor matrix.
                adj[j][i] = (((i + j) % 2 == 0) ? 1 : -1) * fsmlib::linalg::determinant(support);
            }
        }
        return adj;
    }
}

/// @brief Computes the inverse of a matrix.
/// @param matrix The input matrix.
/// @returns The inverse of the matrix if it exists, otherwise a zero matrix.
template <typename T, std::size_t N>
[[nodiscard]] constexpr auto inverse(const fsmlib::Matrix<T, N> &matrix)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Compute the determinant.
    data_type_t det = fsmlib::linalg::determinant(matrix);
    // If determinant is zero, the matrix is singular.
    if (det == 0.) {
        return fsmlib::Matrix<data_type_t, N>();
    }
    // Find adjoint of the matrix.
    auto adjoint = fsmlib::linalg::adjoint(matrix);
    // Create a matrix for the result.
    fsmlib::Matrix<data_type_t, N> inv;
    // Find Inverse using formula "inv(A) = adj(A)/det(A)".
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            inv[r][c] = adjoint[r][c] / det;
        }
    }
    return inv;
}

/// @brief Divides two matrices.
/// @param A The first matrix.
/// @param B The second matrix.
/// @returns The result of A / B.
template <typename T1, typename T2, std::size_t N>
[[nodiscard]] constexpr auto div(const fsmlib::Matrix<T1, N> &A, const fsmlib::Matrix<T2, N> &B)
{
    return fsmlib::multiply(A, linalg::inverse(B));
}

/// @brief Computes the matrix exponential using a scaling and squaring method.
/// @param A The input matrix.
/// @param accuracy The desired accuracy (e.g., 1e-05).
/// @returns The exponential of the matrix.
template <typename T, std::size_t N>
inline auto expm(const fsmlib::Matrix<T, N, N> &A, double accuracy)
{
    const auto [iterations, scale] = fsmlib::linalg::log2_ceil(A);
    const auto scaled_a            = A * scale;

    // Initialize
    fsmlib::Matrix<T, N, N> term = fsmlib::eye<T, N, N>();
    fsmlib::Matrix<T, N, N> ret  = fsmlib::eye<T, N, N>();
    T fac_inv                    = 1.0;

    for (std::size_t k = 1; k <= 50; ++k) { // Max iterations for safety
        fac_inv /= static_cast<double>(k);
        term = fsmlib::multiply(term, scaled_a) * fac_inv;
        ret  = ret + term;

        if (fsmlib::linalg::square_norm(term) < accuracy) {
            break;
        }
    }

    // Raise to power 2^iterations
    for (std::size_t k = 0; k < iterations; ++k) {
        ret = fsmlib::multiply(ret, ret);
    }

    return ret;
}

} // namespace linalg
} // namespace fsmlib
