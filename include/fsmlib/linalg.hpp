
#pragma once

#include "mx/math.hpp"

namespace mx
{

namespace linalg
{

/// @brief Function to get cofactor of the matrix.
/// @param matrix the input matrix.
/// @param p the row that must be removed.
/// @param q the column that must be removed.
/// @returns An [N-1, N-1] matrix, generated by removing row p and column q from
/// the input matrix.
template <typename T, std::size_t N>
[[nodiscard]] constexpr auto cofactor(const mx::Matrix<T, N> &matrix, std::size_t p, std::size_t q)
{
    // Create the output matrix.
    mx::Matrix<T, N - 1> output{};
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

/// @brief Use Gaussian Elimination to create an Upper Diagonalized matrix then
/// multiplying the diagonal to get the determinant.
/// @param matrix the input matrix.
/// @returns the determinant of the matrix.
template <typename CT, std::size_t N>
[[nodiscard]] constexpr auto determinant(const mx::Matrix<CT, N> &matrix)
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
    mx::Matrix<T, N> A(matrix);
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
            mx::details::swap_rows(A, c, k);
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

/// @brief Computes the adjoint of this matrix.
/// @param matrix the input matrix.
/// @returns the adjoint of the matrix.
template <typename T, std::size_t N>
[[nodiscard]] constexpr auto adjoint(const mx::Matrix<T, N> &matrix)
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
                auto support = mx::linalg::cofactor(matrix, i, j);
                // Sign of adj[j][i] positive if sum of row and column indexes is
                // even. Interchanging rows and columns to get the transpose of the
                // cofactor matrix.
                adj[j][i] = (((i + j) % 2 == 0) ? 1 : -1) * mx::linalg::determinant(support);
            }
        }
        return adj;
    }
}

/// @brief Computes the inverse of this matrix.
/// @param matrix the input matrix.
/// @returns the inverse of the matrix.
template <typename T, std::size_t N>
[[nodiscard]] constexpr auto inverse(const mx::Matrix<T, N> &matrix)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Compute the determinant.
    data_type_t det = mx::linalg::determinant(matrix);
    // If determinant is zero, the matrix is singular.
    if (det == 0.) {
        return mx::Matrix<data_type_t, N>();
    }
    // Find adjoint of the matrix.
    auto adjoint = mx::linalg::adjoint(matrix);
    // Create a matrix for the result.
    mx::Matrix<data_type_t, N> inv;
    // Find Inverse using formula "inv(A) = adj(A)/det(A)".
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            inv[r][c] = adjoint[r][c] / det;
        }
    }
    return inv;
}

/// @brief Divides the two matrices.
/// @param A the first matrix.
/// @param B the second matrix.
/// @returns the result of the division.
template <typename T1, typename T2, std::size_t N>
[[nodiscard]] constexpr auto div(const mx::Matrix<T1, N> &A, const mx::Matrix<T2, N> &B)
{
    return mx::multiply(A, linalg::inverse(B));
}

/// @brief Copmutes the exponenation of the input matrix.
/// @param A the input matrix.
/// @param accuracy the desired accuracy (e.g., 1e-05).
/// @returns the result of the exponential.
template <typename T, std::size_t N>
inline auto expm(const mx::Matrix<T, N, N> &A, double accuracy)
{
    // Scale down matrix A by a power of 2, such that norm(A) < 1.
    const auto [iterations, scale] = mx::details::log2_ceil(A);
    // Apply the scaling.
    const auto scaled_a = A * scale;

    // Compute power series for e^(A/(2^iterations))
    // init (k = 0)
    const std::size_t batch_size = N * N;
    const double square_accuracy = accuracy * accuracy * scale * scale;
    mx::Matrix<T, N, N> mtk      = mx::details::eye<T, N, N>(); // scaled_a to the power k
    mx::Matrix<T, N, N> ret      = mx::details::eye<T, N, N>(); // sum of power seriees
    double fac_inv               = 1.0;                         // inverse faculty
    double rel_square_diff       = square_accuracy + 1.0;

    for (std::size_t idx = 1; (rel_square_diff > square_accuracy) && (fac_inv != 0.0); idx += batch_size) {
        auto local_accum = mx::Matrix<T, N, N>();
        for (std::size_t i = 0; i < batch_size; ++i) {
            const double k = static_cast<double>(idx + i);
            fac_inv        = fac_inv * (1.0 / k);
            if (mx::details::approximately_equal(fac_inv, 0.0)) {
                break;
            }
            mtk         = mtk * scaled_a;
            local_accum = local_accum + (mtk * fac_inv);
        }
        ret = ret + local_accum;
        // Caclulate relative change in this iteration.
        // TODO(enrico): properly guard against division by zero
        const mx::Matrix<T, N> rel_error = mx::linalg::div(local_accum * local_accum, ret * ret + accuracy);
        rel_square_diff                  = mx::details::square_norm(rel_error);
    };
    // raise the result
    for (std::size_t k = 0; k < iterations; ++k) {
        ret = ret * ret;
    }
    return ret;
}

} // namespace linalg
} // namespace mx
