/// @file linalg.hpp
/// @brief Provides linear algebra operations and overloads for vector and
/// matrix arithmetic.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include <iostream>
#include <iomanip>

#include <array>
#include <tuple>

#include "fsmlib/view.hpp"
#include "fsmlib/math.hpp"

namespace fsmlib
{

/// @brief Linear algebra operations and utilities.
namespace linalg
{

/// @brief Computes the infinity norm of a matrix.
/// @param A The input matrix.
/// @returns The largest infinity norm among the rows of the matrix.
template <typename T, std::size_t N1, std::size_t N2 = N1>
[[nodiscard]] constexpr inline auto infinity_norm(const fsmlib::MatrixBase<T, N1, N2> &A)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Initialize the maximum value and the accumulator.
    data_type_t max{}, accum{};
    for (std::size_t r = 0; r < N1; ++r) {
        accum = 0.;
        for (std::size_t c = 0; c < N2; ++c) {
            accum += std::abs(A(r, c));
        }
        max = std::max(max, accum);
    }
    return max;
}

/// @brief Computes the square norm of a vector.
/// @param v The input vector.
/// @returns The square norm of the vector.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto square_norm(const fsmlib::Vector<T, N> &v)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Initialize the accumulator.
    data_type_t accum = 0;
    for (std::size_t i = 0; i < N; ++i) {
        accum += v[i] * v[i];
    }
    return std::sqrt(accum);
}

/// @brief Computes the Frobenius norm of a matrix.
/// @param A The input matrix.
/// @returns The Frobenius norm of the matrix.
template <typename T, std::size_t N1, std::size_t N2 = N1>
[[nodiscard]] constexpr inline auto square_norm(const fsmlib::MatrixBase<T, N1, N2> &A)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Initialize the accumulator.
    data_type_t accum = 0;
    // Compute the sum of squares of the elements of the given matrix.
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N2; ++c) {
            accum += A(r, c) * A(r, c);
        }
    }
    // Return the square root of the sum of squares.
    return std::sqrt(accum);
}

/// @brief Computes the Frobenius norm of a matrix.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param mat The input matrix.
/// @return The Frobenius norm of the matrix.
template <typename T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr inline auto frobenius_norm(const fsmlib::MatrixBase<T, Rows, Cols> &mat)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Initialize the accumulator.
    data_type_t sum = 0;
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            sum += mat(i, j) * mat(i, j);
        }
    }
    return std::sqrt(sum);
}

/// @brief Scales down a matrix by a power of 2 such that norm(A) < 1.
/// @param A The input matrix.
/// @returns A pair containing the number of scaling iterations and the scaling factor.
template <typename T, std::size_t N1, std::size_t N2 = N1>
[[nodiscard]] constexpr inline auto scale_to_unit_norm(const fsmlib::MatrixBase<T, N1, N2> &A)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Initialize the scaling factor and the number of iterations.
    std::size_t iterations = 0;
    data_type_t scale      = 1.0;
    const auto norm        = fsmlib::linalg::infinity_norm(A);
    while ((norm * scale) >= 1.0) {
        scale *= 0.5;
        ++iterations;
    }
    return std::make_pair(iterations, scale);
}

/// @brief Scales a matrix such that its Frobenius norm equals a target value.
/// @tparam T The type of the matrix elements.
/// @tparam N1 The number of rows in the matrix.
/// @tparam N2 The number of columns in the matrix (default = N1 for square matrices).
/// @param A The input matrix.
/// @param target_norm The target norm to scale the matrix to (default = 1.0).
/// @returns A pair containing the scaling factor applied and the scaled matrix.
template <typename T, std::size_t N1, std::size_t N2 = N1>
[[nodiscard]] constexpr inline auto scale_to_target_norm(const fsmlib::MatrixBase<T, N1, N2> &A, T target_norm = 1.0)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Compute the Frobenius norm of the matrix.
    data_type_t frobenius_norm = fsmlib::linalg::frobenius_norm(A);

    // Compute the scaling factor to achieve the target norm
    data_type_t scale_factor = frobenius_norm > 0 ? (target_norm / frobenius_norm) : 1.0;

    return std::make_pair(scale_factor, A * scale_factor);
}

/// @brief Balances a matrix to improve numerical stability by scaling rows and columns.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the matrix (square: NxN).
/// @param mat The input square matrix to balance.
/// @returns The balanced matrix.
/// @details This method reduces the condition number of the matrix by scaling rows
///          and columns to have similar norms. The matrix is modified such that:
///          - The row and column norms are approximately equal.
///          - The scaling factors are powers of 2 to ensure no loss of precision.
///          This process is iterative and stops when no further significant improvement is observed.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto balance(const fsmlib::MatrixBase<T, N, N> &mat)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Initialize the result matrix.
    fsmlib::Matrix<data_type_t, N, N> result = mat;
    // Initialize the scaling vector.
    auto scale = fsmlib::ones<data_type_t, N>();
    // Define the gamma factor for scaling.
    constexpr data_type_t gamma = 2.0;
    // Define the convergence flag.
    bool converged = false;

    // Iterate until convergence.
    while (!converged) {
        converged = true;

        for (std::size_t i = 0; i < N; ++i) {
            // Calculate row and column norms
            data_type_t row_norm = 0.0;
            data_type_t col_norm = 0.0;

            for (std::size_t j = 0; j < N; ++j) {
                if (i != j) {
                    row_norm += std::abs(result(i, j));
                    col_norm += std::abs(result(j, i));
                }
            }

            // Compute the scaling factor
            if (row_norm > 0.0 && col_norm > 0.0) {
                data_type_t factor = std::sqrt(col_norm / row_norm);

                // Limit scaling factor to powers of gamma
                if (factor > 1.0 / gamma && factor < gamma) {
                    factor = 1.0;
                }

                // Apply scaling if it improves the balance.
                if (!fsmlib::feq::approximately_equal(factor, 1.0)) {
                    converged = false;
                    for (std::size_t j = 0; j < N; ++j) {
                        result(i, j) *= factor;
                        result(j, i) /= factor;
                    }
                    scale[i] *= factor;
                }
            }
        }
    }

    return result;
}

/// @brief Computes the transpose of a matrix.
/// @param m The input matrix.
/// @returns The transposed matrix.
template <class T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr inline fsmlib::Matrix<T, Cols, Rows> transpose(const fsmlib::MatrixBase<T, Rows, Cols> &m)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Initialize the transposed matrix.
    fsmlib::Matrix<data_type_t, Cols, Rows> ret{};
    for (std::size_t r = 0; r < Rows; ++r) {
        for (std::size_t c = 0; c < Cols; ++c) {
            ret(c, r) = m(r, c);
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
[[nodiscard]] constexpr inline auto cofactor(const fsmlib::MatrixBase<T, N> &matrix, std::size_t p, std::size_t q)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Create the output matrix.
    fsmlib::Matrix<data_type_t, N - 1> output{};
    // Create the indexing variables.
    std::size_t i, j, row, col;
    // Looping for each element of the matrix.
    for (i = 0, j = 0, row = 0, col = 0; row < N; ++row) {
        for (col = 0; col < N; ++col) {
            // Copying only those element which are not in given row and column.
            if ((row != p) && (col != q)) {
                output(i, j++) = matrix(row, col);
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
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto determinant(const fsmlib::MatrixBase<T, N> &matrix)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Fast exit with a 1x1.
    if constexpr (N == 1) {
        return matrix(0, 0);
    }
    // Fast exit with a 2x2.
    if constexpr (N == 2) {
        return matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);
    }
    // Fast exit with a 3x3.
    if constexpr (N == 3) {
        return matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) - matrix(2, 1) * matrix(1, 2)) -
               matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
               matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
    }
    // Fast exit with a 4x4.
    if constexpr (N == 4) {
        return matrix(0, 3) * matrix(1, 2) * matrix(2, 1) * matrix(3, 0) -
               matrix(0, 2) * matrix(1, 3) * matrix(2, 1) * matrix(3, 0) -
               matrix(0, 3) * matrix(1, 1) * matrix(2, 2) * matrix(3, 0) +
               matrix(0, 1) * matrix(1, 3) * matrix(2, 2) * matrix(3, 0) +
               matrix(0, 2) * matrix(1, 1) * matrix(2, 3) * matrix(3, 0) -
               matrix(0, 1) * matrix(1, 2) * matrix(2, 3) * matrix(3, 0) -
               matrix(0, 3) * matrix(1, 2) * matrix(2, 0) * matrix(3, 1) +
               matrix(0, 2) * matrix(1, 3) * matrix(2, 0) * matrix(3, 1) +
               matrix(0, 3) * matrix(1, 0) * matrix(2, 2) * matrix(3, 1) -
               matrix(0, 0) * matrix(1, 3) * matrix(2, 2) * matrix(3, 1) -
               matrix(0, 2) * matrix(1, 0) * matrix(2, 3) * matrix(3, 1) +
               matrix(0, 0) * matrix(1, 2) * matrix(2, 3) * matrix(3, 1) +
               matrix(0, 3) * matrix(1, 1) * matrix(2, 0) * matrix(3, 2) -
               matrix(0, 1) * matrix(1, 3) * matrix(2, 0) * matrix(3, 2) -
               matrix(0, 3) * matrix(1, 0) * matrix(2, 1) * matrix(3, 2) +
               matrix(0, 0) * matrix(1, 3) * matrix(2, 1) * matrix(3, 2) +
               matrix(0, 1) * matrix(1, 0) * matrix(2, 3) * matrix(3, 2) -
               matrix(0, 0) * matrix(1, 1) * matrix(2, 3) * matrix(3, 2) -
               matrix(0, 2) * matrix(1, 1) * matrix(2, 0) * matrix(3, 3) +
               matrix(0, 1) * matrix(1, 2) * matrix(2, 0) * matrix(3, 3) +
               matrix(0, 2) * matrix(1, 0) * matrix(2, 1) * matrix(3, 3) -
               matrix(0, 0) * matrix(1, 2) * matrix(2, 1) * matrix(3, 3) -
               matrix(0, 1) * matrix(1, 0) * matrix(2, 2) * matrix(3, 3) +
               matrix(0, 0) * matrix(1, 1) * matrix(2, 2) * matrix(3, 3);
    }
    // We need to create a temporary.
    fsmlib::Matrix<data_type_t, N> A(matrix);
    // Create the indexing variables.
    std::size_t c, r = 0, k;
    // Initialize the determinant, and create both pivot and ratio variable.
    data_type_t det = static_cast<data_type_t>(1.), pivot, ratio;
    // We convert the temporary to upper triangular form.
    for (c = 0; c < N; ++c) {
        // If we have a negative value on the diagonal, we need to move it
        // somewhere else.
        if (fsmlib::feq::approximately_equal_to_zero(A(c, c))) {
            // Right now, I'm trying to find a place below the current
            k = c + 1;
            while ((k < N) && fsmlib::feq::approximately_equal_to_zero(A(k, c))) {
                k++;
            }
            // If we did not find a non-zero value, we have a singular matrix.
            if (k == N) {
                return static_cast<data_type_t>(0.);
            }
            // Swap the rows.
            fsmlib::swap_rows(A, c, k);
            // Every time we swap rows, we need to change the sign to the
            // determinant.
            det *= -1;
        }
        // Store the pivot.
        pivot = A(c, c);
        for (r = c + 1; r < N; ++r) {
            ratio = A(r, c) / pivot;
            for (k = c; k < N; ++k) {
                A(r, k) -= ratio * A(c, k);
            }
        }
        // Multiply the pivot for the determinant variable.
        det *= pivot;
    }
    return det;
}

/// @brief Computes the adjoint of a matrix.
/// @param A The input matrix.
/// @returns The adjoint of the matrix.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto adjoint(const fsmlib::MatrixBase<T, N> &A)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Return 1 for 1x1 matrix.
    if constexpr (N == 1) {
        Matrix<data_type_t, 1, 1> adj;
        adj(0, 0) = A(0, 0);
        return adj;
    } else {
        Matrix<data_type_t, N, N> cofactors;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                // Compute the cofactor for element (i, j)
                auto view       = fsmlib::cofactor_view(A, i, j);
                cofactors(i, j) = (((i + j) % 2 == 0) ? 1 : -1) * fsmlib::linalg::determinant(view);
            }
        }

        // Transpose the cofactor matrix to get the adjoint
        return fsmlib::linalg::transpose(cofactors);
    }
}

/// @brief Computes the inverse of a matrix.
/// @param matrix The input matrix.
/// @returns The inverse of the matrix if it exists, otherwise a zero matrix.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto inverse(const fsmlib::MatrixBase<T, N> &matrix)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Compute the determinant.
    data_type_t det = fsmlib::linalg::determinant(matrix);
    // If determinant is zero, the matrix is singular.
    if (fsmlib::feq::approximately_equal_to_zero(det)) {
        return fsmlib::Matrix<data_type_t, N>();
    }
    // Find adjoint of the matrix.
    auto adjoint = fsmlib::linalg::adjoint(matrix);
    // Create a matrix for the result.
    fsmlib::Matrix<data_type_t, N> inv;
    // Find Inverse using formula "inv(A) = adj(A)/det(A)".
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            inv(r, c) = adjoint(r, c) / det;
        }
    }
    return inv;
}

/// @brief Divides two matrices (A / B), equivalent to A * B^-1.
/// @tparam T1 The type of elements in matrix A.
/// @tparam T2 The type of elements in matrix B.
/// @tparam N1 The number of rows in matrix A.
/// @tparam N2 The number of columns in matrix A and rows in matrix B.
/// @tparam N3 The number of columns in matrix B.
/// @param A The first matrix.
/// @param B The second matrix.
/// @returns A matrix of size N1 x N3 resulting from A * B^-1.
template <typename T1, typename T2, std::size_t N1, std::size_t N2, std::size_t N3>
[[nodiscard]] inline constexpr auto div(const fsmlib::MatrixBase<T1, N1, N2> &A,
                                        const fsmlib::MatrixBase<T2, N2, N3> &B)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<std::common_type_t<T1, T2>>;
    return fsmlib::multiply(A, linalg::inverse<data_type_t>(B));
}

/// @brief Computes the rank of a matrix using Gaussian elimination.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param mat The input matrix.
/// @return The rank of the matrix.
/// @details
/// This function calculates the rank of the input matrix by performing Gaussian elimination.
/// The rank is defined as the number of linearly independent rows or columns in the matrix.
/// A small tolerance (\( epsilon = 1e-9 \)) is used to handle floating-point precision issues.
///
/// The rank is determined as the number of non-zero rows in the row-echelon form of the matrix.
///
/// Example usage:
/// @code
/// fsmlib::Matrix<double, 4, 4> mat = {
///     { {1.0, 2.0, 3.0, 4.0},
///       {5.0, 6.0, 7.0, 8.0},
///       {9.0, 10.0, 11.0, 12.0},
///       {13.0, 14.0, 15.0, 16.0} }
/// };
/// auto r = fsmlib::linalg::rank(mat);
/// @endcode
/// @note The matrix is modified in-place during the Gaussian elimination process.
template <typename T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr inline auto rank(fsmlib::Matrix<T, Rows, Cols> mat)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Small value to handle floating-point precision issues.
    constexpr data_type_t epsilon = 1e-9;
    std::size_t rank              = 0;

    // Perform Gaussian elimination
    for (std::size_t col = 0, row = 0; col < Cols && row < Rows; ++col) {
        // Find the pivot
        std::size_t pivot = row;
        for (std::size_t i = row + 1; i < Rows; ++i) {
            if (std::abs(mat(i, col)) > std::abs(mat(pivot, col))) {
                pivot = i;
            }
        }

        // If the pivot is effectively zero, skip this column
        if (std::abs(mat(pivot, col)) < epsilon) {
            continue;
        }

        // Swap the current row with the pivot row
        for (std::size_t j = 0; j < Cols; ++j) {
            std::swap(mat(row, j), mat(pivot, j));
        }

        // Normalize the pivot row
        data_type_t pivot_value = mat(row, col);
        for (std::size_t j = 0; j < Cols; ++j) {
            mat(row, j) /= pivot_value;
        }

        // Eliminate the column below the pivot
        for (std::size_t i = row + 1; i < Rows; ++i) {
            data_type_t factor = mat(i, col);
            for (std::size_t j = 0; j < Cols; ++j) {
                mat(i, j) -= factor * mat(row, j);
            }
        }

        // Increment the rank and move to the next row
        ++rank;
        ++row;
    }

    return rank;
}

/// @brief Performs the QR decomposition of a fixed-size matrix.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param A The input matrix to decompose.
/// @return A pair of matrices (Q, R) representing the QR decomposition.
template <typename T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr inline auto qr_decomposition(const fsmlib::MatrixBase<T, Rows, Cols> &A)
{
    static_assert(Rows >= Cols, "The input matrix must have at least as many rows as columns.");
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;
    // Initialize R as a copy of A and Q as the identity matrix.
    Matrix<data_type_t, Rows, Cols> R;
    Matrix<data_type_t, Rows, Rows> Q;
    for (std::size_t k = 0; k < Cols; ++k) {
        // Compute the k-th column of Q.
        for (std::size_t i = 0; i < Rows; ++i) {
            Q(i, k) = A(i, k);
        }
        for (std::size_t j = 0; j < k; ++j) {
            data_type_t dot_product = 0;
            for (std::size_t i = 0; i < Rows; ++i) {
                dot_product += Q(i, j) * A(i, k);
            }
            R(j, k) = dot_product;

            for (std::size_t i = 0; i < Rows; ++i) {
                Q(i, k) -= R(j, k) * Q(i, j);
            }
        }
        // Normalize the k-th column of Q.
        data_type_t norm = 0;
        for (std::size_t i = 0; i < Rows; ++i) {
            norm += Q(i, k) * Q(i, k);
        }
        R(k, k) = std::sqrt(norm);
        for (std::size_t i = 0; i < Rows; ++i) {
            Q(i, k) /= R(k, k);
        }
        if (R(k, k) > 0) {
            R(k, k) = -R(k, k);
            for (std::size_t i = 0; i < Rows; ++i) {
                Q(i, k) = -Q(i, k);
            }
        }
    }
    return std::make_pair(Q, R);
}

/// @brief Performs the LU decomposition of a fixed-size matrix.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix (must equal Rows for LU decomposition).
/// @param A The input square matrix to decompose.
/// @return A pair of matrices (L, U) representing the LU decomposition, where L is lower triangular and U is upper triangular.
template <typename T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr inline auto lu_decomposition(const fsmlib::MatrixBase<T, Rows, Cols> &A)
{
    static_assert(Rows == Cols, "LU decomposition requires a square matrix.");
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Initialize L and U as zero matrices.
    Matrix<data_type_t, Rows, Cols> L = {};
    Matrix<data_type_t, Rows, Cols> U = {};

    for (std::size_t i = 0; i < Rows; ++i) {
        // Compute the upper triangular matrix U.
        for (std::size_t k = i; k < Cols; ++k) {
            data_type_t sum = 0;
            for (std::size_t j = 0; j < i; ++j) {
                sum += L(i, j) * U(j, k);
            }
            U(i, k) = A(i, k) - sum;
        }

        // Compute the lower triangular matrix L.
        for (std::size_t k = i; k < Rows; ++k) {
            if (i == k) {
                // Diagonal elements of L are set to 1.
                L(i, i) = 1;
            } else {
                data_type_t sum = 0;
                for (std::size_t j = 0; j < i; ++j) {
                    sum += L(k, j) * U(j, i);
                }
                L(k, i) = (A(k, i) - sum) / U(i, i);
            }
        }
    }

    return std::make_pair(L, U);
}

/// @brief Computes the Cholesky decomposition of a symmetric positive-definite matrix.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the square matrix (N x N).
/// @param mat The input symmetric positive-definite matrix.
/// @return A lower triangular matrix (L) such that (A=L*L^T).
/// @throws std::runtime_error If the input matrix is not symmetric or not positive definite.
/// @details
/// This function computes the Cholesky decomposition, which decomposes a symmetric positive-definite
/// matrix \( A \) into the product of a lower triangular matrix \( L \) and its transpose \( L^T \),
/// i.e., \( A = L * L^T \). The input matrix must satisfy the following conditions:
/// - Symmetry: \( A_{ij} = A_{ji} \).
/// - Positive definiteness: All eigenvalues of \( A \) must be positive.
///
/// Example usage:
/// @code
/// fsmlib::Matrix<double, 3, 3> A = {
///     { { 4.0, 12.0, -16.0 },
///       { 12.0, 37.0, -43.0 },
///       { -16.0, -43.0, 98.0 } }
/// };
/// auto L = fsmlib::linalg::cholesky_decomposition(A);
/// @endcode
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto cholesky_decomposition(const fsmlib::MatrixBase<T, N, N> &mat)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Ensure the input matrix is symmetric.
    if (!fsmlib::is_symmetric(mat)) {
        throw std::runtime_error("Cholesky decomposition: Matrix is not symmetric.");
    }

    fsmlib::Matrix<data_type_t, N, N> lower = {}; // Lower triangular matrix

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            data_type_t sum = 0;

            // Summation for diagonal and non-diagonal elements
            for (std::size_t k = 0; k < j; ++k) {
                sum += lower(i, k) * lower(j, k);
            }

            if (i == j) {
                // Diagonal elements
                data_type_t diag = mat(i, i) - sum;
                if (diag <= 0) {
                    throw std::runtime_error("Cholesky decomposition: Matrix is not positive definite.");
                }
                lower(i, j) = std::sqrt(diag);
            } else {
                // Non-diagonal elements
                lower(i, j) = (mat(i, j) - sum) / lower(j, j);
            }
        }
    }

    return lower;
}

/// @brief Solves a linear system Ax = b using LU decomposition.
/// @tparam T The type of the matrix and vector elements.
/// @tparam Rows The number of rows in the matrix A.
/// @tparam Cols The number of columns in the matrix A (must equal Rows for LU decomposition).
/// @param A The square coefficient matrix.
/// @param b The right-hand side vector.
/// @return The solution vector x such that Ax = b.
template <typename T, std::size_t Rows, std::size_t Cols>
[[nodiscard]] constexpr inline auto solve(const fsmlib::MatrixBase<T, Rows, Cols> &A, const fsmlib::Vector<T, Rows> &b)
{
    static_assert(Rows == Cols, "solve requires a square matrix for LU decomposition.");

    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Perform LU decomposition
    auto [L, U] = fsmlib::linalg::lu_decomposition(A);

    // Forward substitution to solve Ly = b
    fsmlib::Vector<data_type_t, Rows> y;
    for (std::size_t i = 0; i < Rows; ++i) {
        data_type_t sum = 0;
        for (std::size_t j = 0; j < i; ++j) {
            sum += L(i, j) * y[j];
        }
        y[i] = b[i] - sum;
    }

    // Back substitution to solve Ux = y
    fsmlib::Vector<data_type_t, Rows> x;
    for (std::size_t i = Rows; i-- > 0;) {
        data_type_t sum = 0;
        for (std::size_t j = i + 1; j < Rows; ++j) {
            sum += U(i, j) * x[j];
        }
        x[i] = (y[i] - sum) / U(i, i);
    }

    return x;
}

/// @brief Finds the dominant eigenvalue and eigenvector using power iteration.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the square matrix.
/// @param mat The input matrix.
/// @param max_iterations The maximum number of iterations.
/// @param tolerance The convergence tolerance.
/// @return A pair containing the dominant eigenvalue and its eigenvector.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto
power_iteration(const fsmlib::MatrixBase<T, N, N> &mat, std::size_t max_iterations = 1000, T tolerance = 1e-9)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Start with a vector of ones.
    auto v             = fsmlib::ones<data_type_t, N>();
    data_type_t lambda = 0;

    // Normalize the initial vector
    data_type_t norm = std::sqrt(fsmlib::inner_product(v, v));
    for (std::size_t i = 0; i < N; ++i) {
        v[i] /= norm;
    }

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        auto Av                = fsmlib::multiply(mat, v);     // Compute A * v
        data_type_t new_lambda = fsmlib::inner_product(v, Av); // Compute the Rayleigh quotient

        // Normalize Av
        norm = std::sqrt(fsmlib::inner_product(Av, Av));
        for (std::size_t i = 0; i < N; ++i) {
            v[i] = Av[i] / norm;
        }

        // Check for convergence in eigenvalue
        if (std::abs(new_lambda - lambda) < tolerance) {
            lambda = new_lambda;
            break;
        }

        lambda = new_lambda;
    }

    // Normalize the eigenvector sign to ensure consistency
    data_type_t max_val = *std::max_element(v.begin(), v.end(), [](T a, T b) { return std::abs(a) < std::abs(b); });
    if (max_val < 0) {
        // Ensure the largest absolute value is positive.
        for (std::size_t j = 0; j < N; ++j) {
            v[j] = -v[j];
        }
    }

    return std::make_pair(lambda, v);
}

/// @brief Deflates a matrix by subtracting the contribution of an eigenvector.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the square matrix.
/// @param mat The input matrix.
/// @param lambda The eigenvalue.
/// @param v The eigenvector.
/// @return The deflated matrix.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto
deflate(const fsmlib::MatrixBase<T, N, N> &mat, T lambda, const fsmlib::Vector<T, N> &v)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Normalize the eigenvector
    data_type_t norm                            = std::sqrt(fsmlib::inner_product(v, v));
    fsmlib::Vector<data_type_t, N> normalized_v = v;
    for (std::size_t i = 0; i < N; ++i) {
        normalized_v[i] /= norm;
    }

    // Compute the outer product
    auto vvT = fsmlib::outer_product(normalized_v, normalized_v);

    // Subtract the contribution of the eigenpair
    fsmlib::Matrix<data_type_t, N, N> result = mat;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            result(i, j) -= lambda * vvT(i, j);
        }
    }

    return result;
}

/// @brief Sorts eigenvalues and reorders eigenvectors accordingly.
/// @tparam T The type of the elements.
/// @tparam N The size of the matrix.
/// @param eigenvalues The vector of eigenvalues.
/// @param eigenvectors The matrix of eigenvectors.
/// @return A pair of sorted eigenvalues and reordered eigenvectors.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto sort_eigenpairs(fsmlib::Vector<T, N> eigenvalues,
                                                    fsmlib::Matrix<T, N, N> eigenvectors)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Create a vector of pairs to store eigenvalue-eigenvector pairs.
    std::array<std::pair<data_type_t, fsmlib::Vector<data_type_t, N>>, N> eig_pairs;

    for (std::size_t i = 0; i < N; ++i) {
        eig_pairs[i].first  = eigenvalues[i];
        eig_pairs[i].second = fsmlib::column(eigenvectors, i);
    }

    std::sort(eig_pairs.begin(), eig_pairs.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

    for (std::size_t i = 0; i < N; ++i) {
        eigenvalues[i] = eig_pairs[i].first;
        for (std::size_t j = 0; j < N; ++j) {
            eigenvectors(j, i) = eig_pairs[i].second[j];
        }
    }

    // Step 1: Sort eigenvalues and reorder eigenvectors
    auto indices = fsmlib::sort_indices<false>(eigenvalues);
    eigenvalues  = fsmlib::reorder(eigenvalues, indices);
    eigenvectors = fsmlib::reorder<true>(eigenvectors, indices);

    return std::make_pair(eigenvalues, eigenvectors);
}

/// @brief Computes the matrix exponential using a scaling and squaring method.
/// @param A The input matrix.
/// @param accuracy The desired accuracy (e.g., 1e-05).
/// @returns The exponential of the matrix.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto expm(const fsmlib::MatrixBase<T, N, N> &A, double accuracy)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Scale the matrix to have unit norm.
    const auto [iterations, scale] = fsmlib::linalg::scale_to_unit_norm(A);
    const auto scaled_a            = A * scale;

    // Initialize the series expansion.
    auto term           = fsmlib::eye<data_type_t, N, N>();
    auto ret            = fsmlib::eye<data_type_t, N, N>();
    data_type_t fac_inv = 1.0;

    // Max iterations for safety.
    for (std::size_t k = 1; k <= 1000; ++k) {
        fac_inv /= static_cast<double>(k);
        term = fsmlib::multiply(term, scaled_a) * fac_inv;
        ret  = ret + term;
        if (fsmlib::linalg::square_norm(term) < (accuracy * scale)) {
            break;
        }
    }

    // Raise to power 2^iterations.
    for (std::size_t k = 0; k < iterations; ++k) {
        ret = fsmlib::multiply(ret, ret);
    }
    return ret;
}

/// @brief Computes the matrix power A^p for a square matrix A.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the square matrix A (N x N).
/// @param A The input square matrix.
/// @param p The non-negative integer power to raise the matrix to.
/// @return The resulting matrix A^p.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto powm(const fsmlib::MatrixBase<T, N, N> &A, std::size_t p)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Initialize the result as the identity matrix.
    auto result = fsmlib::eye<data_type_t, N, N>();

    // Temporary variable to store intermediate results.
    fsmlib::Matrix<data_type_t, N, N> base = A;

    // Perform exponentiation by squaring.
    while (p > 0) {
        if (p % 2 == 1) {
            // Multiply result by the current base matrix.
            Matrix<data_type_t, N, N> temp = {};
            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t j = 0; j < N; ++j) {
                    data_type_t sum = 0;
                    for (std::size_t k = 0; k < N; ++k) {
                        sum += result(i, k) * base(k, j);
                    }
                    temp(i, j) = sum;
                }
            }
            result = temp;
        }

        // Square the base matrix.
        Matrix<data_type_t, N, N> temp = {};
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                data_type_t sum = 0;
                for (std::size_t k = 0; k < N; ++k) {
                    sum += base(i, k) * base(k, j);
                }
                temp(i, j) = sum;
            }
        }
        base = temp;

        // Divide the power by 2
        p /= 2;
    }

    return result;
}

/// @brief Computes the eigenvalues and eigenvectors of a symmetric matrix.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the square matrix (N x N).
/// @param mat The input symmetric matrix.
/// @param max_iterations The maximum number of iterations.
/// @param tolerance The convergence tolerance.
/// @return A pair consisting of:
///         - A vector of eigenvalues.
///         - A matrix whose columns are the corresponding eigenvectors.
/// @throws std::runtime_error If the input matrix is not symmetric.
/// @details
/// This function calculates the eigenvalues and eigenvectors of a symmetric
/// matrix using a combination of power iteration and Rayleigh quotient
/// iteration for dominant eigenpair extraction, followed by deflation to
/// compute the remaining eigenpairs.
///
/// The input matrix must be symmetric (\( A_{ij} = A_{ji} \)), which guarantees
/// real eigenvalues and orthogonal eigenvectors. The output is as follows:
/// - The eigenvalues are returned in a vector, sorted in descending order of magnitude.
/// - The eigenvectors are returned as columns of the resulting matrix.
///
/// The decomposition satisfies the equation:
/// \f[
/// A * v = \lambda * v
/// \f]
/// where \f$ \lambda \f$ is an eigenvalue, and \f$ v \f$ is the corresponding eigenvector.
///
/// Example usage:
/// @code
/// fsmlib::Matrix<double, 3, 3> A = {
///     { { 6.0, 2.0, 1.0 },
///       { 2.0, 3.0, 1.0 },
///       { 1.0, 1.0, 1.0 } }
/// };
/// auto [eigenvalues, eigenvectors] = fsmlib::linalg::eigen(A);
/// std::cout << "Eigenvalues:\n" << eigenvalues << "\n";
/// std::cout << "Eigenvectors:\n" << eigenvectors << "\n";
/// @endcode
/// @note The matrix must be symmetric for this function to work correctly.
/// @note This implementation is optimized for fixed-size matrices and may not
/// handle large, ill-conditioned matrices efficiently.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto
eigen(const fsmlib::MatrixBase<T, N, N> &mat, std::size_t max_iterations = 1000, T tolerance = 1e-24)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    if (!fsmlib::is_symmetric(mat, tolerance)) {
        throw std::runtime_error("Eigen decomposition: Matrix is not symmetric.");
    }

    // Initialize the eigenvectors matrix.
    auto eigenvectors = fsmlib::zeros<data_type_t, N, N>();
    // Initialize the eigenvalues vector.
    fsmlib::Vector<data_type_t, N> eigenvalues = {};

    // Balance the matrix to improve numerical stability.
    auto A = fsmlib::linalg::balance(mat);

    // Scale the matrix to have unit norm.
    auto [scale_iterations, scale_factor] = fsmlib::linalg::scale_to_unit_norm(A);

    // Scale the matrix.
    A *= scale_factor;

    for (std::size_t k = 0; k < N; ++k) {
        auto [lambda, v] = fsmlib::linalg::power_iteration(A, max_iterations, tolerance);

        eigenvalues[k] = lambda;
        for (std::size_t i = 0; i < N; ++i) {
            eigenvectors(i, k) = v[i];
        }

        A = fsmlib::linalg::deflate(A, lambda, v);
    }

    // Scale back eigenvalues.
    eigenvalues /= scale_factor;

    auto indices = fsmlib::sort_indices<false>(eigenvalues);
    eigenvalues  = fsmlib::reorder(eigenvalues, indices);
    eigenvectors = fsmlib::reorder<true>(eigenvectors, indices);

    // return fsmlib::linalg::sort_eigenpairs(eigenvalues, eigenvectors);
    return std::make_pair(eigenvalues, eigenvectors);
}

/// @brief Computes the Singular Value Decomposition (SVD) of a matrix.
/// @tparam T The type of the matrix elements.
/// @tparam M The number of rows in the matrix.
/// @tparam N The number of columns in the matrix.
/// @param mat The input matrix.
/// @return A tuple containing:
///         - U     : Left singular vectors (MxM orthogonal matrix).
///         - Sigma : Singular values (diagonal elements in a vector).
///         - V     : Right singular vectors (NxN orthogonal matrix).
template <typename T, std::size_t M, std::size_t N>
[[nodiscard]] constexpr inline auto svd(const fsmlib::MatrixBase<T, M, N> &mat)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Compute the minimum dimension.
    constexpr std::size_t MinDim = std::min(M, N);

    // U: Left singular vectors.
    auto U = fsmlib::eye<data_type_t, M>();

    // V: Right singular vectors.
    auto V = fsmlib::eye<data_type_t, N>();

    // Sigma: Singular values (diagonal elements in a vector).
    fsmlib::Vector<data_type_t, MinDim> Sigma = {};

    // std::cout << "mat:\n" << mat << "\n";

    // Balance the matrix to improve numerical stability.
    auto A = fsmlib::linalg::balance(mat);

    // Step 1: Compute \( A^T A \) and \( A A^T \)
    auto AtA = fsmlib::multiply(fsmlib::linalg::transpose(A), A); // \( N \times N \)
    auto AAt = fsmlib::multiply(A, fsmlib::linalg::transpose(A)); // \( M \times M \)

    // Step 2: Compute eigenvalues and eigenvectors of \( A^T A \) (for V) and \( A A^T \) (for U)
    auto [eigenvalues_AtA, eigenvectors_AtA] = fsmlib::linalg::eigen(AtA);
    auto [eigenvalues_AAt, eigenvectors_AAt] = fsmlib::linalg::eigen(AAt);

    // Step 3: Singular values are square roots of eigenvalues of \( A^T A \) (or \( A A^T \))
    for (std::size_t i = 0; i < MinDim; ++i) {
        Sigma[i] = std::sqrt(std::abs(eigenvalues_AtA[i]));
    }

    // Step 4: Assign eigenvectors to U and V
    U = eigenvectors_AAt; // U is formed by eigenvectors of \( A A^T \)
    V = eigenvectors_AtA; // V is formed by eigenvectors of \( A^T A \)

    // Step 5: Sort singular values and reorder U and V
    auto indices = fsmlib::sort_indices<false>(Sigma);
    Sigma        = fsmlib::reorder(Sigma, indices);
    U            = fsmlib::reorder<true>(U, indices);
    V            = fsmlib::reorder<true>(V, indices);

    // Step 6: Normalize and Align Signs
    for (std::size_t i = 0; i < MinDim; ++i) {
        // Normalize columns of U and V
        data_type_t norm_U = std::sqrt(fsmlib::dot(fsmlib::column(U, i), fsmlib::column(U, i)));
        data_type_t norm_V = std::sqrt(fsmlib::dot(fsmlib::column(V, i), fsmlib::column(V, i)));
        for (std::size_t j = 0; j < M; ++j) {
            U(j, i) /= norm_U;
        }
        for (std::size_t j = 0; j < N; ++j) {
            V(j, i) /= norm_V;
        }

        // Align signs to ensure \( Av_i = \Sigma_i u_i \)
        auto Av   = fsmlib::multiply(A, fsmlib::column(V, i)); // Compute \( A v_i \)
        auto sign = (fsmlib::inner_product(Av, fsmlib::column(U, i)) < 0) ? -1.0 : 1.0;
        for (std::size_t j = 0; j < M; ++j) {
            U(j, i) *= sign;
        }
        for (std::size_t j = 0; j < N; ++j) {
            V(j, i) *= sign;
        }
    }

    return std::make_tuple(U, Sigma, V);
}

/// @brief Computes the coefficients of the polynomial whose roots are the elements of a.
/// @tparam T The type of the elements in the input vector.
/// @tparam N The size of the input vector.
/// @param a The input vector containing the roots of the polynomial.
/// @return A vector of coefficients of the polynomial.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto poly(const fsmlib::Vector<T, N> &a)
{
    // Initialize the coefficients vector with size N + 1 (degree of polynomial + 1).
    fsmlib::Vector<T, N + 1> c = {};
    // The leading coefficient is always 1 for monic polynomials.
    c[0] = 1;
    // Compute the polynomial coefficients using the recurrence relation.
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t i = j + 1; i >= 1; --i) {
            c[i] -= a[j] * c[i - 1];
        }
    }
    return c;
}

/// @brief Reduces a polynomial coefficient vector to a minimum number of terms
/// by stripping off any leading zeros.
/// @tparam T The type of the elements in the input vector.
/// @tparam N The size of the input vector.
/// @param a The input vector of coefficients.
/// @return A reduced vector with leading zeros removed.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto polyreduce(const fsmlib::Vector<T, N> &a)
{
    fsmlib::Vector<T, N> result = {};
    std::size_t first_nonzero   = N;
    // Find the first non-zero element
    for (std::size_t i = 0; i < N; ++i) {
        if (!fsmlib::feq::approximately_equal_to_zero(a[i])) {
            first_nonzero = i;
            break;
        }
    }
    // Copy the reduced coefficients to the result
    for (std::size_t i = first_nonzero; i < N; ++i) {
        result[i - first_nonzero] = a[i];
    }
    return result;
}

/// @brief Computes the characteristic polynomial of a square matrix.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the matrix (N x N).
/// @param A The input square matrix.
/// @return A vector of coefficients representing the characteristic polynomial.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto characteristic_poly(fsmlib::Matrix<T, N, N> A)
{
    // Select the right type.
    using data_type_t = std::remove_const_t<T>;

    // Create a vector to store the polynomial coefficients.
    fsmlib::Vector<data_type_t, N + 1> poly{};

    // Initialize the identity matrix.
    const auto I = fsmlib::eye<data_type_t, N>();

    // Initialize C = A.
    fsmlib::Matrix<data_type_t, N, N> C = A;

    // Compute coefficients iteratively.
    for (std::size_t k = 1; k <= N; ++k) {
        if (k > 1) {
            // Update C = A * (C + poly[k - 1] * I).
            C = fsmlib::multiply(A, C + poly[k - 1] * I);
        }
        // Compute the next coefficient.
        poly[k] = -fsmlib::trace(C) / static_cast<data_type_t>(k);
    }

    // Set the leading coefficient (s^N) to 1 for monic polynomials.
    poly[0] = static_cast<data_type_t>(1);

    return poly;
}

} // namespace linalg
} // namespace fsmlib
