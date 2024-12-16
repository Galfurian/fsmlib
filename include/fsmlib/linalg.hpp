/// @file linalg.hpp
/// @brief Provides linear algebra operations and overloads for vector and
/// matrix arithmetic.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include "fsmlib/math.hpp"
#include <array>

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
constexpr inline auto scale_to_unit_norm(const fsmlib::Matrix<T, N1, N2> &A)
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

/// @brief Computes the Frobenius norm of a matrix.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param mat The input matrix.
/// @return The Frobenius norm of the matrix.
template <typename T, std::size_t Rows, std::size_t Cols>
constexpr T frobenius_norm(const fsmlib::Matrix<T, Rows, Cols> &mat)
{
    T sum = 0;
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            sum += mat[i][j] * mat[i][j];
        }
    }
    return std::sqrt(sum);
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

/// @brief Computes the rank of a matrix using Gaussian elimination.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param mat The input matrix.
/// @return The rank of the matrix.
/// @details
/// This function calculates the rank of the input matrix by performing Gaussian elimination.
/// The rank is defined as the number of linearly independent rows or columns in the matrix.
/// A small tolerance (\( \epsilon = 1e-9 \)) is used to handle floating-point precision issues.
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
[[nodiscard]] constexpr std::size_t rank(fsmlib::Matrix<T, Rows, Cols> mat)
{
    constexpr T epsilon = 1e-9; // Small value to handle floating-point precision issues
    std::size_t rank    = 0;

    // Perform Gaussian elimination
    for (std::size_t col = 0, row = 0; col < Cols && row < Rows; ++col) {
        // Find the pivot
        std::size_t pivot = row;
        for (std::size_t i = row + 1; i < Rows; ++i) {
            if (std::abs(mat[i][col]) > std::abs(mat[pivot][col])) {
                pivot = i;
            }
        }

        // If the pivot is effectively zero, skip this column
        if (std::abs(mat[pivot][col]) < epsilon) {
            continue;
        }

        // Swap the current row with the pivot row
        for (std::size_t j = 0; j < Cols; ++j) {
            std::swap(mat[row][j], mat[pivot][j]);
        }

        // Normalize the pivot row
        T pivot_value = mat[row][col];
        for (std::size_t j = 0; j < Cols; ++j) {
            mat[row][j] /= pivot_value;
        }

        // Eliminate the column below the pivot
        for (std::size_t i = row + 1; i < Rows; ++i) {
            T factor = mat[i][col];
            for (std::size_t j = 0; j < Cols; ++j) {
                mat[i][j] -= factor * mat[row][j];
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
constexpr auto qr_decomposition(const Matrix<T, Rows, Cols> &A)
{
    static_assert(Rows >= Cols, "The input matrix must have at least as many rows as columns.");
    using DataType = std::remove_const_t<T>;
    // Initialize R as a copy of A and Q as the identity matrix.
    Matrix<DataType, Rows, Cols> R;
    Matrix<DataType, Rows, Rows> Q;
    for (std::size_t k = 0; k < Cols; ++k) {
        // Compute the k-th column of Q.
        for (std::size_t i = 0; i < Rows; ++i) {
            Q[i][k] = A[i][k];
        }
        for (std::size_t j = 0; j < k; ++j) {
            T dot_product = 0;
            for (std::size_t i = 0; i < Rows; ++i) {
                dot_product += Q[i][j] * A[i][k];
            }
            R[j][k] = dot_product;

            for (std::size_t i = 0; i < Rows; ++i) {
                Q[i][k] -= R[j][k] * Q[i][j];
            }
        }
        // Normalize the k-th column of Q.
        T norm = 0;
        for (std::size_t i = 0; i < Rows; ++i) {
            norm += Q[i][k] * Q[i][k];
        }
        R[k][k] = std::sqrt(norm);
        for (std::size_t i = 0; i < Rows; ++i) {
            Q[i][k] /= R[k][k];
        }
        if (R[k][k] > 0) {
            R[k][k] = -R[k][k];
            for (std::size_t i = 0; i < Rows; ++i) {
                Q[i][k] = -Q[i][k];
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
constexpr auto lu_decomposition(const Matrix<T, Rows, Cols> &A)
{
    static_assert(Rows == Cols, "LU decomposition requires a square matrix.");
    using DataType = std::remove_const_t<T>;

    // Initialize L and U as zero matrices.
    Matrix<DataType, Rows, Cols> L = {};
    Matrix<DataType, Rows, Cols> U = {};

    for (std::size_t i = 0; i < Rows; ++i) {
        // Compute the upper triangular matrix U.
        for (std::size_t k = i; k < Cols; ++k) {
            T sum = 0;
            for (std::size_t j = 0; j < i; ++j) {
                sum += L[i][j] * U[j][k];
            }
            U[i][k] = A[i][k] - sum;
        }

        // Compute the lower triangular matrix L.
        for (std::size_t k = i; k < Rows; ++k) {
            if (i == k) {
                // Diagonal elements of L are set to 1.
                L[i][i] = 1;
            } else {
                T sum = 0;
                for (std::size_t j = 0; j < i; ++j) {
                    sum += L[k][j] * U[j][i];
                }
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }

    return std::make_pair(L, U);
}

/// @brief Computes the Cholesky decomposition of a symmetric positive-definite matrix.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the square matrix (N x N).
/// @param mat The input symmetric positive-definite matrix.
/// @return A lower triangular matrix \( L \) such that \( A = L \cdot L^T \).
/// @throws std::runtime_error If the input matrix is not symmetric or not positive definite.
/// @details
/// This function computes the Cholesky decomposition, which decomposes a symmetric positive-definite
/// matrix \( A \) into the product of a lower triangular matrix \( L \) and its transpose \( L^T \),
/// i.e., \( A = L \cdot L^T \). The input matrix must satisfy the following conditions:
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
[[nodiscard]] constexpr fsmlib::Matrix<T, N, N> cholesky_decomposition(const fsmlib::Matrix<T, N, N> &mat)
{
    // Ensure the input matrix is symmetric
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (mat[i][j] != mat[j][i]) {
                throw std::runtime_error("Cholesky decomposition: Matrix is not symmetric.");
            }
        }
    }

    fsmlib::Matrix<T, N, N> lower = {}; // Lower triangular matrix

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            T sum = 0;

            // Summation for diagonal and non-diagonal elements
            for (std::size_t k = 0; k < j; ++k) {
                sum += lower[i][k] * lower[j][k];
            }

            if (i == j) {
                // Diagonal elements
                T diag = mat[i][i] - sum;
                if (diag <= 0) {
                    throw std::runtime_error("Cholesky decomposition: Matrix is not positive definite.");
                }
                lower[i][j] = std::sqrt(diag);
            } else {
                // Non-diagonal elements
                lower[i][j] = (mat[i][j] - sum) / lower[j][j];
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
constexpr auto solve(const Matrix<T, Rows, Cols> &A, const Vector<T, Rows> &b)
{
    static_assert(Rows == Cols, "solve requires a square matrix for LU decomposition.");

    // Perform LU decomposition
    auto [L, U] = lu_decomposition(A);

    // Forward substitution to solve Ly = b
    Vector<T, Rows> y;
    for (std::size_t i = 0; i < Rows; ++i) {
        T sum = 0;
        for (std::size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = b[i] - sum;
    }

    // Back substitution to solve Ux = y
    Vector<T, Rows> x;
    for (std::size_t i = Rows; i-- > 0;) {
        T sum = 0;
        for (std::size_t j = i + 1; j < Rows; ++j) {
            sum += U[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i][i];
    }

    return x;
}

/// @brief Computes the matrix exponential using a scaling and squaring method.
/// @param A The input matrix.
/// @param accuracy The desired accuracy (e.g., 1e-05).
/// @returns The exponential of the matrix.
template <typename T, std::size_t N>
inline auto expm(const fsmlib::Matrix<T, N, N> &A, double accuracy)
{
    const auto [iterations, scale] = fsmlib::linalg::scale_to_unit_norm(A);
    const auto scaled_a            = A * scale;

    // Initialize.
    fsmlib::Matrix<T, N, N> term = fsmlib::eye<T, N, N>();
    fsmlib::Matrix<T, N, N> ret  = fsmlib::eye<T, N, N>();
    T fac_inv                    = 1.0;

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
/// @tparam Size The size of the square matrix A (Size x Size).
/// @param A The input square matrix.
/// @param p The non-negative integer power to raise the matrix to.
/// @return The resulting matrix A^p.
template <typename T, std::size_t Size>
constexpr Matrix<T, Size, Size> powm(const Matrix<T, Size, Size> &A, std::size_t p)
{
    // Initialize the result as the identity matrix
    Matrix<T, Size, Size> result = {};
    for (std::size_t i = 0; i < Size; ++i) {
        result[i][i] = 1;
    }

    // Temporary variable to store intermediate results
    Matrix<T, Size, Size> base = A;

    // Perform exponentiation by squaring
    while (p > 0) {
        if (p % 2 == 1) {
            // Multiply result by the current base matrix
            Matrix<T, Size, Size> temp = {};
            for (std::size_t i = 0; i < Size; ++i) {
                for (std::size_t j = 0; j < Size; ++j) {
                    T sum = 0;
                    for (std::size_t k = 0; k < Size; ++k) {
                        sum += result[i][k] * base[k][j];
                    }
                    temp[i][j] = sum;
                }
            }
            result = temp;
        }

        // Square the base matrix
        Matrix<T, Size, Size> temp = {};
        for (std::size_t i = 0; i < Size; ++i) {
            for (std::size_t j = 0; j < Size; ++j) {
                T sum = 0;
                for (std::size_t k = 0; k < Size; ++k) {
                    sum += base[i][k] * base[k][j];
                }
                temp[i][j] = sum;
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
/// @param tol The convergence tolerance.
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
/// A \cdot v = \lambda \cdot v
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
constexpr std::pair<fsmlib::Vector<T, N>, fsmlib::Matrix<T, N, N>> eigen(const fsmlib::Matrix<T, N, N> &mat, std::size_t max_iterations = 1000, T tolerance = 1e-9)
{
    // Ensure the input matrix is symmetric.
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (std::abs(mat[i][j] - mat[j][i]) > tolerance) {
                throw std::runtime_error("Eigen decomposition: Matrix is not symmetric.");
            }
        }
    }

    fsmlib::Vector<T, N> eigenvalues     = {};                       // Eigenvalues
    fsmlib::Matrix<T, N, N> eigenvectors = fsmlib::zeros<T, N, N>(); // Eigenvectors

    // Copy of the matrix to perform deflation
    fsmlib::Matrix<T, N, N> A = mat;

    for (std::size_t k = 0; k < N; ++k) {
        // Start with a random initial vector
        fsmlib::Vector<T, N> v = fsmlib::ones<T, N>(); // Use a vector of ones as the initial guess
        v[0]                   = 1.0;                  // Ensure non-zero

        // Normalize the initial vector
        T norm = std::sqrt(fsmlib::inner_product(v, v));
        for (std::size_t i = 0; i < N; ++i) {
            v[i] /= norm;
        }

        T lambda = 0; // Eigenvalue approximation

        for (std::size_t iter = 0; iter < max_iterations; ++iter) {
            // Multiply A * v
            auto Av = fsmlib::multiply(A, v);

            // Compute Rayleigh quotient for the eigenvalue
            T new_lambda = fsmlib::inner_product(v, Av);

            // Normalize Av to get the next iteration of v
            norm = std::sqrt(fsmlib::inner_product(Av, Av));
            for (std::size_t i = 0; i < N; ++i) {
                v[i] = Av[i] / norm;
            }

            // Check for convergence
            if (std::abs(new_lambda - lambda) < tolerance) {
                break;
            }

            lambda = new_lambda;
        }

        // Store the eigenvalue and eigenvector
        eigenvalues[k] = lambda;
        for (std::size_t i = 0; i < N; ++i) {
            eigenvectors[i][k] = v[i];
        }

        // Deflate the matrix to find the next eigenpair
        auto vvT = fsmlib::outer_product(v, v);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                A[i][j] -= lambda * vvT[i][j];
            }
        }
    }

    // Sorting eigenvalues in ascending order and reordering eigenvectors
    std::array<std::pair<T, fsmlib::Vector<T, N>>, N> eig_pairs;
    for (std::size_t i = 0; i < N; ++i) {
        eig_pairs[i].first  = eigenvalues[i];
        eig_pairs[i].second = fsmlib::column(eigenvectors, i);
    }

    std::sort(eig_pairs.begin(), eig_pairs.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    for (std::size_t i = 0; i < N; ++i) {
        eigenvalues[i] = eig_pairs[i].first;
        for (std::size_t j = 0; j < N; ++j) {
            eigenvectors[j][i] = eig_pairs[i].second[j];
        }
    }

    return { eigenvalues, eigenvectors };
}

/// @brief Computes the dominant eigenvalue and eigenvector of a matrix using power iteration.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the square matrix.
/// @param mat The input matrix.
/// @param max_iterations The maximum number of iterations.
/// @param tolerance The convergence tolerance.
/// @return A pair containing:
///         - The dominant eigenvalue.
///         - The corresponding eigenvector.
template <typename T, std::size_t N>
constexpr std::pair<T, fsmlib::Vector<T, N>> power_iteration(const fsmlib::Matrix<T, N, N> &mat, std::size_t max_iterations = 1000, T tolerance = 1e-6)
{
    fsmlib::Vector<T, N> eigenvector = fsmlib::ones<T, N>();                      // Initial guess
    eigenvector                      = eigenvector / frobenius_norm(eigenvector); // Normalize

    T eigenvalue = 0;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        auto next_vector = mat * eigenvector; // Matrix-vector multiplication
        T next_norm      = fsmlib::linalg::frobenius_norm(next_vector);
        next_vector      = next_vector / next_norm; // Normalize

        if (std::abs(next_norm - eigenvalue) < tolerance) {
            break;
        }

        eigenvalue  = next_norm;
        eigenvector = next_vector;
    }

    return { eigenvalue, eigenvector };
}

} // namespace linalg
} // namespace fsmlib
