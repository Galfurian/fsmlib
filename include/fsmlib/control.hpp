/// @file control.hpp
/// @brief Provides data structures and functions for continuous and discrete
/// state-space models.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include "fsmlib/linalg.hpp"

namespace fsmlib
{

namespace control
{

/// @brief Represents a continuous-time state-space model.
/// @tparam T The type of the elements in the matrices.
/// @tparam N_state The number of states.
/// @tparam N_input The number of inputs.
/// @tparam N_output The number of outputs.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
struct StateSpace {
    fsmlib::Matrix<T, N_state, N_state> A;  ///< System matrix.
    fsmlib::Matrix<T, N_state, N_input> B;  ///< Input matrix.
    fsmlib::Matrix<T, N_output, N_state> C; ///< Output matrix.
    fsmlib::Matrix<T, N_output, N_input> D; ///< Feedforward matrix.
};

/// @brief Represents a discrete-time state-space model, extending the continuous model.
/// @tparam T The type of the elements in the matrices.
/// @tparam N_state The number of states.
/// @tparam N_input The number of inputs.
/// @tparam N_output The number of outputs.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
struct DiscreteStateSpace {
    fsmlib::Matrix<T, N_state, N_state> A;  ///< System matrix.
    fsmlib::Matrix<T, N_state, N_input> B;  ///< Input matrix.
    fsmlib::Matrix<T, N_output, N_state> C; ///< Output matrix.
    fsmlib::Matrix<T, N_output, N_input> D; ///< Feedforward matrix.
    T sample_time;                          ///< The sample time used for discretization.
};

/// @brief Discretizes a continuous-time state-space model using a given sample time.
/// @tparam T The type of the elements in the matrices.
/// @tparam N_state The number of states.
/// @tparam N_input The number of inputs.
/// @tparam N_output The number of outputs.
/// @param sys The continuous-time state-space model.
/// @param sample_time The sample time for discretization.
/// @returns The discretized state-space model.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
[[nodiscard]] constexpr inline auto
c2d(const StateSpace<T, N_state, N_input, N_output> &sys, T sample_time)
{
    DiscreteStateSpace<T, N_state, N_input, N_output> dsys;

    // Discretize the system matrix using matrix exponential.
    dsys.A = fsmlib::linalg::expm(sys.A * sample_time, 1e-06);

    // Discretize the input matrix using the zero-order hold method.
    dsys.B = fsmlib::multiply(
        fsmlib::multiply(fsmlib::linalg::inverse(sys.A), dsys.A - fsmlib::eye<T, N_state>()), sys.B);

    // Copy the output and feedforward matrices.
    dsys.C = sys.C;
    dsys.D = sys.D;

    // Set the sample time.
    dsys.sample_time = sample_time;

    return dsys;
}

/// @brief Simulates one step of a discrete-time state-space model.
/// @tparam T The type of the elements in the matrices and vectors.
/// @tparam N_state The number of states.
/// @tparam N_input The number of inputs.
/// @tparam N_output The number of outputs.
/// @param dsys The discretized state-space model.
/// @param x The current state vector.
/// @param u The input vector.
/// @param state Reference to a vector that will hold the next state after simulation.
/// @param output Reference to a vector that will hold the computed output.
/// @details This function computes the next state and output of a discrete-time state-space model
/// based on the provided current state and input. The next state is computed as:
/// \f$ x_{next} = A \cdot x + B \cdot u \f$
/// and the output is computed as:
/// \f$ y = C \cdot x + D \cdot u \f$.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
constexpr inline void
step(
    const DiscreteStateSpace<T, N_state, N_input, N_output> &dsys,
    const Vector<T, N_state> &x,
    const Vector<T, N_input> &u,
    Vector<T, N_state> &state,
    Vector<T, N_output> &output)
{
    // Compute the next state: x_next = A * x + B * u
    state = fsmlib::multiply(dsys.A, x) + fsmlib::multiply(dsys.B, u);
    // Compute the output: y = C * x + D * u
    output = fsmlib::multiply(dsys.C, x) + fsmlib::multiply(dsys.D, u);
}

/// @brief Computes the controllability matrix of a state-space system.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the square matrix A (NxN).
/// @tparam Q The number of columns in the input matrix B (NxQ).
/// @param A The state matrix of the system (NxN).
/// @param B The input matrix of the system (NxQ).
/// @return The controllability matrix.
/// @details CM = [ B    A*B    A^2*B    ...    A^(N-1)*B ]
template <typename T, std::size_t N, std::size_t Q>
[[nodiscard]] constexpr inline auto
ctrb(const Matrix<T, N, N> &A, const Matrix<T, N, Q> &B)
{
    // Initialize the controllability matrix with zeros.
    Matrix<T, N, N * Q> result = {};
    // Copy the first block (B) into the result.
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < Q; ++j) {
            result[i][j] = B[i][j];
        }
    }
    // Construct the controllability matrix
    for (std::size_t p = 1; p < N; ++p) {
        // Compute A^p * B
        auto Ap  = fsmlib::linalg::powm(A, p); // Compute A^p
        auto ApB = fsmlib::multiply(Ap, B);    // Multiply A^p with B
        // Insert ApB into the result matrix
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < Q; ++j) {
                result[i][p * Q + j] = ApB[i][j];
            }
        }
    }
    return result;
}

/// @brief Computes the observability matrix of a state-space system.
/// @tparam T The type of the matrix elements.
/// @tparam N The size of the square matrix A (NxN).
/// @tparam P The number of rows in the output matrix C (PxN).
/// @param A The state matrix of the system (NxN).
/// @param C The output matrix of the system (PxN).
/// @return The observability matrix.
/// @details
/// The observability matrix is constructed as:
///      | C         |
///      | C*A       |
/// OM = | C*A^2     |
///      | ...       |
///      | C*A^(n-1) |
template <typename T, std::size_t N, std::size_t P>
[[nodiscard]] constexpr inline auto
obsv(const Matrix<T, N, N> &A, const Matrix<T, P, N> &C)
{
    // Initialize the observability matrix with the first block (C).
    Matrix<T, P * N, N> result = {};
    // Copy the first block (C) into the result.
    for (std::size_t i = 0; i < P; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            result[i][j] = C[i][j];
        }
    }
    // Construct the observability matrix.
    for (std::size_t p = 1; p < N; ++p) {
        // Compute C * A^p
        auto Ap = fsmlib::linalg::powm(A, p); // Compute A^p.
        auto CA = fsmlib::multiply(C, Ap);    // Multiply C with A^p.

        // Insert CA into the result matrix.
        for (std::size_t i = 0; i < P; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                result[p * P + i][j] = CA[i][j];
            }
        }
    }
    return result;
}

/// @brief Computes the coefficients of the polynomial whose roots are the elements of a.
/// @tparam T The type of the elements in the input vector.
/// @tparam N The size of the input vector.
/// @param a The input vector containing the roots of the polynomial.
/// @return A vector of coefficients of the polynomial.
template <typename T, std::size_t N>
[[nodiscard]] constexpr inline auto
poly(const fsmlib::Vector<T, N> &a)
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
[[nodiscard]] constexpr inline auto
polyreduce(const fsmlib::Vector<T, N> &a)
{
    fsmlib::Vector<T, N> result = {};
    std::size_t first_nonzero   = N;
    // Find the first non-zero element
    for (std::size_t i = 0; i < N; ++i) {
        if (a[i] != 0) {
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

/// @brief Pole placement using Ackermann method.
/// @tparam T The type of the elements in the matrices.
/// @tparam Rows The number of rows in the state matrix A.
/// @tparam Cols The number of columns in the input matrix B.
/// @tparam NumPoles The number of desired poles.
/// @param A The state matrix of the system (Rows x Rows).
/// @param B The input matrix of the system (Rows x Cols).
/// @param poles The desired poles for generating the closed-loop behavior (fixed-size vector).
/// @return Gains \( K \) such that \( A - BK \) has the given eigenvalues.
template <typename T, std::size_t Rows, std::size_t Cols, std::size_t NumPoles>
[[nodiscard]] constexpr inline auto
acker(const fsmlib::Matrix<T, Rows, Rows> &A, const fsmlib::Matrix<T, Rows, Cols> &B, const fsmlib::Vector<T, NumPoles> &poles)
{
    static_assert(Rows == NumPoles, "The number of poles must match the system order.");
    // Ensure the system is controllable.
    auto ct = ctrb(A, B);
    if (fsmlib::linalg::determinant(ct) == 0) {
        throw std::runtime_error("acker: system not controllable, pole placement invalid.");
    }
    // Compute the desired characteristic polynomial.
    auto p = poly(poles);
    // Construct the Ackermann matrix.
    fsmlib::Matrix<T, Rows, Rows> Ap = {};
    for (std::size_t i = 0; i < (Rows + 1); ++i) {
        Ap += fsmlib::linalg::powm(A, Rows - i) * p[i];
    }
    // Selection matrix to extract the last row of the controllability matrix
    fsmlib::Matrix<T, 1, Rows> selection = {};
    selection[0][Rows - 1]               = 1;
    // Compute the gain matrix.
    return fsmlib::multiply(selection, fsmlib::multiply(fsmlib::linalg::inverse(ct), Ap));
}

} // namespace control

} // namespace fsmlib
