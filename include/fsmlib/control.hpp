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

/// @brief Namespace for control systems.
namespace control
{

/// @brief Represents a continuous-time state-space model.
/// @tparam T The type of the elements in the matrices.
/// @tparam N_state The number of states.
/// @tparam N_input The number of inputs.
/// @tparam N_output The number of outputs.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output> struct StateSpace {
    fsmlib::Matrix<T, N_state, N_state> A;  ///< System matrix.
    fsmlib::Matrix<T, N_state, N_input> B;  ///< Input matrix.
    fsmlib::Matrix<T, N_output, N_state> C; ///< Output matrix.
    fsmlib::Matrix<T, N_output, N_input> D; ///< Feedforward matrix.

    /// @brief Default constructor to initialize the continuous-time state-space model.
    StateSpace()
        : A{}
        , B{}
        , C{}
        , D{}
    {
        // Nothing to do.
    }

    /// @brief Constructor to initialize the continuous-time state-space model with given matrices.
    /// @param _A System matrix.
    /// @param _B Input matrix.
    /// @param _C Output matrix.
    /// @param _D Feedforward matrix.
    StateSpace(
        const fsmlib::Matrix<T, N_state, N_state> &_A,
        const fsmlib::Matrix<T, N_state, N_input> &_B,
        const fsmlib::Matrix<T, N_output, N_state> &_C,
        const fsmlib::Matrix<T, N_output, N_input> &_D)
        : A(_A)
        , B(_B)
        , C(_C)
        , D(_D)
    {
        // Nothing to do.
    }
};

/// @brief Represents a discrete-time state-space model, extending the continuous model.
/// @tparam T The type of the elements in the matrices.
/// @tparam N_state The number of states.
/// @tparam N_input The number of inputs.
/// @tparam N_output The number of outputs.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output> struct DiscreteStateSpace {
    fsmlib::Matrix<T, N_state, N_state> A;  ///< System matrix.
    fsmlib::Matrix<T, N_state, N_input> B;  ///< Input matrix.
    fsmlib::Matrix<T, N_output, N_state> C; ///< Output matrix.
    fsmlib::Matrix<T, N_output, N_input> D; ///< Feedforward matrix.
    T sample_time;                          ///< The sample time used for discretization.

    /// @brief Default constructor to initialize the discrete-time state-space model.
    DiscreteStateSpace()
        : A{}
        , B{}
        , C{}
        , D{}
        , sample_time{}
    {
        // Nothing to do.
    }

    /// @brief Constructor to initialize the discrete-time state-space model with given matrices.
    /// @param _A System matrix.
    /// @param _B Input matrix.
    /// @param _C Output matrix.
    /// @param _D Feedforward matrix.
    /// @param _sample_time The sample time used for discretization.
    DiscreteStateSpace(
        const fsmlib::Matrix<T, N_state, N_state> &_A,
        const fsmlib::Matrix<T, N_state, N_input> &_B,
        const fsmlib::Matrix<T, N_output, N_state> &_C,
        const fsmlib::Matrix<T, N_output, N_input> &_D,
        T _sample_time)
        : A(_A)
        , B(_B)
        , C(_C)
        , D(_D)
        , sample_time(_sample_time)
    {
        // Nothing to do.
    }
};

/// @brief Represents a single transfer function for a MIMO system.
/// @tparam T The type of the coefficients (e.g., double, float).
/// @tparam N_num The number of numerator coefficients (polynomial order + 1).
/// @tparam N_den The number of denominator coefficients (polynomial order + 1).
template <typename T, std::size_t N_num, std::size_t N_den> struct TransferFunction {
    fsmlib::Vector<T, N_num> numerator;   ///< Fixed-size vector for numerator coefficients.
    fsmlib::Vector<T, N_den> denominator; ///< Fixed-size vector for denominator coefficients.

    /// @brief Constructor to initialize a transfer function with zero coefficients.
    constexpr TransferFunction()
        : numerator{}
        , denominator{}
    {
    }

    /// @brief Constructor to initialize a transfer function with given coefficients.
    /// @param num The numerator coefficients.
    /// @param den The denominator coefficients.
    constexpr TransferFunction(const fsmlib::Vector<T, N_num> &num, const fsmlib::Vector<T, N_den> &den)
        : numerator(num)
        , denominator(den)
    {
    }
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
[[nodiscard]]
constexpr inline auto c2d(const StateSpace<T, N_state, N_input, N_output> &sys, T sample_time)
{
    DiscreteStateSpace<T, N_state, N_input, N_output> dsys;

    // Discretize the system matrix using matrix exponential.
    dsys.A = fsmlib::linalg::expm(sys.A * sample_time, 1e-06);

    // Discretize the input matrix using the zero-order hold method.
    dsys.B =
        fsmlib::multiply(fsmlib::multiply(fsmlib::linalg::inverse(sys.A), dsys.A - fsmlib::eye<T, N_state>()), sys.B);

    // Copy the output and feedforward matrices.
    dsys.C = sys.C;
    dsys.D = sys.D;

    // Set the sample time.
    dsys.sample_time = sample_time;

    return dsys;
}

/// @brief Simulates one step of a state-space model.
/// @tparam T The type of the elements in the matrices and vectors.
/// @tparam N_state The number of states.
/// @tparam N_input The number of inputs.
/// @tparam N_output The number of outputs.
/// @param sys The state-space model.
/// @param x The current state vector.
/// @param u The input vector.
/// @param next Reference to a vector that will hold the next state after simulation.
/// @param y Reference to a vector that will hold the computed output.
/// @details This function computes the next state and output of a state-space model
/// based on the provided current state and input. The next state is computed as:
///     x_{next} = A * x + B * u
/// and the output is computed as:
///     y = C * x + D * u
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
constexpr inline void step(
    const StateSpace<T, N_state, N_input, N_output> &sys,
    const Vector<T, N_state> &x,
    const Vector<T, N_input> &u,
    Vector<T, N_state> &next,
    Vector<T, N_output> &y)
{
    // Compute the next state: x_next = A * x + B * u.
    next = fsmlib::multiply(sys.A, x) + fsmlib::multiply(sys.B, u);
    // Compute the output: y = C * x + D * u.
    y    = fsmlib::multiply(sys.C, x) + fsmlib::multiply(sys.D, u);
}

/// @brief Simulates one step of a state-space model.
/// @tparam T The type of the elements in the matrices and vectors.
/// @tparam N_state The number of states.
/// @tparam N_input The number of inputs.
/// @tparam N_output The number of outputs.
/// @param sys The state-space model.
/// @param x The current state vector.
/// @param u The input vector.
/// @param next Reference to a vector that will hold the next state after simulation.
/// @param y Reference to a vector that will hold the computed output.
/// @details This function computes the next state and output of a state-space model
/// based on the provided current state and input. The next state is computed as:
///  x_{next} = A * x + B * u
/// and the output is computed as:
///  y = C * x + D * u .
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
constexpr inline void dstep(
    const DiscreteStateSpace<T, N_state, N_input, N_output> &sys,
    const Vector<T, N_state> &x,
    const Vector<T, N_input> &u,
    Vector<T, N_state> &next,
    Vector<T, N_output> &y)
{
    // Compute the next state: x_next = A * x + B * u.
    next = fsmlib::multiply(sys.A, x) + fsmlib::multiply(sys.B, u);
    // Compute the output: y = C * x + D * u.
    y    = fsmlib::multiply(sys.C, x) + fsmlib::multiply(sys.D, u);
}

/// @brief Simulates the continuous-time state-space model.
/// @tparam T The type of the elements in the matrices.
/// @tparam N_state The number of states.
/// @tparam N_input The number of inputs.
/// @tparam N_output The number of outputs.
/// @tparam TimeSteps The number of time steps.
/// @param ss The state-space model.
/// @param input The input signal.
/// @param x0 The initial state vector.
/// @param time_step The time step for numerical integration.
/// @return The system output as a matrix.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output, std::size_t TimeSteps>
fsmlib::Matrix<T, N_output, TimeSteps> lsim(
    const StateSpace<T, N_state, N_input, N_output> &ss,
    const fsmlib::Matrix<T, N_input, TimeSteps> &input,
    const fsmlib::Vector<T, N_state> &x0,
    T time_step)
{ // Output matrix to store the results.
    fsmlib::Matrix<T, N_output, TimeSteps> output{};
    // State vector initialized to the initial state x0.
    fsmlib::Vector<T, N_state> x = x0;
    // Temporary vector to hold the state derivative (x_dot).
    fsmlib::Vector<T, N_state> x_dot{};
    // Iterate over the time steps.
    for (std::size_t k = 0; k < TimeSteps; ++k) {
        // Extract the column vector for the current time step input.
        fsmlib::Vector<T, N_input> u{};
        for (std::size_t i = 0; i < N_input; ++i) {
            u[i] = input(i, k);
        }
        // Extract the column vector for the current time step output.
        fsmlib::Vector<T, N_output> y{};
        // Compute the state derivative and output using the step function.
        fsmlib::control::step(ss, x, u, x_dot, y);
        // Store the computed output back into the output matrix.
        for (std::size_t i = 0; i < N_output; ++i) {
            output(i, k) = y[i];
        }
        // Scale the state derivative by the time step for the continuous update.
        x += x_dot * time_step;
    }
    return output;
}

/// @brief Simulates the discrete-time state-space model.
/// @tparam T The type of the elements in the matrices.
/// @tparam N_state The number of states.
/// @tparam N_input The number of inputs.
/// @tparam N_output The number of outputs.
/// @tparam TimeSteps The number of time steps.
/// @param dsys The discrete-time state-space model.
/// @param input The input signal.
/// @param x0 The initial state vector.
/// @return The system output as a matrix.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output, std::size_t TimeSteps>
fsmlib::Matrix<T, N_output, TimeSteps> dlsim(
    const DiscreteStateSpace<T, N_state, N_input, N_output> &dsys,
    const fsmlib::Matrix<T, N_input, TimeSteps> &input,
    const fsmlib::Vector<T, N_state> &x0)
{
    // Output matrix to store the results.
    fsmlib::Matrix<T, N_output, TimeSteps> output{};
    // State vector initialized to the initial state x0.
    fsmlib::Vector<T, N_state> x = x0;
    // Temporary vectors for the next state and output.
    fsmlib::Vector<T, N_state> x_next{};
    fsmlib::Vector<T, N_input> u{};
    fsmlib::Vector<T, N_output> y{};
    // Iterate over time steps.
    for (std::size_t k = 0; k < TimeSteps; ++k) {
        // Extract the column vector for the current time step input.
        for (std::size_t i = 0; i < N_input; ++i) {
            u[i] = input(i, k);
        }
        // Compute the next state and output using the step function.
        fsmlib::control::dstep(dsys, x, u, x_next, y);
        // Store the computed output back into the output matrix.
        for (std::size_t i = 0; i < N_output; ++i) {
            output(i, k) = y[i];
        }
        // Update the current state to the next state.
        x = x_next;
    }
    return output;
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
[[nodiscard]]
constexpr inline auto ctrb(const Matrix<T, N, N> &A, const Matrix<T, N, Q> &B)
{
    // Initialize the controllability matrix with zeros.
    Matrix<T, N, N * Q> result = {};
    // Copy the first block (B) into the result.
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < Q; ++j) {
            result(i, j) = B(i, j);
        }
    }
    // Construct the controllability matrix.
    for (std::size_t p = 1; p < N; ++p) {
        // Compute A^p * B.
        auto Ap  = fsmlib::linalg::powm(A, p); // Compute A^p.
        auto ApB = fsmlib::multiply(Ap, B);    // Multiply A^p with B.
        // Insert ApB into the result matrix.
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < Q; ++j) {
                result(i, p * Q + j) = ApB(i, j);
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
[[nodiscard]]
constexpr inline auto obsv(const Matrix<T, N, N> &A, const Matrix<T, P, N> &C)
{
    // Initialize the observability matrix with the first block (C).
    Matrix<T, P * N, N> result = {};
    // Copy the first block (C) into the result.
    for (std::size_t i = 0; i < P; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            result(i, j) = C(i, j);
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
                result(p * P + i, j) = CA(i, j);
            }
        }
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
[[nodiscard]]
constexpr inline auto acker(
    const fsmlib::Matrix<T, Rows, Rows> &A,
    const fsmlib::Matrix<T, Rows, Cols> &B,
    const fsmlib::Vector<T, NumPoles> &poles)
{
    static_assert(Rows == NumPoles, "The number of poles must match the system order.");
    // Ensure the system is controllable.
    auto ct = ctrb(A, B);
    if (fsmlib::feq::approximately_equal_to_zero(fsmlib::linalg::determinant(ct))) {
        throw std::runtime_error("acker: system not controllable, pole placement invalid.");
    }
    // Compute the desired characteristic polynomial.
    auto p                           = fsmlib::linalg::poly(poles);
    // Construct the Ackermann matrix.
    fsmlib::Matrix<T, Rows, Rows> Ap = {};
    for (std::size_t i = 0; i < (Rows + 1); ++i) {
        Ap += fsmlib::linalg::powm(A, Rows - i) * p[i];
    }
    // Selection matrix to extract the last row of the controllability matrix
    fsmlib::Matrix<T, 1, Rows> selection = {};
    selection(0, Rows - 1)               = 1;
    // Compute the gain matrix.
    return fsmlib::multiply(selection, fsmlib::multiply(fsmlib::linalg::inverse(ct), Ap));
}

} // namespace control

} // namespace fsmlib
