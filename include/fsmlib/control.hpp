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
struct state_space_t {
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
struct discrete_state_space_t {
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
inline auto c2d(const state_space_t<T, N_state, N_input, N_output> &sys, T sample_time)
{
    discrete_state_space_t<T, N_state, N_input, N_output> dsys;

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
inline void step(
    const discrete_state_space_t<T, N_state, N_input, N_output> &dsys,
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

} // namespace control

} // namespace fsmlib
