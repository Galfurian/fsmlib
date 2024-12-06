

#pragma once

#include "mx/math.hpp"
#include "mx/linalg.hpp"

namespace mx
{

namespace control
{

/// @brief Continuous-time state space model.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
struct state_space_t {
    mx::Matrix<T, N_state, N_state> A;  ///< System matrix.
    mx::Matrix<T, N_state, N_input> B;  ///< Input matrix.
    mx::Matrix<T, N_output, N_state> C; ///< Output matrix.
    mx::Matrix<T, N_output, N_input> D; ///< Feedforward matrix.
};

/// @brief Continuous-time state space model.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
struct discrete_state_space_t : state_space_t<T, N_state, N_input, N_output> {
    T sample_time; ///< Sample time used for the discretization.
};

/// @brief Discretize the state space model, with the given time-step.
/// @param sys the continuous-time state space model.
/// @param sample_time the sample time step.
/// @returns the discretized state space.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
inline auto c2d(const state_space_t<T, N_state, N_input, N_output> &sys, T sample_time)
{
    // Create the discrete time state-space model.
    discrete_state_space_t<T, N_state, N_input, N_output> dsys;
    // Compute the discretized matrices.
    dsys.A = mx::linalg::expm(sys.A * sample_time, 1e-05);
    dsys.B = mx::multiply(mx::multiply(mx::linalg::inverse(sys.A), dsys.A - mx::details::eye<T, N_state>()), sys.B);
    dsys.C = sys.C;
    dsys.D = sys.D;
    // Set the sampling period.
    dsys.sample_time = sample_time;
    return dsys;
}

/// @brief Simulate one step with the given discretized system.
/// @param sys the system to simulate.
/// @param x the current state.
/// @param u the current input.
/// @returns a tuple containing the next state, and the output.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
inline std::pair<mx::Vector<T, N_state>, Vector<T, N_state>> step(
    const state_space_t<T, N_state, N_input, N_output> &dsys,
    const Vector<T, N_state> &x,
    const Vector<T, N_input> &u)
{
    return std::make_pair(
        mx::multiply(dsys.A, x) + mx::multiply(dsys.B, u),
        mx::multiply(dsys.C, x) + mx::multiply(dsys.D, u));
}

} // namespace control

} // namespace mx
