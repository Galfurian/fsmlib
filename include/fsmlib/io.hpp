

#pragma once

#include "mx/base.hpp"
#include "mx/control.hpp"

#include <iostream>

// Overload the << operator for Vector
template <typename T, std::size_t N>
std::ostream &operator<<(std::ostream &os, const mx::Vector<T, N> &vec)
{
    os << "{ ";
    for (const auto &elem : vec) {
        os << elem << " ";
    }
    os << "}";
    return os;
}

// Overload the << operator for Matrix
template <typename T, std::size_t N1, std::size_t N2>
std::ostream &operator<<(std::ostream &os, const mx::Matrix<T, N1, N2> &mat)
{
    os << "{\n";
    for (const auto &row : mat) {
        os << "  " << row << "\n";
    }
    os << "}";
    return os;
}

/// @brief Output stream function.
/// @param os the stream.
/// @param ss the state space model.
/// @returns the original stream.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
inline std::ostream &operator<<(std::ostream &os, const mx::control::state_space_t<T, N_state, N_input, N_output> &ss)
{
    os << "A =\n"
       << ss.A << "\n"
       << "B =\n"
       << ss.B << "\n"
       << "C =\n"
       << ss.C << "\n"
       << "D =\n"
       << ss.D << "\n"
       << "Continuous-time state-space model.\n";
    return os;
}

/// @brief Output stream function.
/// @param os the stream.
/// @param ss the state space model.
/// @returns the original stream.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
inline std::ostream &operator<<(std::ostream &os, const mx::control::discrete_state_space_t<T, N_state, N_input, N_output> &ss)
{
    os << "A =\n"
       << ss.A << "\n"
       << "B =\n"
       << ss.B << "\n"
       << "C =\n"
       << ss.C << "\n"
       << "D =\n"
       << ss.D << "\n"
       << "Sample time: " << ss.sample_time << " seconds\n"
       << "Discrete-time state-space model.\n";
    return os;
}
