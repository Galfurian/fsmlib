/// @file io.hpp
/// @brief Defines output stream operators for vectors, matrices, and state-space
/// models.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include "fsmlib/fsmlib.hpp"
#include "fsmlib/control.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>

/// @brief Overload the << operator for Vector.
/// @tparam T The type of the vector elements.
/// @tparam N The number of elements in the vector.
/// @param os The output stream.
/// @param vec The vector to print.
/// @returns The output stream with the vector contents.
template <typename T, std::size_t N>
std::ostream &operator<<(std::ostream &os, const fsmlib::Vector<T, N> &vec)
{
    // Determine the maximum width of the elements for alignment
    std::size_t max_width = 0;
    for (std::size_t i = 0; i < N; ++i) {
        std::ostringstream ss;
        ss << std::setprecision(6) << std::fixed << vec[i];
        max_width = std::max(max_width, ss.str().length());
    }
    for (std::size_t i = 0; i < N; ++i) {
        os << std::setw(static_cast<int>(max_width)) << std::setprecision(6) << std::fixed << vec[i];
        // Space between elements.
        if (i < N - 1) {
            os << " ";
        }
    }
    return os;
}

/// @brief Overload the << operator for Matrix.
/// @tparam T The type of the matrix elements.
/// @tparam N1 The number of rows in the matrix.
/// @tparam N2 The number of columns in the matrix.
/// @param os The output stream.
/// @param mat The matrix to print.
/// @returns The output stream with the matrix contents.
template <typename T, std::size_t N1, std::size_t N2>
std::ostream &operator<<(std::ostream &os, const fsmlib::Matrix<T, N1, N2> &mat)
{
    // Determine the maximum width of the elements for alignment
    std::size_t max_width = 0;
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N2; ++c) {
            std::ostringstream ss;
            ss << std::setprecision(6) << std::fixed << mat[r][c];
            max_width = std::max(max_width, ss.str().length());
        }
    }
    // Format and print the matrix.
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N2; ++c) {
            os << std::setw(static_cast<int>(max_width)) << std::setprecision(6) << std::fixed << mat[r][c];
            // Space between columns.
            if (c < N2 - 1) {
                os << " ";
            }
        }
        os << "\n";
    }
    return os;
}

/// @brief Overload the << operator for continuous-time state-space models.
/// @tparam T The type of the state-space model elements.
/// @tparam N_state The number of state variables.
/// @tparam N_input The number of input variables.
/// @tparam N_output The number of output variables.
/// @param os The output stream.
/// @param ss The continuous-time state-space model.
/// @returns The output stream with the state-space model contents.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
inline std::ostream &operator<<(std::ostream &os, const fsmlib::control::state_space_t<T, N_state, N_input, N_output> &ss)
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

/// @brief Overload the << operator for discrete-time state-space models.
/// @tparam T The type of the state-space model elements.
/// @tparam N_state The number of state variables.
/// @tparam N_input The number of input variables.
/// @tparam N_output The number of output variables.
/// @param os The output stream.
/// @param ss The discrete-time state-space model.
/// @returns The output stream with the state-space model contents.
template <typename T, std::size_t N_state, std::size_t N_input, std::size_t N_output>
inline std::ostream &operator<<(std::ostream &os, const fsmlib::control::discrete_state_space_t<T, N_state, N_input, N_output> &ss)
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