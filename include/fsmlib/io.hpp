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
/// @return The output stream with the vector contents.
template <typename T, std::size_t N>
std::ostream &operator<<(std::ostream &os, const fsmlib::Vector<T, N> &vec)
{
    // Determine the maximum width of the elements for alignment.
    std::size_t max_width = 0;
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::ostringstream ss;
        ss << std::setprecision(6) << std::fixed << vec[i];
        max_width = std::max(max_width, ss.str().length());
    }

    // Print the vector elements with consistent alignment.
    for (std::size_t i = 0; i < vec.size(); ++i) {
        os << std::setw(static_cast<int>(max_width)) << std::setprecision(6) << std::fixed << vec[i];
        if (i < vec.size() - 1) {
            os << " ";
        }
    }

    return os;
}

/// @brief Overload the << operator for View.
/// @tparam T The type of the view elements.
/// @tparam N The number of elements in the view.
/// @param os The output stream.
/// @param vec The view to print.
/// @return The output stream with the view contents.
template <typename T, std::size_t N>
std::ostream &operator<<(std::ostream &os, const fsmlib::View<T, N> &vec)
{
    // Determine the maximum width of the elements for alignment.
    std::size_t max_width = 0;
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::ostringstream ss;
        ss << std::setprecision(6) << std::fixed << vec[i];
        max_width = std::max(max_width, ss.str().length());
    }

    // Print the view elements with consistent alignment.
    for (std::size_t i = 0; i < vec.size(); ++i) {
        os << std::setw(static_cast<int>(max_width)) << std::setprecision(6) << std::fixed << vec[i];
        if (i < vec.size() - 1) {
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
/// @return The output stream with the matrix contents.
template <typename T, std::size_t N1, std::size_t N2>
std::ostream &operator<<(std::ostream &os, const fsmlib::Matrix<T, N1, N2> &mat)
{
    // Determine the maximum width of the elements for alignment.
    std::size_t max_width = 0;
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N2; ++c) {
            std::ostringstream ss;
            ss << std::setprecision(6) << std::fixed << mat[r][c];
            max_width = std::max(max_width, ss.str().length());
        }
    }

    // Print the matrix rows.
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N2; ++c) {
            os << std::setw(static_cast<int>(max_width)) << std::setprecision(6) << std::fixed << mat[r][c];
            if (c < N2 - 1) {
                os << " ";
            }
        }
        if (r < N1 - 1) {
            os << "\n";
        }
    }

    return os;
}

/// @brief Overload the << operator for MatrixView.
/// @tparam T The type of the matrix view elements.
/// @tparam N1 The number of rows in the view.
/// @tparam N2 The number of columns in the view.
/// @param os The output stream.
/// @param mat The matrix view to print.
/// @return The output stream with the matrix view contents.
template <typename T, std::size_t N1, std::size_t N2>
std::ostream &operator<<(std::ostream &os, const fsmlib::MatrixView<T, N1, N2> &mat)
{
    // Determine the maximum width of the elements for alignment.
    std::size_t max_width = 0;
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N2; ++c) {
            std::ostringstream ss;
            ss << std::setprecision(6) << std::fixed << mat[r][c];
            max_width = std::max(max_width, ss.str().length());
        }
    }

    // Print the matrix view rows.
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N2; ++c) {
            os << std::setw(static_cast<int>(max_width)) << std::setprecision(6) << std::fixed << mat[r][c];
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
inline std::ostream &operator<<(std::ostream &os, const fsmlib::control::StateSpace<T, N_state, N_input, N_output> &ss)
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
inline std::ostream &operator<<(std::ostream &os,
                                const fsmlib::control::DiscreteStateSpace<T, N_state, N_input, N_output> &ss)
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

namespace fsmlib
{
#include <sstream>
#include <iomanip>

/// @brief Converts a fixed-size vector to an Octave-compatible variable assignment.
/// @tparam T The type of the vector elements.
/// @tparam N The size of the vector.
/// @param name The name of the variable.
/// @param vec The input vector.
/// @returns A string representing the vector in Octave format.
template <typename T, std::size_t N>
inline std::string to_octave(const std::string &name, const fsmlib::Vector<T, N> &vec)
{
    std::ostringstream ss;
    ss << name << " = [";
    for (std::size_t i = 0; i < N; ++i) {
        ss << std::setprecision(6) << vec[i];
        if (i < N - 1) {
            ss << ", ";
        }
    }
    ss << "];";
    return ss.str();
}

/// @brief Converts a fixed-size matrix to an Octave-compatible variable assignment.
/// @tparam T The type of the matrix elements.
/// @tparam Rows The number of rows in the matrix.
/// @tparam Cols The number of columns in the matrix.
/// @param name The name of the variable.
/// @param mat The input matrix.
/// @returns A string representing the matrix in Octave format.
template <typename T, std::size_t Rows, std::size_t Cols>
inline std::string to_octave(const std::string &name, const fsmlib::Matrix<T, Rows, Cols> &mat)
{
    std::ostringstream ss;
    ss << name << " = [";
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            ss << std::setprecision(6) << mat[i][j];
            if (j < Cols - 1) {
                ss << ", ";
            }
        }
        ss << ";";
    }
    ss << "];";
    return ss.str();
}

} // namespace fsmlib
