/// @file io.hpp
/// @brief Defines output stream operators for vectors, matrices, and state-space
/// models.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "fsmlib/fsmlib.hpp"
#include "fsmlib/control.hpp"
#include "fsmlib/view.hpp"

/// @brief Overload the << operator for Vector.
/// @tparam T The type of the vector elements.
/// @tparam N The number of elements in the vector.
/// @param os The output stream.
/// @param vec The vector to print.
/// @return The output stream with the vector contents.
template <typename T, std::size_t N>
std::ostream &operator<<(std::ostream &os, const fsmlib::VectorBase<T, N> &vec)
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

/// @brief Overload the << operator for Matrix.
/// @tparam T The type of the matrix elements.
/// @tparam N1 The number of rows in the matrix.
/// @tparam N2 The number of columns in the matrix.
/// @param os The output stream.
/// @param mat The matrix to print.
/// @return The output stream with the matrix contents.
template <typename T, std::size_t N1, std::size_t N2>
std::ostream &operator<<(std::ostream &os, const fsmlib::MatrixBase<T, N1, N2> &mat)
{
    // Determine the maximum width of the elements for alignment.
    std::size_t max_width = 0;
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N2; ++c) {
            std::ostringstream ss;
            ss << std::setprecision(6) << std::fixed << mat(r, c);
            max_width = std::max(max_width, ss.str().length());
        }
    }

    // Print the matrix rows.
    for (std::size_t r = 0; r < N1; ++r) {
        for (std::size_t c = 0; c < N2; ++c) {
            os << std::setw(static_cast<int>(max_width)) << std::setprecision(6) << std::fixed << mat(r, c);
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

/// @brief Converts a fixed-size vector to an Octave-compatible variable assignment.
/// @tparam T The type of the vector elements.
/// @tparam N The size of the vector.
/// @param name The name of the variable.
/// @param vec The input vector.
/// @returns A string representing the vector in Octave format.
template <typename T, std::size_t N>
inline std::string to_octave(const std::string &name, const fsmlib::VectorBase<T, N> &vec)
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
inline std::string to_octave(const std::string &name, const fsmlib::MatrixBase<T, Rows, Cols> &mat)
{
    std::ostringstream ss;
    ss << name << " = [";
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j) {
            ss << std::setprecision(6) << mat(i, j);
            if (j < Cols - 1) {
                ss << ", ";
            }
        }
        ss << ";";
    }
    ss << "];";
    return ss.str();
}

/// @brief Converts a matrix to LaTeX tabular format.
///
/// @param mat The matrix to convert.
/// @return std::string The LaTeX string representation of the matrix.
template <typename T, size_t Rows, size_t Cols>
std::string to_latex(const fsmlib::MatrixBase<T, Rows, Cols> &mat)
{
    std::ostringstream oss;
    oss << "\\begin{bmatrix}\n";

    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Cols; ++j) {
            oss << mat(i, j);
            if (j < Cols - 1) {
                oss << " & "; // Separate columns with "&".
            }
        }
        if (i < Rows - 1) {
            oss << " \\\\\n"; // New row.
        }
    }

    oss << "\n\\end{bmatrix}";
    return oss.str();
}

/// @brief Converts a matrix to Markdown table format.
///
/// @param mat The matrix to convert.
/// @return std::string The Markdown string representation of the matrix.
template <typename T, size_t Rows, size_t Cols>
std::string to_markdown(const fsmlib::MatrixBase<T, Rows, Cols> &mat)
{
    std::ostringstream oss;

    // Create the header line.
    for (size_t j = 0; j < Cols; ++j) {
        oss << "| Col" << j + 1;
    }
    oss << " |\n";

    // Create the separator line.
    for (size_t j = 0; j < Cols; ++j) {
        oss << "|---";
    }
    oss << " |\n";

    // Create the data rows.
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Cols; ++j) {
            oss << "| " << mat(i, j);
        }
        oss << " |\n";
    }

    return oss.str();
}

/// @brief Saves the matrix to a CSV file.
///
/// @param filename The name of the file to save the matrix to.
/// @param mat The matrix to save.
/// @return int 0 on success, non-zero on error.
template <typename T, size_t Rows, size_t Cols>
int save_matrix_to_csv(const std::string &filename, const fsmlib::MatrixBase<T, Rows, Cols> &mat)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        return -1; // Error opening file.
    }

    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Cols; ++j) {
            file << mat(i, j);
            if (j < Cols - 1)
                file << ","; // Separate columns with commas.
        }
        file << "\n"; // New line for each row.
    }

    file.close();
    return 0; // Success.
}

/// @brief Saves the matrix to a binary file.
///
/// @param filename The name of the file to save the matrix to.
/// @param mat The matrix to save.
/// @return int 0 on success, non-zero on error.
template <typename T, size_t Rows, size_t Cols>
int save_matrix_to_binary(const std::string &filename, const fsmlib::MatrixBase<T, Rows, Cols> &mat)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return -1; // Error opening file.
    }

    // Store dimensions in local variables.
    size_t rows = Rows;
    size_t cols = Cols;

    // Write dimensions to the file.
    file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

    // Write matrix data.
    file.write(reinterpret_cast<const char *>(mat), sizeof(mat));

    file.close();
    return 0; // Success.
}

/// @brief Loads a matrix from a CSV file.
///
/// @param filename The name of the file to load the matrix from.
/// @param matrix The matrix to load into.
/// @return int 0 on success, non-zero on error.
template <typename T, size_t Rows, size_t Cols>
int load_matrix_from_csv(const std::string &filename, fsmlib::MatrixBase<T, Rows, Cols> &mat)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        return -1; // Error opening file.
    }

    std::string line;
    size_t row = 0;

    while (std::getline(file, line) && row < Rows) {
        std::stringstream ss(line);
        std::string value;
        size_t col = 0;

        while (std::getline(ss, value, ',') && col < Cols) {
            std::istringstream(value) >> mat(row, col);
            ++col;
        }

        ++row;
    }

    file.close();
    return 0; // Success.
}

/// @brief Loads a matrix from a binary file.
///
/// @param filename The name of the file to load the matrix from.
/// @param matrix The matrix to load into.
/// @return int 0 on success, non-zero on error.
template <typename T, size_t Rows, size_t Cols>
int load_matrix_from_binary(const std::string &filename, fsmlib::MatrixBase<T, Rows, Cols> &mat)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return -1; // Error opening file.
    }

    // Load dimensions in local variables.
    size_t rows;
    size_t cols;

    // Read dimensions.
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

    if (rows != Rows || cols != Cols) {
        file.close();
        return -2; // Dimension mismatch.
    }

    // Read matrix data.
    file.read(reinterpret_cast<char *>(mat), sizeof(mat));

    file.close();
    return 0; // Success.
}

} // namespace fsmlib
