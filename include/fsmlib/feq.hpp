/// @file feq.hpp
/// @author Enrico Fraccaroli (enry.frak@gmail.com)
/// @brief Functions for checking equality between floating point values.

#pragma once

#include <cmath>
#include <limits>
#include <complex>
#include <type_traits>

namespace fsmlib
{

namespace feq
{

static inline auto &tolerance()
{
    static double tol = 1e-06;
    return tol;
}

/// @brief Checks if the two floating point values are equal.
/// @param a the first value.
/// @param b the second value.
/// @returns true if they are approximately equal.
/// @returns false otherwise.
template <typename T1, typename T2>
inline bool approximately_equal(T1 a, T2 b)
{
    return std::abs(a - b) <= tolerance() * std::max(std::abs(a), std::abs(b));
}

/// @brief Checks if the fir floating point value is lesser than or equal to the second one.
/// @param a the first value.
/// @param b the second value.
/// @returns true if (a <= b).
/// @returns false otherwise.
template <typename T1, typename T2>
inline bool approximately_lesser_than_equal(T1 a, T2 b)
{
    return (a < b) || (fsmlib::feq::approximately_equal(a, b));
}

/// @brief Checks if the fir floating point value is greater than or equal to the second one.
/// @param a the first value.
/// @param b the second value.
/// @returns true if (a >= b).
/// @returns false otherwise.
template <typename T1, typename T2>
inline bool approximately_greater_than_equal(T1 a, T2 b)
{
    return (a > b) || (fsmlib::feq::approximately_equal(a, b));
}

/// @brief Checks if the value is approximately equal to zero.
/// @param a the value.
/// @returns true if the value is approximately equal to zero.
/// @returns false otherwise.
template <typename T>
inline bool approximately_equal_to_zero(T a)
{
    return std::abs(a) <= fsmlib::feq::tolerance() * std::abs(a);
}

} // namespace feq

} // namespace fsmlib
