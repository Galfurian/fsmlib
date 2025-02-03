/// @file traits.hpp
/// @brief Contains type traits and helper variables for the FSM library.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include <cstddef>
#include <type_traits>

namespace fsmlib
{

/// @brief Support for type traits.
namespace traits
{

/// @brief Marker class to identify valid container types.
/// @details
/// Classes intended to be recognized as valid containers should inherit
/// from this class.
class valid_container_t
{
};

/// @brief Helper variable template to check if a type is a valid container.
/// @tparam T The type to check.
/// @retval true if T derives from valid_container_t; false otherwise.
template <typename T>
inline constexpr bool is_valid_container_v = std::is_base_of<fsmlib::traits::valid_container_t, T>::value;

/// @brief Checks if the combination of two types is valid for element-wise operations.
///
/// @details This trait verifies if the types `E1` and `E2` are a valid combination
/// for element-wise operations. A valid combination is defined as:
/// - Both `E1` and `E2` are valid containers.
/// - `E1` is a valid container, and `E2` is an arithmetic type.
/// - `E1` is an arithmetic type, and `E2` is a valid container.
///
/// @tparam E1 The first type to check.
/// @tparam E2 The second type to check.
template <typename E1, typename E2>
inline constexpr bool is_valid_combination_v =
    (fsmlib::traits::is_valid_container_v<E1> && fsmlib::traits::is_valid_container_v<E2>) ||
    (fsmlib::traits::is_valid_container_v<E1> && std::is_arithmetic_v<E2>) ||
    (std::is_arithmetic_v<E1> && fsmlib::traits::is_valid_container_v<E2>);

/// @brief Checks if the combination of two types is valid for mutable element-wise operations.
///
/// @details This trait verifies if the types `E1` and `E2` are a valid combination
/// for element-wise operations where `E1` is mutable (e.g., a non-const reference).
/// A valid combination is defined as:
/// - Both `E1` and `E2` are valid containers.
/// - `E1` is a valid container, and `E2` is an arithmetic type.
///
/// @tparam E1 The first type to check (expected to be mutable).
/// @tparam E2 The second type to check.
template <typename E1, typename E2>
inline constexpr bool is_valid_mutable_combination_v =
    (fsmlib::traits::is_valid_container_v<E1> && fsmlib::traits::is_valid_container_v<E2>) ||
    (fsmlib::traits::is_valid_container_v<E1> && std::is_arithmetic_v<E2>);

/// @brief Trait to determine the fixed size of a container.
/// @tparam Vec The container type.
template <typename Vec, typename = std::void_t<typename Vec::size_type>> struct fixed_size {
    /// @brief The fixed size of the container.
    static constexpr std::size_t value = Vec::size_type::value;
};

/// @brief Helper variable template for fixed_size.
/// @tparam Vec The container type.
template <typename Vec> inline constexpr std::size_t fixed_size_v = fsmlib::traits::fixed_size<Vec>::value;

/// @brief Helper variable template to trigger a static assertion for invalid types.
/// @tparam ... Args The types to evaluate.
template <typename... Args> inline constexpr bool always_false_v = false;

} // namespace traits

} // namespace fsmlib
