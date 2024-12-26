/// @file traits.hpp
/// @brief
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include <type_traits>
#include <cstddef>

namespace fsmlib
{

/// @brief Marker class to identify valid container types.
/// @details
/// Classes intended to be recognized as valid containers should inherit
/// from this class.
class valid_container_t {};

/// @brief Helper variable template to check if a type is a valid container.
/// @tparam T The type to check.
/// @retval true if T derives from valid_container_t; false otherwise.
template <typename T>
inline constexpr bool is_valid_container_v = std::is_base_of<valid_container_t, T>::value;

/// @brief Trait to determine the fixed size of a container.
/// @tparam Vec The container type.
template <typename Vec, typename = std::void_t<typename Vec::size_type>>
struct fixed_size {
    /// @brief The fixed size of the container.
    static constexpr std::size_t value = Vec::size_type::value;
};

/// @brief Helper variable template for fixed_size.
/// @tparam Vec The container type.
template <typename Vec>
inline constexpr std::size_t fixed_size_v = fixed_size<Vec>::value;

/// @brief Helper variable template to trigger a static assertion for invalid types.
/// @tparam ... Args The types to evaluate.
template <typename... Args>
inline constexpr bool always_false_v = false;

} // namespace fsmlib
