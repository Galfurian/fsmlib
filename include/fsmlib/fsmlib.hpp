/// @file fsmlib.hpp
/// @brief Defines fixed-size vector and matrix types for linear algebra operations.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>

namespace fsmlib
{

/// @brief Alias for a fixed-size vector.
/// @tparam T The type of the vector elements.
/// @tparam N The number of elements in the vector.
template <typename T, std::size_t N>
class Vector {
private:
    T data[N]; ///< Internal storage for the vector elements.

public:
    /// @brief Default constructor.
    constexpr Vector()
        : data{}
    {
    }

    /// @brief Constructor from an initializer list.
    constexpr Vector(std::initializer_list<T> init)
    {
        if (init.size() != N) {
            throw std::out_of_range("Initializer list size does not match vector size");
        }
        std::copy(init.begin(), init.end(), data);
    }

    /// @brief Access an element by index (const version).
    constexpr const T &operator[](std::size_t index) const
    {
        return data[index];
    }

    /// @brief Access an element by index (non-const version).
    constexpr T &operator[](std::size_t index)
    {
        return data[index];
    }

    /// @brief Returns the size of the vector.
    constexpr std::size_t size() const noexcept
    {
        return N;
    }

    /// @brief Begin iterator.
    constexpr T *begin() noexcept
    {
        return data;
    }

    /// @brief End iterator.
    constexpr T *end() noexcept
    {
        return data + N;
    }

    /// @brief Const begin iterator.
    constexpr const T *begin() const noexcept
    {
        return data;
    }

    /// @brief Const end iterator.
    constexpr const T *end() const noexcept
    {
        return data + N;
    }
};

template <class T, std::size_t N1, std::size_t N2 = N1>
using Matrix = Vector<Vector<T, N2>, N1>;

} // namespace fsmlib
