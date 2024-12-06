
#pragma once

#include <array>

namespace mx
{

template <typename T, std::size_t N>
using Vector = std::array<T, N>;

template <class T, std::size_t N1, std::size_t N2 = N1>
using Matrix = std::array<std::array<T, N2>, N1>;

} // namespace mx
