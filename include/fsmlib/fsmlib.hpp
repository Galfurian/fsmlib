/// @file fsmlib.hpp
/// @brief Main header file for the Fixed-Size Matrix Library (fsmlib).
/// @details This umbrella header includes all the core components of fsmlib,
///          providing a convenient single include for users who want access
///          to the complete library functionality.
/// @copyright
/// Copyright (c) 2024. All rights reserved.
/// Licensed under the MIT License. See LICENSE file in the project root for details.

#pragma once

// Core data structures (Vector, Matrix classes)
#include "fsmlib/core.hpp"

// Type traits and metaprogramming utilities
#include "fsmlib/traits.hpp"

// Floating-point equality comparison utilities
#include "fsmlib/feq.hpp"

// Mathematical operations and operator overloads
#include "fsmlib/math.hpp"

// Linear algebra operations (decompositions, eigenvalues, norms, etc.)
#include "fsmlib/linalg.hpp"

// View classes for matrix/vector subsets
#include "fsmlib/view.hpp"

// Input/output stream operators
#include "fsmlib/io.hpp"

// Control system data structures and functions
#include "fsmlib/control.hpp"
