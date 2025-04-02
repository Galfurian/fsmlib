# Fixed-Size Matrix Library (fsmlib)

[![Ubuntu](https://github.com/Galfurian/fsmlib/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/Galfurian/fsmlib/actions/workflows/ubuntu.yml)
[![Windows](https://github.com/Galfurian/fsmlib/actions/workflows/windows.yml/badge.svg)](https://github.com/Galfurian/fsmlib/actions/workflows/windows.yml)
[![MacOS](https://github.com/Galfurian/fsmlib/actions/workflows/macos.yml/badge.svg)](https://github.com/Galfurian/fsmlib/actions/workflows/macos.yml)
[![Documentation](https://github.com/Galfurian/fsmlib/actions/workflows/documentation.yml/badge.svg)](https://github.com/Galfurian/fsmlib/actions/workflows/documentation.yml)

`fsmlib` is a lightweight, header-only C++ library for performing efficient
operations on fixed-size matrices and vectors. Designed with performance and
usability in mind, `fsmlib` leverages modern C++ features to enable
high-performance linear algebra operations at compile time.

## Features

- **Fixed-size vectors and matrices** with static storage.
- **Compile-time dimensions** for efficiency and type safety.
- Support for **arithmetic operations** (addition, subtraction, multiplication, division).
- Flexible **templated design** for custom scalar types.
- Fully **header-only**, with no external dependencies.

## Getting Started

### Installation

`fsmlib` is a header-only library. Simply clone the repository and include the
header files in your project:

```bash
git clone https://github.com/yourusername/fsmlib.git
```

Then include the library in your code based on your needs.

### Prerequisites

- C++17 or higher compiler.
- Basic knowledge of template programming for custom use cases.

## Library Contents

The library is divided into modular headers to keep the codebase organized and
provide flexibility to users. Here's an overview of the primary files:

- `fsmlib/fsmlib.hpp`
  - Contains base utilities and foundational types used throughout the library,
    such as common traits and helper functions.
- `fsmlib/control.hpp`
  - Provides utility functions for managing and controlling matrix/vector data,
    likely including error handling or bounds checking.
- `fsmlib/io.hpp`
  - Implements input/output functionality for matrices and vectors, such as
    streaming operations and formatting.
- `fsmlib/linalg.hpp`
  - Focused on linear algebra operations, including matrix multiplications,
    inversions, and possibly eigenvalue computations.
- `fsmlib/math.hpp`
  - Implements core mathematical operations, including element-wise addition,
    subtraction, and other operations between matrices, vectors, and scalars.

## Example Usage

### Basic Vector Operations

```cpp
#include "fsmlib/fsmlib.hpp"
#include "fsmlib/math.hpp"

#include <iostream>

int main() {
    fsmlib::Vector<int, 3> vec1 = {1, 2, 3};
    fsmlib::Vector<int, 3> vec2 = {4, 5, 6};
    
    auto result = vec1 + vec2;

    for (const auto& val : result) {
        std::cout << val << " ";
    }
    return 0;
}
```

Output:

```bash
5 7 9
```

### Basic Matrix Operations

```cpp
#include "fsmlib/fsmlib.hpp"
#include "fsmlib/math.hpp"

#include <iostream>

int main() {
    fsmlib::Matrix<int, 2, 2> mat1 = {{1, 2}, {3, 4}};
    fsmlib::Matrix<int, 2, 2> mat2 = {{5, 6}, {7, 8}};
    
    auto result = mat1 + mat2;

    for (const auto& row : result) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    return 0;
}
```

Output:

```bash
 6  8
10 12
```

## Documentation

Detailed documentation and examples are available in the docs folder.

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or new features
to propose, feel free to open an issue or submit a pull request.

Steps to Contribute:

- Fork the repository.
- Create a new branch: `git checkout -b my-feature-branch`
- Commit your changes: `git commit -m 'Add some feature'`
- Push the branch: `git push origin my-feature-branch`
- Open a pull request.

## License

`fsmlib` is licensed under the MIT License.
