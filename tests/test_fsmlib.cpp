#include <fsmlib/fsmlib.hpp>

#include <iostream>
#include <iomanip>

int main()
{
    try {
        int test_count = 1; // Test counter variable

        {
            fsmlib::Vector<int, 5> vec = { 1, 2, 3, 4, 5 };
            if (vec.size() != 5 || vec[0] != 1 || vec[4] != 5) {
                throw std::runtime_error("Test failed: Vector initialization");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector initialization\n";
        }

        {
            fsmlib::Vector<double, 3> vec = { 1.1, 2.2, 3.3 };
            vec[0]                        = 4.4;
            vec[2]                        = 6.6;
            if (vec[0] != 4.4 || vec[1] != 2.2 || vec[2] != 6.6) {
                throw std::runtime_error("Test failed: Vector modification");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector modification\n";
        }

        {
            fsmlib::Matrix<int, 3, 3> mat = { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } };
            if (mat.size() != 3 || mat[0].size() != 3 || mat[0][0] != 1 || mat[2][2] != 9) {
                throw std::runtime_error("Test failed: Square matrix initialization");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Square matrix initialization\n";
        }

        // Additional tests for access...
        std::cout << "All access tests passed!\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
