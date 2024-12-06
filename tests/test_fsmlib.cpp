#include <fsmlib/fsmlib.hpp>

#include <iostream>

int main()
{
    try
    {
        // Test 1: Initialization of Vector
        {
            fsmlib::Vector<int, 5> vec = {1, 2, 3, 4, 5};
            if (vec.size() != 5 || vec[0] != 1 || vec[4] != 5)
            {
                throw std::runtime_error("Test 1 failed: Vector initialization");
            }
            std::cout << "Test 1 passed: Vector initialization\n";
        }

        // Test 2: Modification of Vector elements
        {
            fsmlib::Vector<double, 3> vec = {1.1, 2.2, 3.3};
            vec[0] = 4.4;
            vec[2] = 6.6;
            if (vec[0] != 4.4 || vec[1] != 2.2 || vec[2] != 6.6)
            {
                throw std::runtime_error("Test 2 failed: Vector modification");
            }
            std::cout << "Test 2 passed: Vector modification\n";
        }

        // Test 3: Initialization of square Matrix
        {
            fsmlib::Matrix<int, 3, 3> mat = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
            if (mat.size() != 3 || mat[0].size() != 3 || mat[0][0] != 1 || mat[2][2] != 9)
            {
                throw std::runtime_error("Test 3 failed: Square matrix initialization");
            }
            std::cout << "Test 3 passed: Square matrix initialization\n";
        }

        // Additional tests for access...
        std::cout << "All access tests passed!\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
