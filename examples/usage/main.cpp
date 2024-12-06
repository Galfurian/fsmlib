
#include "mx/mx.hpp"
#include "mx/io.hpp"
#include "mx/math.hpp"

#include <cassert>

void test_access(void)
{
    // Test 1: Initialization of Vector
    {
        mx::Vector<int, 5> vec = { 1, 2, 3, 4, 5 };
        assert(vec.size() == 5);
        assert(vec[0] == 1);
        assert(vec[4] == 5);
        std::cout << "Test 1 passed: Vector initialization\n";
    }
    // Test 2: Modification of Vector elements
    {
        mx::Vector<double, 3> vec = { 1.1, 2.2, 3.3 };
        vec[0]                    = 4.4;
        vec[2]                    = 6.6;
        assert(vec[0] == 4.4);
        assert(vec[1] == 2.2);
        assert(vec[2] == 6.6);
        std::cout << "Test 2 passed: Vector modification\n";
    }

    // Test 3: Initialization of square Matrix
    {
        mx::Matrix<int, 3> mat = { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } };
        assert(mat.size() == 3);
        assert(mat[0].size() == 3);
        assert(mat[0][0] == 1);
        assert(mat[2][2] == 9);
        std::cout << "Test 3 passed: Square matrix initialization\n";
    }

    // Test 4: Initialization of rectangular Matrix
    {
        mx::Matrix<int, 2, 3> mat = { { { 1, 2, 3 }, { 4, 5, 6 } } };
        assert(mat.size() == 2);
        assert(mat[0].size() == 3);
        assert(mat[1][2] == 6);
        std::cout << "Test 4 passed: Rectangular matrix initialization\n";
    }

    // Test 5: Modification of Matrix elements
    {
        mx::Matrix<double, 2, 2> mat = { { { 1.1, 2.2 }, { 3.3, 4.4 } } };
        mat[0][1]                    = 5.5;
        mat[1][0]                    = 6.6;
        assert(mat[0][1] == 5.5);
        assert(mat[1][0] == 6.6);
        std::cout << "Test 5 passed: Matrix element modification\n";
    }

    // Test 6: Edge cases with empty dimensions
    {
        mx::Vector<int, 0> emptyVec;
        assert(emptyVec.size() == 0);
        mx::Matrix<int, 0> emptyMat;
        assert(emptyMat.size() == 0);
        std::cout << "Test 6 passed: Edge cases with empty dimensions\n";
    }

    // Test 7: Large dimensions
    {
        constexpr unsigned largeSize = 1000;
        mx::Matrix<int, largeSize, largeSize> largeMat;
        largeMat[999][999] = 42;
        assert(largeMat[999][999] == 42);
        std::cout << "Test 7 passed: Large dimensions\n";
    }

    std::cout << "All tests passed!\n";
}

void test_math(void)
{
    // Test 1: Vector addition
    {
        mx::Vector<int, 3> vec1     = { 1, 2, 3 };
        mx::Vector<int, 3> vec2     = { 4, 5, 6 };
        mx::Vector<int, 3> expected = { 5, 7, 9 };
        auto result                 = vec1 + vec2;
        assert(result == expected);
        std::cout << "Test 1 passed: Vector addition\n";
    }

    // Test 2: Vector subtraction
    {
        mx::Vector<float, 2> vec1     = { 3.5, 2.5 };
        mx::Vector<float, 2> vec2     = { 1.5, 0.5 };
        mx::Vector<float, 2> expected = { 2.0, 2.0 };
        auto result                   = vec1 - vec2;
        assert(result == expected);
        std::cout << "Test 2 passed: Vector subtraction\n";
    }

    // Test 3: Vector multiplication
    {
        mx::Vector<int, 4> vec1     = { 1, 2, 3, 4 };
        mx::Vector<int, 4> vec2     = { 2, 3, 4, 5 };
        mx::Vector<int, 4> expected = { 2, 6, 12, 20 };
        auto result                 = vec1 * vec2;
        assert(result == expected);
        std::cout << "Test 3 passed: Vector multiplication\n";
    }

    // Test 4: Vector division
    {
        mx::Vector<double, 3> vec1     = { 4.0, 9.0, 16.0 };
        mx::Vector<double, 3> vec2     = { 2.0, 3.0, 4.0 };
        mx::Vector<double, 3> expected = { 2.0, 3.0, 4.0 };
        auto result                    = vec1 / vec2;
        assert(result == expected);
        std::cout << "Test 4 passed: Vector division\n";
    }

    // Test 5: Matrix addition
    {
        mx::Matrix<int, 2, 2> mat1     = { { { 1, 2 }, { 3, 4 } } };
        mx::Matrix<int, 2, 2> mat2     = { { { 5, 6 }, { 7, 8 } } };
        mx::Matrix<int, 2, 2> expected = { { { 6, 8 }, { 10, 12 } } };
        auto result                    = mat1 + mat2;
        assert(result == expected);
        std::cout << "Test 5 passed: Matrix addition\n";
    }

    // Test 6: Matrix subtraction
    {
        mx::Matrix<int, 2, 3> mat1     = { { { 10, 20, 30 }, { 40, 50, 60 } } };
        mx::Matrix<int, 2, 3> mat2     = { { { 1, 2, 3 }, { 4, 5, 6 } } };
        mx::Matrix<int, 2, 3> expected = { { { 9, 18, 27 }, { 36, 45, 54 } } };
        auto result                    = mat1 - mat2;
        assert(result == expected);
        std::cout << "Test 6 passed: Matrix subtraction\n";
    }

    // Test 7: Matrix multiplication (element-wise)
    {
        mx::Matrix<int, 3, 3> mat1     = { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } };
        mx::Matrix<int, 3, 3> mat2     = { { { 9, 8, 7 }, { 6, 5, 4 }, { 3, 2, 1 } } };
        mx::Matrix<int, 3, 3> expected = { { { 9, 16, 21 }, { 24, 25, 24 }, { 21, 16, 9 } } };
        auto result                    = mat1 * mat2;
        assert(result == expected);
        std::cout << "Test 7 passed: Matrix multiplication\n";
    }

    // Test 8: Matrix division (element-wise)
    {
        mx::Matrix<double, 2, 2> mat1     = { { { 4.0, 9.0 }, { 16.0, 25.0 } } };
        mx::Matrix<double, 2, 2> mat2     = { { { 2.0, 3.0 }, { 4.0, 5.0 } } };
        mx::Matrix<double, 2, 2> expected = { { { 2.0, 3.0 }, { 4.0, 5.0 } } };
        auto result                       = mat1 / mat2;
        assert(result == expected);
        std::cout << "Test 8 passed: Matrix division\n";
    }

    std::cout << "All tests passed!\n";
}

int main(int, char *[])
{
    test_access();
    test_math();
    return 0;
}