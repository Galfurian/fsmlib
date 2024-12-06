
#include <fsmlib/fsmlib.hpp>
#include <fsmlib/math.hpp>
#include <fsmlib/io.hpp>

#include <iostream>

int main()
{
    try {
        // Test 1: Vector addition
        {
            fsmlib::Vector<int, 3> vec1     = { 1, 2, 3 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 5, 7, 9 };
            auto result                     = vec1 + vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test 1 failed: Vector addition");
            }
            std::cout << "Test 1 passed: Vector addition\n";
        }

        // Test 2: Vector subtraction
        {
            fsmlib::Vector<int, 3> vec1     = { 7, 8, 9 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 3, 3, 3 };
            auto result                     = vec1 - vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test 2 failed: Vector subtraction");
            }
            std::cout << "Test 2 passed: Vector subtraction\n";
        }

        // Test 3: Vector element-wise multiplication
        {
            fsmlib::Vector<int, 3> vec1     = { 1, 2, 3 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 4, 10, 18 };
            auto result                     = vec1 * vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test 3 failed: Vector multiplication");
            }
            std::cout << "Test 3 passed: Vector multiplication\n";
        }

        // Test 4: Vector element-wise division
        {
            fsmlib::Vector<int, 3> vec1     = { 6, 8, 10 };
            fsmlib::Vector<int, 3> vec2     = { 3, 2, 5 };
            fsmlib::Vector<int, 3> expected = { 2, 4, 2 };
            auto result                     = vec1 / vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test 4 failed: Vector division");
            }
            std::cout << "Test 4 passed: Vector division\n";
        }

        // Test 5: Matrix addition
        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 1, 2 }, { 3, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 5, 6 }, { 7, 8 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 6, 8 }, { 10, 12 } } };
            auto result                        = mat1 + mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test 5 failed: Matrix addition");
            }
            std::cout << "Test 5 passed: Matrix addition\n";
        }

        // Test 6: Matrix subtraction
        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 6, 8 }, { 10, 12 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 5, 6 }, { 7, 8 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 1, 2 }, { 3, 4 } } };
            auto result                        = mat1 - mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test 6 failed: Matrix subtraction");
            }
            std::cout << "Test 6 passed: Matrix subtraction\n";
        }

        // Test 7: Matrix element-wise multiplication
        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 1, 2 }, { 3, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 2, 2 }, { 2, 2 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 2, 4 }, { 6, 8 } } };
            auto result                        = mat1 * mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test 7 failed: Matrix multiplication");
            }
            std::cout << "Test 7 passed: Matrix multiplication\n";
        }

        // Test 8: Matrix element-wise division
        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 10, 20 }, { 30, 40 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 2, 4 }, { 5, 10 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 5, 5 }, { 6, 4 } } };
            auto result                        = mat1 / mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test 8 failed: Matrix division");
            }
            std::cout << "Test 8 passed: Matrix division\n";
        }

        // Test 9: Matrix-Vector Multiplication
        {
            fsmlib::Matrix<int, 2, 3> mat   = { { { 1, 2, 3 }, { 4, 5, 6 } } };
            fsmlib::Vector<int, 3> vec      = { 1, 2, 3 };
            fsmlib::Vector<int, 2> expected = { 14, 32 }; // [1+4+9, 4+10+18]
            auto result                     = fsmlib::multiply(mat, vec);
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test 9 failed: Matrix-Vector multiplication");
            }
            std::cout << "Test 9 passed: Matrix-Vector multiplication\n";
        }

        // Test 10: Matrix-Matrix Multiplication
        {
            fsmlib::Matrix<int, 2, 3> mat1     = { { { 1, 2, 3 }, { 4, 5, 6 } } };
            fsmlib::Matrix<int, 3, 2> mat2     = { { { 1, 4 }, { 2, 5 }, { 3, 6 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 14, 32 }, { 32, 77 } } };
            auto result                        = fsmlib::multiply(mat1, mat2);
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test 10 failed: Matrix-Matrix multiplication");
            }
            std::cout << "Test 10 passed: Matrix-Matrix multiplication\n";
        }

        // Test 11: Vector += Vector
        {
            fsmlib::Vector<int, 3> vec1     = { 1, 2, 3 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 5, 7, 9 };
            vec1 += vec2;
            if (fsmlib::any(vec1 != expected)) {
                throw std::runtime_error("Test 11 failed: Vector += Vector");
            }
            std::cout << "Test 11 passed: Vector += Vector\n";
        }

        // Test 12: Vector -= Vector
        {
            fsmlib::Vector<int, 3> vec1     = { 7, 8, 9 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 3, 3, 3 };
            vec1 -= vec2;
            if (fsmlib::any(vec1 != expected)) {
                throw std::runtime_error("Test 12 failed: Vector -= Vector");
            }
            std::cout << "Test 12 passed: Vector -= Vector\n";
        }

        // Test 13: Vector *= Vector
        {
            fsmlib::Vector<int, 3> vec1     = { 1, 2, 3 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 4, 10, 18 };
            vec1 *= vec2;
            if (fsmlib::any(vec1 != expected)) {
                throw std::runtime_error("Test 13 failed: Vector *= Vector");
            }
            std::cout << "Test 13 passed: Vector *= Vector\n";
        }

        // Test 14: Vector /= Vector
        {
            fsmlib::Vector<int, 3> vec1     = { 6, 8, 10 };
            fsmlib::Vector<int, 3> vec2     = { 3, 2, 5 };
            fsmlib::Vector<int, 3> expected = { 2, 4, 2 };
            vec1 /= vec2;
            if (fsmlib::any(vec1 != expected)) {
                throw std::runtime_error("Test 14 failed: Vector /= Vector");
            }
            std::cout << "Test 14 passed: Vector /= Vector\n";
        }

        // Test 15: Matrix += Matrix
        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 1, 2 }, { 3, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 5, 6 }, { 7, 8 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 6, 8 }, { 10, 12 } } };
            mat1 += mat2;
            if (fsmlib::any(mat1 != expected)) {
                throw std::runtime_error("Test 15 failed: Matrix += Matrix");
            }
            std::cout << "Test 15 passed: Matrix += Matrix\n";
        }

        // Test 16: Matrix -= Matrix
        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 6, 8 }, { 10, 12 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 5, 6 }, { 7, 8 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 1, 2 }, { 3, 4 } } };
            mat1 -= mat2;
            if (fsmlib::any(mat1 != expected)) {
                throw std::runtime_error("Test 16 failed: Matrix -= Matrix");
            }
            std::cout << "Test 16 passed: Matrix -= Matrix\n";
        }

        // Test 17: Matrix *= Matrix (element-wise)
        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 1, 2 }, { 3, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 2, 2 }, { 2, 2 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 2, 4 }, { 6, 8 } } };
            mat1 *= mat2;
            if (fsmlib::any(mat1 != expected)) {
                throw std::runtime_error("Test 17 failed: Matrix *= Matrix");
            }
            std::cout << "Test 17 passed: Matrix *= Matrix\n";
        }

        // Test 18: Matrix /= Matrix (element-wise)
        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 10, 20 }, { 30, 40 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 2, 4 }, { 5, 10 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 5, 5 }, { 6, 4 } } };
            mat1 /= mat2;
            if (fsmlib::any(mat1 != expected)) {
                throw std::runtime_error("Test 18 failed: Matrix /= Matrix");
            }
            std::cout << "Test 18 passed: Matrix /= Matrix\n";
        }

        std::cout << "All math tests passed!\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
