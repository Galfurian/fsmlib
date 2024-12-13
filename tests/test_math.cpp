
#include <fsmlib/fsmlib.hpp>
#include <fsmlib/math.hpp>
#include <fsmlib/io.hpp>

#include <iostream>
#include <iomanip>
#include <array>

int main()
{
    try {
        int test_count = 1; // Test counter variable

        {
            fsmlib::Vector<int, 3> vec1     = { 1, 2, 3 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 5, 7, 9 };
            auto result                     = vec1 + vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector addition");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector addition\n";
        }

        {
            fsmlib::Vector<int, 3> vec1     = { 7, 8, 9 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 3, 3, 3 };
            auto result                     = vec1 - vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector subtraction");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector subtraction\n";
        }

        {
            fsmlib::Vector<int, 3> vec1     = { 1, 2, 3 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 4, 10, 18 };
            auto result                     = vec1 * vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector multiplication");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector multiplication\n";
        }

        {
            fsmlib::Vector<int, 3> vec1     = { 6, 8, 10 };
            fsmlib::Vector<int, 3> vec2     = { 3, 2, 5 };
            fsmlib::Vector<int, 3> expected = { 2, 4, 2 };
            auto result                     = vec1 / vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector division");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector division\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 1, 2 }, { 3, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 5, 6 }, { 7, 8 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 6, 8 }, { 10, 12 } } };
            auto result                        = mat1 + mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix addition");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix addition\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 6, 8 }, { 10, 12 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 5, 6 }, { 7, 8 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 1, 2 }, { 3, 4 } } };
            auto result                        = mat1 - mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix subtraction");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix subtraction\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 1, 2 }, { 3, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 2, 2 }, { 2, 2 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 2, 4 }, { 6, 8 } } };
            auto result                        = mat1 * mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix multiplication");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix multiplication\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 10, 20 }, { 30, 40 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 2, 4 }, { 5, 10 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 5, 5 }, { 6, 4 } } };
            auto result                        = mat1 / mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix division");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix division\n";
        }

        {
            fsmlib::Matrix<int, 2, 3> mat   = { { { 1, 2, 3 }, { 4, 5, 6 } } };
            fsmlib::Vector<int, 3> vec      = { 1, 2, 3 };
            fsmlib::Vector<int, 2> expected = { 14, 32 }; // [1+4+9, 4+10+18]
            auto result                     = fsmlib::multiply(mat, vec);
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix-Vector multiplication");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix-Vector multiplication\n";
        }

        {
            fsmlib::Matrix<int, 2, 3> mat1     = { { { 1, 2, 3 }, { 4, 5, 6 } } };
            fsmlib::Matrix<int, 3, 2> mat2     = { { { 1, 4 }, { 2, 5 }, { 3, 6 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 14, 32 }, { 32, 77 } } };
            auto result                        = fsmlib::multiply(mat1, mat2);
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix-Matrix multiplication");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix-Matrix multiplication\n";
        }

        {
            fsmlib::Vector<int, 3> vec1     = { 1, 2, 3 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 5, 7, 9 };
            vec1 += vec2;
            if (fsmlib::any(vec1 != expected)) {
                throw std::runtime_error("Test failed: Vector += Vector");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector += Vector\n";
        }

        {
            fsmlib::Vector<int, 3> vec1     = { 7, 8, 9 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 3, 3, 3 };
            vec1 -= vec2;
            if (fsmlib::any(vec1 != expected)) {
                throw std::runtime_error("Test failed: Vector -= Vector");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector -= Vector\n";
        }

        {
            fsmlib::Vector<int, 3> vec1     = { 1, 2, 3 };
            fsmlib::Vector<int, 3> vec2     = { 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 4, 10, 18 };
            vec1 *= vec2;
            if (fsmlib::any(vec1 != expected)) {
                throw std::runtime_error("Test failed: Vector *= Vector");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector *= Vector\n";
        }

        {
            fsmlib::Vector<int, 3> vec1     = { 6, 8, 10 };
            fsmlib::Vector<int, 3> vec2     = { 3, 2, 5 };
            fsmlib::Vector<int, 3> expected = { 2, 4, 2 };
            vec1 /= vec2;
            if (fsmlib::any(vec1 != expected)) {
                throw std::runtime_error("Test failed: Vector /= Vector");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector /= Vector\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 1, 2 }, { 3, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 5, 6 }, { 7, 8 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 6, 8 }, { 10, 12 } } };
            mat1 += mat2;
            if (fsmlib::any(mat1 != expected)) {
                throw std::runtime_error("Test failed: Matrix += Matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix += Matrix\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 6, 8 }, { 10, 12 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 5, 6 }, { 7, 8 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 1, 2 }, { 3, 4 } } };
            mat1 -= mat2;
            if (fsmlib::any(mat1 != expected)) {
                throw std::runtime_error("Test failed: Matrix -= Matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix -= Matrix\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 1, 2 }, { 3, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 2, 2 }, { 2, 2 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 2, 4 }, { 6, 8 } } };
            mat1 *= mat2;
            if (fsmlib::any(mat1 != expected)) {
                throw std::runtime_error("Test failed: Matrix *= Matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix *= Matrix\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1     = { { { 10, 20 }, { 30, 40 } } };
            fsmlib::Matrix<int, 2, 2> mat2     = { { { 2, 4 }, { 5, 10 } } };
            fsmlib::Matrix<int, 2, 2> expected = { { { 5, 5 }, { 6, 4 } } };
            mat1 /= mat2;
            if (fsmlib::any(mat1 != expected)) {
                throw std::runtime_error("Test failed: Matrix /= Matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix /= Matrix\n";
        }

        {
            fsmlib::Vector<int, 3> vec1      = { 5, 3, 7 };
            int scalar                       = 4;
            fsmlib::Vector<bool, 3> expected = { true, false, true };
            auto result                      = vec1 > scalar;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector > Scalar");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector > Scalar\n";
        }

        {
            fsmlib::Vector<int, 3> vec1      = { 5, 3, 7 };
            fsmlib::Vector<int, 3> vec2      = { 4, 3, 8 };
            fsmlib::Vector<bool, 3> expected = { true, false, false };
            auto result                      = vec1 > vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector > Vector");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector > Vector\n";
        }

        {
            fsmlib::Vector<int, 3> vec1      = { 5, 3, 7 };
            int scalar                       = 6;
            fsmlib::Vector<bool, 3> expected = { true, true, false };
            auto result                      = vec1 < scalar;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector < Scalar");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector < Scalar\n";
        }

        {
            fsmlib::Vector<int, 3> vec1      = { 5, 3, 7 };
            fsmlib::Vector<int, 3> vec2      = { 4, 3, 8 };
            fsmlib::Vector<bool, 3> expected = { false, false, true };
            auto result                      = vec1 < vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector < Vector");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector < Vector\n";
        }

        {
            fsmlib::Vector<int, 3> vec1      = { 5, 3, 7 };
            int scalar                       = 5;
            fsmlib::Vector<bool, 3> expected = { true, false, true };
            auto result                      = vec1 >= scalar;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector >= Scalar");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector >= Scalar\n";
        }

        {
            fsmlib::Vector<int, 3> vec1      = { 5, 3, 7 };
            fsmlib::Vector<int, 3> vec2      = { 5, 2, 7 };
            fsmlib::Vector<bool, 3> expected = { true, true, true };
            auto result                      = vec1 >= vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector >= Vector");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector >= Vector\n";
        }

        {
            fsmlib::Vector<int, 3> vec1      = { 5, 3, 7 };
            int scalar                       = 5;
            fsmlib::Vector<bool, 3> expected = { true, true, false };
            auto result                      = vec1 <= scalar;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector <= Scalar");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector <= Scalar\n";
        }

        {
            fsmlib::Vector<int, 3> vec1      = { 5, 3, 7 };
            fsmlib::Vector<int, 3> vec2      = { 5, 4, 8 };
            fsmlib::Vector<bool, 3> expected = { true, true, true };
            auto result                      = vec1 <= vec2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Vector <= Vector");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector <= Vector\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat       = { { { 5, 3 }, { 7, 4 } } };
            int scalar                          = 4;
            fsmlib::Matrix<bool, 2, 2> expected = { { { true, false }, { true, false } } };
            auto result                         = mat > scalar;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix > Scalar");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix > Scalar\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1      = { { { 5, 3 }, { 7, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2      = { { { 4, 3 }, { 8, 3 } } };
            fsmlib::Matrix<bool, 2, 2> expected = { { { true, false }, { false, true } } };
            auto result                         = mat1 > mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix > Matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix > Matrix\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat       = { { { 5, 3 }, { 7, 4 } } };
            int scalar                          = 6;
            fsmlib::Matrix<bool, 2, 2> expected = { { { true, true }, { false, true } } };
            auto result                         = mat < scalar;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix < Scalar");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix < Scalar\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1      = { { { 5, 3 }, { 7, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2      = { { { 6, 2 }, { 8, 5 } } };
            fsmlib::Matrix<bool, 2, 2> expected = { { { true, false }, { true, true } } };
            auto result                         = mat1 < mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix < Matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix < Matrix\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat       = { { { 5, 3 }, { 7, 4 } } };
            int scalar                          = 5;
            fsmlib::Matrix<bool, 2, 2> expected = { { { true, false }, { true, false } } };
            auto result                         = mat >= scalar;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix >= Scalar");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix >= Scalar\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1      = { { { 5, 3 }, { 7, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2      = { { { 5, 2 }, { 6, 4 } } };
            fsmlib::Matrix<bool, 2, 2> expected = { { { true, true }, { true, true } } };
            auto result                         = mat1 >= mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix >= Matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix >= Matrix\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat       = { { { 5, 3 }, { 7, 4 } } };
            int scalar                          = 5;
            fsmlib::Matrix<bool, 2, 2> expected = { { { true, true }, { false, true } } };
            auto result                         = mat <= scalar;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix <= Scalar");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix <= Scalar\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat1      = { { { 5, 3 }, { 7, 4 } } };
            fsmlib::Matrix<int, 2, 2> mat2      = { { { 5, 4 }, { 7, 5 } } };
            fsmlib::Matrix<bool, 2, 2> expected = { { { true, true }, { true, true } } };
            auto result                         = mat1 <= mat2;
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix <= Matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix <= Matrix\n";
        }
        {
            fsmlib::Vector<int, 3> vec1     = { 1, 2, 3 };
            fsmlib::Vector<int, 5> vec2     = { 10, 20, 30, 40, 50 };
            fsmlib::Vector<int, 3> expected = { 31, 42, 53 };

            auto view   = fsmlib::view<int, 3>(vec2, 2);
            auto result = vec1 + view;

            if (fsmlib::any(result != expected)) {
                std::cerr << "Expected: " << expected << "\n";
                std::cerr << "Result:   " << result << "\n";
                throw std::runtime_error("Test failed: Vector + View addition with offset");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector + View addition with offset\n";
        }

        {
            fsmlib::Vector<int, 3> vec1     = { 50, 60, 70 };
            fsmlib::Vector<int, 5> vec2     = { 10, 20, 30, 40, 50 };
            fsmlib::Vector<int, 3> expected = { 30, 30, 30 };

            auto view   = fsmlib::view<int, 3>(vec2, 1);
            auto result = vec1 - view;

            if (fsmlib::any(result != expected)) {
                std::cerr << expected << "\n";
                std::cerr << result << "\n";
                throw std::runtime_error("Test failed: Vector - View subtraction with offset");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector - View subtraction with offset\n";
        }

        {
            fsmlib::Vector<int, 3> vec1     = { 2, 3, 4 };
            fsmlib::Vector<int, 5> vec2     = { 1, 2, 3, 4, 5 };
            fsmlib::Vector<int, 3> expected = { 6, 12, 20 };

            auto view   = fsmlib::view<int, 3>(vec2, 2);
            auto result = vec1 * view;

            if (fsmlib::any(result != expected)) {
                std::cerr << expected << "\n";
                std::cerr << result << "\n";
                throw std::runtime_error("Test failed: Vector * View multiplication with offset");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector * View multiplication with offset\n";
        }

        {
            fsmlib::Vector<int, 3> vec1     = { 40, 50, 60 };
            fsmlib::Vector<int, 5> vec2     = { 1, 2, 4, 5, 6 };
            fsmlib::Vector<int, 3> expected = { 10, 10, 10 };

            auto view   = fsmlib::view<int, 3>(vec2, 2);
            auto result = vec1 / view;

            if (fsmlib::any(result != expected)) {
                std::cerr << expected << "\n";
                std::cerr << result << "\n";
                throw std::runtime_error("Test failed: Vector / View division with offset");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector / View division with offset\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> A = { { 1.0, 2.0 }, { 3.0, 4.0 } };
            double tr                      = trace(A);
            double expected                = 1.0 + 4.0;
            if (std::abs(tr - expected) > 1e-6) {
                std::cerr << "Expected trace: " << expected << "\nGot trace: " << tr << "\n";
                throw std::runtime_error("Test failed: trace of a 2x2 matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: trace of a 2x2 matrix\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> B = { { 5.0, 2.0, 3.0 }, { 1.0, 6.0, 4.0 }, { 7.0, 8.0, 9.0 } };
            double tr                      = trace(B);
            double expected                = 5.0 + 6.0 + 9.0;
            if (std::abs(tr - expected) > 1e-6) {
                std::cerr << "Expected trace: " << expected << "\nGot trace: " << tr << "\n";
                throw std::runtime_error("Test failed: trace of a 3x3 matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: trace of a 3x3 matrix\n";
        }

        std::cout << "All math tests passed!\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
