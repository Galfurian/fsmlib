
#include <fsmlib/fsmlib.hpp>
#include <fsmlib/linalg.hpp>

#include <iostream>
#include <iomanip>

int main()
{
    try {
        int test_count = 1; // Test counter variable

        {
            fsmlib::Matrix<int, 2, 3> mat      = { { { 1, 2, 3 }, { 4, 5, 6 } } };
            fsmlib::Matrix<int, 3, 2> expected = { { { 1, 4 }, { 2, 5 }, { 3, 6 } } };
            auto result                        = fsmlib::linalg::transpose(mat);
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Matrix transpose");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix transpose\n";
        }

        {
            fsmlib::Vector<int, 3> vec1 = { 1, 2, 3 };
            fsmlib::Vector<int, 3> vec2 = { 4, 5, 6 };
            int expected                = 32; // 1*4 + 2*5 + 3*6
            auto result                 = fsmlib::inner_product(vec1, vec2);
            if (result != expected) {
                throw std::runtime_error("Test failed: Inner product");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Inner product\n";
        }

        {
            fsmlib::Vector<int, 2> vec1        = { 1, 2 };
            fsmlib::Vector<int, 3> vec2        = { 3, 4, 5 };
            fsmlib::Matrix<int, 2, 3> expected = { { { 3, 4, 5 }, { 6, 8, 10 } } };
            auto result                        = fsmlib::outer_product(vec1, vec2);
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Outer product");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Outer product\n";
        }

        {
            auto result                        = fsmlib::eye<int, 3>();
            fsmlib::Matrix<int, 3, 3> expected = { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } };
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Identity matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Identity matrix\n";
        }
        {
            fsmlib::Vector<int, 3> vec = { 3, 4, 5 };
            auto result                = fsmlib::linalg::square_norm(vec);
            double expected            = std::sqrt(3 * 3 + 4 * 4 + 5 * 5);
            if (!fsmlib::details::approximately_equal(result, expected)) {
                throw std::runtime_error("Test failed: Square norm of a vector");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Square norm of a vector\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> mat = { { { 1, 2 }, { 3, 4 } } };
            auto result                   = fsmlib::linalg::square_norm(mat);
            double expected               = std::sqrt(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4);
            if (!fsmlib::details::approximately_equal(result, expected)) {
                throw std::runtime_error("Test failed: Frobenius norm of a matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Frobenius norm of a matrix\n";
        }

        {
            fsmlib::Matrix<int, 3> mat      = { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } };
            auto result                     = fsmlib::linalg::cofactor(mat, 1, 1);
            fsmlib::Matrix<int, 2> expected = { { { 1, 3 }, { 7, 9 } } };
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Cofactor matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Cofactor matrix\n";
        }

        {
            fsmlib::Matrix<int, 2> mat = { { { 1, 2 }, { 3, 4 } } };
            auto result                = fsmlib::linalg::determinant(mat);
            int expected               = -2; // 1*4 - 2*3
            if (result != expected) {
                throw std::runtime_error("Test failed: Determinant of a matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Determinant of a matrix\n";
        }

        {
            fsmlib::Matrix<int, 2> mat      = { { { 1, 2 }, { 3, 4 } } };
            auto result                     = fsmlib::linalg::adjoint(mat);
            fsmlib::Matrix<int, 2> expected = { { { 4, -2 }, { -3, 1 } } };
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Adjoint of a matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Adjoint of a matrix\n";
        }

        {
            fsmlib::Matrix<double, 2> mat      = { { { 1, 2 }, { 3, 4 } } };
            auto result                        = fsmlib::linalg::inverse(mat);
            fsmlib::Matrix<double, 2> expected = { { { -2.0, 1.0 }, { 1.5, -0.5 } } };
            if (!fsmlib::details::approximately_equal(result[0][0], expected[0][0]) ||
                !fsmlib::details::approximately_equal(result[0][1], expected[0][1]) ||
                !fsmlib::details::approximately_equal(result[1][0], expected[1][0]) ||
                !fsmlib::details::approximately_equal(result[1][1], expected[1][1])) {
                throw std::runtime_error("Test failed: Matrix inverse");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix inverse\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> mat = {
                { { 0.1, 0.2 }, { 0.3, 0.4 } }
            };
            auto [iterations, scale] = fsmlib::linalg::log2_ceil(mat);

            if (iterations != 0 || scale != 1.0) {
                std::cerr << "Expected: iterations = 0, scale = 1.0\n";
                std::cerr << "Got: iterations = " << iterations << ", scale = " << scale << "\n";
                throw std::runtime_error("Test failed: log2_ceil");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: log2_ceil\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> mat = {
                { { 1024.0, 512.0 }, { 512.0, 1024.0 } }
            };
            auto [iterations, scale] = fsmlib::linalg::log2_ceil(mat);

            if (iterations != 11 || std::abs(scale - 0.000488281) > 1e-9) {
                std::cerr << "Expected: iterations = 11, scale = 0.000488281\n";
                std::cerr << "Got: iterations = " << iterations << ", scale = " << scale << "\n";
                throw std::runtime_error("Test failed: log2_ceil");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: log2_ceil\n";
        }

        std::cout << "All linear algebra tests passed!\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
