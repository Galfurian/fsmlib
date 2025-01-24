

#include <iostream>
#include <iomanip>

#include <fsmlib/io.hpp>
#include <fsmlib/fsmlib.hpp>
#include <fsmlib/linalg.hpp>

int main()
{
    try {
        int test_count = 1; // Test counter variable

        {
            fsmlib::Matrix<int, 2, 3> mat = {
                { 1, 2, 3 },
                { 4, 5, 6 },
            };
            fsmlib::Matrix<int, 3, 2> expected = {
                { 1, 4 },
                { 2, 5 },
                { 3, 6 },
            };
            auto result = fsmlib::linalg::transpose(mat);
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
            fsmlib::Matrix<int, 2, 3> expected = {
                { 3, 4, 5 },
                { 6, 8, 10 },
            };
            auto result = fsmlib::outer_product(vec1, vec2);
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Outer product");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Outer product\n";
        }

        {
            auto result                        = fsmlib::eye<int, 3>();
            fsmlib::Matrix<int, 3, 3> expected = {
                { 1, 0, 0 },
                { 0, 1, 0 },
                { 0, 0, 1 },
            };
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
            fsmlib::Matrix<int, 2, 2> mat = {
                { 1, 2 },
                { 3, 4 },
            };
            auto result     = fsmlib::linalg::square_norm(mat);
            double expected = std::sqrt(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4);
            if (!fsmlib::details::approximately_equal(result, expected)) {
                throw std::runtime_error("Test failed: Frobenius norm of a matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: Frobenius norm of a matrix\n";
        }

        {
            fsmlib::Matrix<int, 3> mat = {
                { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 },
            };
            auto result                     = fsmlib::linalg::cofactor(mat, 1, 1);
            fsmlib::Matrix<int, 2> expected = {
                { 1, 3 },
                { 7, 9 },
            };
            if (fsmlib::any(result != expected)) {
                throw std::runtime_error("Test failed: Cofactor matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Cofactor matrix\n";
        }

        {
            fsmlib::Matrix<int, 2> mat = {
                { 1, 2 },
                { 3, 4 },
            };
            auto result  = fsmlib::linalg::determinant(mat);
            int expected = -2; // 1*4 - 2*3
            if (result != expected) {
                std::cerr << "Test failed: determinant for matrix\n"
                          << "Expected: " << expected << ", Got: " << result << "\n";
                throw std::runtime_error("Test failed: Determinant of a matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Determinant of a matrix\n";
        }
        {
            // Define a 2x2 matrix
            fsmlib::Matrix<double, 2, 2> A = { { 1.5, -2.3 }, { 3.1, 0.7 } };
            // Compute determinant
            double det = fsmlib::linalg::determinant(A);
            // Expected result
            double expected = 1.5 * 0.7 - (-2.3) * 3.1; // 1.05 + 7.13 = 8.18
            // Validate
            if (std::abs(det - expected) > 1e-06) {
                std::cerr << "Test failed: determinant for 2x2 matrix\n"
                          << "Expected: " << expected << ", Got: " << det << "\n";
                throw std::runtime_error("Test failed: determinant for 2x2 matrix");
            }
            std::cout << "Test 1 passed: determinant for 2x2 matrix\n";
        }
        {
            // Define a 3x3 matrix
            fsmlib::Matrix<double, 3, 3> A = { { 2.1, -1.2, 0.5 }, { 4.3, 1.7, -3.0 }, { 0.0, 2.2, -0.8 } };
            // Compute determinant
            double det = fsmlib::linalg::determinant(A);
            // Expected result (computed using Octave or another trusted source)
            double expected = 2.1 * (1.7 * -0.8 - (-3.0) * 2.2) - (-1.2) * (4.3 * -0.8 - 0.0 * 2.2) +
                              0.5 * (4.3 * 2.2 - 0.0 * 1.7); // Expected: -6.276
            // Validate
            if (std::abs(det - expected) > 1e-06) {
                std::cerr << "Test failed: determinant for 3x3 matrix\n"
                          << "Expected: " << expected << ", Got: " << det << "\n";
                throw std::runtime_error("Test failed: determinant for 3x3 matrix");
            }
            std::cout << "Test 2 passed: determinant for 3x3 matrix\n";
        }
        {
            // Define a 4x4 matrix
            fsmlib::Matrix<double, 4, 4> A = {
                { 1.5, -2.3, 0.7, 1.2 }, { 3.1, 0.4, -1.2, -0.8 }, { 0.0, 4.0, -0.5, 2.6 }, { 1.3, 2.2, -3.3, 0.9 }
            };
            // Compute determinant
            double det = fsmlib::linalg::determinant(A);
            // Expected result (computed using Octave or another trusted source)
            double expected = 100.118; // Example placeholder
            // Validate
            if (std::abs(det - expected) > 1e-03) {
                std::cerr << "Test failed: determinant for 4x4 matrix\n"
                          << "Expected: " << expected << ", Got: " << det << "\n";
                throw std::runtime_error("Test failed: determinant for 4x4 matrix");
            }
            std::cout << "Test 3 passed: determinant for 4x4 matrix\n";
        }
        {
            // Define a 5x5 matrix
            fsmlib::Matrix<double, 5, 5> A = { { 1, 2, 3, 4, 5 },
                                               { 6, 7, 8, 9, 10 },
                                               { 11, 12, 13, 14, 15 },
                                               { 16, 17, 18, 19, 20 },
                                               { 21, 22, 23, 24, 25 } };

            // Compute determinant
            double det = fsmlib::linalg::determinant(A);
            // Expected result (computed using Octave or another trusted source)
            double expected = 0.0; // Matrix is singular
            // Validate
            if (std::abs(det - expected) > 1e-06) {
                std::cerr << "Test failed: determinant for 5x5 singular matrix\n"
                          << "Expected: " << expected << ", Got: " << det << "\n";
                throw std::runtime_error("Test failed: determinant for 5x5 singular matrix");
            }
            std::cout << "Test 4 passed: determinant for 5x5 singular matrix\n";
        }
        {
            // Define a 5x5 matrix
            fsmlib::Matrix<double, 5, 5> A = { { 1.1, 0.5, -1.2, 3.4, 0.9 },
                                               { -2.3, 4.7, 5.5, -0.6, 1.0 },
                                               { 3.8, -1.2, 2.1, 4.0, -3.3 },
                                               { 5.6, -4.1, 0.0, 1.3, -2.2 },
                                               { 2.2, 3.9, -1.1, 0.8, 0.5 } };
            // Compute determinant
            double det = fsmlib::linalg::determinant(A);
            // Expected result (computed using Octave or another trusted source)
            double expected = 1646.54744; // Example placeholder
            // Validate
            if (std::abs(det - expected) > 1e-06) {
                std::cerr << "Test failed: determinant for 5x5 matrix\n"
                          << "Expected: " << expected << ", Got: " << det << "\n";
                throw std::runtime_error("Test failed: determinant for 5x5 matrix");
            }
            std::cout << "Test 5 passed: determinant for 5x5 matrix\n";
        }

        {
            fsmlib::Matrix<int, 2> mat = {
                { 1, 2 },
                { 3, 4 },
            };
            auto result                     = fsmlib::linalg::adjoint(mat);
            fsmlib::Matrix<int, 2> expected = {
                { 4, -2 },
                { -3, 1 },
            };
            if (fsmlib::any(result != expected)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: Adjoint of a matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Adjoint of a matrix\n";
        }

        {
            fsmlib::Matrix<double, 2> mat = {
                { 1, 2 },
                { 3, 4 },
            };
            auto result                        = fsmlib::linalg::inverse(mat);
            fsmlib::Matrix<double, 2> expected = {
                { -2.0, 1.0 },
                { 1.5, -0.5 },
            };
            if (!fsmlib::details::approximately_equal(result(0, 0), expected(0, 0)) ||
                !fsmlib::details::approximately_equal(result(0, 1), expected(0, 1)) ||
                !fsmlib::details::approximately_equal(result(1, 0), expected(1, 0)) ||
                !fsmlib::details::approximately_equal(result(1, 1), expected(1, 1))) {
                throw std::runtime_error("Test failed: Matrix inverse");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix inverse\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> mat = {
                { 0.1, 0.2 },
                { 0.3, 0.4 },
            };
            auto [iterations, scale] = fsmlib::linalg::scale_to_unit_norm(mat);

            if (iterations != 0 || !fsmlib::feq::approximately_equal(scale, 1.0)) {
                std::cerr << "Expected: iterations = 0, scale = 1.0\n";
                std::cerr << "Got: iterations = " << iterations << ", scale = " << scale << "\n";
                throw std::runtime_error("Test failed: scale_to_unit_norm");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: scale_to_unit_norm\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> mat = {
                { 1024.0, 512.0 },
                { 512.0, 1024.0 },
            };
            auto [iterations, scale] = fsmlib::linalg::scale_to_unit_norm(mat);

            if (iterations != 11 || std::abs(scale - 0.000488281) > 1e-9) {
                std::cerr << "Expected: iterations = 11, scale = 0.000488281\n";
                std::cerr << "Got: iterations = " << iterations << ", scale = " << scale << "\n";
                throw std::runtime_error("Test failed: scale_to_unit_norm");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: scale_to_unit_norm\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> mat = {
                { 0, 1 },
                { -2, -3 },
            };
            auto result = fsmlib::linalg::expm(mat, 1e-12);

            // Updated expected result from Octave
            fsmlib::Matrix<double, 2, 2> expected = {
                { 0.600424, 0.232544 },
                { -0.465088, -0.097209 },
            };

            if (fsmlib::any(fsmlib::abs(result - expected) > 0.1)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: expm with simple 2x2 matrix");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: expm with simple 2x2 matrix\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> mat = {
                { 0.0, 1.0 },
                { -10.0, -2.0 },
            };

            auto result = fsmlib::linalg::expm(mat, 1e-12);

            // Updated expected result from Octave
            fsmlib::Matrix<double, 2, 2> expected = {
                { -0.346893, 0.017305 },
                { -0.173050, -0.381503 },
            };

            if (fsmlib::any(fsmlib::abs(result - expected) > 0.1)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: expm with simple 2x2 matrix");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: expm with simple 2x2 matrix\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> mat = {
                { 1.0, 2.0 },
                { 3.0, 4.0 },
            };
            auto [Q, R] = fsmlib::linalg::qr_decomposition(mat);

            fsmlib::Matrix<double, 2, 2> expected_Q = {
                { -0.316228, -0.948683 },
                { -0.948683, 0.316228 },
            };
            fsmlib::Matrix<double, 2, 2> expected_R = {
                { -3.162278, -4.427189 },
                { 0.000000, -0.632456 },
            };

            auto reconstructed = fsmlib::multiply(Q, R);

            if (fsmlib::any(fsmlib::abs(Q - expected_Q) > 1e-3)) {
                std::cerr << "Expected Q:\n" << expected_Q << "\nGot Q:\n" << Q << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with simple 2x2 matrix");
            }
            if (fsmlib::any(fsmlib::abs(R - expected_R) > 1e-3)) {
                std::cerr << "Expected R:\n" << expected_R << "\nGot R:\n" << R << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with simple 2x2 matrix");
            }
            if (fsmlib::any(fsmlib::abs(reconstructed - mat) > 1e-3)) {
                std::cerr << "Expected mat:\n" << mat << "\nGot Reconstructed:\n" << reconstructed << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with simple 2x2 matrix");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: qr_decomposition with simple 2x2 matrix\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> mat = {
                { 12.0, -51.0, 4.0 },
                { 6.0, 167.0, -68.0 },
                { -4.0, 24.0, -41.0 },
            };
            auto [Q, R] = fsmlib::linalg::qr_decomposition(mat);

            fsmlib::Matrix<double, 3, 3> expected_Q = {
                { -0.857143, 0.394286, 0.331429 },
                { -0.428571, -0.902857, -0.034286 },
                { 0.285714, -0.171429, 0.942857 },
            };
            fsmlib::Matrix<double, 3, 3> expected_R = {
                { -14.000000, -21.000000, 14.000000 },
                { 0.000000, -175.000000, 70.000000 },
                { 0.000000, 0.000000, -35.000000 },
            };

            auto reconstructed = fsmlib::multiply(Q, R);

            if (fsmlib::any(fsmlib::abs(Q - expected_Q) > 1e-3)) {
                std::cerr << "Expected Q:\n" << expected_Q << "\nGot Q:\n" << Q << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with 3x3 matrix");
            }
            if (fsmlib::any(fsmlib::abs(R - expected_R) > 1e-3)) {
                std::cerr << "Expected R:\n" << expected_R << "\nGot R:\n" << R << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with 3x3 matrix");
            }
            if (fsmlib::any(fsmlib::abs(reconstructed - mat) > 1e-3)) {
                std::cerr << "Expected mat:\n" << mat << "\nGot Reconstructed:\n" << reconstructed << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with 3x3 matrix");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: qr_decomposition with 3x3 matrix\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> mat = {
                { 2.0, -1.0, -2.0 },
                { -4.0, 6.0, 3.0 },
                { -4.0, -2.0, 8.0 },
            };

            auto [L, U] = fsmlib::linalg::lu_decomposition(mat);

            fsmlib::Matrix<double, 3, 3> expected_L = {
                { 1.000000, 0.000000, 0.000000 },
                { -2.000000, 1.000000, 0.000000 },
                { -2.000000, -1.000000, 1.000000 },
            };
            fsmlib::Matrix<double, 3, 3> expected_U = {
                { 2.000000, -1.000000, -2.000000 },
                { 0.000000, 4.000000, -1.000000 },
                { 0.000000, 0.000000, 3.000000 },
            };

            auto reconstructed = fsmlib::multiply(L, U);

            if (fsmlib::any(fsmlib::abs(L - expected_L) > 1e-3)) {
                std::cerr << "Expected L:\n" << expected_L << "\nGot L:\n" << L << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 3x3 matrix (L)");
            }
            if (fsmlib::any(fsmlib::abs(U - expected_U) > 1e-3)) {
                std::cerr << "Expected U:\n" << expected_U << "\nGot U:\n" << U << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 3x3 matrix (U)");
            }
            if (fsmlib::any(fsmlib::abs(reconstructed - mat) > 1e-3)) {
                std::cerr << "Expected mat:\n" << mat << "\nGot Reconstructed:\n" << reconstructed << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 3x3 matrix (Reconstruction)");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: lu_decomposition with 3x3 matrix\n";
        }

        {
            fsmlib::Matrix<double, 4, 4> mat = {
                { 4.0, 3.0, 2.0, 1.0 },
                { 3.0, 4.0, 1.0, 2.0 },
                { 2.0, 1.0, 4.0, 3.0 },
                { 1.0, 2.0, 3.0, 4.0 },
            };

            auto [L, U] = fsmlib::linalg::lu_decomposition(mat);

            fsmlib::Matrix<double, 4, 4> expected_L = {
                { 1.0000, 0.0000, 0.0000, 0.0000 },
                { 0.7500, 1.0000, 0.0000, 0.0000 },
                { 0.5000, -0.2857, 1.0000, 0.0000 },
                { 0.2500, 0.7143, 1.0000, 1.0000 },
            };
            fsmlib::Matrix<double, 4, 4> expected_U = {
                { 4.0000, 3.0000, 2.0000, 1.0000 },
                { 0.0000, 1.7500, -0.5000, 1.2500 },
                { 0.0000, 0.0000, 2.8571, 2.8571 },
                { 0.0000, 0.0000, 0.0000, 0.0000 },
            };

            auto reconstructed = fsmlib::multiply(L, U);

            if (fsmlib::any(fsmlib::abs(L - expected_L) > 1e-3)) {
                std::cerr << "Expected L:\n" << expected_L << "\nGot L:\n" << L << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 4x4 matrix (L)");
            }
            if (fsmlib::any(fsmlib::abs(U - expected_U) > 1e-3)) {
                std::cerr << "Expected U:\n" << expected_U << "\nGot U:\n" << U << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 4x4 matrix (U)");
            }
            if (fsmlib::any(fsmlib::abs(reconstructed - mat) > 1e-3)) {
                std::cerr << "Expected mat:\n" << mat << "\nGot Reconstructed:\n" << reconstructed << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 4x4 matrix (Reconstruction)");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: lu_decomposition with 4x4 matrix\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> A = {
                { 3.0, 2.0 },
                { 1.0, 4.0 },
            };
            fsmlib::Vector<double, 2> b = { 10.0, 11.0 };

            auto x = fsmlib::linalg::solve(A, b);

            fsmlib::Vector<double, 2> expected_x = { 1.8, 2.3 };

            if (fsmlib::any(fsmlib::abs(x - expected_x) > 1e-3)) {
                std::cerr << "Expected x:\n" << expected_x << "\nGot x:\n" << x << "\n";
                throw std::runtime_error("Test failed: solve with 2x2 system");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: solve with 2x2 system\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> A = {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 },
                { 7.0, 8.0, 10.0 },
            };
            fsmlib::Vector<double, 3> b = { 6.0, 15.0, 25.0 };

            auto x = fsmlib::linalg::solve(A, b);

            fsmlib::Vector<double, 3> expected_x = { 1.0, 1.0, 1.0 };

            if (fsmlib::any(fsmlib::abs(x - expected_x) > 1e-3)) {
                std::cerr << "Expected x:\n" << expected_x << "\nGot x:\n" << x << "\n";
                throw std::runtime_error("Test failed: solve with 3x3 system");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: solve with 3x3 system\n";
        }

        {
            fsmlib::Matrix<double, 4, 4> A = {
                { 1.5, -3.2, 2.7, 0.8 },
                { -4.1, 5.6, -1.9, 3.3 },
                { 2.0, 3.7, -6.4, -2.5 },
                { -1.3, -2.6, 4.5, 7.8 },
            };
            fsmlib::Vector<double, 4> b = { 4.8, -3.2, 5.9, -7.6 };

            auto x = fsmlib::linalg::solve(A, b);

            fsmlib::Vector<double, 4> expected_x = { 10.0840, 10.0891, 8.3556, -0.7512 };

            auto reconstructed = fsmlib::multiply(A, x);

            if (fsmlib::any(fsmlib::abs(x - expected_x) > 1e-3)) {
                std::cerr << "Expected x:\n" << expected_x << "\nGot x:\n" << x << "\n";
                throw std::runtime_error("Test failed: solve with 4x4 system (solution)");
            }
            if (fsmlib::any(fsmlib::abs(reconstructed - b) > 1e-3)) {
                std::cerr << "Expected b:\n" << b << "\nGot Reconstructed:\n" << reconstructed << "\n";
                throw std::runtime_error("Test failed: solve with 4x4 system (reconstruction)");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: solve with 4x4 system\n";
        }

        {
            // Define a symmetric positive-definite matrix
            fsmlib::Matrix<double, 3, 3> A = {
                { 4.0, 12.0, -16.0 },
                { 12.0, 37.0, -43.0 },
                { -16.0, -43.0, 98.0 },
            };

            // Perform Cholesky decomposition
            auto L = fsmlib::linalg::cholesky_decomposition(A);

            // Expected lower triangular matrix
            fsmlib::Matrix<double, 3, 3> expected_L = {
                { 2.0, 0.0, 0.0 },
                { 6.0, 1.0, 0.0 },
                { -8.0, 5.0, 3.0 },
            };

            // Reconstruct the original matrix from L
            auto reconstructed_A = fsmlib::multiply(L, fsmlib::linalg::transpose(L));

            // Check if the decomposition matches the expected result
            if (fsmlib::any(fsmlib::abs(L - expected_L) > 1e-3)) {
                std::cerr << "Expected L:\n" << expected_L << "\nGot L:\n" << L << "\n";
                throw std::runtime_error("Test failed: Cholesky decomposition (L mismatch)");
            }

            // Check if reconstruction matches the original matrix
            if (fsmlib::any(fsmlib::abs(reconstructed_A - A) > 1e-3)) {
                std::cerr << "Expected A:\n" << A << "\nGot Reconstructed A:\n" << reconstructed_A << "\n";
                throw std::runtime_error("Test failed: Cholesky decomposition (reconstruction mismatch)");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Cholesky decomposition\n";
        }

        {
            // Define a 4x4 matrix
            fsmlib::Matrix<double, 4, 4> mat = {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0, 16.0 },
            };

            // Compute the rank
            std::size_t computed_rank = fsmlib::linalg::rank(mat);

            // Expected rank
            std::size_t expected_rank = 2;

            // Verify the result
            if (computed_rank != expected_rank) {
                std::cerr << "Expected rank: " << expected_rank << "\nGot rank: " << computed_rank << "\n";
                throw std::runtime_error("Test failed: rank computation for 4x4 matrix");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: rank computation for 4x4 matrix\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> A = {
                { 4.0, 1.0 },
                { 1.0, 3.0 },
            };
            auto [lambda, v]                     = fsmlib::linalg::power_iteration(A);
            double expected_lambda               = 4.6180;
            fsmlib::Vector<double, 2> expected_v = { 0.851, 0.526 };
            if (std::abs(lambda - expected_lambda) > 1e-3) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_lambda << "\n";
                std::cerr << "Got      :\n" << lambda << "\n";
                throw std::runtime_error("Test failed: power_iteration - lambda mismatch");
            }
            if (fsmlib::any(fsmlib::abs(v - expected_v) > 1e-3)) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_v << "\n";
                std::cerr << "Got      :\n" << v << "\n";
                throw std::runtime_error("Test failed: power_iteration - eigenvector mismatch");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: Power iteration on 2x2 symmetric matrix\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> A = {
                { 2.0, -1.0, 0.0 },
                { -1.0, 2.0, -1.0 },
                { 0.0, -1.0, 2.0 },
            };
            auto [lambda, v]                     = fsmlib::linalg::power_iteration(A);
            double expected_lambda               = 3.4142;
            fsmlib::Vector<double, 3> expected_v = { -0.5, 0.707, -0.5 };
            if (std::abs(lambda - expected_lambda) > 1e-3) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_lambda << "\n";
                std::cerr << "Got      :\n" << lambda << "\n";
                throw std::runtime_error("Test failed: power_iteration - lambda mismatch");
            }
            if (fsmlib::any(fsmlib::abs(v - expected_v) > 1e-3)) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_v << "\n";
                std::cerr << "Got      :\n" << v << "\n";
                throw std::runtime_error("Test failed: power_iteration - eigenvector mismatch");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: Power iteration on 3x3 symmetric matrix\n";
        }

        {
            fsmlib::Matrix<double, 4, 4> A = {
                { 10.0, 2.0, 3.0, 1.0 },
                { 2.0, 6.0, 2.0, 2.0 },
                { 3.0, 2.0, 7.0, 3.0 },
                { 1.0, 2.0, 3.0, 8.0 },
            };
            auto [lambda, v]                     = fsmlib::linalg::power_iteration(A);
            double expected_lambda               = 14.5130;
            fsmlib::Vector<double, 4> expected_v = { 0.6157, 0.3744, 0.5260, 0.4518 };
            if (std::abs(lambda - expected_lambda) > 1e-3) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_lambda << "\n";
                std::cerr << "Got      :\n" << lambda << "\n";
                throw std::runtime_error("Test failed: power_iteration - lambda mismatch");
            }
            if (fsmlib::any(fsmlib::abs(v - expected_v) > 1e-3)) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_v << "\n";
                std::cerr << "Got      :\n" << v << "\n";
                throw std::runtime_error("Test failed: power_iteration - eigenvector mismatch");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: Power iteration on 4x4 symmetric matrix\n";
        }

        {
            fsmlib::Matrix<double, 4, 4> A = {
                { 5.5, -2.3, 1.7, 3.9 },
                { -2.3, 6.8, 2.1, -1.4 },
                { 1.7, 2.1, 7.2, 0.5 },
                { 3.9, -1.4, 0.5, 4.3 },
            };
            auto [lambda, v]                     = fsmlib::linalg::power_iteration(A);
            double expected_lambda               = 10.688;
            fsmlib::Vector<double, 4> expected_v = { 0.655906, -0.538693, 0.070480, 0.524050 };
            if (std::abs(lambda - expected_lambda) > 1e-3) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_lambda << "\n";
                std::cerr << "Got      :\n" << lambda << "\n";
                throw std::runtime_error("Test failed: power_iteration - lambda mismatch");
            }
            if (fsmlib::any(fsmlib::abs(v - expected_v) > 1e-3)) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_v << "\n";
                std::cerr << "Got      :\n" << v << "\n";
                throw std::runtime_error("Test failed: power_iteration - eigenvector mismatch");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: Power iteration on 4x4 symmetric matrix\n";
        }

        {
            // Define a 2x2 symmetric matrix
            fsmlib::Matrix<double, 2, 2> A = {
                { 4.0, 1.0 },
                { 1.0, 3.0 },
            };
            // Perform eigen decomposition
            auto [eigenvalues, eigenvectors] = fsmlib::linalg::eigen(A, 2000, 1e-32);
            // Expected eigenvalues
            fsmlib::Vector<double, 2> expected_eigenvalues = { 4.618, 2.382 };
            // Expected eigenvectors
            fsmlib::Matrix<double, 2, 2> expected_eigenvectors = {
                { 0.8507, -0.5257 },
                { 0.5257, 0.8507 },
            };

            // Verify eigenvalues by comparing with expected values.
            if (fsmlib::any(fsmlib::abs(fsmlib::abs(eigenvalues) - fsmlib::abs(expected_eigenvalues)) > 1e-3)) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_eigenvalues << "\n";
                std::cerr << "Got      :\n" << eigenvalues << "\n";
                throw std::runtime_error("Test failed: eigenvalues mismatch");
            }
            // Verify eigenvectors by comparing each element of the matrix
            if (fsmlib::any(fsmlib::abs(fsmlib::abs(eigenvectors) - fsmlib::abs(expected_eigenvectors)) > 1e-3)) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_eigenvectors << "\n";
                std::cerr << "Got      :\n" << eigenvectors << "\n";
                throw std::runtime_error("Test failed: eigenvectors mismatch");
            }
            // If no mismatches are found, the test passes
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: eigen decomposition for 2x2 matrix\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> A = {
                { 2.0, -1.0, 0.0 },
                { -1.0, 2.0, -1.0 },
                { 0.0, -1.0, 2.0 },
            };
            // Perform eigen decomposition
            auto [eigenvalues, eigenvectors] = fsmlib::linalg::eigen(A, 2000, 1e-32);
            // Expected eigenvalues
            fsmlib::Vector<double, 3> expected_eigenvalues = { 3.4142, 2.0000, 0.5858 };
            // Expected eigenvectors
            fsmlib::Matrix<double, 3, 3> expected_eigenvectors = {
                { -0.500000, -0.707110, +0.500000 },
                { +0.707110, +0.000000, +0.707110 },
                { -0.500000, +0.707110, +0.500000 },
            };

            // Verify eigenvalues by comparing with expected values.
            if (fsmlib::any(fsmlib::abs(eigenvalues - expected_eigenvalues) > 1e-3)) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_eigenvalues << "\n";
                std::cerr << "Got      :\n" << eigenvalues << "\n";
                throw std::runtime_error("Test failed: eigenvalues mismatch");
            }
            // Verify eigenvectors by comparing each element of the matrix
            if (fsmlib::any(fsmlib::abs(eigenvectors - expected_eigenvectors) > 1e-3)) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_eigenvectors << "\n";
                std::cerr << "Got      :\n" << eigenvectors << "\n";
                throw std::runtime_error("Test failed: eigenvectors mismatch");
            }
            // If no mismatches are found, the test passes
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: eigen decomposition for 3x3 matrix\n";
        }

        {
            // Define a 4x4 symmetric matrix with more complex floating-point values
            fsmlib::Matrix<double, 4, 4> A = {
                { 5.5, -2.3, 1.7, 3.9 },
                { -2.3, 6.8, 2.1, -1.4 },
                { 1.7, 2.1, 7.2, 0.5 },
                { 3.9, -1.4, 0.5, 4.3 },
            };

            // Perform eigen decomposition
            auto [eigenvalues, eigenvectors] = fsmlib::linalg::eigen(A, 2000, 1e-32);

            // Expected eigenvalues.
            fsmlib::Vector<double, 4> expected_eigenvalues = { 10.6876, 8.9685, 3.4125, 0.7314 };

            // Expected eigenvectors.
            fsmlib::Matrix<double, 4, 4> expected_eigenvectors = {
                { 0.655906, 0.218389, 0.185699, 0.698290 },
                { -0.538693, 0.494653, 0.658882, 0.176076 },
                { 0.070480, 0.832130, -0.516567, -0.189077 },
                { 0.524050, 0.123223, 0.514344, -0.667561 },
            };

            // Verify eigenvalues by comparing with expected values.
            if (fsmlib::any(fsmlib::abs(eigenvalues - expected_eigenvalues) > 1e-3)) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected : \n" << expected_eigenvalues << "\n";
                std::cerr << "Got      :\n" << eigenvalues << "\n";
                throw std::runtime_error("Test failed: eigenvalues mismatch");
            }

            // Verify eigenvectors by comparing each element of the matrix
            if (fsmlib::any(fsmlib::abs(eigenvectors - expected_eigenvectors) > 1e-3)) {
                std::cerr << "Input    :\n" << A << "\n";
                std::cerr << "Expected :\n" << expected_eigenvectors << "\n";
                std::cerr << "Got      :\n" << eigenvectors << "\n";
                throw std::runtime_error("Test failed: eigenvectors mismatch");
            }

            // If no mismatches are found, the test passes
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: eigen decomposition for 4x4 matrix\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> A = {
                { -1.0, 2.0, 3.0 },
                { 4.0, -5.0, 6.0 },
                { 7.0, -8.0, -9.0 },
            };

            auto [U, Sigma, V] = fsmlib::linalg::svd(A);

            // Expected values (adapted from the given output)
            fsmlib::Matrix<double, 3, 3> expected_U = {
                { +0.246400, +0.11774, +0.96199 },
                { -0.095432, +0.99072, -0.09681 },
                { -0.964460, -0.06795, +0.25535 },
            };
            fsmlib::Vector<double, 3> expected_Sigma = { 14.428, 8.7473, 0.57051 };
            fsmlib::Matrix<double, 3, 3> expected_V  = {
                { +0.51147, +0.38520, +0.768130 },
                { -0.60201, -0.47723, +0.640180 },
                { -0.61318, +0.78985, +0.012195 },
            };

            // Verify U
            if (fsmlib::any(fsmlib::abs(U - expected_U) > 1e-3)) {
                std::cerr << "Expected : \n" << expected_U << "\n";
                std::cerr << "Got      : \n" << U << "\n";
                throw std::runtime_error("Test failed: U mismatch");
            }

            // Verify Sigma
            if (fsmlib::any(fsmlib::abs(Sigma - expected_Sigma) > 1e-3)) {
                std::cerr << "Expected : \n" << expected_Sigma << "\n";
                std::cerr << "Got      : \n" << Sigma << "\n";
                throw std::runtime_error("Test failed: Sigma mismatch");
            }

            // Verify V
            if (fsmlib::any(fsmlib::abs(V - expected_V) > 1e-3)) {
                std::cerr << "Expected : \n" << expected_V << "\n";
                std::cerr << "Got      : \n" << V << "\n";
                throw std::runtime_error("Test failed: V mismatch");
            }

            std::cout << "Test passed: SVD for 3x3 matrix\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> A = {
                { -1.0, +2.0, +3.0 },
                { +4.0, -5.0, +6.0 },
                { +7.0, -8.0, +9.0 },
            };

            auto [U, Sigma, V] = fsmlib::linalg::svd(A);

            // Expected values (adapted from the given output)
            fsmlib::Matrix<double, 3, 3> expected_U = {
                { +0.021466, +0.995670, +0.090492 },
                { +0.532700, +0.065204, -0.843790 },
                { +0.846030, -0.066318, +0.529000 },
            };
            fsmlib::Vector<double, 3> expected_Sigma = { 16.46, 3.7411, 0.29232 };
            fsmlib::Matrix<double, 3, 3> expected_V  = {
                { +0.48796, -0.32052, -0.81189 },
                { -0.57042, +0.58696, -0.57455 },
                { +0.66070, +0.74347, +0.10358 },
            };

            // Verify U
            if (fsmlib::any(fsmlib::abs(U - expected_U) > 1e-3)) {
                std::cerr << "Expected : \n" << expected_U << "\n";
                std::cerr << "Got      : \n" << U << "\n";
                throw std::runtime_error("Test failed: U mismatch");
            }

            // Verify Sigma
            if (fsmlib::any(fsmlib::abs(Sigma - expected_Sigma) > 1e-3)) {
                std::cerr << "Expected : \n" << expected_Sigma << "\n";
                std::cerr << "Got      : \n" << Sigma << "\n";
                throw std::runtime_error("Test failed: Sigma mismatch");
            }

            // Verify V
            if (fsmlib::any(fsmlib::abs(V - expected_V) > 1e-3)) {
                std::cerr << "Expected : \n" << expected_V << "\n";
                std::cerr << "Got      : \n" << V << "\n";
                throw std::runtime_error("Test failed: V mismatch");
            }

            std::cout << "Test passed: SVD for 3x3 matrix\n";
        }

        {
            fsmlib::Matrix<double, 5, 5> A = {
                { 1.1, -2.2, 3.3, 4.4, 5.5 },  { 6.6, 7.7, -8.8, 9.9, -1.1 }, { 2.2, 3.3, -4.4, 5.5, 6.6 },
                { 7.7, -8.8, 9.9, -1.1, 2.2 }, { 3.3, 4.4, 5.5, 6.6, 7.7 },
            };

            auto [U, Sigma, V] = fsmlib::linalg::svd(A);

            // Expected values (precomputed using Octave)
            fsmlib::Matrix<double, 5, 5> expected_U = {
                { +1.7755e-03, +4.2528e-01, -2.1239e-01, -3.7320e-01, +7.9671e-01 },
                { +7.4867e-01, +6.7568e-02, +6.1823e-01, +1.3111e-01, +1.8849e-01 },
                { +4.0561e-01, +2.5516e-01, -2.1896e-01, -6.7773e-01, -5.1294e-01 },
                { -4.8084e-01, +5.9542e-01, +6.0147e-01, -1.0302e-01, -2.0468e-01 },
                { +2.0919e-01, +6.2844e-01, -4.0370e-01, +6.1123e-01, -1.5723e-01 },
            };
            fsmlib::Vector<double, 5> expected_Sigma = { 20.9996, 17.5865, 8.9045, 4.8630, 2.4274 };
            fsmlib::Matrix<double, 5, 5> expected_V  = {
                { +0.134451, +0.462498, +0.748385, -0.038579, -0.454367 },
                { +0.583402, -0.116447, -0.287961, -0.655973, -0.364499 },
                { -0.570338, +0.513874, -0.162131, -0.604286, +0.138566 },
                { +0.550492, +0.422840, +0.073624, -0.015585, +0.715892 },
                { +0.115059, +0.574173, -0.570343, +0.450341, -0.359150 },
            };

            // Verify U
            if (fsmlib::any(fsmlib::abs(U - expected_U) > 1e-3)) {
                std::cerr << "Expected : \n" << expected_U << "\n";
                std::cerr << "Got      : \n" << U << "\n";
                throw std::runtime_error("Test failed: U mismatch");
            }

            // Verify Sigma
            if (fsmlib::any(fsmlib::abs(Sigma - expected_Sigma) > 1e-3)) {
                std::cerr << "Expected : \n" << expected_Sigma << "\n";
                std::cerr << "Got      : \n" << Sigma << "\n";
                throw std::runtime_error("Test failed: Sigma mismatch");
            }

            // Verify V
            if (fsmlib::any(fsmlib::abs(V - expected_V) > 1e-3)) {
                std::cerr << "Expected : \n" << expected_V << "\n";
                std::cerr << "Got      : \n" << V << "\n";
                throw std::runtime_error("Test failed: V mismatch");
            }

            std::cout << "Test passed: SVD for 5x5 matrix\n";
        }

        {
            // Define a 2x2 matrix
            fsmlib::Matrix<double, 2, 2> A = { { 1.5, -2.3 }, { 3.1, 0.7 } };

            // Compute characteristic polynomial
            auto poly = fsmlib::linalg::characteristic_poly(A);

            // Corrected expected coefficients based on Octave result
            fsmlib::Vector<double, 3> expected = { 1.0, -2.2, 8.18 };

            // Validate the result
            if (fsmlib::any(fsmlib::abs(poly - expected) > 1e-06)) {
                std::cerr << "Expected characteristic polynomial:\n" << expected << "\nGot:\n" << poly << "\n";
                throw std::runtime_error("Test failed: characteristic_poly for 2x2 matrix");
            }

            std::cout << "Test passed: characteristic_poly for 2x2 matrix\n";
        }

        {
            // Define a 3x3 matrix with floating-point values.
            fsmlib::Matrix<double, 3, 3> A = { { 2.1, -1.2, 0.5 }, { 4.3, 1.7, -3.0 }, { 0.0, 2.2, -0.8 } };

            // Compute the characteristic polynomial.
            auto poly = fsmlib::linalg::characteristic_poly(A);

            // Expected characteristic polynomial coefficients.
            fsmlib::Vector<double, 4> expected = {
                1.0,    // Coefficient for s^3
                -3.0,   // Coefficient for s^2
                12.29,  // Coefficient for s^1
                -11.606 // Coefficient for s^0
            };

            // Validate the result.
            if (fsmlib::any(fsmlib::abs(poly - expected) > 1e-06)) {
                std::cerr << "Expected characteristic polynomial:\n" << expected << "\nGot:\n" << poly << "\n";
                throw std::runtime_error("Test failed: characteristic_poly for 3x3 matrix");
            }

            std::cout << "Test  2 passed: characteristic_poly for 3x3 matrix\n";
        }

        {
            // Define a 4x4 matrix with floating-point values.
            fsmlib::Matrix<double, 4, 4> A = {
                { 1.5, -2.3, 0.7, 1.2 }, { 3.1, 0.4, -1.2, -0.8 }, { 0.0, 4.0, -0.5, 2.6 }, { 1.3, 2.2, -3.3, 0.9 }
            };

            // Compute the characteristic polynomial.
            auto poly = fsmlib::linalg::characteristic_poly(A);

            // Expected characteristic polynomial coefficients.
            fsmlib::Vector<double, 5> expected = {
                1.0,     // Coefficient for s^4
                -2.3,    // Coefficient for s^3
                21.62,   // Coefficient for s^2
                -57.293, // Coefficient for s^1
                100.1181 // Coefficient for s^0
            };

            // Validate the result.
            if (fsmlib::any(fsmlib::abs(poly - expected) > 1e-06)) {
                std::cerr << "Expected characteristic polynomial:\n" << expected << "\nGot:\n" << poly << "\n";
                throw std::runtime_error("Test failed: characteristic_poly for 4x4 matrix");
            }

            std::cout << "Test  3 passed: characteristic_poly for 4x4 matrix\n";
        }

        std::cout << "All linear algebra tests passed!\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
