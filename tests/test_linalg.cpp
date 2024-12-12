
#include <fsmlib/fsmlib.hpp>
#include <fsmlib/linalg.hpp>
#include <fsmlib/io.hpp>

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
            auto [iterations, scale] = fsmlib::linalg::scale_to_unit_norm(mat);

            if (iterations != 0 || scale != 1.0) {
                std::cerr << "Expected: iterations = 0, scale = 1.0\n";
                std::cerr << "Got: iterations = " << iterations << ", scale = " << scale << "\n";
                throw std::runtime_error("Test failed: scale_to_unit_norm");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: scale_to_unit_norm\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> mat = {
                { { 1024.0, 512.0 }, { 512.0, 1024.0 } }
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
                { { 0, 1 }, { -2, -3 } }
            };
            auto result = fsmlib::linalg::expm(mat, 1e-12);

            // Updated expected result from Octave
            fsmlib::Matrix<double, 2, 2> expected = {
                { { 0.600424, 0.232544 }, { -0.465088, -0.097209 } }
            };

            if (fsmlib::any(fsmlib::abs(result - expected) > 0.1)) {
                std::cerr << "Expected:\n"
                          << expected << "\nGot:\n"
                          << result << "\n";
                throw std::runtime_error("Test failed: expm with simple 2x2 matrix");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: expm with simple 2x2 matrix\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> mat = { { { 0.0, 1.0 }, { -10.0, -2.0 } } };

            auto result = fsmlib::linalg::expm(mat, 1e-12);

            // Updated expected result from Octave
            fsmlib::Matrix<double, 2, 2> expected = {
                { { -0.346893, 0.017305 }, { -0.173050, -0.381503 } }
            };

            if (fsmlib::any(fsmlib::abs(result - expected) > 0.1)) {
                std::cerr << "Expected:\n"
                          << expected << "\nGot:\n"
                          << result << "\n";
                throw std::runtime_error("Test failed: expm with simple 2x2 matrix");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: expm with simple 2x2 matrix\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> mat = {
                { { 1.0, 2.0 },
                  { 3.0, 4.0 } }
            };
            auto [Q, R] = fsmlib::linalg::qr_decomposition(mat);

            fsmlib::Matrix<double, 2, 2> expected_Q = {
                { { -0.316228, -0.948683 },
                  { -0.948683, 0.316228 } }
            };
            fsmlib::Matrix<double, 2, 2> expected_R = {
                { { -3.162278, -4.427189 },
                  { 0.000000, -0.632456 } }
            };

            auto reconstructed = fsmlib::multiply(Q, R);

            if (fsmlib::any(fsmlib::abs(Q - expected_Q) > 1e-3)) {
                std::cerr << "Expected Q:\n"
                          << expected_Q << "\nGot Q:\n"
                          << Q << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with simple 2x2 matrix");
            }
            if (fsmlib::any(fsmlib::abs(R - expected_R) > 1e-3)) {
                std::cerr << "Expected R:\n"
                          << expected_R << "\nGot R:\n"
                          << R << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with simple 2x2 matrix");
            }
            if (fsmlib::any(fsmlib::abs(reconstructed - mat) > 1e-3)) {
                std::cerr << "Expected mat:\n"
                          << mat << "\nGot Reconstructed:\n"
                          << reconstructed << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with simple 2x2 matrix");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: qr_decomposition with simple 2x2 matrix\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> mat = {
                { { 12.0, -51.0, 4.0 },
                  { 6.0, 167.0, -68.0 },
                  { -4.0, 24.0, -41.0 } }
            };
            auto [Q, R] = fsmlib::linalg::qr_decomposition(mat);

            fsmlib::Matrix<double, 3, 3> expected_Q = {
                { { -0.857143, 0.394286, 0.331429 },
                  { -0.428571, -0.902857, -0.034286 },
                  { 0.285714, -0.171429, 0.942857 } }
            };
            fsmlib::Matrix<double, 3, 3> expected_R = {
                { { -14.000000, -21.000000, 14.000000 },
                  { 0.000000, -175.000000, 70.000000 },
                  { 0.000000, 0.000000, -35.000000 } }
            };

            auto reconstructed = fsmlib::multiply(Q, R);

            if (fsmlib::any(fsmlib::abs(Q - expected_Q) > 1e-3)) {
                std::cerr << "Expected Q:\n"
                          << expected_Q << "\nGot Q:\n"
                          << Q << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with 3x3 matrix");
            }
            if (fsmlib::any(fsmlib::abs(R - expected_R) > 1e-3)) {
                std::cerr << "Expected R:\n"
                          << expected_R << "\nGot R:\n"
                          << R << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with 3x3 matrix");
            }
            if (fsmlib::any(fsmlib::abs(reconstructed - mat) > 1e-3)) {
                std::cerr << "Expected mat:\n"
                          << mat << "\nGot Reconstructed:\n"
                          << reconstructed << "\n";
                throw std::runtime_error("Test failed: qr_decomposition with 3x3 matrix");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: qr_decomposition with 3x3 matrix\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> mat = {
                { { 2.0, -1.0, -2.0 },
                  { -4.0, 6.0, 3.0 },
                  { -4.0, -2.0, 8.0 } }
            };

            auto [L, U] = fsmlib::linalg::lu_decomposition(mat);

            fsmlib::Matrix<double, 3, 3> expected_L = {
                { { 1.000000, 0.000000, 0.000000 },
                  { -2.000000, 1.000000, 0.000000 },
                  { -2.000000, -1.000000, 1.000000 } }
            };
            fsmlib::Matrix<double, 3, 3> expected_U = {
                { { 2.000000, -1.000000, -2.000000 },
                  { 0.000000, 4.000000, -1.000000 },
                  { 0.000000, 0.000000, 3.000000 } }
            };

            auto reconstructed = fsmlib::multiply(L, U);

            if (fsmlib::any(fsmlib::abs(L - expected_L) > 1e-3)) {
                std::cerr << "Expected L:\n"
                          << expected_L << "\nGot L:\n"
                          << L << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 3x3 matrix (L)");
            }
            if (fsmlib::any(fsmlib::abs(U - expected_U) > 1e-3)) {
                std::cerr << "Expected U:\n"
                          << expected_U << "\nGot U:\n"
                          << U << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 3x3 matrix (U)");
            }
            if (fsmlib::any(fsmlib::abs(reconstructed - mat) > 1e-3)) {
                std::cerr << "Expected mat:\n"
                          << mat << "\nGot Reconstructed:\n"
                          << reconstructed << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 3x3 matrix (Reconstruction)");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: lu_decomposition with 3x3 matrix\n";
        }

        {
            fsmlib::Matrix<double, 4, 4> mat = {
                { { 4.0, 3.0, 2.0, 1.0 },
                  { 3.0, 4.0, 1.0, 2.0 },
                  { 2.0, 1.0, 4.0, 3.0 },
                  { 1.0, 2.0, 3.0, 4.0 } }
            };

            auto [L, U] = fsmlib::linalg::lu_decomposition(mat);

            fsmlib::Matrix<double, 4, 4> expected_L = {
                { { 1.0000, 0.0000, 0.0000, 0.0000 },
                  { 0.7500, 1.0000, 0.0000, 0.0000 },
                  { 0.5000, -0.2857, 1.0000, 0.0000 },
                  { 0.2500, 0.7143, 1.0000, 1.0000 } }
            };
            fsmlib::Matrix<double, 4, 4> expected_U = {
                { { 4.0000, 3.0000, 2.0000, 1.0000 },
                  { 0.0000, 1.7500, -0.5000, 1.2500 },
                  { 0.0000, 0.0000, 2.8571, 2.8571 },
                  { 0.0000, 0.0000, 0.0000, 0.0000 } }
            };

            auto reconstructed = fsmlib::multiply(L, U);

            if (fsmlib::any(fsmlib::abs(L - expected_L) > 1e-3)) {
                std::cerr << "Expected L:\n"
                          << expected_L << "\nGot L:\n"
                          << L << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 4x4 matrix (L)");
            }
            if (fsmlib::any(fsmlib::abs(U - expected_U) > 1e-3)) {
                std::cerr << "Expected U:\n"
                          << expected_U << "\nGot U:\n"
                          << U << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 4x4 matrix (U)");
            }
            if (fsmlib::any(fsmlib::abs(reconstructed - mat) > 1e-3)) {
                std::cerr << "Expected mat:\n"
                          << mat << "\nGot Reconstructed:\n"
                          << reconstructed << "\n";
                throw std::runtime_error("Test failed: lu_decomposition with 4x4 matrix (Reconstruction)");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: lu_decomposition with 4x4 matrix\n";
        }

        {
            fsmlib::Matrix<double, 2, 2> A = {
                { { 3.0, 2.0 },
                  { 1.0, 4.0 } }
            };
            fsmlib::Vector<double, 2> b = { 10.0, 11.0 };

            auto x = fsmlib::linalg::solve(A, b);

            fsmlib::Vector<double, 2> expected_x = { 1.8, 2.3 };

            if (fsmlib::any(fsmlib::abs(x - expected_x) > 1e-3)) {
                std::cerr << "Expected x:\n"
                          << expected_x << "\nGot x:\n"
                          << x << "\n";
                throw std::runtime_error("Test failed: solve with 2x2 system");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: solve with 2x2 system\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> A = {
                { { 1.0, 2.0, 3.0 },
                  { 4.0, 5.0, 6.0 },
                  { 7.0, 8.0, 10.0 } }
            };
            fsmlib::Vector<double, 3> b = { 6.0, 15.0, 25.0 };

            auto x = fsmlib::linalg::solve(A, b);

            fsmlib::Vector<double, 3> expected_x = { 1.0, 1.0, 1.0 };

            if (fsmlib::any(fsmlib::abs(x - expected_x) > 1e-3)) {
                std::cerr << "Expected x:\n"
                          << expected_x << "\nGot x:\n"
                          << x << "\n";
                throw std::runtime_error("Test failed: solve with 3x3 system");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: solve with 3x3 system\n";
        }

        {
            fsmlib::Matrix<double, 4, 4> A = {
                { { 1.5, -3.2, 2.7, 0.8 },
                  { -4.1, 5.6, -1.9, 3.3 },
                  { 2.0, 3.7, -6.4, -2.5 },
                  { -1.3, -2.6, 4.5, 7.8 } }
            };
            fsmlib::Vector<double, 4> b = { 4.8, -3.2, 5.9, -7.6 };

            auto x = fsmlib::linalg::solve(A, b);

            fsmlib::Vector<double, 4> expected_x = { 10.0840, 10.0891, 8.3556, -0.7512 };

            auto reconstructed = fsmlib::multiply(A, x);

            if (fsmlib::any(fsmlib::abs(x - expected_x) > 1e-3)) {
                std::cerr << "Expected x:\n"
                          << expected_x << "\nGot x:\n"
                          << x << "\n";
                throw std::runtime_error("Test failed: solve with 4x4 system (solution)");
            }
            if (fsmlib::any(fsmlib::abs(reconstructed - b) > 1e-3)) {
                std::cerr << "Expected b:\n"
                          << b << "\nGot Reconstructed:\n"
                          << reconstructed << "\n";
                throw std::runtime_error("Test failed: solve with 4x4 system (reconstruction)");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: solve with 4x4 system\n";
        }

        std::cout << "All linear algebra tests passed!\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
