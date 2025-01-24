#include <fsmlib/fsmlib.hpp>
#include <fsmlib/math.hpp>
#include <fsmlib/io.hpp>

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
            fsmlib::Vector<double, 3> expected = { 4.4, 2.2, 6.6 };
            if (fsmlib::any(fsmlib::abs(vec - expected) > 1e-06)) {
                throw std::runtime_error("Test failed: Vector modification");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Vector modification\n";
        }
        {
            fsmlib::Matrix<int, 3, 3> mat = { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } };
            if (mat(0, 0) != 1 || mat(2, 2) != 9) {
                throw std::runtime_error("Test failed: Square matrix initialization");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: Square matrix initialization\n";
        }
        {
            const fsmlib::Vector<int, 5> vec = { 1, 2, 3, 4, 5 };
            // Create a view of elements 2, 3, 4 (offset 1, size 3).
            auto view = fsmlib::view<int, 3>(vec, 1);
            // Check the size and access elements via the view.
            if (view.size() != 3 || view[0] != 2 || view[1] != 3 || view[2] != 4) {
                throw std::runtime_error("Test failed: VectorView creation and access");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: VectorView creation and access\n";
        }
        {
            fsmlib::Vector<int, 5> vec = { 1, 2, 3, 4, 5 };
            // Create a view of elements 2, 3, 4 (offset 1, size 3).
            auto view = fsmlib::view<int, 3>(vec, 1);
            // Modify elements through the view.
            view[0] = 20; // Modify the first element in the view.
            view[2] = 40; // Modify the last element in the view.
            // Verify changes reflect in the original vector.
            if (vec[1] != 20 || vec[3] != 40 || vec[0] != 1 || vec[4] != 5) {
                throw std::runtime_error("Test failed: VectorView modification and underlying vector");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: VectorView modification and underlying vector\n";
        }
        {
            fsmlib::Vector<int, 5> vec = { 10, 20, 30, 40, 50 };
            // Create a full view of the vector (offset 0, size 5).
            auto full_view = fsmlib::view<int, 5>(vec, 0);
            // Validate the size and elements in the full view.
            if (full_view.size() != 5 || full_view[0] != 10 || full_view[4] != 50) {
                throw std::runtime_error("Test failed: Full VectorView creation and access");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: Full VectorView creation and access\n";
        }
        {
            // Define a 3x3 matrix.
            fsmlib::Matrix<int, 3, 3> mat = {
                { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 },
            };
            // Create a view over the second row (row 1, all columns).
            auto view = fsmlib::view<int, 1, 3>(mat, 1, 0);
            // Access elements in the row view.
            if (view[0] != 4 || view[1] != 5 || view[2] != 6) {
                std::cerr << "Test failed: Matrix row view creation and access\n";
                std::cerr << "Contents of MatrixView:\n";
                for (std::size_t j = 0; j < view.size(); ++j) {
                    std::cerr << view[j] << std::endl;
                }
                throw std::runtime_error("Test failed: Matrix row view creation and access");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: Matrix row view creation and access\n";
        }
        {
            // Define a 5x5 matrix
            fsmlib::Matrix<int, 5, 5> mat = { { { 1, 2, 3, 4, 5 },
                                                { 6, 7, 8, 9, 10 },
                                                { 11, 12, 13, 14, 15 },
                                                { 16, 17, 18, 19, 20 },
                                                { 21, 22, 23, 24, 25 } } };
            // Create a 2x2 view starting at row 1, column 1
            auto view = fsmlib::view<int, 2, 2>(mat, 1, 1); // 2x2 submatrix from (1,1) to (2,2)
            // Access and modify elements within the view
            view(0, 0) = 99;
            view(1, 1) = 88;

            // Check modifications in the view
            if (view(0, 0) != 99) {
                std::cerr << "Test failed: MatrixView modification\n";
                std::cerr << "Value at (0, 0): " << view(0, 0) << ", Expected: 99\n";
                throw std::runtime_error("MatrixView modification failed at (0, 0)");
            }
            if (view(0, 1) != 8) {
                std::cerr << "Test failed: MatrixView modification\n";
                std::cerr << "Value at (0, 1): " << view(0, 1) << ", Expected: 8\n";
                throw std::runtime_error("MatrixView modification failed at (0, 1)");
            }
            if (view(1, 0) != 12) {
                std::cerr << "Test failed: MatrixView modification\n";
                std::cerr << "Value at (1, 0): " << view(1, 0) << ", Expected: 12\n";
                throw std::runtime_error("MatrixView modification failed at (1, 0)");
            }
            if (view(1, 1) != 88) {
                std::cerr << "Test failed: MatrixView modification\n";
                std::cerr << "Value at (1, 1): " << view(1, 1) << ", Expected: 88\n";
                throw std::runtime_error("MatrixView modification failed at (1, 1)");
            }

            // Check modifications in the original matrix
            if (mat(1, 1) != 99) {
                std::cerr << "Test failed: MatrixView reflects changes in the original matrix\n";
                std::cerr << "Value at (1, 1): " << mat(1, 1) << ", Expected: 99\n";
                throw std::runtime_error("MatrixView reflection failed at (1, 1)");
            }
            if (mat(1, 2) != 8) {
                std::cerr << "Test failed: MatrixView reflects changes in the original matrix\n";
                std::cerr << "Value at (1, 2): " << mat(1, 2) << ", Expected: 8\n";
                throw std::runtime_error("MatrixView reflection failed at (1, 2)");
            }
            if (mat(2, 1) != 12) {
                std::cerr << "Test failed: MatrixView reflects changes in the original matrix\n";
                std::cerr << "Value at (2, 1): " << mat(2, 1) << ", Expected: 12\n";
                throw std::runtime_error("MatrixView reflection failed at (2, 1)");
            }
            if (mat(2, 2) != 88) {
                std::cerr << "Test failed: MatrixView reflects changes in the original matrix\n";
                std::cerr << "Value at (2, 2): " << mat(2, 2) << ", Expected: 88\n";
                throw std::runtime_error("MatrixView reflection failed at (2, 2)");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: MatrixView modification and original matrix reflection\n";
        }
        {
            // Define a 4x4 matrix
            fsmlib::Matrix<int, 4, 4> mat = {
                { { 10, 20, 30, 40 }, { 50, 60, 70, 80 }, { 90, 100, 110, 120 }, { 130, 140, 150, 160 } }
            };
            // Create a 3x2 view starting at row 0, column 1
            auto view = fsmlib::view<int, 3, 2>(mat, 0, 1); // 3x2 submatrix from (0,1) to (2,2)
            // Access and check the view
            if (view(0, 0) != 20 || view(0, 1) != 30 || view(1, 0) != 60 || view(1, 1) != 70 || view(2, 0) != 100 ||
                view(2, 1) != 110) {
                throw std::runtime_error("Test failed: MatrixView creation and access");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: MatrixView creation and access\n";
        }

        {
            fsmlib::Matrix<int, 2, 2> A = { { { 1, 2 }, { 3, 4 } } };
            fsmlib::Matrix<int, 2, 3> B = { { { 5, 6, 7 }, { 8, 9, 10 } } };

            auto result = fsmlib::hstack(A, B);

            fsmlib::Matrix<int, 2, 5> expected = { { { 1, 2, 5, 6, 7 }, { 3, 4, 8, 9, 10 } } };

            if (fsmlib::any(result != expected)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: hstack");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: hstack\n";
        }

        {
            fsmlib::Matrix<int, 2, 3> A = { { { 1, 2, 3 }, { 4, 5, 6 } } };
            fsmlib::Matrix<int, 1, 3> B = { { { 7, 8, 9 } } };

            auto result = fsmlib::vstack(A, B);

            fsmlib::Matrix<int, 3, 3> expected = { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } };

            if (fsmlib::any(result != expected)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: vstack");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: vstack\n";
        }

        // Additional tests for access...
        std::cout << "All access tests passed!\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
