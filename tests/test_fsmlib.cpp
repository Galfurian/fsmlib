#include <fsmlib/fsmlib.hpp>
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

        {
            const fsmlib::Vector<int, 5> vec = { 1, 2, 3, 4, 5 };
            auto view                        = fsmlib::view<int, 3>(vec, 1); // View of elements 2, 3, 4
            if (view.size() != 3 || view[0] != 2 || view[1] != 3 || view[2] != 4) {
                throw std::runtime_error("Test failed: View creation and access");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: View creation and access\n";
        }

        {
            fsmlib::Vector<int, 5> vec = { 1, 2, 3, 4, 5 };
            auto view                  = fsmlib::view<int, 3>(vec, 1); // View of elements 2, 3, 4
            view[0]                    = 20;                           // Modify the first element in the view
            view[2]                    = 40;                           // Modify the last element in the view
            if (vec[1] != 20 || vec[3] != 40 || vec[0] != 1 || vec[4] != 5) {
                throw std::runtime_error("Test failed: View modification and underlying vector");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: View modification and underlying vector\n";
        }

        {
            fsmlib::Vector<int, 5> vec = { 10, 20, 30, 40, 50 };
            auto full_view             = fsmlib::view<int, 5>(vec); // Full view of the vector
            if (full_view.size() != 5 || full_view[0] != 10 || full_view[4] != 50) {
                throw std::runtime_error("Test failed: Full view creation and access");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Full view creation and access\n";
        }

        try {
            fsmlib::Vector<int, 5> vec = { 1, 2, 3, 4, 5 };
            auto view                  = fsmlib::view<int, 3>(vec, 2); // View of elements 3, 4, 5
            [[maybe_unused]] int x     = view[3];                      // Access out-of-bounds
            throw std::runtime_error("Test failed: Out-of-bounds access not detected");
        } catch (const std::out_of_range &) {
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Out-of-bounds access detected\n";
        }

        try {
            fsmlib::Vector<int, 5> vec = { 1, 2, 3, 4, 5 };
            fsmlib::view<int, 3>(vec, 4); // Exceeds vector bounds
            throw std::runtime_error("Test failed: Invalid view creation not detected");
        } catch (const std::out_of_range &) {
            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Invalid view creation detected\n";
        }

        {
            // Define a 3x3 matrix
            fsmlib::Matrix<int, 3, 3> mat = {
                { { 1, 2, 3 },
                  { 4, 5, 6 },
                  { 7, 8, 9 } }
            };

            // Create a view over the second row
            auto row_view = fsmlib::view<int, 3>(mat[1]); // View of the second row: {4, 5, 6}

            // Access elements in the row view
            if (row_view[0] != 4 || row_view[1] != 5 || row_view[2] != 6) {
                throw std::runtime_error("Test failed: Matrix row view creation and access");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Matrix row view creation and access\n";
        }

        {
            // Define a 5x5 matrix
            fsmlib::Matrix<int, 5, 5> mat = {
                { { 1, 2, 3, 4, 5 },
                  { 6, 7, 8, 9, 10 },
                  { 11, 12, 13, 14, 15 },
                  { 16, 17, 18, 19, 20 },
                  { 21, 22, 23, 24, 25 } }
            };

            // Create a 2x2 view starting at row 1, column 1
            auto view = fsmlib::view<int, 2, 2, 5, 5>(mat, 1, 1);

            // Access and modify elements within the view
            view[0][0] = 99;
            view[1][1] = 88;

            // Check modifications in the view
            if (view[0][0] != 99 || view[0][1] != 8 || view[1][0] != 12 || view[1][1] != 88) {
                throw std::runtime_error("Test failed: MatrixView modification");
            }

            // Check modifications in the original matrix
            if (mat[1][1] != 99 || mat[1][2] != 8 || mat[2][1] != 12 || mat[2][2] != 88) {
                throw std::runtime_error("Test failed: MatrixView reflects changes in the original matrix");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: MatrixView modification and original matrix reflection\n";
        }

        {
            // Define a 4x4 matrix
            fsmlib::Matrix<int, 4, 4> mat = {
                { { 10, 20, 30, 40 },
                  { 50, 60, 70, 80 },
                  { 90, 100, 110, 120 },
                  { 130, 140, 150, 160 } }
            };

            // Create a 3x2 view starting at row 0, column 1
            auto view = fsmlib::view<int, 3, 2, 4, 4>(mat, 0, 1);

            // Access and check the view
            if (view[0][0] != 20 || view[0][1] != 30 ||
                view[1][0] != 60 || view[1][1] != 70 ||
                view[2][0] != 100 || view[2][1] != 110) {
                throw std::runtime_error("Test failed: MatrixView creation and access");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: MatrixView creation and access\n";
        }

        // Additional tests for access...
        std::cout << "All access tests passed!\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
