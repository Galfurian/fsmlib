

#include <fsmlib/control.hpp>
#include <fsmlib/io.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>

int main()
{
    try {
        int test_count = 1; // Test counter variable

        // Test 1: Continuous-to-Discrete Conversion
        {
            // Define a continuous-time state-space model
            fsmlib::control::StateSpace<double, 2, 1, 1> sys = {
                {{0.0, 1.0}, {-10.0, -2.0}}, // A matrix
                {{0.0}, {1.0}},              // B matrix
                {{1.0, 0.0}},                // C matrix
                {{0.0}}                      // D matrix
            };

            // Define the sample time
            double sample_time = 0.1;

            // Convert to discrete-time
            auto dsys = fsmlib::control::c2d(sys, sample_time);

            // Expected results
            fsmlib::Matrix<double, 2, 2> expected_A = {
                {0.953557, 0.089133},
                {-0.891326, 0.775292},
            };
            fsmlib::Matrix<double, 2, 1> expected_B = {{4.6443e-03}, {8.9133e-02}};

            // Validate results
            if (fsmlib::any(fsmlib::abs(dsys.A - expected_A) > 1e-3)) {
                std::ostringstream error;
                error << "Test failed: Incorrect A matrix in Continuous-to-Discrete Conversion.\n"
                      << "Expected A:\n"
                      << expected_A << "\n"
                      << "Got A:\n"
                      << dsys.A << "\n";
                throw std::runtime_error(error.str());
            }
            if (fsmlib::any(fsmlib::abs(dsys.B - expected_B) > 1e-3)) {
                std::ostringstream error;
                error << "Test failed: Incorrect B matrix in Continuous-to-Discrete Conversion.\n"
                      << "Expected B:\n"
                      << expected_B << "\n"
                      << "Got B:\n"
                      << dsys.B << "\n";
                throw std::runtime_error(error.str());
            }
            if (fsmlib::any(fsmlib::abs(dsys.C - sys.C) > 1e-3)) {
                std::ostringstream error;
                error << "Test failed: Incorrect C matrix in Continuous-to-Discrete Conversion.\n"
                      << "Expected C:\n"
                      << sys.C << "\n"
                      << "Got C:\n"
                      << dsys.C << "\n";
                throw std::runtime_error(error.str());
            }
            if (fsmlib::any(fsmlib::abs(dsys.D - sys.D) > 1e-3)) {
                std::ostringstream error;
                error << "Test failed: Incorrect D matrix in Continuous-to-Discrete Conversion.\n"
                      << "Expected D:\n"
                      << sys.D << "\n"
                      << "Got D:\n"
                      << dsys.D << "\n";
                throw std::runtime_error(error.str());
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: Continuous-to-Discrete Conversion\n";
        }

        // Test 2: Simulate one step of a discrete-time model
        {
            // Define a discrete-time state-space model
            fsmlib::control::DiscreteStateSpace<double, 2, 1, 1> dsys = {
                {{0.904837, 0.0951626}, {-0.951626, 0.809685}},
                {{0.00483742}, {0.0951626}},
                {{1.0, 0.0}},
                {{0.0}},
                0.1};

            // Define initial state and input
            fsmlib::Vector<double, 2> x = {1.0, 0.0};
            fsmlib::Vector<double, 1> u = {0.0};

            // Simulate one step
            fsmlib::Vector<double, 2> x_next;
            fsmlib::Vector<double, 1> y;
            fsmlib::control::dstep(dsys, x, u, x_next, y);

            // Expected results
            fsmlib::Vector<double, 2> expected_x_next = {0.904837, -0.951626};
            fsmlib::Vector<double, 1> expected_y      = {1.0};

            // Validate results
            if (fsmlib::any(fsmlib::abs(x_next - expected_x_next) > 1e-06)) {
                throw std::runtime_error("Test failed: Incorrect next state in Simulation step");
            }
            if (fsmlib::any(fsmlib::abs(y - expected_y) > 1e-06)) {
                throw std::runtime_error("Test failed: Incorrect output in Simulation step");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: Simulate one step of a discrete-time model\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> A = {
                {8., 1., 6.},
                {3., 5., 7.},
                {4., 9., 2.},
            };
            fsmlib::Matrix<double, 3, 1> B        = {{1.0}, {0.0}, {0.0}};
            auto result                           = fsmlib::control::ctrb(A, B);
            fsmlib::Matrix<double, 3, 3> expected = {
                {1., 8., 91.},
                {0., 3., 67.},
                {0., 4., 67.},
            };
            if (fsmlib::any(fsmlib::abs(result - expected) > 1e-06)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: ctrb with 3x3 system");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: ctrb with 3x3 system\n";
        }
        {
            fsmlib::Matrix<double, 4, 4> A = {
                {1.12, 2.34, 3.45, 4.56},
                {0.78, 1.23, 4.56, 5.67},
                {0.89, 0.12, 2.34, 6.78},
                {0.01, 0.23, 0.45, 1.89},
            };
            fsmlib::Matrix<double, 4, 1> B = {
                {1.00},
                {0.00},
                {0.00},
                {0.00},
            };

            auto result                           = fsmlib::control::ctrb(A, B);
            fsmlib::Matrix<double, 4, 4> expected = {
                {1.0000, 1.1200, 6.1957, 34.8201},
                {0.0000, 0.7800, 5.9481, 30.3856},
                {0.0000, 0.8900, 3.2408, 17.9472},
                {0.0000, 0.0100, 0.6100, 4.0413},
            };

            if (fsmlib::any(fsmlib::abs(result - expected) > 1e-03)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: ctrb with 4x4 system (single input)");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: ctrb with 4x4 system (single input)\n";
        }
        {
            fsmlib::Matrix<double, 3, 3> A = {
                {2.34, 1.23, 0.45},
                {0.78, 1.12, 1.56},
                {0.34, 0.67, 3.21},
            };
            fsmlib::Matrix<double, 3, 2> B = {
                {1.23, 0.45},
                {0.78, 1.12},
                {0.34, 0.67},
            };

            auto result                           = fsmlib::control::ctrb(A, B);
            fsmlib::Matrix<double, 3, 6> expected = {
                {1.2300, 0.4500, 3.9906, 2.7321, 13.1595, 11.0277},
                {0.7800, 1.1200, 2.3634, 2.6506, 8.9299, 9.8641},
                {0.3400, 0.6700, 2.0322, 3.0541, 9.4636, 12.5085},
            };

            if (fsmlib::any(fsmlib::abs(result - expected) > 1e-03)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: ctrb with 3x3 system (two inputs)");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: ctrb with 3x3 system (two inputs)\n";
        }
        {
            fsmlib::Matrix<double, 5, 5> A = {
                {0.12, 1.34, 0.56, 0.78, 0.90}, {0.23, 0.45, 1.23, 0.67, 0.89}, {0.34, 0.56, 0.78, 1.34, 1.45},
                {0.45, 0.67, 0.89, 1.12, 1.23}, {0.56, 0.78, 1.34, 0.90, 1.11},
            };
            fsmlib::Matrix<double, 5, 2> B = {
                {1.12, 0.45}, {0.78, 1.34}, {0.34, 0.67}, {0.23, 0.89}, {0.56, 1.12},
            };

            auto result                            = fsmlib::control::ctrb(A, B);
            fsmlib::Matrix<double, 5, 10> expected = {
                {1.1200, 0.4500, 2.0534, 3.9270, 7.7731, 14.0235, 33.6552, 60.9582, 143.9747, 260.1241},
                {0.7800, 1.3400, 1.6793, 3.1237, 7.7049, 14.0277, 33.2281, 59.9712, 141.7037, 256.1486},
                {0.3400, 0.6700, 2.2030, 4.2426, 10.0599, 17.9957, 42.6182, 77.0898, 181.9789, 328.9044},
                {0.2300, 0.8900, 2.2756, 4.0710, 9.6579, 17.4097, 41.0462, 74.2130, 175.3315, 316.9085},
                {0.5600, 1.1200, 2.5198, 4.2392, 10.2568, 18.6901, 43.9202, 79.3237, 187.5662, 339.0554},
            };

            if (fsmlib::any(fsmlib::abs(result - expected) > 1e-03)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: ctrb with 5x5 system (two inputs)");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: ctrb with 5x5 system (two inputs)\n";
        }

        {
            // Define a state matrix A and output matrix C
            fsmlib::Matrix<double, 3, 3> A = {
                {1.0, 2.0, 0.0},
                {0.0, 1.0, 3.0},
                {0.0, 0.0, 1.0},
            };

            fsmlib::Matrix<double, 2, 3> C = {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
            };
            // Compute the observability matrix
            auto result                           = fsmlib::control::obsv(A, C);
            // Expected result (manually calculated or verified with Octave)
            fsmlib::Matrix<double, 6, 3> expected = {
                {1., 0., 0.}, {0., 1., 0.}, {1., 2., 0.}, {0., 1., 3.}, {1., 4., 6.}, {0., 1., 6.},
            };
            // Verify the result
            if (fsmlib::any(fsmlib::abs(result - expected) > 1e-06)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: obsv with 3x3 state matrix and 2x3 output matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: obsv with 3x3 state matrix and 2x3 output matrix\n";
        }

        {
            // Define the roots of the polynomial
            fsmlib::Vector<double, 3> roots    = {2.0, -3.0, 1.5};
            // Compute the polynomial coefficients
            auto result                        = fsmlib::linalg::poly(roots);
            // Expected coefficients for polynomial.
            fsmlib::Vector<double, 4> expected = {1.0, -0.5, -7.5, 9.0};
            // Verify the result
            if (fsmlib::any(fsmlib::abs(result - expected) > 1e-06)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: poly function with 3 roots");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: poly function with 3 roots\n";
        }

        {
            fsmlib::Vector<double, 6> coefficients = {0, 0, 0, 1.0, -0.5, 9.0};
            // Reduce the polynomial coefficients
            auto result                            = fsmlib::linalg::polyreduce(coefficients);
            // Expected result
            fsmlib::Vector<double, 6> expected     = {1.0, -0.5, 9.0, 0., 0., 0.};
            if (fsmlib::any(fsmlib::abs(result - expected) > 1e-06)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: polyreduce");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: polyreduce\n";
        }

        {
            // Define the state matrix A and input matrix B
            fsmlib::Matrix<double, 2, 2> A        = {{1.0, 2.0}, {3.0, 4.0}};
            fsmlib::Matrix<double, 2, 1> B        = {{0.0}, {1.0}};
            // Define the desired poles.
            fsmlib::Vector<double, 2> poles       = {-3.0, -2.0};
            // Compute the gain matrix using the acker function
            auto result                           = fsmlib::control::acker(A, B, poles);
            // Expected gain matrix
            fsmlib::Matrix<double, 1, 2> expected = {{9.0, 10.0}};
            if (fsmlib::any(fsmlib::abs(result - expected) > 1e-06)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: acker with 2x2 system and 2 poles");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: acker with 2x2 system and 2 poles\n";
        }

        {
            // Define the state matrix A and input matrix B.
            fsmlib::Matrix<double, 3, 3> A        = {{1.5, -2.3, 0.7}, {3.1, 0.4, -1.2}, {0.0, 4.0, -0.5}};
            fsmlib::Matrix<double, 3, 1> B        = {{0.0}, {1.0}, {0.5}};
            // Define the desired poles.
            fsmlib::Vector<double, 3> poles       = {-4.0, -3.0, -2.0};
            // Compute the gain matrix using the acker function.
            auto result                           = fsmlib::control::acker(A, B, poles);
            // Expected gain matrix (precomputed for the given input).
            fsmlib::Matrix<double, 1, 3> expected = {{30.3346, -1.3159, 23.4318}};
            // Validate the result.
            if (fsmlib::any(fsmlib::abs(result - expected) > 1e-03)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: acker with 3x3 system and 3 poles");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: acker with 3x3 system and 3 poles\n";
        }

        std::cout << "All control tests passed!\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
