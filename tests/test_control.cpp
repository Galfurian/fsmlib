

#include <fsmlib/control.hpp>
#include <fsmlib/io.hpp>

#include <stdexcept>
#include <iostream>
#include <sstream>

int main()
{
    try {
        int test_count = 1; // Test counter variable

        // Test 1: Continuous-to-Discrete Conversion
        {
            // Define a continuous-time state-space model
            fsmlib::control::StateSpace<double, 2, 1, 1> sys = {
                { { 0.0, 1.0 }, { -10.0, -2.0 } }, // A matrix
                { { 0.0 }, { 1.0 } },              // B matrix
                { { 1.0, 0.0 } },                  // C matrix
                { { 0.0 } }                        // D matrix
            };

            // Define the sample time
            double sample_time = 0.1;

            // Convert to discrete-time
            auto dsys = fsmlib::control::c2d(sys, sample_time);

            // Expected results
            fsmlib::Matrix<double, 2, 2> expected_A = {
                { 0.953557, 0.089133 },
                { -0.891326, 0.775292 },
            };
            fsmlib::Matrix<double, 2, 1> expected_B = { { 4.6443e-03 }, { 8.9133e-02 } };

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
            fsmlib::control::DiscreteStateSpace<double, 2, 1, 1> dsys = { { { 0.904837, 0.0951626 },
                                                                            { -0.951626, 0.809685 } },
                                                                          { { 0.00483742 }, { 0.0951626 } },
                                                                          { { 1.0, 0.0 } },
                                                                          { { 0.0 } },
                                                                          0.1 };

            // Define initial state and input
            fsmlib::Vector<double, 2> x = { 1.0, 0.0 };
            fsmlib::Vector<double, 1> u = { 0.0 };

            // Simulate one step
            fsmlib::Vector<double, 2> x_next;
            fsmlib::Vector<double, 1> y;
            fsmlib::control::dstep(dsys, x, u, x_next, y);

            // Expected results
            fsmlib::Vector<double, 2> expected_x_next = { 0.904837, -0.951626 };
            fsmlib::Vector<double, 1> expected_y      = { 1.0 };

            // Validate results
            if (fsmlib::any(x_next != expected_x_next)) {
                throw std::runtime_error("Test failed: Incorrect next state in Simulation step");
            }
            if (fsmlib::any(y != expected_y)) {
                throw std::runtime_error("Test failed: Incorrect output in Simulation step");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: Simulate one step of a discrete-time model\n";
        }

        {
            fsmlib::Matrix<double, 3, 3> A = {
                { 8., 1., 6. },
                { 3., 5., 7. },
                { 4., 9., 2. },
            };
            fsmlib::Matrix<double, 3, 1> B        = { { 1.0 }, { 0.0 }, { 0.0 } };
            auto result                           = fsmlib::control::ctrb(A, B);
            fsmlib::Matrix<double, 3, 3> expected = {
                { 1., 8., 91. },
                { 0., 3., 67. },
                { 0., 4., 67. },
            };
            if (fsmlib::any(result != expected)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: ctrb with 3x3 system");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: ctrb with 3x3 system\n";
        }
        {
            // Define a state matrix A and output matrix C
            fsmlib::Matrix<double, 3, 3> A = {
                { 1.0, 2.0, 0.0 },
                { 0.0, 1.0, 3.0 },
                { 0.0, 0.0, 1.0 },
            };

            fsmlib::Matrix<double, 2, 3> C = {
                { 1.0, 0.0, 0.0 },
                { 0.0, 1.0, 0.0 },
            };
            // Compute the observability matrix
            auto result = fsmlib::control::obsv(A, C);
            // Expected result (manually calculated or verified with Octave)
            fsmlib::Matrix<double, 6, 3> expected = {
                { 1., 0., 0. }, { 0., 1., 0. }, { 1., 2., 0. }, { 0., 1., 3. }, { 1., 4., 6. }, { 0., 1., 6. },
            };
            // Verify the result
            if (fsmlib::any(result != expected)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: obsv with 3x3 state matrix and 2x3 output matrix");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: obsv with 3x3 state matrix and 2x3 output matrix\n";
        }

        {
            // Define the roots of the polynomial
            fsmlib::Vector<double, 3> roots = { 2.0, -3.0, 1.5 };
            // Compute the polynomial coefficients
            auto result = fsmlib::linalg::poly(roots);
            // Expected coefficients for polynomial.
            fsmlib::Vector<double, 4> expected = { 1.0, -0.5, -7.5, 9.0 };
            // Verify the result
            if (fsmlib::any(result != expected)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: poly function with 3 roots");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: poly function with 3 roots\n";
        }

        {
            fsmlib::Vector<double, 6> coefficients = { 0, 0, 0, 1.0, -0.5, 9.0 };
            // Reduce the polynomial coefficients
            auto result = fsmlib::linalg::polyreduce(coefficients);
            // Expected result
            fsmlib::Vector<double, 6> expected = { 1.0, -0.5, 9.0, 0., 0., 0. };
            if (fsmlib::any(result != expected)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: polyreduce");
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: polyreduce\n";
        }

        {
            // Define the state matrix A and input matrix B
            fsmlib::Matrix<double, 2, 2> A = { { 1.0, 2.0 }, { 3.0, 4.0 } };
            fsmlib::Matrix<double, 2, 1> B = { { 0.0 }, { 1.0 } };
            // Define the desired poles.
            fsmlib::Vector<double, 2> poles = { -3.0, -2.0 };
            // Compute the gain matrix using the acker function
            auto result = fsmlib::control::acker(A, B, poles);
            // Expected gain matrix
            fsmlib::Matrix<double, 1, 2> expected = { { 9.0, 10.0 } };
            if (fsmlib::any(result != expected)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: acker with 2x2 system and 2 poles");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: acker with 2x2 system and 2 poles\n";
        }

        {
            // Define the state matrix A and input matrix B.
            fsmlib::Matrix<double, 3, 3> A = { { 1.5, -2.3, 0.7 }, { 3.1, 0.4, -1.2 }, { 0.0, 4.0, -0.5 } };
            fsmlib::Matrix<double, 3, 1> B = { { 0.0 }, { 1.0 }, { 0.5 } };
            // Define the desired poles.
            fsmlib::Vector<double, 3> poles = { -4.0, -3.0, -2.0 };
            // Compute the gain matrix using the acker function.
            auto result = fsmlib::control::acker(A, B, poles);
            // Expected gain matrix (precomputed for the given input).
            fsmlib::Matrix<double, 1, 3> expected = { { 30.3346, -1.3159, 23.4318 } };
            // Validate the result.
            if (fsmlib::any(fsmlib::abs(result - expected) > 1e-03)) {
                std::cerr << "Expected:\n" << expected << "\nGot:\n" << result << "\n";
                throw std::runtime_error("Test failed: acker with 3x3 system and 3 poles");
            }
            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: acker with 3x3 system and 3 poles\n";
        }

        {
            fsmlib::control::StateSpace<double, 2, 2, 2> ss = {
                .A = { { 0.0, 1.0 }, { -4.0, -5.0 } },
                .B = { { 1.0, 0.0 }, { 0.0, 1.0 } },
                .C = { { 1.0, 0.0 }, { 0.0, 1.0 } },
                .D = { { 0.0, 0.0 }, { 0.0, 0.0 } },
            };

            // Compute the transfer function representation.
            auto transfer_functions = fsmlib::control::ss2tf(ss);

            // Expected transfer functions for each input-output pair.
            fsmlib::control::TransferFunction<double, 3, 3> expected_tf[2][2] = {
                {
                    // Row 1 (Output 1)
                    { { 1.0, 5.0, 0.0 }, { 1.0, 5.0, 4.0 } }, // H11
                    { { 1.0, 0.0, 0.0 }, { 1.0, 5.0, 4.0 } }  // H12
                },
                {
                    // Row 2 (Output 2)
                    { { -4.0, 0.0, 0.0 }, { 1.0, 5.0, 4.0 } }, // H21
                    { { 1.0, 0.0, 0.0 }, { 1.0, 5.0, 4.0 } }   // H22
                }
            };

            // Validate transfer functions for all input-output pairs.
            for (std::size_t i = 0; i < 2; ++i) {
                for (std::size_t j = 0; j < 2; ++j) {
                    const auto &tf       = transfer_functions[i][j];
                    const auto &expected = expected_tf[i][j];
                    // Check numerator
                    if (fsmlib::any(fsmlib::abs(tf.numerator - expected.numerator) > 1e-06)) {
                        std::cerr << "Expected numerator for H(" << i + 1 << ", " << j + 1 << "):\n"
                                  << expected.numerator << "\nGot numerator:\n"
                                  << tf.numerator << "\n";
                        throw std::runtime_error("Test failed: numerator mismatch in MIMO transfer function");
                    }
                    // Check denominator
                    if (fsmlib::any(fsmlib::abs(tf.denominator - expected.denominator) > 1e-06)) {
                        std::cerr << "Expected denominator for H(" << i + 1 << ", " << j + 1 << "):\n"
                                  << expected.denominator << "\nGot denominator:\n"
                                  << tf.denominator << "\n";
                        throw std::runtime_error("Test failed: denominator mismatch in MIMO transfer function");
                    }
                }
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++
                      << " passed: ss2tf with 2x2 system and MIMO model\n";
        }

        std::cout << "All control tests passed!\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
