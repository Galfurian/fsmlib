

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
            fsmlib::control::state_space_t<double, 2, 1, 1> sys = {
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
            fsmlib::Matrix<double, 2, 1> expected_B = {
                { 4.6443e-03 },
                { 8.9133e-02 }
            };

            // Validate results
            if (fsmlib::any(dsys.A != expected_A)) {
                std::ostringstream error;
                error << "Test failed: Incorrect A matrix in Continuous-to-Discrete Conversion.\n"
                      << "Expected A:\n"
                      << expected_A << "\n"
                      << "Got A:\n"
                      << dsys.A << "\n";
                throw std::runtime_error(error.str());
            }
            if (fsmlib::any(dsys.B != expected_B)) {
                std::ostringstream error;
                error << "Test failed: Incorrect B matrix in Continuous-to-Discrete Conversion.\n"
                      << "Expected B:\n"
                      << expected_B << "\n"
                      << "Got B:\n"
                      << dsys.B << "\n";
                throw std::runtime_error(error.str());
            }
            if (fsmlib::any(dsys.C != sys.C)) {
                std::ostringstream error;
                error << "Test failed: Incorrect C matrix in Continuous-to-Discrete Conversion.\n"
                      << "Expected C:\n"
                      << sys.C << "\n"
                      << "Got C:\n"
                      << dsys.C << "\n";
                throw std::runtime_error(error.str());
            }
            if (fsmlib::any(dsys.D != sys.D)) {
                std::ostringstream error;
                error << "Test failed: Incorrect D matrix in Continuous-to-Discrete Conversion.\n"
                      << "Expected D:\n"
                      << sys.D << "\n"
                      << "Got D:\n"
                      << dsys.D << "\n";
                throw std::runtime_error(error.str());
            }

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Continuous-to-Discrete Conversion\n";
        }

        // Test 2: Simulate one step of a discrete-time model
        {
            // Define a discrete-time state-space model
            fsmlib::control::discrete_state_space_t<double, 2, 1, 1> dsys = {
                { { 0.904837, 0.0951626 }, { -0.951626, 0.809685 } },
                { { 0.00483742 }, { 0.0951626 } },
                { { 1.0, 0.0 } },
                { { 0.0 } },
                0.1
            };

            // Define initial state and input
            fsmlib::Vector<double, 2> x = { 1.0, 0.0 };
            fsmlib::Vector<double, 1> u = { 0.0 };

            // Simulate one step
            fsmlib::Vector<double, 2> x_next;
            fsmlib::Vector<double, 1> y;
            fsmlib::control::step(dsys, x, u, x_next, y);

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

            std::cout << "Test " << std::setw(2) << std::right << test_count++ << " passed: Simulate one step of a discrete-time model\n";
        }

        std::cout << "All control tests passed!\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
