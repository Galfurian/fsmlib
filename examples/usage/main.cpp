
#include <fsmlib/fsmlib.hpp>
#include <fsmlib/math.hpp>
#include <fsmlib/io.hpp>

#include <cassert>

int main(int, char *[])
{
    {
        fsmlib::Vector<int, 3> vec1 = { 1, 2, 3 };
        fsmlib::Vector<int, 3> vec2 = { 4, 5, 6 };

        auto result = vec1 + vec2;

        std::cout << result << "\n";
    }
    {
        fsmlib::Matrix<int, 2, 2> mat1 = { { { 1, 2 }, { 3, 4 } } };
        fsmlib::Matrix<int, 2, 2> mat2 = { { { 5, 6 }, { 7, 8 } } };

        auto result = mat1 + mat2;

        std::cout << result << "\n";
    }
    {
        fsmlib::Vector<int, 3> vec = { 1, 2, 3 };
        int value = 2;

        auto result = vec + value;

        std::cout << result << "\n";
    }
    {
        fsmlib::Matrix<int, 2, 2> mat = { { { 1, 2 }, { 3, 4 } } };
        int value = 2;

        auto result = mat + value;

        std::cout << result << "\n";
    }
    return 0;
}