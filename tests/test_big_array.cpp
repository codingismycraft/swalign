//  Tests the BigArray class.
//
#include <iostream>
#include "big_array.h"


int main(){

    const size_t ROWS = 5;
    const size_t COLS = 5;


    auto p_big_array = BigArray::make_new("test_file.bin", ROWS, COLS);
    std::cout << p_big_array->get_filename() << std::endl;

    for (size_t row = 0; row < ROWS; ++row) {
        for (size_t col = 0; col < COLS; ++col) {
            p_big_array->set( row, col, row*col + col);
        }
    }

    for (size_t row = 0; row < ROWS; ++row) {
        for (size_t col = 0; col < COLS; ++col) {
            const int v = p_big_array->get( row, col);
            std::cout << "Value at (" <<row  << ", " << col << ") = " << v << std::endl;

        }
    }
}
