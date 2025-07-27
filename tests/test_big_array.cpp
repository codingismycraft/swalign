//  Tests the BigArray class.
//
#include <iostream>
#include "big_array.h"


int main(){

    const uint64_t ROWS = 5;
    const uint64_t COLS = 5;


    auto p_big_array = BigArray::make_new("test_file.bin", ROWS, COLS);
    std::cout << p_big_array->get_filename() << std::endl;

    for (uint64_t row = 0; row < ROWS; ++row) {
        for (uint64_t col = 0; col < COLS; ++col) {
            p_big_array->set( row, col, row*col + col);
        }
    }

    for (uint64_t row = 0; row < ROWS; ++row) {
        for (uint64_t col = 0; col < COLS; ++col) {
            const int v = p_big_array->get( row, col);
            std::cout << "Value at (" <<row  << ", " << col << ") = " << v << std::endl;

        }
    }
}
