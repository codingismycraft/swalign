//  Tests the BigArray class.
//
#include <assert.h>
#include <iostream>
#include <string>
#include "big_array.h"

#define ROWS 50
#define COLS 50


std::string create_file() {
    auto p_big_array = BigArray::make_new(ROWS, COLS);
    for (size_t row = 0; row < ROWS; ++row) {
        for (size_t col = 0; col < COLS; ++col) {
            p_big_array->set( row, col, row * COLS + col);
        }
    }
    for (size_t row = 0; row < ROWS; ++row) {
        for (size_t col = 0; col < COLS; ++col) {
            const auto expected = int32_t(row * COLS + col);
            const auto retrieved = p_big_array->get( row, col);
            assert(retrieved == expected);
        }
    }
    return p_big_array->get_filename();
}


void read_file(const std::string& filename) {
    auto p_big_array = BigArray::load(filename);

    for (size_t row = 0; row < ROWS; ++row) {
        for (size_t col = 0; col < COLS; ++col) {
            const auto expected = int32_t(row * COLS + col);
            const auto retrieved = p_big_array->get( row, col);
            assert(retrieved == expected);
        }
    }
}


int main(){
    auto filename = create_file();
    read_file(filename);
    std::cout << "BigArray: all tests passed!" << std::endl;
}
