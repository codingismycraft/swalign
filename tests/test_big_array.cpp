//  Tests the BigArray class.
//
#include <assert.h>
#include <iostream>
#include "big_array.h"

#define TESTING_FILE_NAME "test_file.bin"
#define ROWS 50
#define COLS 50


void create_file() {
    auto p_big_array = BigArray::make_new(TESTING_FILE_NAME, ROWS, COLS);
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
}


void read_file() {
    auto p_big_array = BigArray::load(TESTING_FILE_NAME);

    for (size_t row = 0; row < ROWS; ++row) {
        for (size_t col = 0; col < COLS; ++col) {
            const auto expected = int32_t(row * COLS + col);
            const auto retrieved = p_big_array->get( row, col);
            assert(retrieved == expected);
        }
    }
}


int main(){
    create_file();
    read_file();
    std::cout << "BigArray: all tests passed!" << std::endl;
}
