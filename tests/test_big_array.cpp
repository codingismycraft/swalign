//  Tests the BigArray class.
//
#include <assert.h>
#include <iostream>
#include <string>
#include "big_array.h"

#define ROWS 50
#define COLS 50


std::string create_file() {
    auto p_big_array = make_new(ROWS, COLS);
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
    auto p_big_array = load(filename);

    for (size_t row = 0; row < ROWS; ++row) {
        for (size_t col = 0; col < COLS; ++col) {
            const auto expected = int32_t(row * COLS + col);
            const auto retrieved = p_big_array->get( row, col);
            assert(retrieved == expected);
        }
    }
}

// void test_antidiagonals() {
//     size_t rows = 2;
//     size_t cols = 3;
//
//     auto p_big_array = make_new(rows, cols);
//
//     assert(p_big_array->antidiagonals_count() == 4);
//
//     assert(p_big_array->antidiagonal_size(0) == 1);
//     assert(p_big_array->antidiagonal_size(1) == 2);
//     assert(p_big_array->antidiagonal_size(2) == 2);
//     assert(p_big_array->antidiagonal_size(3) == 1);
//
//     try {
//         p_big_array->antidiagonal_size(4);
//         assert(false); // Should not reach here
//     } catch (const std::out_of_range&) {
//         std::cout << "Antidiagonal index out of bounds caught as expected." << std::endl;
//     }
//}


int main(){
    auto filename = create_file();
    read_file(filename);
    //test_antidiagonals();
    std::cout << "BigArray: all tests passed!" << std::endl;
}
