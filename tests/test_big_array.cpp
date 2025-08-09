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

void test_antidiagonals() {
    size_t rows = 2;
    size_t cols = 3;
    auto p_big_array = make_new_antidiagonal(rows, cols);

    std::cout << "Testing antidiagonals for a 2x3 array..." << std::endl;

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            p_big_array->set( row, col, row * cols + col);
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            const auto expected = int32_t(row * cols + col);
            const auto retrieved = p_big_array->get( row, col);
            assert(retrieved == expected);
        }
    }

    auto p_big_array1 = load_antidiagonal(p_big_array->get_filename());

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            const auto expected = int32_t(row * cols + col);
            const auto retrieved = p_big_array1->get( row, col);
            if (retrieved != expected) {
                std::cout << "Mismatch at (" << row << ", " << col << "): "
                          << "expected " << expected << ", got " << retrieved << std::endl;
            }
            assert(retrieved == expected);
        }
    }

    const size_t expected_antidiagonals = 4;
    assert(p_big_array1->get_antidiagonals_count() == expected_antidiagonals);

    assert(p_big_array1->get_antidiagonal_size(0) == 1);
    assert(p_big_array1->get_antidiagonal_size(1) == 2);
    assert(p_big_array1->get_antidiagonal_size(2) == 2);
    assert(p_big_array1->get_antidiagonal_size(3) == 1);

    try{
        auto x = p_big_array1->get_antidiagonal_size(1023);
        std::cout << "Unexpectedly retrieved size: " << x << std::endl;
        assert(false); // Should not reach here
    } catch (const std::out_of_range& e) {
        // Expected
    }

    const size_t max_size = p_big_array1->get_max_antidiagonal_size() * sizeof(int32_t);
    int32_t* buffer = (int32_t*)malloc(max_size);

    size_t copied = p_big_array1->copy_diagonal(0, buffer, max_size);
    assert(copied == 1);
    assert(buffer[0] == 0);

    copied = p_big_array1->copy_diagonal(1, buffer, max_size);
    assert(copied == 2);
    assert(buffer[0] == 1);
    assert(buffer[1] == 3);

    copied = p_big_array1->copy_diagonal(2, buffer, max_size);
    assert(copied == 2);
    assert(buffer[0] == 2);
    assert(buffer[1] == 4);

    copied = p_big_array1->copy_diagonal(3, buffer, max_size);
    assert(copied == 1);
    assert(buffer[0] == 5);

    try{
        auto x = p_big_array1->copy_diagonal(4, buffer, max_size);
        std::cout << "Copy invalid diagonal did not threw exception. Seems that it copied" << x << std::endl;
        assert(false); // Should not reach here
    } catch (const std::out_of_range& e) {
        // Expected
    }

    int32_t buffer_const[10] ;
    buffer_const[0] = 123;
    copied = p_big_array1->assign_from_diagonal(0, buffer_const);
    const auto x = p_big_array1->get(0,0);
    assert(x == 123);
    std::cout << "Value at (0,0) after assign from diagonal 0: " << x << std::endl;

    free(buffer);
    buffer = nullptr;

}


int main(){
    auto filename = create_file();
    read_file(filename);
    test_antidiagonals();
    std::cout << "BigArray: all tests passed!" << std::endl;
}
