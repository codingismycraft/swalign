/*
 ******************************************************************************
 *
 *  big_array.h
 *
 *  Copyright (c) 2025 John Pazarzis
 *  Licensed under the GPL License.
 *
 * This header defines the BigArray class, which allows for the creation and
 * use of large, memory-mapped arrays stored in files.
 *
 ******************************************************************************
 */

#ifndef BIG_ARRAY_INCLUDED
#define BIG_ARRAY_INCLUDED

#include <string>
#include <cstddef>
#include <memory>

class IBigArray{
    public:
        virtual ~IBigArray() = default;
        virtual int32_t get(size_t row, size_t col) const = 0;
        virtual void set(size_t row, size_t col, int32_t value) = 0;
        virtual const std::string& get_filename() const = 0;
};

class IBigArrayAntidiagonal : public IBigArray {
    public:
        virtual size_t get_antidiagonals_count() const = 0;
        virtual size_t get_antidiagonal_size(size_t antidiagonal_index) const = 0;
        virtual size_t get_max_antidiagonal_size() const = 0;
        virtual size_t copy_diagonal(size_t antidiagonal_index, int32_t* const buffer, size_t buffer_length) const = 0;
        virtual size_t assign_from_diagonal(size_t antidiagonal_index, const int32_t* const buffer) = 0;
};


std::unique_ptr<IBigArray> make_new(size_t rows, size_t  cols);
std::unique_ptr<IBigArray> load(const std::string& filename);

std::unique_ptr<IBigArrayAntidiagonal> make_new_antidiagonal(size_t rows, size_t  cols);
std::unique_ptr<IBigArrayAntidiagonal> load_antidiagonal(const std::string& filename);


#endif // BIG_ARRAY_INCLUDED
