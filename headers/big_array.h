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
        virtual void set(size_t row, size_t col, int value) = 0;
        virtual const std::string& get_filename() const = 0;
};


std::unique_ptr<IBigArray> make_new(size_t rows, size_t  cols);
std::unique_ptr<IBigArray> make_new_antidiagonal(size_t rows, size_t  cols);
std::unique_ptr<IBigArray> load(const std::string& filename);


#endif // BIG_ARRAY_INCLUDED
