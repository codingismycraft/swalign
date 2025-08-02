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
#include <cstddef> // for size_t
#include <memory>


class BigArray{

    public:
        static std::unique_ptr<BigArray> make_new(const std::string& filename, size_t rows, size_t  cols);
        static std::unique_ptr<BigArray> load(const std::string& filename);

        ~BigArray();

        int32_t get(size_t row, size_t col) const;
        void set(size_t row, size_t col, int value);
        const std::string& get_filename() const;
    private:
        BigArray();
        BigArray(const BigArray&) = delete;
        BigArray& operator=(const BigArray&) = delete;
        BigArray(BigArray&&) = delete;
        BigArray& operator=(BigArray&&) = delete;

        void create_new(const std::string& filename, size_t rows, size_t cols);
        void load_from_file(const std::string& filename);
    private:
        std::string m_filename;
        size_t m_rows;
        size_t m_cols;
        size_t m_file_size;
        int m_fd;
        void* m_mmapped_ptr;
        int32_t* m_data;
};


#endif // BIG_ARRAY_INCLUDED
