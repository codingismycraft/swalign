#include "big_array.h"
#include "utils.h"
#include <cstring>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <algorithm>
#include <sys/stat.h>

constexpr size_t HEADER_SIZE = sizeof(size_t) * 2;
constexpr std::string BASE_DIR = "/tmp/";


class BigArrayBase: public IBigArray {

    public:
        BigArrayBase();
        virtual ~BigArrayBase();

        int32_t get(size_t row, size_t col) const override;
        void set(size_t row, size_t col, int32_t value) override;
        const std::string& get_filename() const override;
        virtual size_t find_flat_index(size_t row, size_t col)  const = 0;


        virtual void create_new(const std::string& filename, size_t rows, size_t cols);
        virtual void load_from_file(const std::string& filename);

        inline size_t get_rows_count() const { return m_rows; }
        inline size_t get_cols_count() const { return m_cols; }

    private:
        BigArrayBase(const BigArrayBase&) = delete;
        BigArrayBase& operator=(const BigArrayBase&) = delete;
        BigArrayBase(BigArrayBase&&) = delete;
        BigArrayBase& operator=(BigArrayBase&&) = delete;

     private:
        std::string m_filename;
        size_t m_rows;
        size_t m_cols;
        size_t m_file_size;
        int m_fd;
        void* m_mmapped_ptr;
      protected:
        int32_t* m_data;
};


BigArrayBase::BigArrayBase() :
    m_filename(""),
    m_rows(0),
    m_cols(0),
    m_file_size(0),
    m_fd(-1),
    m_mmapped_ptr(nullptr),
    m_data(nullptr)
{
}


void BigArrayBase::create_new(const std::string& filename, size_t rows, size_t cols)
{
    if (m_fd != -1) {
        throw std::runtime_error("BigArray already initialized, cannot create new instance");
    }

    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Rows and columns must be greater than zero");
    }

    m_filename = filename;
    m_rows = rows;
    m_cols = cols;

    const size_t array_size = m_rows * m_cols;

    if (m_cols != 0 && array_size / m_cols != m_rows) {
        throw std::runtime_error("Row or column count overflow");
    }

    m_file_size = HEADER_SIZE + array_size * sizeof(int32_t);

    if (m_file_size > SIZE_MAX) {
        throw std::runtime_error("File size exceeds SIZE_MAX, cannot map.");
    }

    m_fd = open(m_filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);

    if (m_fd == -1) {
        throw std::runtime_error("Failed to open file for writing");
    }

    if (ftruncate(m_fd, m_file_size) == -1) {
        throw std::runtime_error("Failed to set file size");
    }


    m_mmapped_ptr = mmap(
        NULL, m_file_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0
    );

    if (m_mmapped_ptr == MAP_FAILED) {
        throw std::runtime_error("Failed to map file into memory");
    }

    // Write header
    memcpy(m_mmapped_ptr, &m_rows, sizeof(size_t));
    memcpy((char*)m_mmapped_ptr + sizeof(size_t), &m_cols, sizeof(size_t));

    m_data = (int32_t*)((char*)m_mmapped_ptr + HEADER_SIZE);

}

void BigArrayBase::load_from_file(const std::string& filename)
{
    if (m_fd != -1) {
        throw std::runtime_error("BigArray already initialized, cannot create new instance");
    }
    m_filename = filename;
    m_fd = open(m_filename.c_str(), O_RDWR, 0);

    if (m_fd == -1) {
        throw std::runtime_error("Failed to open file for reading");
    }

    struct stat st;

    if (fstat(m_fd, &st) == -1) {
        throw std::runtime_error("Failed to get file status");
    }

    if (static_cast<size_t>(st.st_size) < HEADER_SIZE) {
        throw std::runtime_error("File too small to be a valid BigArray");
    }

    m_file_size = st.st_size;

    if (m_file_size > SIZE_MAX) {
        throw std::runtime_error("File size exceeds SIZE_MAX, cannot map.");
    }

    m_mmapped_ptr = mmap(
            NULL,
            (size_t)m_file_size,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            m_fd,
            0
    );

    if (m_mmapped_ptr == MAP_FAILED) {
        throw std::runtime_error("Failed to map file into memory");
    }

    memcpy(&m_rows, m_mmapped_ptr, sizeof(size_t));
    memcpy(&m_cols, (char*)m_mmapped_ptr + sizeof(size_t), sizeof(size_t));

    m_data = (int32_t*)((char*)m_mmapped_ptr + HEADER_SIZE);
}


BigArrayBase::~BigArrayBase(){
  if (m_mmapped_ptr && m_file_size) {
      msync(m_mmapped_ptr, (size_t)m_file_size, MS_SYNC);
  }

  if (m_mmapped_ptr) {
      munmap(m_mmapped_ptr, (size_t)m_file_size);
      m_mmapped_ptr = nullptr;
  }

  if (m_fd != -1) {
      close(m_fd);
      m_fd = -1;
  }
}


int32_t BigArrayBase::get(size_t row, size_t col) const {
    if (row >= m_rows || col >= m_cols) {
        throw std::out_of_range("Index out of bounds");
    }
    const auto index = find_flat_index(row, col);
    return m_data[index];
}

void BigArrayBase::set(size_t row, size_t col, int32_t value) {
    if (row >= m_rows || col >= m_cols) {
        throw std::out_of_range("Index out of bounds");
    }
    const auto index = find_flat_index(row, col);
    m_data[index] = value;
}

const std::string& BigArrayBase::get_filename() const {
    return m_filename;
}


class BigArrayRect: public BigArrayBase {
    protected:
        inline size_t find_flat_index(size_t row, size_t col) const override {
            const auto index = row * get_cols_count() + col;
            return index;
        }
};

class BigArrayAntidiagonal: public IBigArrayAntidiagonal, public BigArrayBase {
    public:
        virtual ~BigArrayAntidiagonal() = default;

        int32_t get(size_t row, size_t col) const override {
            return BigArrayBase::get(row, col);
        }

        void set(size_t row, size_t col, int32_t value) override {
            BigArrayBase::set(row, col, value);
        }

        const std::string& get_filename() const override {
            return BigArrayBase::get_filename();
        }

        size_t get_antidiagonals_count() const override {
            return m_antidiagonal_size.size();
        }

        size_t get_antidiagonal_size(size_t antidiagonal_index) const override {
            return m_antidiagonal_size.at(antidiagonal_index);
        }

        size_t get_max_antidiagonal_size() const override {
            return m_max_antidiagonal_size;
        }

        size_t copy_diagonal(size_t antidiagonal_index, int32_t* const buffer, size_t buffer_length) const override {
            if (antidiagonal_index >= get_antidiagonals_count()) {
                throw std::out_of_range("Antidiagonal index out of bounds");
            }

            const auto antidiagonal_size = get_antidiagonal_size(antidiagonal_index);
            const size_t size = antidiagonal_size * sizeof(int32_t);

            if (buffer_length < size) {
                throw std::out_of_range("Buffer index out of bounds");
            }

            size_t offset = 0;
            for (size_t k = 0; k < antidiagonal_index; ++k){
                offset += m_antidiagonal_size[k];
            }

            memcpy(buffer, m_data + offset, size * sizeof(int32_t));
            return antidiagonal_size;
        }

        size_t assign_from_diagonal(size_t antidiagonal_index, const int32_t* const buffer) override {
            if (antidiagonal_index >= get_antidiagonals_count()) {
                throw std::out_of_range("Antidiagonal index out of bounds");
            }

            const auto antidiagonal_size = get_antidiagonal_size(antidiagonal_index);
            size_t offset = 0;
            for (size_t k = 0; k < antidiagonal_index; ++k){
                offset += m_antidiagonal_size[k];
            }

            memcpy(m_data + offset, buffer, antidiagonal_size * sizeof(int32_t));
            return antidiagonal_size;
        }

        void create_new(const std::string& filename, size_t rows, size_t cols) override {
            BigArrayBase::create_new(filename, rows, cols);
            precalc_get_antidiagonal_size_internals();
        }

        void load_from_file(const std::string& filename) override {
            BigArrayBase::load_from_file(filename);
            precalc_get_antidiagonal_size_internals();
        }
    protected:
        inline size_t find_flat_index(size_t row, size_t col) const override {
            const auto index = antidiagonal_index(row, col);
            return index;
        }

    private:

        size_t antidiagonal_offset(size_t row, size_t col) const {
            const size_t d = row + col;
            size_t offset = 0;
            for (size_t k = 0; k < d; ++k)
                offset += m_antidiagonal_size[k];
            return offset;
        }

        size_t antidiagonal_index(size_t row, size_t col) const {
            const size_t d = row + col;
            const size_t cols = get_cols_count();
            // Calculate offset: total elements in previous antidiagonals
            size_t offset = 0;
            for (size_t k = 0; k < d; ++k)
                offset += m_antidiagonal_size[k];

            // Position within antidiagonal
            const size_t min_val = (d >= (cols - 1)) ? (d - (cols - 1)) : 0;
            const size_t pos_in_antidiagonal = row - min_val;

            return offset + pos_in_antidiagonal;
        }

        void precalc_get_antidiagonal_size_internals() {
            m_antidiagonal_size.clear();
            m_max_antidiagonal_size = 0;
            for (size_t i = 0; i < get_antidiagonals_count_internal(); ++i) {
                const auto length = get_antidiagonal_size_internal(i);
                m_antidiagonal_size.push_back(length);
                if (length > m_max_antidiagonal_size) {
                    m_max_antidiagonal_size = length;
                }
            }

        }

        size_t get_antidiagonal_size_internal(size_t antidiagonal_index) const {
            if (antidiagonal_index >= get_antidiagonals_count_internal()) {
                throw std::out_of_range("Antidiagonal index out of bounds");
            }
            const size_t rows = get_rows_count();
            const size_t cols = get_cols_count();

            const size_t min_dim = std::min(rows, cols);
            const size_t max_dim = std::max(rows, cols);

            if (antidiagonal_index < min_dim) {
                return antidiagonal_index + 1;
            } else if (antidiagonal_index < max_dim) {
                return min_dim;
            } else {
                return rows + cols - 1 - antidiagonal_index;
            }
        }

        inline size_t get_antidiagonals_count_internal() const {
            const size_t rows = get_rows_count();
            const size_t cols = get_cols_count();

            if (rows == 0 || cols == 0) {
                return 0;
            }
            return rows + cols - 1;
        }

        std::vector<size_t> m_antidiagonal_size;
        size_t m_max_antidiagonal_size = 0;

};


std::unique_ptr<IBigArray> make_new(size_t rows, size_t cols) {
    const std::string filename = BASE_DIR + generate_random_name();
    auto big_array = std::unique_ptr<BigArrayRect>(new BigArrayRect());
    big_array->create_new(filename, rows, cols);
    return big_array;
}

std::unique_ptr<IBigArray> load(const std::string& filename) {
    auto big_array = std::unique_ptr<BigArrayRect>(new BigArrayRect());
    big_array->load_from_file(filename);
    return big_array;
}

std::unique_ptr<IBigArrayAntidiagonal> make_new_antidiagonal(size_t rows, size_t cols) {
    const std::string filename = BASE_DIR + generate_random_name();
    auto big_array = std::unique_ptr<BigArrayAntidiagonal>(new BigArrayAntidiagonal());
    big_array->create_new(filename, rows, cols);
    return big_array;
}

std::unique_ptr<IBigArrayAntidiagonal> load_antidiagonal(const std::string& filename) {
    auto big_array = std::unique_ptr<BigArrayAntidiagonal>(new BigArrayAntidiagonal());
    big_array->load_from_file(filename);
    return big_array;
}
