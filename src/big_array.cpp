#include "big_array.h"
#include "utils.h"
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

constexpr size_t HEADER_SIZE = sizeof(size_t) * 2;
constexpr std::string BASE_DIR = "/tmp/";


class BigArrayBase: public IBigArray {

    public:
        BigArrayBase();
        virtual ~BigArrayBase();

        int32_t get(size_t row, size_t col) const override;
        void set(size_t row, size_t col, int value) override;
        const std::string& get_filename() const override;

        void create_new(const std::string& filename, size_t rows, size_t cols);
        void load_from_file(const std::string& filename);
        size_t antidiagonal_size(size_t antidiagonal_index) const;
        size_t antidiagonals_count() const;

        virtual size_t find_flat_index(size_t row, size_t col)  const = 0;

    private:
        BigArrayBase(const BigArrayBase&) = delete;
        BigArrayBase& operator=(const BigArrayBase&) = delete;
        BigArrayBase(BigArrayBase&&) = delete;
        BigArrayBase& operator=(BigArrayBase&&) = delete;

     protected:
        std::string m_filename;
        size_t m_rows;
        size_t m_cols;
        size_t m_file_size;
        int m_fd;
        void* m_mmapped_ptr;
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

size_t BigArrayBase::antidiagonals_count() const{
    if (m_rows == 0 || m_cols == 0) {
        return 0;
    }
    return m_rows + m_cols - 1;
}

size_t BigArrayBase::antidiagonal_size(size_t antidiagonal_index) const {
    if (antidiagonal_index >= antidiagonals_count()) {
        throw std::out_of_range("Antidiagonal index out of bounds");
    }
    const size_t min_dim = std::min(m_rows, m_cols);
    const size_t max_dim = std::max(m_rows, m_cols);

    if (antidiagonal_index < min_dim) {
        return antidiagonal_index + 1;
    } else if (antidiagonal_index < max_dim) {
        return min_dim;
    } else {
        return m_rows + m_cols - 1 - antidiagonal_index;
    }
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
    return m_data[find_flat_index(row, col)];
}

void BigArrayBase::set(size_t row, size_t col, int32_t value) {
    if (row >= m_rows || col >= m_cols) {
        throw std::out_of_range("Index out of bounds");
    }
    m_data[find_flat_index(row, col)] = value;
}

const std::string& BigArrayBase::get_filename() const {
    return m_filename;
}


class BigArrayRect: public BigArrayBase {
    protected:
        size_t find_flat_index(size_t row, size_t col) const;
};

size_t BigArrayRect::find_flat_index(size_t row, size_t col) const {
    return row * m_cols + col;
}


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
