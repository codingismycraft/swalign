#include "big_array.h"
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

constexpr uint64_t HEADER_SIZE = sizeof(uint64_t) * 2;

std::unique_ptr<BigArray> BigArray::make_new(const std::string& filename, uint64_t rows, uint64_t cols) {
    auto big_array = std::unique_ptr<BigArray>(new BigArray());
    big_array->create_new(filename, rows, cols);
    if (big_array->m_fd == -1) {
        throw std::runtime_error("Failed to load BigArray from file: " + filename);
    }
    return big_array;
}


std::unique_ptr<BigArray> BigArray::load(const std::string& filename) {
    auto big_array = std::unique_ptr<BigArray>(new BigArray());
    big_array->load_from_file(filename);
    if (big_array->m_fd == -1) {
        throw std::runtime_error("Failed to load BigArray from file: " + filename);
    }
    return big_array;
}

BigArray::BigArray() :
    m_filename(""),
    m_rows(0),
    m_cols(0),
    m_file_size(0),
    m_fd(-1),
    m_mmapped_ptr(nullptr),
    m_data(nullptr)
{
}


void BigArray::create_new(const std::string& filename, uint64_t rows, uint64_t cols)
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

    const uint64_t array_size = m_rows * m_cols;

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
    memcpy(m_mmapped_ptr, &m_rows, sizeof(uint64_t));
    memcpy((char*)m_mmapped_ptr + sizeof(uint64_t), &m_cols, sizeof(uint64_t));

    m_data = (int32_t*)((char*)m_mmapped_ptr + HEADER_SIZE);
}

void BigArray::load_from_file(const std::string& filename)
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

    if (static_cast<uint64_t>(st.st_size) < HEADER_SIZE) {
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

    memcpy(&m_rows, m_mmapped_ptr, sizeof(uint64_t));
    memcpy(&m_cols, (char*)m_mmapped_ptr + sizeof(uint64_t), sizeof(uint64_t));

    m_data = (int32_t*)((char*)m_mmapped_ptr + HEADER_SIZE);
}


BigArray::~BigArray(){
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

int32_t BigArray::get(uint64_t row, uint64_t col) const {
    if (row >= m_rows || col >= m_cols) {
        throw std::out_of_range("Index out of bounds");
    }
    return m_data[row * m_cols + col];
}

void BigArray::set(uint64_t row, uint64_t col, int32_t value) {
    if (row >= m_rows || col >= m_cols) {
        throw std::out_of_range("Index out of bounds");
    }
    m_data[row * m_cols + col] = value;
}

const std::string& BigArray::get_filename() const {
    return m_filename;
}
