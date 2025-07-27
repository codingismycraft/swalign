#include "big_array.h"

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

static const uint64_t _HEADER_SIZE = sizeof(uint64_t) * 2;

std::unique_ptr<BigArray> BigArray::make_new(const std::string& filename, uint64_t rows, uint64_t cols) {
    return std::unique_ptr<BigArray>(new BigArray(filename, rows, cols));
}


std::unique_ptr<BigArray> BigArray::load(const std::string& filename) {
    return std::unique_ptr<BigArray>(new BigArray(filename));
}


BigArray::BigArray(const std::string& filename, uint64_t rows, uint64_t cols)
    : m_filename(filename),
      m_rows(rows),
      m_cols(cols)
{
    const uint64_t array_size = m_rows * m_cols;

    m_file_size = _HEADER_SIZE + array_size * sizeof(int32_t);

    m_fd = open(m_filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);

    if (m_fd == -1) {
        perror("open");
        exit(1);
    }

    if (ftruncate(m_fd, m_file_size) == -1) {
        perror("ftruncate");
        exit(1);
    }

    // Only map up to size_t bytes at a time!
    if (m_file_size > SIZE_MAX) {
        fprintf(stderr, "Cannot map file larger than SIZE_MAX at once.\n");
        exit(1);
    }


    m_mmapped_ptr = mmap(
            NULL, m_file_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0
    );

    if (m_mmapped_ptr == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    // Write header
    memcpy(m_mmapped_ptr, &m_rows, sizeof(uint64_t));
    memcpy((char*)m_mmapped_ptr + sizeof(uint64_t), &m_cols, sizeof(uint64_t));

    m_data = (int32_t*)((char*)m_mmapped_ptr + _HEADER_SIZE);
}

BigArray::BigArray(const std::string& filename)
    : m_filename(filename)
{
    m_fd = open(m_filename.c_str(), O_RDWR, 0);

    if (m_fd == -1) {
        perror("open");
        exit(1);
    }

    struct stat st;

    if (fstat(m_fd, &st) == -1) {
        perror("fstat");
        exit(1);
    }

    m_file_size = st.st_size;

    if (m_file_size > SIZE_MAX) {
        fprintf(stderr, "Cannot map file larger than SIZE_MAX at once.\n");
        exit(1);
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
        perror("mmap");
        exit(1);
    }

    memcpy(&m_rows, m_mmapped_ptr, sizeof(uint64_t));
    memcpy(&m_cols, (char*)m_mmapped_ptr + sizeof(uint64_t), sizeof(uint64_t));

    m_data = (int32_t*)((char*)m_mmapped_ptr + _HEADER_SIZE);
}


BigArray::~BigArray(){
  if (m_mmapped_ptr && m_file_size) {
      msync(m_mmapped_ptr, (size_t)m_file_size, MS_SYNC);
  }

  if (m_mmapped_ptr) {
      munmap(m_mmapped_ptr, (size_t)m_file_size);
  }

  if (m_fd != -1) {
      close(m_fd);
      m_fd = -1;
  }
}

int32_t BigArray::get(uint64_t row, uint64_t col) const {
    return m_data[row * m_cols + col];
}

void BigArray::set(uint64_t row, uint64_t col, int value) {
    m_data[row * m_cols + col] = value;
}

const std::string& BigArray::get_filename() const {
    return m_filename;
}
