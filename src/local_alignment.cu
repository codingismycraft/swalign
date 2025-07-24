
#include "local_alignment.h"

#include <sstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <new>
#include <iomanip>
#include <assert.h>


#define THREADS_PER_BLOCK 256

__device__ __forceinline__ int max_of_three(int a, int b, int c) {

    if (a < 0) {
        a = 0; // Ensure non-negative values
    }

    if (b < 0) {
        b = 0; // Ensure non-negative values
    }

    if (c < 0) {
        c = 0; // Ensure non-negative values
    }


    int max = a;
    if (b > max) max = b;
    if (c > max) max = c;
    return max;
}


__device__  __host__ int get_flat_index(int row, int col, int cols) {
    if (row < 0 || col < 0 || cols <= 0) {
        return -1;
    }
    return row * cols + col;
}

__device__ __host__ int get_value(int* matrix, int row, int col, int cols) {
    const int flat_index = get_flat_index(row, col, cols);
    if (flat_index < 0) {
        return 0; // Return 0 for out-of-bounds indices
    }
    return matrix[flat_index];
}


__global__ void update_cell_in_diagonal(
    int* matrix,
    int d,
    int cols,
    int cells_count,
    const char* strA,
    const char* strB,
    int match_score,
    int mismatch_penalty,
    int gap_penalty)
{
    const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_index < cells_count) {
        const int row = thread_index + ((d >= cols) ? (d - cols + 1) : 0) + 1;
        const int col = d - row + 1;
        const int matrix_flat_index = get_flat_index(row, col, cols );

        const int upper_v = get_value(matrix, row - 1, col, cols) + gap_penalty;
        const int left_v = get_value(matrix, row, col - 1, cols) + gap_penalty;

        int diagonal_v = get_value(matrix, row - 1, col - 1, cols) ;

        if (strA[col-1] == strB[row-1]) {
            diagonal_v += match_score;
        } else {
            diagonal_v += mismatch_penalty;
        }

        const int score = max_of_three(upper_v, left_v, diagonal_v);
        matrix[matrix_flat_index] = score;
    }
}



// Implementation of LocalAlignment class.
LocalAlignmentFinder::LocalAlignmentFinder( const std::string& s1, const std::string& s2,
        int match_score, int mismatch_penalty, int gap_penalty, size_t max_alignments):
            m_sequence1(s1),
            m_sequence2(s2),
            m_match_score(match_score),
            m_mismatch_penalty(mismatch_penalty),
            m_gap_penalty(gap_penalty),
            m_max_alignments(max_alignments),
            m_max_score(0),
            m_rows(s2.length()+1),
            m_cols(s1.length()+1),
            m_matrix_size(long(m_rows) * m_cols* sizeof(int))
{
    std::cout << "LocalAlignmentFinder initialized with sequences of lengths: "
              << m_rows << " and " << m_matrix_size << std::endl;
    if (m_matrix_size <= 0) {
        throw std::invalid_argument("value must be non-negative");
    }

    m_matrix = (int*)malloc(m_matrix_size);
    if (!m_matrix) {
        throw std::bad_alloc();
    }
    memset(m_matrix, 0, m_matrix_size);
    initializeMatrix();
}

LocalAlignmentFinder::~LocalAlignmentFinder() {
    if (m_matrix) {
        free(m_matrix);
        m_matrix = nullptr;
    }
}

int LocalAlignmentFinder::count_anti_diagonal_cells(int anti_diagonal_index) {
    const int start_i = (anti_diagonal_index - (m_cols - 1) > 0) ? (anti_diagonal_index - (m_cols - 1)) : 0;
    const int end_i = (anti_diagonal_index < m_rows - 1) ? anti_diagonal_index : (m_rows - 1);
    const int count = end_i - start_i + 1;
    return (count > 0) ? count : 0;
}

void LocalAlignmentFinder::initializeMatrix() {
    // Allocate device memory
    char *d_strA, *d_strB;
    cudaMalloc((void**)&d_strA, m_rows + 1);
    cudaMalloc((void**)&d_strB, m_cols + 1);

    // Copy strings to device
    cudaMemcpy(d_strA, m_sequence1.c_str(), m_rows + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_strB, m_sequence2.c_str(), m_cols + 1, cudaMemcpyHostToDevice);

    // Allocate memory on the GPU
    int* d_a;

    if (cudaMalloc((void**)&d_a, m_matrix_size) != cudaSuccess) {
        throw std::bad_alloc();
    }

    if (cudaMemcpy(d_a, m_matrix, m_matrix_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_a);
        throw std::bad_alloc();
    }

    for (int index = 1; index < m_cols + m_rows - 1;  index++) {
        const int count = count_anti_diagonal_cells(index);
        const int numBlocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        update_cell_in_diagonal<<<numBlocks, THREADS_PER_BLOCK>>>(
                d_a,
                index,
                m_cols,
                count,
                d_strA,
                d_strB,
                m_match_score,
                m_mismatch_penalty,
                m_gap_penalty
        );
    }

    cudaDeviceSynchronize();

    // Copy the matrix from GPU to CPU
    if (cudaMemcpy(m_matrix, d_a, m_matrix_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_a);
        throw std::bad_alloc();
    }

    // Free the GPU memory
    cudaFree(d_a);
    cudaFree(d_strA);
    cudaFree(d_strB);

    findMaxScores();
     for (const auto& pos : m_max_positions) {
        traceback(pos.first, pos.second, "", "", "");
    }

}

void LocalAlignmentFinder::print_matrix() const {
    for (int row = 0; row < m_rows; row++) {
        for (int col = 0; col < m_cols; col++) {
            printf("%3d ", getScore(row, col));
        }
        printf("\n");
    }
}

void LocalAlignmentFinder::findMaxScores() {
    int current_max = -1;
    m_max_positions.clear();

    for (int row = 0; row < m_rows; row++) {
        for (int col = 0; col < m_cols; col++) {
            const int value = m_matrix[row * m_cols + col];
            if (value > current_max) {
                m_max_positions.clear();
            }
            if (value >= current_max) {
                current_max = value;
                m_max_positions.push_back(std::make_pair(row, col));
            }
        }
    }
}

const std::string& LocalAlignmentFinder::getSequence1() const{
    return m_sequence1;
}


const std::string& LocalAlignmentFinder::getSequence2() const {
    return m_sequence2;
}


int LocalAlignmentFinder::getRowCount() const {
    return m_rows;
}

int LocalAlignmentFinder::getColCount() const {
    return m_cols;
}

int LocalAlignmentFinder::getScore(int row, int col) const {
    if (row < 0 || row >= m_rows || col < 0 || col >= m_cols) {
        throw std::out_of_range("Row or column index out of bounds");
    }
    const int index = row * m_cols + col;
    return m_matrix[index];
}

const std::vector<std::string>& LocalAlignmentFinder::getLocalAlignments() const {
    return m_local_alignments;
}

std::string LocalAlignmentFinder::toString() const {
    std::ostringstream oss;
    const int width = 6;
    if (m_matrix == nullptr) return "";

    // Build horizontal separator
    std::string separator;
    // The separator needs to cover one extra for the row label column.
    for (int col = 0; col < getColCount() + 1; ++col) {
        separator += std::string(width, '_');
        if (col + 1 < getColCount() + 1) separator += "|";
    }
    separator += "\n";

    // Print top separator
    oss << separator;

    // Print header row: empty cell, then m_sequence1 characters as column headers
    oss << std::setw(width) << ' ';

    oss << "|" << std::setw(width) << ' ';
    for (int col = 0; col < getColCount()-1; ++col) {
        oss << "|" << std::setw(width) << m_sequence1[col];
    }
    oss << "\n" << separator;

    // Print all matrix rows
    for (int row = 0; row < getRowCount(); ++row) {
        // First column: empty for first row, else m_sequence2 character
        if (row == 0) {
            oss << std::setw(width) << ' ';
        } else {
            oss << std::setw(width) << m_sequence2[row - 1];
        }
        // Print all matrix columns for this row
        for (int col = 0; col < getRowCount(); ++col) {
            oss << "|" << std::setw(width) << getScore(row, col);
        }
        oss << "\n" << separator;
    }

    return oss.str();
}

void LocalAlignmentFinder::traceback(int row, int col, std::string x1, std::string x2, std::string a) {
    while(row >0 && col >0 && getScore(row, col) > 0) {
        if (m_local_alignments.size() >= m_max_alignments)
        {
            return;
        }

        const int row_coming_in = row;
        const int col_coming_in = col;

        const std::string a_coming_in = a;
        const std::string x1_coming_in = x1;
        const std::string x2_coming_in = x2;

        const int current_score = getScore(row, col);
        const int diagonal_row = row - 1;
        const int diagonal_col = col - 1;

        const bool valid_diagonal = (diagonal_row >= 0 && diagonal_col >= 0);
        const bool valid_up_col = (col - 1 >= 0);
        const bool valid_left_row = (row - 1 >= 0);

        bool already_moved = false;


        if (valid_diagonal && getScore(diagonal_row, diagonal_col) + m_match_score == current_score) {
            if (!already_moved) {
                a = '*' + a_coming_in;
                x1 = m_sequence1[col_coming_in - 1] + x1_coming_in;
                x2 = m_sequence2[row_coming_in  - 1] + x2_coming_in;
                row = diagonal_row;
                col = diagonal_col;
                already_moved = true;
            }
            else {
                assert (false && "Traceback logic error: already_moved should not be true here");
            }
        }

        if (valid_diagonal && getScore(diagonal_row, diagonal_col) + m_mismatch_penalty == current_score) {
            if (!already_moved) {
                a = '|' + a_coming_in;
                x1 = m_sequence1[col_coming_in - 1] + x1_coming_in;
                x2 = m_sequence2[row_coming_in  - 1] + x2_coming_in;
                row = diagonal_row;
                col = diagonal_col;
                already_moved = true;
            }
            else {
                if (m_local_alignments.size() < m_max_alignments) {
                    traceback(
                        diagonal_row,
                        diagonal_col,
                        m_sequence1[col_coming_in - 1] + x1_coming_in,
                        m_sequence2[row_coming_in  - 1] + x2_coming_in,
                        '|' + a_coming_in
                   );
               }
            }
        }


        if (valid_left_row && getScore(row_coming_in  - 1, col_coming_in) + m_gap_penalty == current_score) {
            if (!already_moved) {
                a = ' ' + a;
                x1 = '_' + x1;
                x2= m_sequence2[row_coming_in  - 1] + x2;
                row -= 1;
                already_moved = true;
            }
            else {
                if (m_local_alignments.size() < m_max_alignments) {
                    traceback(
                        row_coming_in  - 1,
                        col_coming_in,
                        '_' + x1_coming_in,
                        m_sequence2[row_coming_in  - 1] + x2_coming_in,
                        ' ' + a_coming_in
                   );
                }
            }
        }

        if (valid_up_col && getScore(row_coming_in, col_coming_in  - 1) + m_gap_penalty == current_score) {
            if (!already_moved) {
                a = ' ' + a;
                x1 = m_sequence1[col - 1] + x1;
                x2 = '_' + x2;
                col  -= 1;
                already_moved = true;
            }
            else {
                if (m_local_alignments.size() < m_max_alignments) {
                    traceback(
                        row_coming_in,
                        col_coming_in  - 1,
                        m_sequence1[col_coming_in - 1] + x1_coming_in,
                        '_' + x2_coming_in,
                        ' ' + a_coming_in
                   );
                }
            }
        }

       assert(already_moved && "Traceback logic error");
       assert (row < row_coming_in || col < col_coming_in);
    }

    m_local_alignments.push_back("\n"  + x2 + "\n" + a + "\n" + x1 + "\n");
}



std::ostream& operator<<(std::ostream& os, const LocalAlignmentFinder& obj) {
    os << "\n***************************************" << std::endl;
    os << "Seq1: " << obj.getSequence1() << std::endl;
    os << "Seq2: " << obj.getSequence2() << std::endl;

    //os << "\nNumber of Alignments ..: " << obj.getNumberOfAlignments()<< std::endl;
    //os << "Max score .............: " << obj.getMaxScore()<< std::endl;

    int counter = 1;
    os << "\n--------------------------" << std::endl;
    for (const auto& aln : obj.getLocalAlignments()) {
         os << "Alignment num: " << counter++ << "\n";
         os << aln << std::endl;
         os << "--------------------------" << std::endl;
    }
    return os;
}

