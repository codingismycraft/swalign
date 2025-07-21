
#include "local_alignment.h"

#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <new>


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
        const int row = thread_index + ((d >= cols) ? (d - cols + 1) : 0);
        const int col = d - row;
        const int matrix_flat_index = get_flat_index(row, col, cols);

        const int upper_v = get_value(matrix, row - 1, col, cols) + gap_penalty;
        const int left_v = get_value(matrix, row, col - 1, cols) + gap_penalty;

        int diagonal_v = get_value(matrix, row - 1, col - 1, cols) ;

        if (strA[col] == strB[row]) {
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
            m_rows(s2.length()),
            m_cols(s1.length()),
            m_matrix_size(m_rows * m_cols* sizeof(int))
{
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

    for (int index = 0; index < m_cols + m_rows - 1;  index++) {
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

}

void LocalAlignmentFinder::print_matrix() const {
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            printf("%3d ", m_matrix[i * m_cols + j]);
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



