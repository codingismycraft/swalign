#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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


__device__  int get_flat_index(int row, int col, int cols) {
    if (row < 0 || col < 0 || cols <= 0) {
        return -1;
    }
    return row * cols + col;
}

__device__  int get_value(int* matrix, int row, int col, int cols) {
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
    int match_score, int mismatch_penalty, int gap_penalty
) {
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

int get_count_of_cells(int index, int rows, int cols) {
    const int start_i = (index - (cols - 1) > 0) ? (index - (cols - 1)) : 0;
    const int end_i = (index < rows - 1) ? index : (rows - 1);
    const int count = end_i - start_i + 1;
    return (count > 0) ? count : 0;
}

int diagonal_count_cells(const char* psz1, const char* psz2) {

    const int match_score = 2;
    const int mismatch_penalty = -1;
    const int gap_penalty = -2;

    const int rows = strlen(psz2);
    const int cols = strlen(psz1);

    // Allocate device memory
    char *d_strA, *d_strB;
    cudaMalloc((void**)&d_strA, rows + 1); // +1 for null terminator
    cudaMalloc((void**)&d_strB, cols + 1);

    // Copy strings to device
    cudaMemcpy(d_strA, psz1, rows + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_strB, psz2, cols + 1, cudaMemcpyHostToDevice);

    const int size = rows * cols * sizeof(int);
    int* matrix = (int*)malloc(size);
    if (!matrix) {
        fprintf(stderr, "Failed to allocate host matrix\n");
        return 1;
    }
    memset(matrix, 0, size);

    // Allocate memory on the GPU
    int* d_a;

    if (cudaMalloc((void**)&d_a, size) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device matrix\n");
        free(matrix);
        return 2;
    }

    if (cudaMemcpy(d_a, matrix, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy to device\n");
        cudaFree(d_a);
        free(matrix);
        return 3;
    }

    for (int index = 0; index < cols + rows - 1;  index++) {
        const int count = get_count_of_cells(index, rows, cols);
        const int numBlocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        update_cell_in_diagonal<<<numBlocks, THREADS_PER_BLOCK>>>(
                d_a,
                index,
                cols,
                count,
                d_strA,
                d_strB,
                match_score,
                mismatch_penalty,
                gap_penalty
        );
    }

    cudaDeviceSynchronize();

     if (cudaMemcpy(matrix, d_a, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy from device\n");
        cudaFree(d_a);
        free(matrix);
        return 4;
    }

    // Copy the matrix from GPU to CPU
    if (cudaMemcpy(matrix, d_a, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy from device\n");
        cudaFree(d_a);
        free(matrix);
        return 4;
    }

    // Free the GPU memory
    cudaFree(d_a);

    cudaFree(d_strA);
    cudaFree(d_strB);

    // Print the resulting matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i * cols + j]);
        }
        printf("\n");
    }

    // Free the allocated memory
    free(matrix);
    return 0;
}

int main() {
    //return diagonal_sum();
    //
    const char* psz1 = "ACGTC";
    const char* psz2 = "ACATC";

    return diagonal_count_cells(psz1, psz2);
}
