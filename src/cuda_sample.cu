#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define THREADS_PER_BLOCK 256

int get_count_of_cells(int index, int rows, int cols) {
    const int start_i = (index - (cols - 1) > 0) ? (index - (cols - 1)) : 0;
    const int end_i = (index < rows - 1) ? index : (rows - 1);
    const int count = end_i - start_i + 1;
    return (count > 0) ? count : 0;
}


__global__ void update_cell_in_diagonal(int* matrix, int d, int cols, int cells_count) {
    const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index < cells_count) {
       const int row = thread_index + (d >= cols ? (d - cols + 1) : 0);
       const int col = d - row;
       const int i = row * cols + col;
       matrix[i] =  cells_count;
    }
}

int diagonal_count_cells() {
    const char* psz1 = "hello";
    const char* psz2 = "hellona";

    const int rows = strlen(psz2);
    const int cols = strlen(psz1);

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
        update_cell_in_diagonal<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, index, cols, count);
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
    return diagonal_count_cells();
}
