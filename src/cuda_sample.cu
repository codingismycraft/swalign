

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void calc_scores(int** matrix, const char* psz1, const char* psz2){
    const int rows = strlen(psz1) + 1;
    const int cols = strlen(psz2) + 1;
    const int size = rows * cols * sizeof(int);
    *matrix = (int*)malloc(size);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            const int index = i * cols + j;
            (*matrix)[index] = index; // Initialize the matrix to zero
        }
    }
}

int main_using_cpu(){
    const char* psz1 = "hello";
    const char* psz2 = "helll";
    int* matrix = NULL;
    const int rows = strlen(psz1) + 1;
    const int cols = strlen(psz2) + 1;
    calc_scores(&matrix, psz1, psz2);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    free(matrix);
    return 0;
}

__global__ void calc_scores_gpu(int* matrix,int rows, int cols) {
    int i = blockIdx.x;
    if (i < rows * cols) {
        int row = i / cols;
        int col = i % cols;
        matrix[i] = row + col; // Example calculation
    }
}



int main_using_gpu() {
    const char* psz1 = "hello";
    const char* psz2 = "hellloword";
    const int rows = strlen(psz1) + 1;
    const int cols = strlen(psz2) + 1;
    const int size = rows * cols * sizeof(int);
    int* matrix = (int*)malloc(size);

    // Allocate memory on the GPU
    int* d_a;
    cudaMalloc((void**)&d_a, size);

    // Copy the matrix from CPU to the GPU
    cudaMemcpy(d_a, matrix, size, cudaMemcpyHostToDevice);

    // Call the kernel to perform the calculation
    //
    calc_scores_gpu<<<rows * cols, 1>>>(d_a, rows, cols);

    cudaDeviceSynchronize();

    // Copy the matrix from GPU to CPU
    cudaMemcpy(matrix, d_a, size, cudaMemcpyDeviceToHost);
    //
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
    return main_using_gpu();
}



