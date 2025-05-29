#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__device__ void gpu_fma_reduction(double* value, double u, double p) {
    *value = fmod(*value * u, p);
}


__global__ void gpu_matrix_fma_reduction_original(double *matrix, int rows, int cols, double u, double p) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols; j += blockDim.y * gridDim.y) {
            gpu_fma_reduction(matrix + i * cols + j, u, p);
        }
    }
}

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    cudaSetDevice(1);
    int rows = 1024*8*2;  
    int cols = 1024*8;  
    double u = 1.0 / 10007.0; 
    double p = 10007.0;        

    int matrix_size = rows * cols * sizeof(double);

    
    double *h_matrix = (double*)malloc(matrix_size);

    
    for (int i = 0; i < rows * cols; ++i) {
        h_matrix[i] = static_cast<double>(rand()) / RAND_MAX * 10000;
    }

    
    double *d_matrix;
    checkCudaError(cudaMalloc(&d_matrix, matrix_size), "cudaMalloc failed");

    
    checkCudaError(cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice), "cudaMemcpy failed");

    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10; i++)
        gpu_matrix_fma_reduction_original<<<numBlocks, threadsPerBlock>>>(d_matrix, rows, cols, u, p);
    checkCudaError(cudaDeviceSynchronize(), "Kernel launch failed for original");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_original = end - start;

    
    checkCudaError(cudaMemcpy(h_matrix, d_matrix, matrix_size, cudaMemcpyDeviceToHost), "cudaMemcpy failed");
    
    std::cout << "Original kernel time: " << elapsed_original.count() << " seconds" << std::endl;

    
    cudaFree(d_matrix);
    free(h_matrix);

    return 0;
}