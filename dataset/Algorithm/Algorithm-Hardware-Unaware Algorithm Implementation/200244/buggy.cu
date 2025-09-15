#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define TILE_SIZE 8

// Check CUDA errors
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Simple Matrix Multiplication Kernel
__global__
void matMultKer(float* Cd, float* Ad, float* Bd, int width) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < width && col < width) {
        float res = 0;
        for (int k = 0; k < width; ++k) {
            res += Ad[row * width + k] * Bd[k * width + col];
        }
        Cd[row * width + col] = res;
    }
}

// Host code to initialize matrices and compare kernel execution time
void initializeMatrix(float* matrix, int width) {
    for (int i = 0; i < width * width; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void compareKernels(int width) {
    // Host matrices
    float *A, *B, *C_basic;
    int size = width * width * sizeof(float);

    // Allocate host memory
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C_basic = (float*)malloc(size);

    // Initialize matrices
    initializeMatrix(A, width);
    initializeMatrix(B, width);

    // Device matrices
    float *Ad, *Bd, *Cd_basic;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&Ad, size));
    CUDA_CHECK(cudaMalloc(&Bd, size));
    CUDA_CHECK(cudaMalloc(&Cd_basic, size));

    // Copy matrices to device
    CUDA_CHECK(cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, (width + TILE_SIZE - 1) / TILE_SIZE);

    // Time the tiled kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Time the basic kernel
    CUDA_CHECK(cudaEventRecord(start));
    for(int i=0;i<10;i++){
        matMultKer<<<dimGrid, dimBlock>>>(Cd_basic, Ad, Bd, width);
    }
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float basicTime;
    CUDA_CHECK(cudaEventElapsedTime(&basicTime, start, stop));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(C_basic, Cd_basic, size, cudaMemcpyDeviceToHost));

    // Compare execution times
    std::cout << "Basic Kernel Time: " << basicTime << " ms\n";

    // Free memory
    free(A);
    free(B);
    free(C_basic);
    CUDA_CHECK(cudaFree(Ad));
    CUDA_CHECK(cudaFree(Bd));
    CUDA_CHECK(cudaFree(Cd_basic));
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    int width = 1024; // Set matrix size (width x width)
    compareKernels(width);

    return 0;
}