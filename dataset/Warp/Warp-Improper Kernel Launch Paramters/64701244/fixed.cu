#include <iostream>
#include <cuda_runtime.h>

__global__ void reduceBlksKernel2(int *in, int n, int *out) {
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;

    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            if (i + stride < n) {
                in[i] += in[i + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = in[blockIdx.x * blockDim.x * 2];
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


void testKernel2(int blockSize, int gridSize, int n) {
    int *d_in, *d_out;
    int *h_in = new int[n];
    int *h_out = new int[gridSize];
    
    
    for (int i = 0; i < n; ++i) {
        h_in[i] = 1;
    }
    
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, gridSize * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice));
    
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    for(int i=0;i<10;i++){
        reduceBlksKernel2<<<gridSize, blockSize>>>(d_in, n, d_out);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
    }
    
    
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    
    CUDA_CHECK(cudaMemcpy(h_out, d_out, gridSize * sizeof(int), cudaMemcpyDeviceToHost));

    
    std::cout << "Block Size: " << blockSize << ", Grid Size: " << gridSize << ", Execution Time: " << milliseconds << " ms" << std::endl;

    
    delete[] h_in;
    delete[] h_out;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}


int main() {
    int n = 1024 * 128 * 2; 

    testKernel2(256, n/256/2, n);

    return 0;
}