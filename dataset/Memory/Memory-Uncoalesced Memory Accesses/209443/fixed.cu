#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>    
#include <chrono>
#include <algorithm>
#include <iostream>

#define N 8
#define MAX 1024*1024*512

__global__ void kernel2(int*x){
    #pragma unroll
    for(auto i = 0; i < N; i++) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int j = i * gridDim.x * blockDim.x + idx; 
        if(j > MAX) break;
        x[j] = j; /* do something with x[j] */
    }
}


int main() {
    cudaSetDevice(1);
    int Q = MAX / (N * 512);
    int *d_x2;
    int *h_x2 = new int[MAX];

    cudaMalloc(&d_x2, sizeof(int) * MAX);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run kernel2
    cudaEventRecord(start);
    for(int i=0;i<10;i++){
        kernel2<<<Q, 512>>>(d_x2);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time2;
    cudaEventElapsedTime(&time2, start, stop);
    std::cout << "Kernel2 time: " << time2 << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_x2, d_x2, sizeof(int) * MAX, cudaMemcpyDeviceToHost);

    // Clean up
    delete[] h_x2;
    cudaFree(d_x2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}