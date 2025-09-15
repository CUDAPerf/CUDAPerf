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

__global__ void kernel1(int *x){
    int idx = blockDim.x* blockIdx.x + threadIdx.x;
    #pragma unroll
    for(auto i =0; i<N; i++){
        int j = idx*N + i;
        if(j>MAX) break;
        x[j] = j; /* do something with x[j] */
    }
}


int main() {
    cudaSetDevice(1);
    int Q = MAX / (N * 512);
    int *d_x1;
    int *h_x1 = new int[MAX];

    cudaMalloc(&d_x1, sizeof(int) * MAX);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run kernel1
    cudaEventRecord(start);
    for(int i=0;i<10;i++){
        kernel1<<<Q, 512>>>(d_x1);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time1;
    cudaEventElapsedTime(&time1, start, stop);
    std::cout << "Kernel1 time: " << time1 << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_x1, d_x1, sizeof(int) * MAX, cudaMemcpyDeviceToHost);


    // Clean up
    delete[] h_x1;
    cudaFree(d_x1);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}