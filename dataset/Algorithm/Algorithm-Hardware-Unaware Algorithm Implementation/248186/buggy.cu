#include <cstdlib>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>

const int N = 1024;

__global__ void forward_pass(double* w, double* b){
    int idx = threadIdx.x;
    __shared__ double w_buffer[N];
    __shared__ double b_buffer[N];
    w_buffer[idx] = w[idx];
    b_buffer[idx] = b[idx];
    __syncthreads();
    if (idx == 0) {
        for (int i = 1; i < N; i++) {
            b_buffer[i] = b_buffer[i] - b_buffer[i - 1] * w_buffer[i];
        }
    }
    __syncthreads();
    b[idx] = b_buffer[idx];
}


using mt = double;
using namespace thrust::placeholders;

int main() {
    //cudaSetDevice(1);
    mt *h_w, *d_w, *h_b, *d_b, *h_r;
    h_w = new mt[N];
    h_r = new mt[N];
    h_b = new mt[N];

    cudaMalloc(&d_b, N * sizeof(d_b[0]));
    cudaMalloc(&d_w, N * sizeof(d_w[0]));
    for (int i = 0; i < N; i++) {
        h_w[i] = rand() / (double)RAND_MAX;
        h_b[i] = rand() / (double)RAND_MAX;
    }
    cudaMemcpy(d_b, h_b, N * sizeof(d_b[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, N * sizeof(d_w[0]), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    forward_pass<<<1, N>>>(d_w, d_b);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_r, d_b, N * sizeof(d_b[0]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; i++) std::cout << h_r[i] << ",";
    std::cout << std::endl;

    // free
    cudaFree(d_b);
    cudaFree(d_w);
    delete[] h_w;
    delete[] h_b;
    delete[] h_r;

    return 0;
}