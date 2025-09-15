#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <cmath>

__global__ void modulus_kernel1(int length, int lines, cufftComplex *PostFFTData, float* z) 
{
    //delete sync
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x<length*lines)
        z[x] = sqrt(PostFFTData[x].x *PostFFTData[x].x + PostFFTData[x].y *PostFFTData[x].y);
}


void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}


int main() {
    cudaSetDevice(1);
    int FFTlength = 1024; // Example FFT length
    int lines = 1024; // Example number of lines

    // Define block and grid sizes
    dim3 dimBlock(256);
    dim3 dimGrid((FFTlength * lines + dimBlock.x - 1) / dimBlock.x);

    // Allocate host memory
    cufftComplex* h_PostFFTData = new cufftComplex[FFTlength * lines];
    float* h_z = new float[FFTlength * lines];

    // Initialize host memory with some example values
    for (int i = 0; i < FFTlength * lines; ++i) {
        h_PostFFTData[i].x = static_cast<float>(i);
        h_PostFFTData[i].y = static_cast<float>(i) / 2.0f;
    }

    // Allocate device memory
    cufftComplex* d_PostFFTData;
    float* d_z;
    checkCudaError(cudaMalloc((void**)&d_PostFFTData, FFTlength * lines * sizeof(cufftComplex)), "cudaMalloc d_PostFFTData");
    checkCudaError(cudaMalloc((void**)&d_z, FFTlength * lines * sizeof(float)), "cudaMalloc d_z");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_PostFFTData, h_PostFFTData, FFTlength * lines * sizeof(cufftComplex), cudaMemcpyHostToDevice), "cudaMemcpy h_PostFFTData to d_PostFFTData");
    // Launch kernel
    for(int i=0;i<10;i++){
        modulus_kernel1<<<dimGrid, dimBlock>>>(FFTlength, lines, d_PostFFTData, d_z);
    }
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy results from device to host
    checkCudaError(cudaMemcpy(h_z, d_z, FFTlength * lines * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_z to h_z");


    // Clean up
    delete[] h_PostFFTData;
    delete[] h_z;
    checkCudaError(cudaFree(d_PostFFTData), "cudaFree d_PostFFTData");
    checkCudaError(cudaFree(d_z), "cudaFree d_z");

    return 0;
}


