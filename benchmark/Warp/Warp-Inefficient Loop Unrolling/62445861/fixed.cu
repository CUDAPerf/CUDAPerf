#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define SIZE 512


__global__ void copy_unroll(float *input, float *output) {
    int tidx = blockIdx.y * blockDim.x + threadIdx.x;
    int stride = 1024 * 1024;
    for (int i = 0; i < SIZE; i++) {
        int idx = i * stride + tidx;
        float x = input[idx];
        float y = 0;

        #pragma unroll
        for (int j = 0; j < 50; j += 10) {
            x = x + sqrtf(float(j));
            y = sqrtf(fabsf(x)) + sinf(x) + cosf(x);

            x = x + sqrtf(float(j+1));
            y = sqrtf(fabsf(x)) + sinf(x) + cosf(x);

            x = x + sqrtf(float(j+2));
            y = sqrtf(fabsf(x)) + sinf(x) + cosf(x);

            x = x + sqrtf(float(j+3));
            y = sqrtf(fabsf(x)) + sinf(x) + cosf(x);

            x = x + sqrtf(float(j+4));
            y = sqrtf(fabsf(x)) + sinf(x) + cosf(x);

            x = x + sqrtf(float(j+5));
            y = sqrtf(fabsf(x)) + sinf(x) + cosf(x);

            x = x + sqrtf(float(j+6));
            y = sqrtf(fabsf(x)) + sinf(x) + cosf(x);

            x = x + sqrtf(float(j+7));
            y = sqrtf(fabsf(x)) + sinf(x) + cosf(x);

            x = x + sqrtf(float(j+8));
            y = sqrtf(fabsf(x)) + sinf(x) + cosf(x);

            x = x + sqrtf(float(j+9));
            y = sqrtf(fabsf(x)) + sinf(x) + cosf(x);
        }

        output[idx] = y;
    }
}


int main() {
    const int size = 1024 * 1024 * SIZE; 

    
    float *h_input = new float[size];
    float *h_output = new float[size];

    
    for (int i = 0; i < size; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(1024);
    dim3 gridSize(1, 1024);

    
    for(int i=0; i<10; i++){
        copy_unroll<<<gridSize, blockSize>>>(d_input, d_output);
    }
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_output[" << i << "] = " << h_output[i] << std::endl;
    }

    
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}