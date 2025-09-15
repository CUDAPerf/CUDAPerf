#include <iostream>
#include <cuda_runtime.h>

#define N 1024*8

__global__ void kernel2(float* ma, float* mb, float* mc)
{
    uint32_t const row{blockIdx.x * blockDim.x + threadIdx.x};
    uint32_t const col{blockIdx.y * blockDim.y + threadIdx.y};

    uint32_t offset{N * row};
    float result{0.0f};

    for (uint32_t s{0}; s < N; ++s)
    {
        result += ma[offset + s] * mb[col + s * N];
    }

    mc[offset + col] = result;
}


__global__ void transpose(float* src, float* dest)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N)
    {
        dest[y * N + x] = src[x * N + y];
    }
}


int main()
{
    cudaSetDevice(1);
    
    float *h_a = new float[N * N];
    float *h_b = new float[N * N];
    float *h_c2 = new float[N * N];

    for (int i = 0; i < N * N; ++i)
    {
        h_a[i] = 1.0f; 
        h_b[i] = 2.0f; 
    }

    
    float *d_a, *d_b, *d_c2, *d_b_transpose, *d_a_transpose;
    cudaMalloc(&d_a, N * N * sizeof(float));
    cudaMalloc(&d_b, N * N * sizeof(float));
    cudaMalloc(&d_c2, N * N * sizeof(float));
    cudaMalloc(&d_b_transpose, N * N * sizeof(float));
    cudaMalloc(&d_a_transpose, N * N * sizeof(float));

    
    cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    transpose<<<numBlocks, threadsPerBlock>>>(d_b, d_b_transpose);
    transpose<<<numBlocks, threadsPerBlock>>>(d_a, d_a_transpose);
    cudaDeviceSynchronize();

    cudaEventRecord(start); 
    for(int i=0;i<10;i++){
        kernel2<<<numBlocks, threadsPerBlock>>>(d_a_transpose, d_b_transpose, d_c2);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2, start, stop); 
    std::cout << "Kernel2 execution time: " << elapsedTime2 << " ms\n";

    
    cudaMemcpy(h_c2, d_c2, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c2);
    cudaFree(d_b_transpose);
    cudaFree(d_a_transpose);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c2;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
