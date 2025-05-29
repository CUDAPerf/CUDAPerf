#include <iostream>
#include <cuda.h>

#define N (1<<29) // 1M elements
#define BLOCK_SIZE 512

__device__ void warpReduce_origin(volatile int* sdata, int tid) {
sdata[tid] += sdata[tid + 32];
sdata[tid] += sdata[tid + 16];
sdata[tid] += sdata[tid + 8];
sdata[tid] += sdata[tid + 4];
sdata[tid] += sdata[tid + 2];
sdata[tid] += sdata[tid + 1];
}

__global__ void originalReduce(int *input, int *output) {
    __shared__ int sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    sdata[tid] = input[i] + input[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce_origin(sdata, tid);
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

__inline__ __device__ int warpReduce_optimize(int localSum){
    localSum += __shfl_xor_sync(0xFFFFFFFF,localSum, 16);
    localSum += __shfl_xor_sync(0xFFFFFFFF,localSum, 8);
    localSum += __shfl_xor_sync(0xFFFFFFFF,localSum, 4);
    localSum += __shfl_xor_sync(0xFFFFFFFF,localSum, 2);
    localSum += __shfl_xor_sync(0xFFFFFFFF,localSum, 1);
    return localSum;
}

__global__ void optimizedReduce(int *input, int *output) {
    __shared__ int sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    sdata[tid] = input[i] + input[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    int val = sdata[tid];
    if (tid < 32) {
        val = warpReduce_optimize(val);
    }

    if (tid == 0) output[blockIdx.x] = val;
}

int test() {
    int *h_input, *h_output;
    int *d_input, *d_output;
    size_t size = N * sizeof(int);

    // Allocate host memory
    h_input = (int*)malloc(size);
    h_output = (int*)malloc(BLOCK_SIZE * sizeof(int));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % 100;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, BLOCK_SIZE * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(N / block.x / 2, 1, 1);

    // Original reduce
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    originalReduce<<<grid, block>>>(d_input, d_output);
    cudaEventRecord(stop);

    cudaMemcpy(h_output, d_output, BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Original Reduce Time: " << milliseconds << " ms" << std::endl;

    // Optimized reduce
    cudaEventRecord(start);
    optimizedReduce<<<grid, block>>>(d_input, d_output);
    cudaEventRecord(stop);

    cudaMemcpy(h_output, d_output, BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Optimized Reduce Time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

int main(){
    cudaSetDevice(1);
    for (int i=0;i<10;i++){
        test();
    }
    return 0;
}
