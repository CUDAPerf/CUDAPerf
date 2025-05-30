//nvcc -arch=sm_61 -lineinfo -g -O3 -std=c++17
#include <cassert>
#include <chrono>
#include <iostream>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <thrust/fill.h>
#include <thrust/execution_policy.h>

namespace cg = cooperative_groups;


__global__
void printkernel(float* C1, float* C2){
    for(int i = 0; i < 10; i++){
        printf("%f ", C1[i]);
    }
    printf("\n");
    for(int i = 0; i < 10; i++){
        printf("%f ", C2[i]);
    }
    printf("\n");
}


template<int M>
__device__
float f(const float* x){
    float result = 0;
    for(int i = 0; i < M; i++){
        result += expf(sin(x[i]));
    }
    return result;
}

template<int M, class Group>
__device__
float f(Group group, const float* x){
    float result = 0;
    for(int i = group.thread_rank(); i < M; i += group.size()){
        result += expf(sin(x[i]));
    }
    result = cg::reduce(group, result, cg::plus<float>{});
    return result;
}


template<int M>
__global__
void kernel1(const float* x_head, float* output){
    __shared__ float x_cache[32][M];
    constexpr auto WORK_PER_THREAD = (M + 31) / 32;
    for (int i = 0;i<32;i++)
    {
        const float* x = x_head + M * (blockIdx.x * blockDim.x + i);
        for(int j = 0;j<WORK_PER_THREAD;j++)
        {
            x_cache[i][j * 32 + threadIdx.x] = x[j * 32 + threadIdx.x];
        }
    }
    __syncthreads();
    
    float y = f<M>(&x_cache[threadIdx.x][0]);
    output[blockIdx.x * blockDim.x + threadIdx.x] = y;
}


int main(){
    cudaSetDevice(1);
    constexpr size_t numColumns = 128;
    size_t numRows = 1024*128;

    float* A = nullptr; 
    float* C1 = nullptr;
    float* C2 = nullptr;
    cudaMalloc(&A, sizeof(float) * numColumns * numRows);
    cudaMalloc(&C1, sizeof(float) * numRows);
    cudaMalloc(&C2, sizeof(float) * numRows);

    thrust::fill(
        thrust::device,
        A,
        A + numColumns * numRows,
        0.01f
    );

    int deviceId = 0;
    int numSMs = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);

    //32 threads per block. 1 thread per x
    dim3 block1(32);
    dim3 grid1((numRows + 31) / 32);
    auto t1 = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++){
        kernel1<numColumns><<<grid1, block1>>>(A, C1);
        cudaDeviceSynchronize();
    }
    auto t2 = std::chrono::system_clock::now();
    std::cout << "Timing 1 thread per x: " << std::chrono::duration<double>(t2 - t1).count() << "s\n";

    printkernel<<<1,1>>>(C1, C2);
    cudaDeviceSynchronize();
}