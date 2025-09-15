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


template<int M, int blocksize, int groupsize>
__global__
void kernel4(const float* x_head, float* output, int numX){
    constexpr int numGroupsPerBlock = blocksize / groupsize;

    auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
    const int numGroupsInGrid = (blockDim.x * gridDim.x) / groupsize;
    const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;

    
    //only need 1 x array smem per group
    __shared__ float x_cache[numGroupsPerBlock][M];
    
    //each group processes processes 1 x / each block processes numGroupsPerBlock x
    for(int xIndex = groupIdInGrid; xIndex < numX; xIndex += numGroupsInGrid){
        //load x vector of group to smem
        const float* x = x_head + M * xIndex;
        for(int j = group.thread_rank(); j < M; j += group.size()){
            x_cache[group.meta_group_rank()][j] = x[j];
        }
        //wait for shared memory
        group.sync();

        //process x vector with group
        float y = f<M>(group, &x_cache[group.meta_group_rank()][0]);
        if(group.thread_rank() == 0){
            output[xIndex] = y;
        }
        //wait untill shared memory can be reused
        group.sync();
    }
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

    //128 threads per block, 1 group per x
    constexpr int blocksize4 = 32;
    constexpr int groupsize4 = 32;
    constexpr int xPerBlock4 = blocksize4 / groupsize4;

    int maxBlocksPerSM4 = 0;   
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM4,
        kernel4<numColumns, blocksize4, groupsize4>,
        blocksize4, 
        0
    );
    dim3 block4(blocksize4);
    dim3 grid4(std::min(size_t(maxBlocksPerSM4) * numSMs, (numRows + xPerBlock4 - 1) / xPerBlock4));
    auto t3 = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++){
        kernel4<numColumns, blocksize4, groupsize4><<<grid4, block4>>>(A, C2, numRows);
        cudaDeviceSynchronize();
    }
    auto t4 = std::chrono::system_clock::now();
    std::cout << "Timing 32 threads per x: " << std::chrono::duration<double>(t4 - t3).count() << "s\n";

    printkernel<<<1,1>>>(C1, C2);
    cudaDeviceSynchronize();
}