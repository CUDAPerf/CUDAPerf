#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <thrust/complex.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

using CoeffT = thrust::complex<float>;


//compute sum of coefficients per polynomial, use groupsize threads (1,2,4,8,16,or 32) per polynomial
template<int groupsize>
__global__
void coeffSumKernel3(CoeffT* __restrict__ output, const CoeffT* __restrict__ coeffs, int N, int maxDegree){
    auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
    const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;

    if(groupId < N){
        const CoeffT* myCoeffs = coeffs + groupId * maxDegree;
        CoeffT result = 0;
        const int numIters = (maxDegree + groupsize - 1) / groupsize;
        for(int iter = 0; iter < numIters; iter++){
            const int index = iter * groupsize + group.thread_rank();
            CoeffT val = index < maxDegree ? myCoeffs[index] : 0;
            result += cg::reduce(group, val, cg::plus<CoeffT>());
        }
        if(group.thread_rank() == 0){
            output[groupId] = result;
        }
    }
}



int test(){
    const int N = 1024*128;
    const int maxDegree = 10;
    const int maxCoeffs = N * maxDegree;

    //Approach 3, your data layout, use multiple threads per polynomial
    constexpr int numThreadsPerPoly3 = 8;
    std::vector<CoeffT> coeffs3(maxCoeffs);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < maxDegree; j++){
            coeffs3[i * maxDegree + j] = i;
        }
    }

    CoeffT* d_coeffs3; 
    CoeffT* d_result3;
    cudaMalloc(&d_coeffs3, sizeof(CoeffT) * maxCoeffs);
    cudaMalloc(&d_result3, sizeof(CoeffT) * N);
    cudaMemcpy(d_coeffs3, coeffs3.data(), sizeof(CoeffT) * maxCoeffs, cudaMemcpyHostToDevice);    

    coeffSumKernel3<numThreadsPerPoly3><<<((N * numThreadsPerPoly3) + 127)/128, 128>>>(d_result3, d_coeffs3, N, maxDegree);

    std::vector<CoeffT> result3(N);
    cudaMemcpy(result3.data(), d_result3, sizeof(CoeffT) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //assert(result3 == result1);

    cudaFree(d_coeffs3);
    cudaFree(d_result3);

    return 0;

}

int main(){
    //cudaSetDevice(1);
    for(int i=0;i<10;i++){
        test();
    }
}