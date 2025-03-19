#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <thrust/complex.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

using CoeffT0 = thrust::complex<double>;

using CoeffT = thrust::complex<float>;


//compute sum of coefficients per polynomial， use double

__global__
void coeffSumKernel0(CoeffT0* __restrict__ output, const CoeffT0* __restrict__ coeffs, int N, int maxDegree){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N){
        const CoeffT0* myCoeffs = coeffs + tid * maxDegree;
        CoeffT0 result = 0;
        for(int i = 0; i < maxDegree; i++){
            result += myCoeffs[i];
        }
        output[tid] = result;
    }
}


//compute sum of coefficients per polynomial
__global__
void coeffSumKernel(CoeffT* __restrict__ output, const CoeffT* __restrict__ coeffs, int N, int maxDegree){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N){
        const CoeffT* myCoeffs = coeffs + tid * maxDegree;
        CoeffT result = 0;
        for(int i = 0; i < maxDegree; i++){
            result += myCoeffs[i];
        }
        output[tid] = result;
    }
}


//compute sum of coefficients per polynomial, coeffs are transposed
__global__
void coeffSumKernel2(CoeffT* __restrict__ output, const CoeffT* __restrict__ coeffs, int N, int maxDegree){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N){
        const CoeffT* myCoeffs = coeffs + tid;
        CoeffT result = 0;
        for(int i = 0; i < maxDegree; i++){
            result += myCoeffs[i * N];
        }
        output[tid] = result;
    }
}


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

    //Approach 0, this is your current data layout
    
    std::vector<CoeffT0> coeffs0(maxCoeffs);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < maxDegree; j++){
            coeffs0[i * maxDegree + j] = i;
        }
    }

    CoeffT0* d_coeffs0; 
    CoeffT0* d_result0;
    cudaMalloc(&d_coeffs0, sizeof(CoeffT0) * maxCoeffs);
    cudaMalloc(&d_result0, sizeof(CoeffT0) * N);
    cudaMemcpy(d_coeffs0, coeffs0.data(), sizeof(CoeffT0) * maxCoeffs, cudaMemcpyHostToDevice);

    coeffSumKernel0<<<(maxCoeffs + 127)/128, 128>>>(d_result0, d_coeffs0, N, maxDegree);

    std::vector<CoeffT0> result0(N);
    cudaMemcpy(result0.data(), d_result0, sizeof(CoeffT0) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_coeffs0);
    cudaFree(d_result0);

    //Approach 1, this is your current data layout
    std::vector<CoeffT> coeffs1(maxCoeffs);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < maxDegree; j++){
            coeffs1[i * maxDegree + j] = i;
        }
    }

    CoeffT* d_coeffs1; 
    CoeffT* d_result1;
    cudaMalloc(&d_coeffs1, sizeof(CoeffT) * maxCoeffs);
    cudaMalloc(&d_result1, sizeof(CoeffT) * N);
    cudaMemcpy(d_coeffs1, coeffs1.data(), sizeof(CoeffT) * maxCoeffs, cudaMemcpyHostToDevice);

    coeffSumKernel<<<(maxCoeffs + 127)/128, 128>>>(d_result1, d_coeffs1, N, maxDegree);

    std::vector<CoeffT> result1(N);
    cudaMemcpy(result1.data(), d_result1, sizeof(CoeffT) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_coeffs1);
    cudaFree(d_result1);


    //Approach 2, transpose the coefficient matrix. the coefficients of polynomial i are stored in the i-th column
    std::vector<CoeffT> coeffs2(maxCoeffs);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < maxDegree; j++){
            coeffs2[i + j * N] = i;
        }
    }

    CoeffT* d_coeffs2; 
    CoeffT* d_result2;
    cudaMalloc(&d_coeffs2, sizeof(CoeffT) * maxCoeffs);
    cudaMalloc(&d_result2, sizeof(CoeffT) * N);
    cudaMemcpy(d_coeffs2, coeffs2.data(), sizeof(CoeffT) * maxCoeffs, cudaMemcpyHostToDevice);

    coeffSumKernel2<<<(maxCoeffs + 127)/128, 128>>>(d_result2, d_coeffs2, N, maxDegree);

    std::vector<CoeffT> result2(N);
    cudaMemcpy(result2.data(), d_result2, sizeof(CoeffT) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //assert(result2 == result1);
    cudaFree(d_coeffs2);
    cudaFree(d_result2);


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