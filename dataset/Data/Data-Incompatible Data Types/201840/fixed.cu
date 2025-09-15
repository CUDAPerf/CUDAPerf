#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <thrust/complex.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

using CoeffT = thrust::complex<float>;


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



int test(){
    const int N = 1024*128;
    const int maxDegree = 10;
    const int maxCoeffs = N * maxDegree;

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

    return 0;

}

int main(){
    //cudaSetDevice(1);
    for(int i=0;i<10;i++){
        test();
    }
}