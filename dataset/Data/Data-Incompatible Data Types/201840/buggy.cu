#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <thrust/complex.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

using CoeffT0 = thrust::complex<double>;


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

    return 0;

}

int main(){
    //cudaSetDevice(1);
    for(int i=0;i<10;i++){
        test();
    }
}