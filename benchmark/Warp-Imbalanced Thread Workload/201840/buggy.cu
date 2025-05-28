#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <thrust/complex.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;


using CoeffT = thrust::complex<float>;

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



int test(){
    const int N = 1024*128;
    const int maxDegree = 10;
    const int maxCoeffs = N * maxDegree;


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

    return 0;

}

int main(){
    //cudaSetDevice(1);
    for(int i=0;i<10;i++){
        test();
    }
}