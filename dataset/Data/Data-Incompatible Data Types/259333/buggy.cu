#include <cstdio>

int blockNum=1024;

__global__ void test_shfl2(float * A){
    float x = threadIdx.x ;
    float sum = 0;
    // computation
    for(int i = 0; i < 10; ++i){
    x = __cosf(x);
    x = __cosf(1 - x);
    x = __cosf(1 - x);
    x = __cosf(1 - x);
    x = __cosf(1 - x);
    x = __cosf(1 - x);
    x = __cosf(1 - x);
    x = __cosf(1 - x);
    x = __cosf(1 - x);
    x = __cosf(1 - x);
    x = __cosf(1 - x);
    }
#define MY_CONST2 0.001
    sum =  MY_CONST2 * __shfl_xor_sync(0xffffffff, x, 16, 32);
    sum += MY_CONST2 * __shfl_xor_sync(0xffffffff, sum, 8, 32);
    sum += MY_CONST2 * __shfl_xor_sync(0xffffffff, sum, 4, 32);
    sum += MY_CONST2 * __shfl_xor_sync(0xffffffff, sum, 2, 32);
    sum += MY_CONST2 * __shfl_xor_sync(0xffffffff, sum, 1, 32);
    // Memory
    atomicAdd(A, sum);
}

int test2(){
    cudaEvent_t fft_begin, fft_end;
    float elapsed_time;
    float *dA, *A;
    A = (float*)malloc(sizeof(float));
    cudaEventCreate(&fft_begin);
    cudaEventCreate(&fft_end);
    cudaMalloc((void**) &dA, sizeof(float) * 1);
    cudaMemset(dA, 0, sizeof(float));
    cudaEventRecord(fft_begin);

    for(int i = 0; i < 10; ++i){
        test_shfl2 <<<blockNum, 1024>>>(dA);
    }
    cudaEventRecord(fft_end);
    cudaEventSynchronize(fft_begin);
    cudaEventSynchronize(fft_end);
    cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);

    cudaMemcpy((void*)A, (void*)dA, sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f, %f\n", elapsed_time, *A);

    return 0;
}

int main(){
    test2();
    return 0;
}