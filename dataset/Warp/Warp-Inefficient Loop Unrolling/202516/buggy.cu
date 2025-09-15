#include <cstdlib>
#include <assert.h>
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>


// compile using
// /usr/local/cuda/bin/nvcc -g -Xcompiler -O3 -gencode arch=compute_86,code=sm_86 -use_fast_math test_cuda_kernel.cu -o test_cuda_kernel

__device__ __forceinline__ long fmax3(long a, long b, long c)
{
    long r = a;
    if (b > r)
        r = b;
    if (c > r)
        r = c;
    return r;
}

#define max2(a,b) ((a)>(b))? (a):(b)
#define max3(a,b,c) max2((c), max2((b), (a)))

__global__  void test(long* pin, long* pout, long n)
{
   long gid = blockIdx.x * blockDim.x + threadIdx.x;

   long sum = 0;
   for (long i = 0; i < n; i++)
   {
       long idx = gid - n;
       long idx2 = idx +1;
       if (idx > 0 && idx2 < gid)
           sum = fmax3(sum, pin[idx], pin[idx2]);
   }
   pout[gid] = sum;
}


struct timeval tnow;

double dtime(){
    gettimeofday(&tnow, NULL);
    return (double)tnow.tv_sec + (double)tnow.tv_usec * 1.0e-6;}

int main(int argc, char** argv)
{
    //cudaSetDevice(1);
    size_t blockSize = 1024;
    size_t gridSize = 1024;
    size_t size = gridSize * blockSize;  
    long* pin;
    long* pout;
    cudaMalloc(&pin,  size * sizeof(long));
    cudaMalloc(&pout, size * sizeof(long));
    long n = 1000;
    //double t0 = dtime();
    for(int i=0; i<10; i++){
        test<<<gridSize, blockSize>>>(pin, pout, n);
        cudaDeviceSynchronize();
    }
    //double tf = dtime();
    //printf("n: %ld Ellapsed: %f\n", n, (tf-t0));

    return 0;
}