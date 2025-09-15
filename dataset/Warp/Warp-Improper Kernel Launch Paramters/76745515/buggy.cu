#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#define USECPSEC 1000000ULL
#define N 1024*1024

unsigned long long dtime_usec(unsigned long long start=0){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

__global__ void warmup_kernel(int *a, int *b, int *result, int n) {
    //Get the thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //Only calculate if thread index is valid
    if (tid < n) {
       int sum = 0;
       //Perform dot product calculation for this thread's segment of the arrays
       for (int i = tid; i < n; i+= blockDim.x * gridDim.x) {
           sum += a[i] * b[i];
       }
       //Atomically add this thread's result to the shared result
       atomicAdd(result, sum);
    }
}

__global__ void dotProductKernel1(int *a, int *b, int *result, int n)
{
    //Get the thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //Only calculate if thread index is valid
    if (tid < n) {
       int sum = 0;
       //Perform dot product calculation for this thread's segment of the arrays
       for (int i = tid; i < n; i+= blockDim.x * gridDim.x) {
           sum += a[i] * b[i];
       }
       //Atomically add this thread's result to the shared result
       atomicAdd(result, sum);
    }
}


int main(){
  //cudaSetDevice(1);
  const int sz = N;
  int *a, *b, *c;
  cudaMallocManaged(&a, sz*sizeof(a[0]));
  cudaMallocManaged(&b, sz*sizeof(b[0]));
  cudaMallocManaged(&c, sizeof(c[0]));
  for (int i = 0; i < sz; i++) {a[i] = 1; b[i] = 2;}
  c[0] = 0;
  cudaMemPrefetchAsync(a, sz*sizeof(a[0]), 0);
  cudaMemPrefetchAsync(b, sz*sizeof(b[0]), 0);
  cudaMemPrefetchAsync(c, sizeof(c[0]), 0);
  // warm-up
  for(int i=0;i<3;i++){
    warmup_kernel<<<1, 1024>>>(a, b, c, sz);
    cudaDeviceSynchronize();
  }
  for(int i=0;i<10;i++){
    dotProductKernel1<<<1, 1024>>>(a, b, c, sz);
    cudaDeviceSynchronize();
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << cudaGetErrorString(err) << std::endl;
}