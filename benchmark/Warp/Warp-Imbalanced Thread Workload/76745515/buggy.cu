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

__global__ void dotProductKernel2(int *a, int *b, int *results, int n) {
    //Get the thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //Only calculate if thread index is valid
    if (tid < n) {
       int sum = 0;
       //Perform dot product calculation for this thread's segment of the arrays
       for (int i = tid; i < n; i+= blockDim.x * gridDim.x) {
           sum += a[i] * b[i];
       }
       //Store this thread's result in the shared results array
       results[blockIdx.x * blockDim.x + threadIdx.x] = sum;
    } }

__global__ void sumResultsKernel(int *results, int *result, int n) {
    //Get the thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //Only calculate if thread index is valid
    if (tid == 0) {
       int sum = 0;
       //Sum up the partial results
       for (int i = 0; i < n; i++) {
           sum += results[i];
       }
       //Store the final result
       *result = sum;
    } 
}

void test_2() {
    int n = N;

    // Use vectors instead of raw pointers
    std::vector<int> a(n), b(n);
    int *c = (int*) malloc(sizeof(int));

    // Initialize the input vectors
    for (int i = 0; i < n; i++) {
        a[i] = 1; b[i] = 2;
    }
    *c = 0;

    // Determine the grid size and block size
    int blockSize = 1024;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Allocate memory on the GPU
    int* d_a, *d_b, *d_results, *d_result;
    cudaMalloc(&d_a, a.size() * sizeof(int));
    cudaMalloc(&d_b, b.size() * sizeof(int));
    cudaMalloc(&d_results, gridSize * blockSize * sizeof(int));
    cudaMalloc(&d_result, 1 * sizeof(int));

    // Copy vectors to GPU
    cudaMemcpy(d_a, a.data(), a.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), b.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, gridSize * blockSize * sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));
    for(int i=0; i<10;i++){
      // Launch kernel
      dotProductKernel2<<<gridSize, blockSize>>>(d_a, d_b, d_results, n);
      // Sum up the partial results
      sumResultsKernel<<<1, 1024>>>(d_results, d_result, gridSize * blockSize);
      cudaDeviceSynchronize();
    }

    // Copy result back from GPU
    cudaMemcpy(c, d_result, 1 * sizeof(int),cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_results);
    cudaFree(d_result);

    free(c); 
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
  test_2();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << cudaGetErrorString(err) << std::endl;
}