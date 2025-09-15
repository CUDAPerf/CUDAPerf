#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
 
#include "device_launch_parameters.h"
 
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
 
#include <random>
#include <iostream>
#include <chrono>
 
# define CUDA_CHECK {\
    cudaError_t  cu_error = cudaGetLastError();                                 \
    if (cu_error != cudaSuccess) {                                              \
      std::cout << "Cuda error: " << cudaGetErrorString(cu_error) << std::endl; \
    }                                                                           \
  }
 
struct custom_functor{
    float factor;
    custom_functor(float _factor){
      factor = _factor;
    }
    __host__ __device__ int operator()(float &x) const {
        return (int) floor(x / factor);
    }
};
 
 
__global__ void custom_reduce_kernel(float *d_x, float *d_a, float *d_temp_a, int N, int Na, float da)
{
// Get our global thread ID
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
 
  float ix ;
 
  // Compute a
  for(int x = index; x < N; x += stride) {
      ix = floor(d_x[x] / da);
 
      d_temp_a[((int)ix) + Na * index] += 0.5;
      d_temp_a[((int)ix + 1) + Na * index] += 0.5;
  }
  __syncthreads();
 
 
  // Reduce
  for(int l = index; l < Na; l += stride) {
      for(int m = 0; m < stride; m += 1) {
          d_a[l] += d_temp_a[l + Na * m];
      }
  }
  __syncthreads();
}
 
void test_custom_reduce(float *d_x, float *d_a, float *h_a, int N, int Na, float da)
{
  int blockSize = 512; // Number of threads in each thread block
  int gridSize = (int)ceil((float) N /blockSize); // Number of thread blocks in grid
 
  // Create temp d_a array
  float *d_temp_a;
  cudaMalloc((void **) &d_temp_a, Na * blockSize * gridSize * sizeof(float));
  CUDA_CHECK;
 
  custom_reduce_kernel<<<gridSize,blockSize>>>(d_x, d_a, d_temp_a, N, Na, da);
  cudaMemcpy(h_a, d_a, Na * sizeof(float), cudaMemcpyDeviceToHost);
 
  cudaFree(d_temp_a);
}

 
void reference(float* h_x, float* h_a, int N, float da)
{
  for(int j = 0; j < N; j++) {
    float i = floor(h_x[j] / da); // in principle i < size(a)
 
    h_a[(int)i] += 0.5;
    h_a[(int)i+1] += 0.5; // I simplify the problem
  }
}
 
int main()
{
  float da = 0.1f;
  int N = 1024;   
  int Na = 1024;   
 
  float L = 50; // box size
 
  float *h_x = (float *)malloc(N * sizeof(float));
 
  float *h_a1 = (float *)malloc(Na * sizeof(float));
  float *h_a_reference = (float *)malloc(Na * sizeof(float));
 
  /* Initialize random seed: */
  std::default_random_engine generator;
  std::uniform_real_distribution<float> generate_unif_dist(0.0,1.0);
 
  // h_x random initialisation
  for (int x = 0; x < N; x++) {
      float random = generate_unif_dist(generator);
      h_x[x] = random * L;
  }
 
  float *d_x, *d_a; 
  cudaMalloc((void**) &d_x, N * sizeof(float));
  cudaMalloc((void**) &d_a, Na * sizeof(float));
 
  cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
 
 
  std::chrono::steady_clock::time_point current_bi = std::chrono::steady_clock::now();
  reference(h_x, h_a_reference, N, da);
  std::chrono::steady_clock::time_point current_ai = std::chrono::steady_clock::now();
  float time_ref = std::chrono::duration_cast<std::chrono::nanoseconds> (current_ai - current_bi).count();
 
 
  current_bi = std::chrono::steady_clock::now();
  test_custom_reduce(d_x, d_a, h_a1, N, Na, da);
  current_ai = std::chrono::steady_clock::now();
  float time1 = std::chrono::duration_cast<std::chrono::nanoseconds> (current_ai - current_bi).count();

 
  for (int i = 0; i < Na; i++)
  {
    if (fabs(h_a_reference[i]-h_a1[i]) > 0.0001)
      std::cout << "Error 1: " << i << " - " << h_a_reference[i] << " != " << h_a1[i] << std::endl;
  }
 
  std::cout << "Times: " << std::endl;
  std::cout << "- CPU Reference:         " << time_ref << " ms" << std::endl;
  std::cout << "- CUDA Custom reduce:    " << time1 << " ms" << std::endl;
 
  free(h_x);
  free(h_a1);
 
  cudaFree(d_x);
  cudaFree(d_a);
 
  return 0;
}
