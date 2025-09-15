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
 
__global__ void simple_atomicAdd_kernel(const float *d_x, float *d_a, float da, int N, int Na)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
 
  if (index >= N)
    return;
 
  int a_idx = floor(d_x[index] / da); // in principle i < size(a)
 
  atomicAdd(&d_a[a_idx], 0.5f);
  atomicAdd(&d_a[a_idx+1], 0.5f);
} 
void test_simple_atomicAdd(float *d_x, float *d_a, float *h_a, int N, int Na, float da)
{
  cudaMemset(d_a, 0, Na * sizeof(float));
 
  dim3 dimBlock(256);
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);
  simple_atomicAdd_kernel<<<dimGrid, dimBlock>>>(d_x, d_a, da, N, Na);
  cudaMemcpy(h_a, d_a, Na * sizeof(float), cudaMemcpyDeviceToHost);
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
 
  float *h_a3 = (float *)malloc(Na * sizeof(float));
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
  test_simple_atomicAdd(d_x, d_a, h_a3, N, Na, da);
  current_ai = std::chrono::steady_clock::now();
  float time3 = std::chrono::duration_cast<std::chrono::nanoseconds> (current_ai - current_bi).count();
 
  for (int i = 0; i < Na; i++)
  { 
    if (fabs(h_a_reference[i]-h_a3[i]) > 0.0001)
      std::cout << "Error 3: " << i << " - " << h_a_reference[i] << " != " << h_a3[i] << std::endl;
  }
 
  std::cout << "Times: " << std::endl;
  std::cout << "- CPU Reference:         " << time_ref << " ms" << std::endl;
  std::cout << "- CUDA Simple atomicAdd: " << time3 << " ms" << std::endl;
 
  free(h_x);
  free(h_a3);
 
  cudaFree(d_x);
  cudaFree(d_a);
 
  return 0;
}
