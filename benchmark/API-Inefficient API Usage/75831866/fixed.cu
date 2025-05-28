#include <cuda_runtime.h>
#include <string>
#include <chrono>
#include <random>
#include <iostream>
using namespace std;

  int N = 1024*1024;

class MyTimer {
    std::chrono::time_point<std::chrono::system_clock> start;

public:
    void startCounter() {
        start = std::chrono::system_clock::now();
    }

    int64_t getCounterNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - start).count();
    }

    int64_t getCounterMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count();
    }

    double getCounterMsPrecise() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - start).count()
                / 1000000.0;
    }
};

__global__
void HelloWorld()
{
  printf("Hello world\n");
}

volatile double dummy = 0;

__global__
void multiply1(int N, float* __restrict__ output, const float* __restrict__ x, const float* __restrict__ y)
{
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = start; i < N; i += stride) {
    output[i] = x[i] * y[i];
  }
}


int test1()
{
  MyTimer timer;
  srand(time(NULL));
  HelloWorld<<<1,1>>>();
  timer.startCounter();
  float *d_a, *d_b, *d_c;
#ifdef USE_MANAGED
  cudaMallocManaged(&d_a, N * sizeof(float));
  cudaMallocManaged(&d_b, N * sizeof(float));
  cudaMallocManaged(&d_c, N * sizeof(float));
  for (int i = 0; i < N; i++) {
    d_a[i] = float(rand() % 1000000) / (rand() % 1000 + 1);
    d_b[i] = float(rand() % 1000000) / (rand() % 1000 + 1);
    d_c[i] = 0.f;
  }
  cudaMemPrefetchAsync(d_a, N*sizeof(float), 0);
  cudaMemPrefetchAsync(d_b, N*sizeof(float), 0);
  cudaMemPrefetchAsync(d_c, N*sizeof(float), 0);
#else
  float* h_a = new float[N];
  float* h_b = new float[N];
  float* h_res = new float[N];
  for (int i = 0; i < N; i++) {
    h_a[i] = float(rand() % 1000000) / (rand() % 1000 + 1);
    h_b[i] = float(rand() % 1000000) / (rand() % 1000 + 1);
  }
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_c, N * sizeof(float));
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
#endif
  cudaDeviceSynchronize();
  dummy = timer.getCounterMsPrecise();
  cout << "alloc/H2D cost = " << dummy << "\n";
  constexpr int GRID_DIM = 256;
  constexpr int BLOCK_DIM = 256;

  timer.startCounter();
  multiply1<<<GRID_DIM, BLOCK_DIM>>>(N, d_c, d_a, d_b);
  cudaDeviceSynchronize();
  dummy = timer.getCounterMsPrecise();
  cout << "kernel cost = " << dummy << "\n";
  float *res = d_c;
  float *a = d_a;
  float *b = d_b;
#ifndef USE_MANAGED
  timer.startCounter();
  cudaMemcpy(h_res, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  dummy = timer.getCounterMsPrecise();
  cout << "D2H copy cost = " << timer.getCounterMsPrecise() << "\n";
  res = h_res;
  a = h_a;
  b = h_b;
#endif

  for (int i = 0; i < N; i++) if (res[i] != (a[i]*b[i])) {
    cout << "error\n";
    exit(1);
  }
  return 0;
}


int main(){
  for(int i = 0; i < 10; i++){
    test1();
  }
}