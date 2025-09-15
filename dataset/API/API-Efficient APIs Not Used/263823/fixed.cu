#include "cublas_v2.h"
#include "cuda_runtime.h"
#include <iostream>

#define TILE_WIDTH 32
const int Width = 1024;

void testCuBLASMatrixMulKernel(const int Width) {
  float *A_h = (float *)malloc(Width * Width * sizeof(float));
  float *B_h = (float *)malloc(Width * Width * sizeof(float));
  for (auto i = 0; i < Width * Width; i++) {
    A_h[i] = 1.0 * float(i) / 4.0;
    B_h[i] = 2.0 * float(i) / 4.0;
  }
  float *A_d, *B_d, *C_d;
  cudaMalloc((void **)&A_d, Width * Width * sizeof(float));
  cudaMalloc((void **)&B_d, Width * Width * sizeof(float));
  cudaMalloc((void **)&C_d, Width * Width * sizeof(float));

  cudaMemcpy(A_d, A_h, Width * Width * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, Width * Width * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Width, Width, Width, &alpha, A_d, Width, B_d, Width, &beta, C_d, Width);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Elapsed time for testCuBLASMatrixMulKernel : " << milliseconds
            << '\n';

  cublasDestroy(handle);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  free(A_h);
  free(B_h);
}

int main(int argc, const char* argv[]) {
	cudaDeviceSynchronize();
	std::cout << "Arithmetic Intensity : " << 1.0 / 3.0 * float(Width) << '\n';
  for(int i=0;i<1;i++){
    testCuBLASMatrixMulKernel(Width);
  }
	return 0;
}