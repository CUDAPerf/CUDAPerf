#include "cublas_v2.h"
#include "cuda_runtime.h"
#include <iostream>

#define TILE_WIDTH 32
const int Width = 1024;

__global__ void MatrixMulKernel(const float *M, const float *N, float *P,
                                const int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < width && column < width) {
    float Pvalue = .0;
    for (int k = 0; k < width; ++k) {
      Pvalue += M[row * width + k] * N[k * width + column];
    }
    P[row * width + column] = Pvalue;
  }
}

void testMatrixMulKernel(const int Width) {
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

  dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  MatrixMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, Width);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Elapsed time for testMatrixMulKernel : " << milliseconds
            << '\n';

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
    testMatrixMulKernel(Width);
  }
	return 0;
}