#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <ctime>
#include <ratio>
#include <chrono>
#include <random>
#include <time.h>
#include <math.h>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cusolverDn.h>

//#include "Utilities.cuh"

using namespace std;
using namespace std::chrono;

/************************************/
/* COEFFICIENT REARRANGING FUNCTION */
/************************************/
void rearrange(double** vec, int* pivotArray, int N, int numMatrices) {
  for (int nm = 0; nm < numMatrices; nm++) {
    for (int i = 0; i < N; i++) {
      double temp = vec[nm][i];
      vec[nm][i] = vec[nm][pivotArray[N*i + nm] - 1];
      vec[nm][pivotArray[N * i + nm] - 1] = temp;
    }
  }
}


/************************************/
/* MAIN  */
/************************************/
int main() {
  cudaSetDevice(1);

  const int N = 3; 
  const int numMatrices = 1024*1024*512; 

  // random generator to fill matrices and coefficients
  random_device device;
  mt19937 generator(device());
  uniform_real_distribution<double> distribution(1., 5.);
  //ALLOCATE MEMORY - using unified memory
  double** h_A;
  cudaMallocManaged(&h_A, sizeof(double*) * numMatrices);
  for (int nm = 0; nm < numMatrices; nm++) {
    cudaMallocManaged(&(h_A[nm]), sizeof(double) * N * N);
  }

  double** h_b;
  cudaMallocManaged(&h_b, sizeof(double*) * numMatrices);
  for (int nm = 0; nm < numMatrices; nm++) {
    cudaMallocManaged(&(h_b[nm]), sizeof(double) * N );
  }
  cout << " memory allocated" << endl;
  // FILL MATRICES
  for (int nm = 0; nm < numMatrices; nm++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        h_A[nm][j * N + i] = distribution(generator);
      }
    }
  }
  cout << " Matrix filled " << endl;

  // FILL COEFFICIENTS
  for (int nm = 0; nm < numMatrices; nm++) {
    for (int i = 0; i < N; i++) {
      h_b[nm][i] = distribution(generator);
    }
  }
  cout << " Coeff. vector filled " << endl;
  cout << endl;

  // --- CUDA solver initialization
  cublasHandle_t cublas_handle;
  cublasCreate_v2(&cublas_handle);
  int* PivotArray;
  cudaMallocManaged(&PivotArray, N * numMatrices * sizeof(int));
  int* infoArray;
  cudaMallocManaged(&infoArray, numMatrices * sizeof(int));

  //CUBLAS LU SOLVER
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  cublasDgetrfBatched(cublas_handle, N, h_A, N, PivotArray, infoArray, numMatrices);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  cout << "It took " << time_span.count() * 1000. << " milliseconds." << endl;


  for (int i = 0; i < numMatrices; i++)
    if (infoArray[i] != 0) {
      fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
    }

 // rearrange coefficient 
 // (temporarily on CPU, this step will be on a GPU Kernel as well)
  high_resolution_clock::time_point tA = high_resolution_clock::now();
  rearrange(h_b, PivotArray, N, numMatrices);
  high_resolution_clock::time_point tB = high_resolution_clock::now();
  duration<double> time_spanA = duration_cast<duration<double>>(tB - tA);
  cout << "rearrangement took " << time_spanA.count() * 1000. << " milliseconds." << endl;

//INVERT UPPER AND LOWER TRIANGULAR MATRICES 
  // --- Function solves the triangular linear system with multiple right-hand sides
  // --- Function overrides b as a result 
  const double alpha = 1.f;
  high_resolution_clock::time_point t3 = high_resolution_clock::now();
  cublasDtrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, 1, &alpha, h_A, N, h_b, N, numMatrices);
  cublasDtrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, 1, &alpha, h_A, N, h_b, N, numMatrices);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t4 = high_resolution_clock::now();
  duration<double> time_span2 = duration_cast<duration<double>>(t4 - t3);
  cout << "second step took " << time_span2.count() * 1000. << " milliseconds." << endl;
  
  // --- Free resources
  if (h_A) cudaFree(h_A);
  if (h_b) cudaFree(h_b);
 
  cudaDeviceReset();

  return 0;
}