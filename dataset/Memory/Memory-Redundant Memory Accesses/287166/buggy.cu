#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>    
#include <chrono>
#include <algorithm>

__global__ void copyRow_Sheme(float * MatA,float* MatB,int nx,int ny)
{
  __shared__ float tile[16][16 * 2];
  int ix=threadIdx.x+blockDim.x * blockIdx.x * 2;
  int iy=threadIdx.y+blockDim.y * blockIdx.y;
  int idx=ix+iy * nx;

   // loop unroll 2
   if(ix<nx&& iy<ny)
        {
          tile[threadIdx.y][threadIdx.x]=MatA[idx];
          tile[threadIdx.y][threadIdx.x+blockDim.x]=MatA[idx+blockDim.x];

          __syncthreads();

          MatB[idx]=tile[threadIdx.y][threadIdx.x];
          MatB[idx+blockDim.x]=tile[threadIdx.y][threadIdx.x+blockDim.x];
        }
}

using mt = float;
int main(){
  //cudaSetDevice(1);
  size_t sz = 1024*8;
  size_t msz = sz*sz;
  dim3 grid = dim3(sz/16/2, sz/16);
  dim3 block = dim3(16,16);
  mt *d_MatA, *d_MatB;
  cudaMalloc(&d_MatA, sizeof(float)*msz);
  cudaMalloc(&d_MatB, sizeof(float)*msz);
  for(int i=0;i<10;i++){

    copyRow_Sheme<<<grid,block>>>(d_MatA,d_MatB,sz,sz);
    cudaDeviceSynchronize();

  }
  
}
