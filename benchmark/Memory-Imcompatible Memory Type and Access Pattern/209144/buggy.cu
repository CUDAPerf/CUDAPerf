#include <iostream>
#include <cuda_runtime.h>

const int ngroups = 1024*1024/64;
const int groupsz = 64;
const int halfblock = 32;  // choose as the minimum power of 2 greater than or equal to half of the groupsz

// for this kernel we assume that we launch with block size (threads per block) == groupsz
template <typename T>
__global__ void blockReduction(const T * __restrict__ in, T * __restrict__ out){

  __shared__ T sdata[groupsz];
  sdata[threadIdx.x] = in[threadIdx.x + blockIdx.x*groupsz];
  __syncthreads();
  for (int i = halfblock; i>0; i>>=1){
    if ((threadIdx.x < i) && (threadIdx.x+i < groupsz))
      sdata[threadIdx.x] += sdata[threadIdx.x+i];
    __syncthreads();}
  if (!threadIdx.x) out[blockIdx.x] = sdata[0];
}


void initializeData(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand() % 100); 
    }
}


int main() {
    cudaSetDevice(1);
    const int dataSize = ngroups * groupsz;
    float *h_in = new float[dataSize];
    float *h_out1 = new float[ngroups];

    initializeData(h_in, dataSize);

    float *d_in, *d_out1;
    cudaMalloc(&d_in, dataSize * sizeof(float));
    cudaMalloc(&d_out1, ngroups * sizeof(float));

    cudaMemcpy(d_in, h_in, dataSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(ngroups);
    dim3 threads(groupsz);
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for(int i=0;i<10;i++){
      blockReduction<<<blocks, threads>>>(d_in, d_out1);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);

    
    cudaMemcpy(h_out1, d_out1, ngroups * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Time for blockReduction: " << milliseconds1 << " ms" << std::endl;

    
    delete[] h_in;
    delete[] h_out1;
    cudaFree(d_in);
    cudaFree(d_out1);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}