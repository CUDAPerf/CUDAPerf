#include <iostream>
#include <cuda_runtime.h>

const int ngroups = 1024*1024/64;
const int groupsz = 64;


// for this kernel we assume that we launch with block size (threads per block) == groupsz
template <typename T>
__global__ void blockReduction2(const T * __restrict__ in, T * __restrict__ out){

  __shared__ T sdata[2];  // specific for groupsz 64 case
 T val = in[threadIdx.x + blockIdx.x*groupsz];
 // warp-shuffle reduction
  for (int offset = 16; offset > 0; offset >>= 1) 
       val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  if (!(threadIdx.x & 31)) sdata[threadIdx.x >> 5] = val;
  __syncthreads(); // put warp results in shared mem

  if (!threadIdx.x) out[blockIdx.x] = sdata[0] + sdata[1]; // specific for groupsz 64 case
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
    float *h_out2 = new float[ngroups];

    initializeData(h_in, dataSize);

    float *d_in, *d_out2;
    cudaMalloc(&d_in, dataSize * sizeof(float));
    cudaMalloc(&d_out2, ngroups * sizeof(float));

    cudaMemcpy(d_in, h_in, dataSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(ngroups);
    dim3 threads(groupsz);
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i=0;i<10;i++)
      blockReduction2<<<blocks, threads>>>(d_in, d_out2);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start, stop);

    
    cudaMemcpy(h_out2, d_out2, ngroups * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Time for blockReduction2: " << milliseconds2 << " ms" << std::endl;

    
    delete[] h_in;
    delete[] h_out2;
    cudaFree(d_in);
    cudaFree(d_out2);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}