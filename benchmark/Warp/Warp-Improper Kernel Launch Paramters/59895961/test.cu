#include <iostream>
#include <cstdlib>

const int my_bitset_size = 512;
const int my_bunch_size = 1024*2;
typedef unsigned uint;

//using one thread per bitset in the bunch
__global__ void kernelXOR(uint * bitset, uint * bunch, int * set_bits, int bitset_size, int bunch_size) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < bunch_size){      // 1 Thread for each bitset in the 'bunch'
        int sum = 0;
        uint xor_res = 0;
        for (int i = 0; i < bitset_size; ++i){  // Iterate through every uint-block of the bitsets
            xor_res = bitset[i] ^ bunch[bitset_size * tid + i];
            sum += __popc(xor_res);
        }
        set_bits[tid] = sum;
    }
}

__global__ void kernelXOR2(uint * bitset, uint * bunch, int * set_bits, int bitset_size, int bunch_size) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < bunch_size){      // 1 Thread for each bitset in the 'bunch'
        int sum = 0;
        uint xor_res = 0;
        for (int i = 0; i < bitset_size; ++i){  // Iterate through every uint-block of the bitsets
            xor_res = bitset[i] ^ bunch[bitset_size * tid + i];
            sum += __popc(xor_res);
        }
        set_bits[tid] = sum;
    }
}

const int nTPB = 256;
// one block per bitset, multiple bitsets per block
__global__ void kernelXOR_imp(const uint * __restrict__  bitset, const uint * __restrict__  bunch, int * __restrict__  set_bits, int bitset_size, int bunch_size) {

    __shared__ uint sbitset[my_bitset_size];  // could also be dynamically allocated for varying bitset sizes
    __shared__ int ssum[nTPB];
    // load shared, block-stride loop
    for (int idx = threadIdx.x; idx < bitset_size; idx += blockDim.x) sbitset[idx] = bitset[idx];
    __syncthreads();
    // stride across all bitsets in bunch
    for (int bidx = blockIdx.x; bidx < bunch_size; bidx += gridDim.x){
      int my_sum = 0;
      for (int idx = threadIdx.x; idx < bitset_size; idx += blockDim.x) my_sum += __popc(sbitset[idx] ^ bunch[bidx*bitset_size + idx]);
    // block level parallel reduction
      ssum[threadIdx.x] = my_sum;
      for (int ridx = nTPB>>1; ridx > 0; ridx >>=1){
        __syncthreads();
        if (threadIdx.x < ridx) ssum[threadIdx.x] += ssum[threadIdx.x+ridx];}
      if (!threadIdx.x) set_bits[bidx] = ssum[0];}
}



int test(){

// data setup

  uint *d_cbitset, *d_bitsets, *h_cbitset, *h_bitsets;
  int *d_r, *h_r, *h_ri;
  h_cbitset = new uint[my_bitset_size];
  h_bitsets = new uint[my_bitset_size*my_bunch_size];
  h_r  = new int[my_bunch_size];
  h_ri = new int[my_bunch_size];
  for (int i = 0; i < my_bitset_size*my_bunch_size; i++){
    h_bitsets[i] = rand();
    if (i < my_bitset_size) h_cbitset[i] = rand();}
  cudaMalloc(&d_cbitset, my_bitset_size*sizeof(uint));
  cudaMalloc(&d_bitsets, my_bitset_size*my_bunch_size*sizeof(uint));
  cudaMalloc(&d_r,  my_bunch_size*sizeof(int));
  cudaMemcpy(d_cbitset, h_cbitset, my_bitset_size*sizeof(uint), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bitsets, h_bitsets, my_bitset_size*my_bunch_size*sizeof(uint), cudaMemcpyHostToDevice);

  // Grid/Blocks used for kernel invocation
  dim3 block(32);
  dim3 grid((my_bunch_size-1)/32+1);

  kernelXOR2<<<grid, block>>>(d_cbitset, d_bitsets, d_r, my_bitset_size, my_bunch_size);
  cudaMemcpy(h_r, d_r, my_bunch_size*sizeof(int), cudaMemcpyDeviceToHost);

  dim3 iblock(nTPB);
  dim3 igrid((my_bunch_size-1)/nTPB+1);

  kernelXOR<<<igrid, iblock>>>(d_cbitset, d_bitsets, d_r, my_bitset_size, my_bunch_size);
  cudaMemcpy(h_r, d_r, my_bunch_size*sizeof(int), cudaMemcpyDeviceToHost);


  kernelXOR_imp<<<igrid, iblock>>>(d_cbitset, d_bitsets, d_r, my_bitset_size, my_bunch_size);
  cudaMemcpy(h_ri, d_r, my_bunch_size*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < my_bunch_size; i++)
    if (h_r[i] != h_ri[i]) {std::cout << "mismatch at i: " << i << " was: " << h_ri[i] << " should be: " << h_r[i] << std::endl; return 0;}
  std::cout << "Results match." << std::endl;
  return 0;
}

int main(){
  cudaSetDevice(1);
  for(int i = 0; i < 10; i++){
    test();
  }
  return 0;
}