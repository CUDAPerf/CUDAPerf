#include <iostream>
#include <cstdlib>

const int my_bitset_size = 512;
const int my_bunch_size = 1024*2;
typedef unsigned uint;


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