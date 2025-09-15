#include <cub/cub.cuh>   // or equivalently <cub/block/block_radix_sort.cuh>
#include <iostream>
const int ipt=8;
const int tpb=128;
__global__ void ExampleKernel(int *data)
{
    // Specialize BlockRadixSort for a 1D block of 128 threads owning 8 integer items each
    typedef cub::BlockRadixSort<int, tpb, ipt> BlockRadixSort;
    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    // Obtain a segment of consecutive items that are blocked across threads
    int thread_keys[ipt];
    // just create some synthetic data in descending order 1023 1022 1021 1020 ...
    for (int i = 0; i < ipt; i++) thread_keys[i] = (tpb-1-threadIdx.x)*ipt+i;
    // Collectively sort the keys
    BlockRadixSort(temp_storage).Sort(thread_keys);
    __syncthreads();
    // write results to output array
    for (int i = 0; i < ipt; i++) data[blockIdx.x*ipt*tpb + threadIdx.x*ipt+i] = thread_keys[i];
}


int main(){

    const int blks = 1024;
    int *data;
    cudaMalloc(&data, blks*ipt*tpb*sizeof(int));
    for(int i =0;i<10;i++)
        ExampleKernel<<<blks,tpb>>>(data);
    int *h_data = new int[blks*ipt*tpb];
    cudaMemcpy(h_data, data, blks*ipt*tpb*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) std::cout << h_data[i] << " ";
    std::cout << std::endl;
}