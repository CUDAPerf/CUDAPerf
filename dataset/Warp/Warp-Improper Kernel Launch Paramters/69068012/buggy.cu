#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iostream>

void initData_int(int *p, int size){
    for (int t=0; t<size; t++){
        p[t] = (int)(rand()&0xff);
    }
}

__global__ void vecCopy(int *d_in, int *d_out, int size)
{
    int2* in = (int2*)d_in;
    int2* out = (int2*)d_out;
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for (int i = tid; i < size/2; i += blockDim.x*gridDim.x)
    {
        out[i] = in[i];
    }

    if(tid==size/2 && size%2==1)
        d_out[size-1] = d_in[size-1];
}


int main(int argc, char **argv)
{
    cudaSetDevice(1);
    int size = 1024*1024*64;
    //int size = 128;
    int nBytes = size*sizeof(int);
    int *d_h;
    cudaMallocHost((int**)&d_h, nBytes);
    initData_int(d_h, size);

    //printData(d_h, size);

    int *res = (int*)malloc(nBytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int *d_in, *d_out;
    dim3 block(128, 1);
    dim3 grid((size-1)/block.x+1, 1);
    dim3 grid2((size / 2 - 1) / block.x + 1, 1);
    cudaMalloc((int**)&d_in, nBytes);
    cudaMalloc((int**)&d_out, nBytes);

    cudaMemcpyAsync(d_in, d_h, nBytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    
    memset(res, 0, nBytes);
    cudaMemset(d_out, 0, nBytes);
    //vectorized access:
    cudaStreamSynchronize(stream);
    auto s_0 = std::chrono::system_clock::now();
    for(int i=0;i<10;i++){
        vecCopy<<<grid, block, 0, stream>>>(d_in, d_out, size);
        cudaStreamSynchronize(stream);
    }
    auto e_0 = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = e_0 - s_0;
    printf("Vectorized kernel time cost is: %2f.\n", diff.count());
    
    cudaStreamDestroy(stream);
    cudaFree(d_h);
    cudaFree(d_in);
    cudaFree(d_out);
    free(res);

    return 0;
} 