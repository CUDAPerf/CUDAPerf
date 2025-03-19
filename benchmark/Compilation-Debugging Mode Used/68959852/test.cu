#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

void initData_int(int *p, int size){
    for (int t=0; t<size; t++){
        p[t] = (int)(rand()&0xff);
    }
}


__global__ void gpuReduce(int *in, int *out, int size)
{
    int tid = threadIdx.x;
    int* data = in + blockIdx.x*blockDim.x;
    if (tid >= size)
        return;

    for (int stride = 1; stride < blockDim.x; stride*=2)
    {
        if((tid%(2*stride)) == 0){
            data[tid] += data[tid+stride];

        }
        __syncthreads();
    }
    if (tid == 0){
        out[blockIdx.x] = data[0];
    }
}

__global__ void gpuReduceOpt(int *in, int *out, int size)
{
    int tid = threadIdx.x;
    int* data = in + blockIdx.x*blockDim.x;
    if (tid >= size)
        return;

    for (int stride = 1; stride < blockDim.x; stride*=2)
    {
        int index = 2*stride*tid;
        if(index < blockDim.x){
            data[index] += data[index+stride];
        }
        __syncthreads();
    }
    if (tid == 0){
        out[blockIdx.x] = data[0];
    }
}

int test()
{
    cudaSetDevice(1);
    int size = 1<<26;
    int blocksize = 1024;


    dim3 block(blocksize, 1);
    dim3 grid((size-1)/block.x+1, 1);
    int nBytes = sizeof(int)*size;

    int *a_h = (int*)malloc(nBytes);
    int *tmp = (int*)malloc(sizeof(int)*grid.x);
    int *tmp1 = (int*)malloc(sizeof(int)*grid.x);
    initData_int(a_h, size);

    int *a_d, *output;
    cudaMalloc((int**)&a_d, nBytes);
    cudaMalloc((int**)&output, grid.x*sizeof(int));

    int *a_d1, *output1;
    cudaMalloc((int**)&a_d1, nBytes);
    cudaMalloc((int**)&output1, grid.x*sizeof(int));
    cudaMemcpy(a_d1, a_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);

    auto start2 = std::chrono::system_clock::now();
    gpuReduce<<<grid, block>>>(a_d, output, size);
    cudaMemcpy(tmp, output, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_result = 0;

    for (int i =0; i < grid.x; i++)
    {
        gpu_result += tmp[i];
    }
    cudaDeviceSynchronize();
    auto end2 = std::chrono::system_clock::now();
    std::chrono::duration<double>diff2 = end2 - start2;
    printf("Gpu reduce take:%2f s\n", diff2.count());

    auto start3 = std::chrono::system_clock::now();
    gpuReduceOpt<<<grid, block>>>(a_d1, output1, size);
    cudaMemcpy(tmp1, output1, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_result1 = 0;

    for (int i =0; i < grid.x; i++)
    {
        gpu_result1 += tmp1[i];
    }
    cudaDeviceSynchronize();
    auto end3 = std::chrono::system_clock::now();
    std::chrono::duration<double>diff3 = end3 - start3;
    printf("Gpu reduce opt take:%2f s\n", diff3.count());
    printf("Result from gpuReduce and gpuReduceOpt are %6d and %6d\n", gpu_result, gpu_result1);


    cudaFree(a_d);
    cudaFree(output);
    free(a_h);
    free(tmp);
    cudaDeviceReset();
    return 0;
}

int main()
{
    for(int i = 0; i < 10; i++)
        test();
    return 0;
}