#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define ThreadPB 32 // optimal size
dim3 threadsPerBlock(ThreadPB, ThreadPB);

__global__ void initKernel(int *input, int nx, int ny)
{
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = idx_y * nx + idx_x;

    if (idx_x < nx && idx_y < ny) {
        input[idx] = idx_y;
    }

}


__global__ void flipKernel2(int *output,int *input, int nx, int ny)
{
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = idx_y * nx + idx_x;
    
    if (idx_x < nx && idx_y < ny/2) { 
        output[(ny - idx_y - 1) * nx + idx_x] = input[idx_y * nx + idx_x];
        output[idx_y * nx + idx_x] = input[(ny - idx_y - 1) * nx + idx_x];;
    }
}

int main()
{
    //cudaSetDevice(1);
    // time check
    cudaEvent_t start, stop;
    cudaEvent_t start_temp2, stop_temp2;
    float elapsedTime;
    cudaEventCreate(&start);        cudaEventCreate(&stop);
    cudaEventCreate(&start_temp2);  cudaEventCreate(&stop_temp2);

    const int num_x = 1024*16;//1024*8;//1024*2;//1024;
    const int num_y = 1024*32;//1024*8;//1024*4;//1024;/
    const int arraySize = num_x * num_y;

    int *orig, *flip;
    orig = (int *)malloc(sizeof(int) * arraySize);
    flip = (int *)malloc(sizeof(int) * arraySize);

    int *dev_orig = 0;
    int *dev_flip = 0;

    cudaMalloc((void**)&dev_orig, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_flip, arraySize * sizeof(int));
    cudaMemcpy(dev_orig, orig, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_flip, flip, arraySize * sizeof(int), cudaMemcpyHostToDevice);



    dim3 blocksFlip2((num_x + threadsPerBlock.x - 1) / threadsPerBlock.x, ((num_y + threadsPerBlock.y - 1) / threadsPerBlock.y)/2);
    
    cudaEventRecord(start, 0);
    

    for(int i = 0; i < 10; i++){
        flipKernel2 << <blocksFlip2, threadsPerBlock >> > (dev_flip,dev_orig, num_x, num_y);
    }


    // time check end
    cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsedTime, start, stop); printf("flip = %f ms.\n", elapsedTime);


    cudaMemcpy(orig, dev_orig, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(flip, dev_flip, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // check flip works
    printf("FLIP this array { 0, 1, 2, 3, 4 , 5, 6, 7, 8, 9...} \n= { %d, %d, %d, %d, %d, %d, %d, %d, %d, %d...}\n",
        flip[num_x * 0], flip[num_x * 1], flip[num_x * 2], flip[num_x * 3], flip[num_x * 4],
        flip[num_x * 5], flip[num_x * 6], flip[num_x * 7], flip[num_x * 8], flip[num_x * 9]);
    


    return 0;
}
