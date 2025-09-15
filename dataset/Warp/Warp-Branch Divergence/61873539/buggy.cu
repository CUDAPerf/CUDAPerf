#include <iostream>
#include <cuda_runtime.h>
typedef unsigned char U8;
__global__ void ThreadPerRowCounter(int Threshold, int W, int H, U8 **AllPixels, int *AllReturns)
{
    extern __shared__ int row_counts[];//this parameter to kernel call "<<<, ,>>>" sets the size

    //see here for indexing https://blog.usejournal.com/cuda-thread-indexing-fb9910cba084
    int myImage = blockIdx.y * gridDim.x + blockIdx.x;
    int myStartRow = (threadIdx.y * blockDim.x + threadIdx.x);
    unsigned char *imageStart = AllPixels[myImage];

    unsigned char *pixelStart   = imageStart + myStartRow * W;
    unsigned char *pixelEnd     = pixelStart + W;
    unsigned char *pixelItr     = pixelStart;

    int row_count = 0;
    while(pixelItr < pixelEnd)
    {
        if (*pixelItr > Threshold) //REMOVING THIS LINE GIVES 6x PERFORMANCE
        {
            row_count++;
        }
        pixelItr++;
    }
    row_counts[myStartRow] = row_count;

    __syncthreads();

    if (myStartRow == 0)
    {//first thread sums up for the while image

        int image_count = 0;
        for (int i = 0; i < H; i++)
        {
            image_count += row_counts[i];
        }
        AllReturns[myImage] = image_count;
    }
}


void cuda_Benchmark(int nImages, int W, int H, U8** AllPixels, int *AllReturns, int Threshold)
{
    for(int i=0; i<10; i++){
        ThreadPerRowCounter<<<nImages, H, sizeof(int)*H>>> (
            Threshold,
            W, H,
            AllPixels,
            AllReturns);

        //wait for all blocks to finish
        cudaDeviceSynchronize();
    }
}

unsigned next_power_of_2(unsigned v){
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;}

int main(){
    cudaSetDevice(1);
    const int my_W = 512;
    const int my_H = 512;
    const int n_img = 256;
    const int my_thresh = 10;

    U8 **img_p, **img_ph;
    U8 *img, *img_h;
    int *res, *res_h;
    img_ph = (U8 **)malloc(n_img*sizeof(U8*));
    cudaMalloc(&img_p, n_img*sizeof(U8*));
    cudaMalloc(&img, n_img*my_W*my_H*sizeof(U8));
    img_h = new U8[n_img*my_W*my_H];
    for (int i = 0; i < n_img*my_W*my_H; i++) img_h[i] = rand()%20;
    cudaMemcpy(img, img_h, n_img*my_W*my_H*sizeof(U8), cudaMemcpyHostToDevice);
    for (int i = 0; i < n_img; i++) img_ph[i] = img+my_W*my_H*i;
    cudaMemcpy(img_p, img_ph, n_img*sizeof(U8*), cudaMemcpyHostToDevice);
    cudaMalloc(&res, n_img*sizeof(int));
    cuda_Benchmark(n_img, my_W, my_H, img_p, res, my_thresh);
    res_h = new int[n_img];
    cudaMemcpy(res_h, res, n_img*sizeof(int), cudaMemcpyDeviceToHost);
}