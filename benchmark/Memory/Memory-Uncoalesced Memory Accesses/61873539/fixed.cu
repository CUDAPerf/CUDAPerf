#include <iostream>
#include <cuda_runtime.h>
typedef unsigned char U8;

__global__ void ThreadPerColCounter(int Threshold, int W, int H, U8 **AllPixels, int *AllReturns, int rsize)
{
    extern __shared__ int col_counts[];//this parameter to kernel call "<<<, ,>>>" sets the size
    int myImage = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned char *imageStart = AllPixels[myImage];
    int myStartCol = (threadIdx.y * blockDim.x + threadIdx.x);
    int col_count = 0;
    for (int i = 0; i < H; i++) if (imageStart[myStartCol+i*W]> Threshold) col_count++;
    col_counts[threadIdx.x] = col_count;
    __syncthreads();
    for (int i = rsize; i > 0; i>>=1){
      if ((threadIdx.x+i < W) && (threadIdx.x < i)) col_counts[threadIdx.x] += col_counts[threadIdx.x+i];
    __syncthreads();}
    if (!threadIdx.x) AllReturns[myImage] = col_counts[0];
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

void cuda_Benchmark1(int nImages, int W, int H, U8** AllPixels, int *AllReturns, int Threshold)
{
    int rsize = next_power_of_2(W/2);
    for(int i=0; i<10; i++){
        ThreadPerColCounter<<<nImages, W, sizeof(int)*W>>> (
            Threshold,
            W, H,
            AllPixels,
            AllReturns, rsize);

        //wait for all blocks to finish
        cudaDeviceSynchronize();
    }
}

int main(){
    cudaSetDevice(1);
    const int my_W = 512;
    const int my_H = 512;
    const int n_img = 256;
    const int my_thresh = 10;

    U8 **img_p, **img_ph;
    U8 *img, *img_h;
    int *res, *res_h1;
    img_ph = (U8 **)malloc(n_img*sizeof(U8*));
    cudaMalloc(&img_p, n_img*sizeof(U8*));
    cudaMalloc(&img, n_img*my_W*my_H*sizeof(U8));
    img_h = new U8[n_img*my_W*my_H];
    for (int i = 0; i < n_img*my_W*my_H; i++) img_h[i] = rand()%20;
    cudaMemcpy(img, img_h, n_img*my_W*my_H*sizeof(U8), cudaMemcpyHostToDevice);
    for (int i = 0; i < n_img; i++) img_ph[i] = img+my_W*my_H*i;
    cudaMemcpy(img_p, img_ph, n_img*sizeof(U8*), cudaMemcpyHostToDevice);
    cudaMalloc(&res, n_img*sizeof(int));
    cuda_Benchmark1(n_img, my_W, my_H, img_p, res, my_thresh);
    res_h1 = new int[n_img];
    cudaMemcpy(res_h1, res, n_img*sizeof(int), cudaMemcpyDeviceToHost);
}