#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>    
#include <chrono>
#include <algorithm>

#define TILE_W 16
#define TILE_H 16
#define BLOCK 512

inline int get_number_of_blocks(int array_size, int block_size)
{
    return array_size / block_size + ((array_size % block_size > 0) ? 1 : 0);
}

__device__ __constant__ float kernel_gauss5x5[25] =
{
    0.00296902,      0.0133062,       0.0219382,       0.0133062,       0.00296902,
    0.0133062,       0.0596343,       0.0983203,       0.0596343,       0.0133062,
    0.0219382,       0.0983203,       0.162103,        0.0983203,       0.0219382,
    0.0133062,       0.0596343,       0.0983203,       0.0596343,       0.0133062,
    0.00296902,      0.0133062,       0.0219382,       0.0133062,       0.00296902
};

__device__ __constant__ float kernel_gauss7x7[49] =
{
    0.00001965,	0.00023941,	0.00107296,	0.00176901,	0.00107296,	0.00023941,	0.00001965,
    0.00023941,	0.0029166,	0.01307131,	0.02155094,	0.01307131,	0.0029166,	0.00023941,
    0.00107296,	0.01307131,	0.05858154,	0.09658462,	0.05858154,	0.01307131,	0.00107296,
    0.00176901,	0.02155094,	0.09658462,	0.15924113,	0.09658462,	0.02155094,	0.00176901,
    0.00107296,	0.01307131,	0.05858154,	0.09658462,	0.05858154,	0.01307131,	0.00107296,
    0.00023941,	0.0029166,	0.01307131,	0.02155094,	0.01307131,	0.0029166,	0.00023941,
    0.00001965,	0.00023941,	0.00107296,	0.00176901,	0.00107296,	0.00023941,	0.00001965,
};


__global__ void gauss5x5_tiles_kernel(const float* __restrict__ in, float *out, int w, int h)
{   
    const int R = 2;
    const int BLOCK_W = (TILE_W + 2*R);
    const int BLOCK_H = (TILE_H + 2*R);
    __shared__ float smem[BLOCK_W*BLOCK_H];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int offset = blockIdx.x * TILE_W + tx-R;
    int x = offset % w;
    int c = offset / w;
    int y = blockIdx.y * TILE_H + ty-R;

    // clamp to edge of image
    x = max(0, x);
    x = min(x, w-1);
    y = max(y, 0);
    y = min(y, h-1);
    //x = clamp(x, 0, w - 1);
    //y = clamp(y, 0, h - 1);

    unsigned int idx = y*w*4 + c*w + x;
    unsigned int bindex = threadIdx.y*BLOCK_W+threadIdx.x;
    
    // each thread copies its pixel of the block to shared memory
    smem[bindex] = in[idx];
    __syncthreads();

    float sum = 0;

    // only threads inside the apron will write results
    if (threadIdx.x >= R && threadIdx.x < (BLOCK_W-R) && threadIdx.y >= R && threadIdx.y < (BLOCK_H-R))
    {
#pragma unroll
        for(int i = -R; i <=R; i++)
        for(int j = -R; j <=R; j++)
        {
            sum += smem[bindex + (i*blockDim.x) + j] * kernel_gauss5x5[(i + R) * 5 + (j + R)];
        }

        out[idx] = sum;
    }
}

__global__ void gauss7x7_tiles_kernel(const float* __restrict__ in, float *out, int w, int h)
{   
    const int R = 3;
    const int BLOCK_W = (TILE_W + 2*R);
    const int BLOCK_H = (TILE_H + 2*R);
    __shared__ float smem[BLOCK_W*BLOCK_H];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int offset = blockIdx.x * TILE_W + tx-R;
    unsigned int x = offset % w;
    unsigned int c = offset / w;
    int y = blockIdx.y * TILE_H + ty-R;

    // clamp to edge of image
    x = max(0, x);
    x = min(x, w-1);
    y = max(y, 0);
    y = min(y, h-1);

    unsigned int idx = y*w*4 + c*w + x;
    unsigned int bindex = threadIdx.y*BLOCK_W+threadIdx.x;
    
    // each thread copies its pixel of the block to shared memory
    smem[bindex] = in[idx];
    __syncthreads();

    float sum = 0;

    // only threads inside the apron will write results
    if (threadIdx.x >= R && threadIdx.x < (BLOCK_W-R) && threadIdx.y >= R && threadIdx.y < (BLOCK_H-R))
    {
#pragma unroll
        for(int i = -R; i <=R; i++)
        for(int j = -R; j <=R; j++)
        {
            sum += smem[bindex + (i*blockDim.x) + j] *kernel_gauss7x7[(i + R) * 7 + (j + R)];
        }

        out[idx] = sum;
    }
}

extern "C" float* gauss5x5_gpu_tiles(float* d_src, float* d_dest, int w, int h, int cycles)
{
    dim3 dimGrid ((w*3) / TILE_W, h / TILE_H);
    dim3 dimBlock(TILE_W+4, TILE_H+4);

    float* src = d_src,*dst = d_dest, *tmp = d_dest;
    while (cycles--)
    {
        gauss5x5_tiles_kernel << < dimGrid, dimBlock, 0 >> > (src, d_dest, w, h);
        tmp = dst;
        dst = src;
        src = tmp;
    }

    return tmp;
}


extern "C" float* gauss7x7_gpu_tiles(float* d_src, float* d_dest, int w, int h, int cycles)
{
    dim3 dimGrid ((w*3) / TILE_W, h / TILE_H);
    dim3 dimBlock(TILE_W+6, TILE_H+6);

    float* src = d_src,*dst = d_dest, *tmp = d_dest;
    while (cycles--)
    {
        gauss7x7_tiles_kernel << < dimGrid, dimBlock, 0 >> > (src, d_dest, w, h);
        tmp = dst;
        dst = src;
        src = tmp;
    }

    return tmp;
}


void check_error(cudaError_t status)
{
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        printf("\n CUDA Error: %s\n", s);
        getchar();
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status2);
        printf("\n CUDA Error Prev: %s\n", s);
        getchar();
    }
}

void printTime(const char* name, double time)
{
    float fps = 1000 / time;
    printf("%-#40s",name);
    char tmp[32];
    sprintf(tmp, "%0.2f [ms]", time);
    printf("%-#20s%0.2f\n", tmp, fps);
}

#define CHECK_CUDA(X) check_error((cudaError_t)X);

extern "C" float* gauss5x5_gpu_tiles(float* d_src, float* d_dest,  int w, int h, int cycles);
extern "C" float* gauss7x7_gpu_tiles(float* d_src, float* d_dest, int w, int h, int cycles);

int main(void)
{
    cudaSetDevice(1);
    const int IMAGE_W = 2048*4 ; // pixels
    const int IMAGE_H = 2048*4 ;   
    const int N = IMAGE_W * IMAGE_H * 4;
    const int cycles = 10;

    // image is loaded as RGBA. fill with random values
    float* img_cpu = new float[N];
    for (int k = 0; k < N; k++)
        img_cpu[k] = std::rand() % 255;
  
    float* img_gpu = nullptr;
    CHECK_CUDA(cudaMalloc((void **) &img_gpu, (N * sizeof(float))));

    float* temp_gpu = nullptr;
    CHECK_CUDA(cudaMalloc((void **) &temp_gpu, (N * sizeof(float))));

    printf("image size: %d x %d\n", IMAGE_W, IMAGE_H);
    printf("%-#40s%-#20s%0s\n", "filter", "time", "FPS");
    printf("---------------------------------------------------------------------\n");


    CHECK_CUDA(cudaDeviceSynchronize());
    auto timeStart = std::chrono::system_clock::now();
    gauss5x5_gpu_tiles(img_gpu, temp_gpu, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());  
    auto timeEnd = std::chrono::system_clock::now();
    double dProcessingTime = (double)std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count() / cycles;
    printTime("gauss5x5_gpu_tiles", dProcessingTime);



    CHECK_CUDA(cudaDeviceSynchronize());
    timeStart = std::chrono::system_clock::now();
    gauss7x7_gpu_tiles(img_gpu, temp_gpu, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());  
    timeEnd = std::chrono::system_clock::now();
    dProcessingTime = (double)std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count() / cycles;
    printTime("gauss7x7_gpu_tiles", dProcessingTime);


    delete img_cpu;
    cudaFree(img_gpu);
    cudaFree(temp_gpu);

    return 0;
}