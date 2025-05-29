#include <cuda.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

const int imageWidth = 1280;
const int imageHeight = 720;

// The "image" data pY is split into rectangles of the following dimension, with each rectangle processed by its own CUDA thread.
// There are 320 rectangles width-wise and 6 rectangles height-wise to perfectly cover the image area.
const int threadWidth = 4;
const int threadHeight = 120;

#define SQUARE(A)       ((A) * (A))

uint64_t get_time_usec(void)
{
	static struct timeval _time_stamp;
	gettimeofday(&_time_stamp, NULL);
	return (_time_stamp.tv_sec * 1000000ll) + _time_stamp.tv_usec;
}


__global__ void kernel1(uint8_t* pY)
{
    int cudaRow = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelRow = cudaRow * threadHeight;
    int cudaCol = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelCol = cudaCol * threadWidth;
    
    int xConst = (imageWidth / 2) - pixelCol;
    int yConst = (imageHeight / 2) - pixelRow;
    
    for(int x_offset = 0; x_offset < threadWidth; x_offset++)
    {
        int xSq = SQUARE(xConst - x_offset);
        
        for(int y_offset = 0; y_offset < threadHeight; y_offset++)
        {
            uint32_t offset = (pixelRow + y_offset) * imageWidth + pixelCol + x_offset;
            
            int ySq = SQUARE(yConst - y_offset);
            int sumSq = xSq + ySq;
            float distanceFromCentre = sqrtf(sumSq);
            float correction = (1.0749947E-6f * sumSq) - (0.000297173f * distanceFromCentre) + 1.01820957f;
            float pixelVal = (float)pY[offset] * correction;
            
            pY[offset] = (uint8_t)pixelVal;
        }
    }
}


int main(void)
{
    //cudaSetDevice(1);
    //dim3 blocks = 2; // max on Jetson Nano
    dim3 threadsPerBlock = dim3(160, 6);
    int blockWidth = threadWidth * 160;
    int blockHeight = threadHeight * 6;
    dim3 blocks(
        (imageWidth + blockWidth - 1) / blockWidth, 
        (imageHeight + blockHeight - 1) / blockHeight
    );
    uint8_t* pY;
    
    cudaMalloc(&pY, imageWidth * imageHeight * sizeof(uint8_t));
    
    uint64_t start = get_time_usec();
    for(int i=0;i<10;i++){
        kernel1<<<blocks, threadsPerBlock>>>(pY);
        cudaDeviceSynchronize();
    }
    
    
    printf("int/float took %fms.\n", (double)(get_time_usec() - start) / 1000.0);
     
     return 0;
}