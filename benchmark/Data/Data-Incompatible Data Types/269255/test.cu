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


__global__ void kernel2(uint8_t* pY)
{
    int cudaRow = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelRow = cudaRow * threadHeight;
    int cudaCol = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelCol = cudaCol * threadWidth;
    
    // mult is necessary otherwise when the numbers are squared later on to calculate distanceFromCentre there will be an overflow on half.
    half mult = 0.1;
    half a = __float2half(1.0749947E-4);
    half b = __float2half(-0.00297173);
    half c = __float2half(1.01820957);
    half xConst = __short2half_ru((imageWidth / 2) - pixelCol) * mult;
    half yConst = __short2half_ru((int)(imageHeight / 2) - pixelRow) * mult;

    for(int x_offset = 0; x_offset < threadWidth; x_offset++)
    {
        half xTemp = __hfma(__short2half_ru(x_offset), mult, -xConst);
        half xSq = SQUARE(xTemp);
        
        for(int y_offset = 0; y_offset < threadHeight; y_offset++)
        {
            uint32_t offset = (pixelRow + y_offset) * imageWidth + pixelCol + x_offset;
            
            // Use a quadratic equation to adjust the value in pY.
            half yTemp = __hfma(__short2half_ru(y_offset), mult, -yConst);   
            half sumSq = __hfma(yTemp, yTemp, xSq);                          
            half distanceFromCentre = hsqrt(sumSq);                          
            half correction = (a * sumSq) + __hfma(b, distanceFromCentre, c);
            half pixelVal = __short2half_ru(pY[offset]) * correction;        
            
            pY[offset] = __half2short_ru(pixelVal);
        }
    }
}

__global__ void kernel3(uint8_t* pY)
{
    int cudaRow = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelRow = cudaRow * threadHeight;
    int cudaCol = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelCol = cudaCol * threadWidth;
    
    int xConst = (imageWidth / 2) - pixelCol;
    int yConst = (imageHeight / 2) - pixelRow;

    //assuming threadWidth == 4;

    for(int y_offset = 0; y_offset < threadHeight; y_offset++)
    {
        uint32_t offset = (pixelRow + y_offset) * imageWidth + pixelCol;
        uint8_t myPY[4];
        *((char4*)(&myPY[0])) = *((const char4*)(&pY[offset]));

        for(int x = 0; x < 4; x++){
          int xSq = SQUARE(xConst - x);
          int ySq = SQUARE(yConst - y_offset);
          int sumSq = xSq + ySq;
          float distanceFromCentre = sqrtf(sumSq);
          float correction = (1.0749947E-6f * sumSq) - (0.000297173f * distanceFromCentre) + 1.01820957f;
          float pixelVal = (float)myPY[x] * correction;
          myPY[x] = (uint8_t)pixelVal;
        }
        *((char4*)(&pY[offset])) = *((const char4*)(&myPY[0]));
        
        
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
    
    start = get_time_usec();

    for(int i=0;i<10;i++){
        kernel2<<<blocks, threadsPerBlock>>>(pY);
        cudaDeviceSynchronize();
    }
    
     printf("half took %fms.\n", (double)(get_time_usec() - start) / 1000.0);


     start = get_time_usec();
    
    for(int i=0;i<10;i++){
        kernel3<<<blocks, threadsPerBlock>>>(pY);
        cudaDeviceSynchronize();
    }
    
     printf("kernel3 took %fms.\n", (double)(get_time_usec() - start) / 1000.0);
     
     return 0;
}