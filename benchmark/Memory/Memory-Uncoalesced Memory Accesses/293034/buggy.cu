#include <iostream>
#include <cuda_runtime.h>

// define row_access
__global__
void row_access_gmem(int n, float *L, float *r, int lda)
{
    int tx = threadIdx.x;
    int start_L = blockIdx.x * (lda * n);
    int start_r = blockIdx.x * lda;

    for (int j = n - 1; j >= 0; j--) {
        if (tx == 0) {
            r[start_r + j] /= L[start_L + lda * j + j];
        }
        __syncthreads();
        if (tx < j) {
            r[start_r + tx] -= L[start_L + lda * tx + j] * r[start_r + j];
        }
        __syncthreads();
    }
}


float launch(int n, float *d_L, float *d_r, int lda, int batch_count) 
{
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    row_access_gmem<<<batch_count, 256>>>(n, d_L, d_r, lda);
    cudaEventRecord(stop, 0);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}

int main() {
    //cudaSetDevice(1);
    int n = 256; 
    int lda = 256;
    int batch_count = 1024;//1024*2;//1024*4;//1024*8;

    
    float *h_L = new float[lda * n * batch_count];
    float *h_r = new float[lda * batch_count];

    for (int i = 0; i < lda * n * batch_count; ++i) {
        h_L[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < lda * batch_count; ++i) {
        h_r[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    
    float *d_L, *d_r;
    cudaMalloc(&d_L, lda * n * batch_count * sizeof(float));
    cudaMalloc(&d_r, lda * batch_count * sizeof(float));

    
    cudaMemcpy(d_L, h_L, lda * n * batch_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_r, lda * batch_count * sizeof(float), cudaMemcpyHostToDevice);

    
    for(int i=0;i<10;i++)
    {
        float time_row = launch(n, d_L, d_r, lda, batch_count);
        std::cout << "Row-wise access time: " << time_row << " ms" << std::endl;
    }

    
    cudaFree(d_L);
    cudaFree(d_r);
    delete[] h_L;
    delete[] h_r;

    return 0;
}