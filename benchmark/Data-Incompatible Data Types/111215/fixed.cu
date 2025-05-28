#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define N 1024*2  // Adjust the number of elements as needed
#define BLOCK_SIZE 256

// Compute1_1 use float
__global__ void Compute1_1(float2* p0, float2* p1, float2* v, int count) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    float2 fd = {0.0F, 0.0F};
    for (auto j = 0; j < count; ++j) {
        //if (i == j) continue;
        const auto dx = p0[i].x - p0[j].x;
        const auto dy = p0[i].y - p0[j].y;
        const auto f = 0.000000001F / (dx * dx + dy * dy + 0.000000000001F);
        fd.x += dx * f;
        fd.y += dy * f;
    }
    p1[i].x = p0[i].x + (v[i].x -= fd.x);
    p1[i].y = p0[i].y + (v[i].y -= fd.y);
}

int main() {
    cudaSetDevice(1);
    int count = N;

    float2* oldPos, * newPos, * vel;
    float2* d_oldPos, * d_newPos, * d_vel;

    size_t size_float2 = count * sizeof(float2);

    oldPos = (float2*)malloc(size_float2);
    newPos = (float2*)malloc(size_float2);
    vel = (float2*)malloc(size_float2);

    cudaMalloc(&d_oldPos, size_float2);
    cudaMalloc(&d_newPos, size_float2);
    cudaMalloc(&d_vel, size_float2);

    for (int i = 0; i < count; ++i) {      
        oldPos[i] = {i * 1.0f, i * 2.0f}; 
        vel[i] = {1.0f, 1.0f};         
    }

    cudaMemcpy(d_oldPos, oldPos, size_float2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, vel, size_float2, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int i=0;i<10;i++){
        cudaEventRecord(start);
        Compute1_1<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_oldPos, d_newPos, d_vel, count);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsedTime1;
        cudaEventElapsedTime(&elapsedTime1, start, stop);
        std::cout << "Compute1 execution time: " << elapsedTime1 << " ms" << std::endl;
    }

    cudaFree(d_oldPos);
    cudaFree(d_newPos);
    cudaFree(d_vel);

    free(oldPos);
    free(newPos);
    free(vel);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}