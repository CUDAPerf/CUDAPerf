#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define N 1024*2  // Adjust the number of elements as needed
#define BLOCK_SIZE 256

// Compute1
__global__ void Compute1(double2* p0, double2* p1, double2* v, int count) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    double2 fd = {0.0F, 0.0F};
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

    double2* p0, * p1, * v;
    double2* d_p0, * d_p1, * d_v;

    size_t size_double2 = count * sizeof(double2);

    p0 = (double2*)malloc(size_double2);
    p1 = (double2*)malloc(size_double2);
    v = (double2*)malloc(size_double2);

    cudaMalloc(&d_p0, size_double2);
    cudaMalloc(&d_p1, size_double2);
    cudaMalloc(&d_v, size_double2);

    for (int i = 0; i < count; ++i) {
        p0[i] = {i * 1.0, i * 2.0};
        v[i] = {1.0, 1.0};                 
    }

    cudaMemcpy(d_p0, p0, size_double2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size_double2, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int i=0;i<10;i++){
        cudaEventRecord(start);
        Compute1<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_p0, d_p1, d_v, count);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsedTime1;
        cudaEventElapsedTime(&elapsedTime1, start, stop);
        std::cout << "Compute1 execution time: " << elapsedTime1 << " ms" << std::endl;
    }

    cudaFree(d_p0);
    cudaFree(d_p1);
    cudaFree(d_v);

    free(p0);
    free(p1);
    free(v);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}