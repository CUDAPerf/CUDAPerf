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


__global__ void Compute2(float2* p0, float2* p1, float2* v, int count){
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    float2 fd = {0.0F, 0.0F};
#pragma unroll 32
    for (auto j = 0; j < count; ++j){
        const auto dx = p0[i].x - p0[j].x;
        const auto dy = p0[i].y - p0[j].y;
        const auto f = 0.0000000001F/(dx*dx + dy*dy + 0.000000000001F);
        fd.x += dx * f;
        fd.y += dy * f;
    }
    p1[i].x = p0[i].x + (v[i].x -= fd.x);
    p1[i].y = p0[i].y + (v[i].y -= fd.y);
}

// Compute2
__global__ void Compute3(float2* __restrict__ oldPos, float2* __restrict__ newPos, float2* vel, int count) {
    using namespace cooperative_groups;
    const auto cta = this_thread_block();
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    auto me = oldPos[index];
    __shared__ float2 them[64];
    float2 acc = {0.0F, 0.0F};
    for (auto tile = 0U; tile < blockDim.x; ++tile) {
        them[threadIdx.x] = oldPos[tile * blockDim.x + threadIdx.x];
        sync(cta);
#pragma unroll 32
        for (auto counter = 0U; counter < blockDim.x; ++counter) {
            float2 r;
            r.x = me.x - them[counter].x;
            r.y = me.y - them[counter].y;
            const auto f = 0.00000001F / (r.x * r.x + r.y * r.y + 0.000000000001F);
            acc.x += r.x * f;
            acc.y += r.y * f;
        }
        sync(cta);
    }
    me.x += vel[index].x -= acc.x;
    me.y += vel[index].y -= acc.y;
    newPos[index] = me;
}

int main() {
    cudaSetDevice(1);
    int count = N;

    double2* p0, * p1, * v;
    float2* oldPos, * newPos, * vel;
    double2* d_p0, * d_p1, * d_v;
    float2* d_oldPos, * d_newPos, * d_vel;

    size_t size_double2 = count * sizeof(double2);
    size_t size_float2 = count * sizeof(float2);

    p0 = (double2*)malloc(size_double2);
    p1 = (double2*)malloc(size_double2);
    v = (double2*)malloc(size_double2);

    oldPos = (float2*)malloc(size_float2);
    newPos = (float2*)malloc(size_float2);
    vel = (float2*)malloc(size_float2);

    cudaMalloc(&d_p0, size_double2);
    cudaMalloc(&d_p1, size_double2);
    cudaMalloc(&d_v, size_double2);

    cudaMalloc(&d_oldPos, size_float2);
    cudaMalloc(&d_newPos, size_float2);
    cudaMalloc(&d_vel, size_float2);

    for (int i = 0; i < count; ++i) {
        p0[i] = {i * 1.0, i * 2.0};
        v[i] = {1.0, 1.0};        
        oldPos[i] = {i * 1.0f, i * 2.0f}; 
        vel[i] = {1.0f, 1.0f};         
    }

    cudaMemcpy(d_p0, p0, size_double2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size_double2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldPos, oldPos, size_float2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, vel, size_float2, cudaMemcpyHostToDevice);

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

    for(int i=0;i<10;i++){
        cudaEventRecord(start);
        Compute1_1<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_oldPos, d_newPos, d_vel, count);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsedTime1;
        cudaEventElapsedTime(&elapsedTime1, start, stop);
        std::cout << "Compute1 execution time: " << elapsedTime1 << " ms" << std::endl;
    }

    for(int i=0;i<10;i++){
        cudaEventRecord(start);
        Compute2<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_oldPos, d_newPos, d_vel, count);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsedTime2;
        cudaEventElapsedTime(&elapsedTime2, start, stop);
        std::cout << "Compute2 execution time: " << elapsedTime2 << " ms" << std::endl;
    }

    cudaFree(d_p0);
    cudaFree(d_p1);
    cudaFree(d_v);
    cudaFree(d_oldPos);
    cudaFree(d_newPos);
    cudaFree(d_vel);

    free(p0);
    free(p1);
    free(v);
    free(oldPos);
    free(newPos);
    free(vel);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}