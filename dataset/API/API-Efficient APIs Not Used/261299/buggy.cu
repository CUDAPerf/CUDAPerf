#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

__global__ void fill1(float* a, size_t num, float value) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < num) {
        a[gtid] = value;
    }
}

int main() {
    cudaSetDevice(1);

    std::size_t num = 1024*1024;
    float* a_d{};
    cudaMalloc(&a_d, sizeof(float) * num);

    int blockSize{};
    int gridSize{};

    float value = 1.0f;
    int loopCount = 10;

    // Method1: Each thread performs a single set operation.
    for (int i = 0; i < loopCount; ++i) {
        cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, fill1);
        gridSize = (num + blockSize - 1) / blockSize;
        fill1<<<gridSize, blockSize>>>(a_d, num, value);
    }

    cudaDeviceSynchronize();
    cudaFree(a_d);
}