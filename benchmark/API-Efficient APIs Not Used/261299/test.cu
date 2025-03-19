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

__global__ void fill2(float* a, size_t num, float value) {
    float4* aAlt = reinterpret_cast<float4*>(a);

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t numFloor = num / 4 * 4;
    size_t c = numFloor / 4;

    if (gtid < c) {
        aAlt[gtid] = make_float4(value, value, value, value);
    }

    if (gtid == 0) {
        for (int i = 0; i < num - numFloor; ++i) {
            a[num - 1 - i] = value;
        }
    }
}

__global__ void fill3(float* a, size_t num, float value) {
    int numThreads = blockDim.x * gridDim.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = gtid; i < num; i += numThreads) {
        a[i] = value;
    }
}

__global__ void fill4(float* a, size_t num, float value) {
    float4* aAlt = reinterpret_cast<float4*>(a);

    int numThreads = blockDim.x * gridDim.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t numFloor = num / 4 * 4;
    size_t c = numFloor / 4;

    for (int i = gtid; i < c; i += numThreads) {
        aAlt[i] = make_float4(value, value, value, value);
    }

    if (gtid == 0) {
        for (int i = 0; i < num - numFloor; ++i) {
            a[num - 1 - i] = value;
        }
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

    // Method 2: On top of method 1, replace scalar store with vector store.
    for (int i = 0; i < loopCount; ++i) {
        cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, fill2);
        size_t numFloor = num / 4 * 4;
        size_t c = numFloor / 4;
        gridSize = (c + blockSize - 1) / blockSize;
        fill2<<<gridSize, blockSize>>>(a_d, num, value);
    }


    // Method 6: Use thrust API.
    for (int i = 0; i < loopCount; ++i) {
        thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(a_d);
        thrust::fill(dev_ptr, dev_ptr + num, value);
    }

    cudaDeviceSynchronize();
    cudaFree(a_d);
}