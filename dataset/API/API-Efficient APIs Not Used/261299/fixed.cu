#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>


int main() {
    cudaSetDevice(1);

    std::size_t num = 1024*1024;
    float* a_d{};
    cudaMalloc(&a_d, sizeof(float) * num);

    float value = 1.0f;
    int loopCount = 10;

    // Method 6: Use thrust API.
    for (int i = 0; i < loopCount; ++i) {
        thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(a_d);
        thrust::fill(dev_ptr, dev_ptr + num, value);
    }

    cudaDeviceSynchronize();
    cudaFree(a_d);
}