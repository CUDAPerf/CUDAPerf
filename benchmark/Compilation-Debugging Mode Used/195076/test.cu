#include<cuda_runtime.h>

__global__ void clear_buffer(float4* accum_buffer, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        accum_buffer[i] = make_float4(0, 0, 0, 0);
    }
}

int main() {
    float4* buffer;
    int grid_size = 7579;
    int block_size = 128;
    int n = grid_size * block_size;

    cudaMalloc(&buffer, n * 4 * sizeof(float));
    for(int i=0;i<10;i++)
        clear_buffer<<<grid_size, block_size>>>(buffer, n);

    cudaFree(buffer);
}