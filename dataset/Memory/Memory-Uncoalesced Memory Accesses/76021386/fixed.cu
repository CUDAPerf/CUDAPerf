#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__device__ float sqr_dist_rgb(float4 a, float4 b) {
    a.x -= b.x, a.y -= b.y, a.z -= b.z;
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__global__ void fragment_shader(int palette_lim, const float4 *palette, const float4 *input, float4 *output, int width, int height) {
    extern __shared__ float4 colorbuf[];
    const int buf_size = blockDim.x * blockDim.y;
    const int buf_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const int fragment_idx = (blockIdx.y * blockDim.y + threadIdx.y) * width + (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (fragment_idx >= width * height) return;  

    const float4 inputcolor = input[fragment_idx];
    float4 best_color = __ldg(palette);
    const float min_dist_sqr = 3.f;
    float best_dist = sqr_dist_rgb(best_color, inputcolor);
    
    for (int cb = 0, b = 0; cb < palette_lim; b ^= 1, cb += buf_size) {
        float4* cur_buf = b ? colorbuf + buf_size : colorbuf;
        if (cb + buf_idx < palette_lim)
            cur_buf[buf_idx] = __ldg(palette + cb + buf_idx);
        __syncthreads();
        
        const int n = min(buf_size, palette_lim - cb);
        for (int c = 0; c < n; c++) {
            float4 color = cur_buf[c];
            float dist = sqr_dist_rgb(color, inputcolor);
            if (dist < min_dist_sqr && dist < best_dist) {
                best_color = color;
                best_dist = dist;
            }
        }
    }
    output[fragment_idx] = best_color;
}

void run_cuda_test(int palette_lim, int width, int height) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    float *palette, *input, *output_old, *output_new;
    float4 *palette4, *input4, *output4;

    cudaMallocManaged(&palette, palette_lim * 3 * sizeof(float));
    cudaMallocManaged(&input, width * height * 3 * sizeof(float));
    cudaMallocManaged(&output_old, width * height * 3 * sizeof(float));
    cudaMallocManaged(&output_new, width * height * 3 * sizeof(float));

    cudaMallocManaged(&palette4, palette_lim * sizeof(float4));
    cudaMallocManaged(&input4, width * height * sizeof(float4));
    cudaMallocManaged(&output4, width * height * sizeof(float4));

    for (int i = 0; i < palette_lim; i++) {
        palette[i * 3] = rand() % 256;
        palette[i * 3 + 1] = rand() % 256;
        palette[i * 3 + 2] = rand() % 256;
        palette4[i] = make_float4(palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2], 0.0f);
    }
    for (int i = 0; i < width * height; i++) {
        input[i * 3] = rand() % 256;
        input[i * 3 + 1] = rand() % 256;
        input[i * 3 + 2] = rand() % 256;
        input4[i] = make_float4(input[i * 3], input[i * 3 + 1], input[i * 3 + 2], 0.0f);
    }

    
    dim3 blockSize(16, 16);  // 16x16 
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);


    cudaEventRecord(start);
    fragment_shader<<<gridSize, blockSize, 2 * blockSize.x * blockSize.y * sizeof(float4)>>>(palette_lim, palette4, input4, output4, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_new;
    cudaEventElapsedTime(&time_new, start, stop);
    std::cout << "fragment_shader time: " << time_new << " ms\n";

    
    cudaFree(palette);
    cudaFree(input);
    cudaFree(output_old);
    cudaFree(output_new);
    cudaFree(palette4);
    cudaFree(input4);
    cudaFree(output4);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int palette_lim = 256;    
    int width = 1024;          
    int height = 1024;         

    for(int i = 0; i < 10; i++){
        run_cuda_test(palette_lim, width, height);
    }


    return 0;
}