#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__device__ float sqr_dist_rgb(float4 a, float4 b) {
    a.x -= b.x, a.y -= b.y, a.z -= b.z;
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__global__ void fragment_shader_old(int palette_lim, float *palette, float *input, float *output, int width, int height) {
    
    int fragment_idx = (3 * (blockIdx.y * blockDim.y + threadIdx.y) * width) + (3 * (blockIdx.x * blockDim.x + threadIdx.x));
    
    if (fragment_idx >= width * height * 3) return;  

    float min_dist = sqrtf(3);
    float color_dist;
    int best_c = 0;
    
    for (int c = 0; c < palette_lim; c++) {
        color_dist = sqrtf(pow(input[fragment_idx] - palette[c * 3], 2) +
                           pow(input[fragment_idx + 1] - palette[c * 3 + 1], 2) +
                           pow(input[fragment_idx + 2] - palette[c * 3 + 2], 2));
        if (color_dist < min_dist) {
            min_dist = color_dist;
            best_c = c;
        }
    }
    output[fragment_idx] = palette[best_c * 3];
    output[fragment_idx + 1] = palette[best_c * 3 + 1];
    output[fragment_idx + 2] = palette[best_c * 3 + 2];
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
    fragment_shader_old<<<gridSize, blockSize>>>(palette_lim, palette, input, output_old, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_old;
    cudaEventElapsedTime(&time_old, start, stop);
    std::cout << "fragment_shader_old time: " << time_old << " ms\n";

    
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