#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call)                                             \
    {                                                                      \
        const cudaError_t error = call;                                     \
        if (error != cudaSuccess) {                                         \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "   \
                      << cudaGetErrorString(error) << std::endl;           \
            exit(1);                                                        \
        }                                                                  \
    }


__global__ void optimized_kernel_applyweightandbias(float *in, float *weights, float *bias, float *out, const int input_size, const int output_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;

    if (tid < input_size) {
        sum = bias[tid];
        for (int i = 0; i < input_size; i++) {
            sum += in[i] * weights[tid * input_size + i];
        }
        if (sum < 0) sum = 0;
        out[tid] = sum;
    }
}


void initialize_data(float* in, float* weights, float* bias, int input_size, int output_size) {
    for (int i = 0; i < input_size; i++) {
        in[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < output_size * input_size; i++) {
        weights[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < output_size; i++) {
        bias[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}


int test() {
    const int input_size = 1024;
    const int output_size = 512;
    const int num_threads = 256;
    const int num_blocks = (output_size + num_threads - 1) / num_threads;

    
    float *h_in = (float*)malloc(input_size * sizeof(float));
    float *h_weights = (float*)malloc(output_size * input_size * sizeof(float));
    float *h_bias = (float*)malloc(output_size * sizeof(float));
    float *h_out_optimized = (float*)malloc(output_size * sizeof(float));

    
    initialize_data(h_in, h_weights, h_bias, input_size, output_size);

    
    float *d_in, *d_weights, *d_bias, *d_out_optimized;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in, input_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights, output_size * input_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_bias, output_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out_optimized, output_size * sizeof(float)));

    
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, h_weights, output_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias, h_bias, output_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start2, stop2;
    CHECK_CUDA_ERROR(cudaEventCreate(&start2));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop2));

    
    CHECK_CUDA_ERROR(cudaEventRecord(start2));
    optimized_kernel_applyweightandbias<<<num_blocks, num_threads>>>(d_in, d_weights, d_bias, d_out_optimized, input_size, output_size);
    CHECK_CUDA_ERROR(cudaEventRecord(stop2));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop2));
    float time_optimized;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_optimized, start2, stop2));

    CHECK_CUDA_ERROR(cudaMemcpy(h_out_optimized, d_out_optimized, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Optimized kernel execution time: " << time_optimized << " ms" << std::endl;

    
    free(h_in);
    free(h_weights);
    free(h_bias);
    free(h_out_optimized);
    cudaFree(d_in);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_out_optimized);

    return 0;
}

int main(){
    //cudaSetDevice(1);
    for(int i=0;i<10;i++){
        test();
    }
    return 0;
}