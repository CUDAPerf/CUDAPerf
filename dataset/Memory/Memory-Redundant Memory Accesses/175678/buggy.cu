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


__global__ void kernel_applyweightandbias(float *in, float *weights, float *bias, float *out, const int input_size, const int output_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;

    if (tid < input_size) {
        sum = bias[tid];
        for (int i = 0; i < input_size; i++) {
            sum += in[i] * weights[tid * input_size + i];
        }
        out[tid] = sum;
        if (out[tid] <= 0) {
            out[tid] = 0;
        }
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
    float *h_out_original = (float*)malloc(output_size * sizeof(float));

    
    initialize_data(h_in, h_weights, h_bias, input_size, output_size);

    
    float *d_in, *d_weights, *d_bias, *d_out_original;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in, input_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights, output_size * input_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_bias, output_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out_original, output_size * sizeof(float)));

    
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, h_weights, output_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias, h_bias, output_size * sizeof(float), cudaMemcpyHostToDevice));

    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    kernel_applyweightandbias<<<num_blocks, num_threads>>>(d_in, d_weights, d_bias, d_out_original, input_size, output_size);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float time_original;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_original, start, stop));

    
    CHECK_CUDA_ERROR(cudaMemcpy(h_out_original, d_out_original, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    
    std::cout << "Original kernel execution time: " << time_original << " ms" << std::endl;

    
    free(h_in);
    free(h_weights);
    free(h_bias);
    free(h_out_original);
    cudaFree(d_in);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_out_original);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}

int main(){
    //cudaSetDevice(1);
    for(int i=0;i<10;i++){
        test();
    }
    return 0;
}