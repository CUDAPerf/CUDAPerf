#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void setupCurandStates(curandState *states, unsigned long long seed, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void dropr(float *A, curandState *globalstate, uint64_t N, float R) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        curandState localstate = globalstate[i];
        A[i] *= curand_uniform(&localstate) < R ? 0 : 1;
        globalstate[i] = localstate;
    }
}

void measureKernelTime(float *A, curandState *states, uint64_t N, float R) {
    float *d_A;
    curandState *d_states;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_states, N * sizeof(curandState));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);

    // Setup random states
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    setupCurandStates<<<blocks, threadsPerBlock>>>(d_states, 1234, N);

    // Measure time for dropr
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    dropr<<<blocks, threadsPerBlock>>>(d_A, d_states, N, R);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float time1;
    cudaEventElapsedTime(&time1, start, stop);
    std::cout << "Time for dropr: " << time1 << " ms" << std::endl;

    // Reset states and measure time for dropr2
    cudaMemcpy(d_states, d_states, N * sizeof(curandState), cudaMemcpyDeviceToDevice);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    //cudaSetDevice(1);
    const uint64_t N = 1 << 17; // Size of the array
    float R = 0.5f;
    float *A = new float[N];
    curandState *states = new curandState[N];

    // Initialize input data
    for (uint64_t i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i) / N;
    }

    for(int i=0;i<10;i++){
        measureKernelTime(A, states, N, R);
    }


    delete[] A;
    delete[] states;
    return 0;
}