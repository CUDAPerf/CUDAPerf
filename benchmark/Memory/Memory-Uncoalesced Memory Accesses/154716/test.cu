#include <chrono>
#include <iostream>

using namespace std;

#define checkCudaErrors( err ) \
  if( err != cudaSuccess ) { \
    std::cerr << "ERROR: " << cudaGetErrorString( err ) << std::endl; \
    exit( -1 ); \
  }

const int numStrides = 1024*512;
const int numThreads = 256;

// Traditional access, whole warp/block coalesced read/write
__global__
void access01(float* d_mem) {
    for (int i = 0; i < numStrides; i++) {
        const int idx = threadIdx.x + blockDim.x * i;
        float v = d_mem[idx];
        d_mem[idx] = v + 1;
    }
}

// Bad access pattern, thread 0 read/write 0, thread 1 read/write 256, etc..
__global__
void access02(float* d_mem) {
    for (int i = 0; i < numStrides; i++) {
        const int idx = threadIdx.x * numStrides + i;
        float v = d_mem[idx];
        d_mem[idx] = v + 1;
    }
}

// Tuned access, I read that coalesced access only needs to be coalesced based on 32 byte chunks
// This access pattern has the first 8 threads read/write to same chunk and then go to the 32 bytes right next to it
// second 8 threads read/write continuing where first 8 threads will stop after loop
__global__
void access03(float* d_mem) {
    const int numFloatsChunked = 8;

    const int chunk = threadIdx.x / numFloatsChunked;
    const int tid = threadIdx.x % numFloatsChunked;

    for (int i = 0; i < numStrides; i++) {
        const int idx = tid + numFloatsChunked * numStrides * chunk + i*numFloatsChunked;
        float v = d_mem[idx];
        d_mem[idx] = v + 1;
    }
}

void runAccess01(float* d_mem) {
    // ACCESS 01
    //access01<<<1,numThreads>>>(d_mem);
    checkCudaErrors( cudaDeviceSynchronize() );

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        access01<<<256,numThreads>>>(d_mem);
        checkCudaErrors( cudaDeviceSynchronize() );
    }
    checkCudaErrors( cudaDeviceSynchronize() );
    auto end = std::chrono::high_resolution_clock::now();

    cout << "Elapsed time in microseconds 01 : "
    << chrono::duration_cast<chrono::microseconds>(end - start).count()
    << " us" << endl;
}

void runAccess02(float* d_mem) {
    // ACCESS 02
    //access02<<<1,numThreads>>>(d_mem);
    checkCudaErrors( cudaDeviceSynchronize() );

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        access02<<<256,numThreads>>>(d_mem);
        checkCudaErrors( cudaDeviceSynchronize() );
    }
    checkCudaErrors( cudaDeviceSynchronize() );
    auto end = std::chrono::high_resolution_clock::now();

    cout << "Elapsed time in microseconds 02 : "
    << chrono::duration_cast<chrono::microseconds>(end - start).count()
    << " us" << endl;
}


int main() {
    cudaSetDevice(1);
    const int numElems = numStrides * numThreads;
    const int elemSize = numElems * sizeof(float);

    float* h_mem = (float *) malloc(elemSize);
    float* d_mem;
    checkCudaErrors( cudaMalloc((void **) &d_mem, elemSize) );

    for (int i = 0; i < numElems; i++) {
        h_mem[i] = 0;
    }

    checkCudaErrors( cudaMemcpy(d_mem, h_mem, elemSize, cudaMemcpyHostToDevice) );

    runAccess02(d_mem);
    runAccess01(d_mem);

    checkCudaErrors( cudaMemcpy(h_mem, d_mem, elemSize, cudaMemcpyDeviceToHost) );

    const float v = h_mem[0];
    for (int i = 0; i < numElems; i++) {
        if (v != h_mem[i]) {
            cout << "err" << endl;
            exit(1);
        }
    }
    cout << endl;
    cout << "all are " << v << endl;
}