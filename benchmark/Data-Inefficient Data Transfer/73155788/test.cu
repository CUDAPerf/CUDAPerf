#include <iostream>
#include <chrono>

const size_t mb20 = 1024 * 1024 * 512;


    cudaError_t allocate(void** ptr, size_t size, cudaStream_t stream){
        return cudaMallocAsync(ptr, size, stream);
    }

    cudaError_t deallocate(void* ptr, cudaStream_t stream){
        return cudaFreeAsync(ptr, stream);
    }


    cudaError_t allocate2(void** ptr, size_t size, cudaStream_t){
        return cudaMalloc(ptr, size);
    }

    cudaError_t deallocate2(void* ptr, cudaStream_t){
        return cudaFree(ptr);
    }


__global__
void computeNextSize(size_t* size, size_t growBy){
    *size = *size + growBy;
}

void method1(){
    cudaStream_t stream = cudaStreamPerThread;

    size_t* h_size = nullptr;
    size_t* d_size = nullptr;

    cudaMalloc(&d_size, sizeof(size_t));
    cudaMallocHost(&h_size, sizeof(size_t));
    *h_size = mb20;
    cudaMemcpyAsync(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice, stream);

    cudaMemPool_t memPool;
    cudaDeviceGetMemPool(&memPool, 0);
    size_t setVal = UINT64_MAX;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

    void* ptr = nullptr;
    size_t size = mb20;
    allocate(&ptr, size, stream);
    //std::cout << "size: " << size << ", ptr = " << ptr << "\n";
    cudaMemsetAsync(ptr, 0, size); //work with ptr
    computeNextSize<<<1,1,0,stream>>>(d_size, mb20);
    cudaMemcpyAsync(h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); //wait for computation and transfer of next size


    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for(int i = 0; i < 50; i++){
        deallocate(ptr, stream);
        size = *h_size;
        allocate(&ptr, size, stream);
        //std::cout << "size: " << size << ", ptr = " << ptr << "\n";
        cudaMemsetAsync(ptr, 0, size); //work with ptr
        computeNextSize<<<1,1,0,stream>>>(d_size, mb20);
        cudaMemcpyAsync(h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); //wait for computation and transfer of next size
    }
    deallocate(ptr, stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "method1: " << elapsedTime << " ms" << std::endl;
}

void method2(){
    cudaStream_t stream = cudaStreamPerThread;

    size_t* h_size = nullptr;
    size_t* d_size = nullptr;

    cudaMalloc(&d_size, sizeof(size_t));
    cudaMallocHost(&h_size, sizeof(size_t));
    *h_size = mb20;
    cudaMemcpyAsync(d_size, h_size, sizeof(size_t), cudaMemcpyHostToDevice, stream);

    cudaMemPool_t memPool;
    cudaDeviceGetMemPool(&memPool, 0);
    size_t setVal = UINT64_MAX;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

    void* ptr = nullptr;
    size_t size = mb20;
    allocate2(&ptr, size, stream);
    //std::cout << "size: " << size << ", ptr = " << ptr << "\n";
    cudaMemsetAsync(ptr, 0, size); //work with ptr
    computeNextSize<<<1,1,0,stream>>>(d_size, mb20);
    cudaMemcpyAsync(h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); //wait for computation and transfer of next size


    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for(int i = 0; i < 50; i++){
        deallocate2(ptr, stream);
        size = *h_size;
        allocate2(&ptr, size, stream);
        //std::cout << "size: " << size << ", ptr = " << ptr << "\n";
        cudaMemsetAsync(ptr, 0, size); //work with ptr
        computeNextSize<<<1,1,0,stream>>>(d_size, mb20);
        cudaMemcpyAsync(h_size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); //wait for computation and transfer of next size
    }
    deallocate2(ptr, stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "method2: " << elapsedTime << " ms" << std::endl;
}

int main(){
    cudaSetDevice(1);
    method1();
    method2();
}