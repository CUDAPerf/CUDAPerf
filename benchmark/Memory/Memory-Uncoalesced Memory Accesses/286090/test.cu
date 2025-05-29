#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>

#define CHECK_CUDA_ERROR(ans) { assert_cuda( (ans), __LINE__); }

void assert_cuda
(
    cudaError_t error,
    int         line
);

__host__
cudaError_t copy_cuda
(
    void*  dst,
    void*  src,
    size_t bytes
);

__host__
void* malloc_device
(
    size_t bytes
);

__host__
cudaError_t free_device
(
    void* ptr
);

__global__
void store_kernel
(
    int* d_array_load,
    int* d_array_store
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int load_idx = 4 * idx;

    int buf[4];

    buf[0] = d_array_load[load_idx + 0];
    buf[1] = d_array_load[load_idx + 1];
    buf[2] = d_array_load[load_idx + 2];
    buf[3] = d_array_load[load_idx + 3];

    d_array_store[idx] = buf[0] + buf[1] + buf[2] + buf[3];
}

__global__
void store_kernel_warp
(
    int* d_array_load,
    int* d_array_store
)
{
    const int tidx              = threadIdx.x;
    const int idx               = blockIdx.x * blockDim.x + tidx;
    const int warp_id           = idx / 32;
    const int lane_id           = tidx % 32;
    const int starting_load_idx = warp_id * 128;

    int buf[4];

    for (int i = 0; i < 4; i++)
    {
        int load_idx         = starting_load_idx + i * 32 + lane_id;
        int starting_lane_id = i * 8;

        int val = d_array_load[load_idx];

        buf[0] = __shfl_sync(0xffffffff, val, 4 * (lane_id % 8) + 0);
        buf[1] = __shfl_sync(0xffffffff, val, 4 * (lane_id % 8) + 1);
        buf[2] = __shfl_sync(0xffffffff, val, 4 * (lane_id % 8) + 2);
        buf[3] = __shfl_sync(0xffffffff, val, 4 * (lane_id % 8) + 3);

        if (lane_id >= starting_lane_id && lane_id < starting_lane_id + 8)
        {
            d_array_store[idx] = buf[0] + buf[1] + buf[2] + buf[3];
        }
    }
}

__global__
void store_kernel_vector
(
    int* d_array_load,
    int* d_array_store
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int4 vec = reinterpret_cast<int4*>(d_array_load)[idx];

    d_array_store[idx] = vec.x + vec.y + vec.z + vec.w;
}

int main
(
    int    argc,
    char** argv
)
{
    cudaSetDevice(1);
    const int num_elems_load  = 1024*1024*512;
    const int num_elems_store = num_elems_load/4;

    const int block_size  = 256;
    const int num_threads = num_elems_store;
    const int num_blocks  = num_threads / block_size;

    size_t bytes_load  = num_elems_load  * sizeof(int);
    size_t bytes_store = num_elems_store * sizeof(int);

    int* h_array_load  = new int[num_elems_load];
    int* h_array_store = new int[num_elems_store];
    int* d_array_load  = (int*)malloc_device(bytes_load);
    int* d_array_store = (int*)malloc_device(bytes_store);

    for (int i = 0; i < num_elems_load; i++)
    {
         h_array_load[i] = i % 4;
    }

    for(int i=0;i<10;i++){
        copy_cuda(d_array_load, h_array_load, bytes_load);
        store_kernel<<<num_blocks, block_size>>>(d_array_load, d_array_store);
        store_kernel_warp<<<num_blocks, block_size>>>(d_array_load, d_array_store);
        store_kernel_vector<<<num_blocks, block_size>>>(d_array_load, d_array_store);
    }
    
    
    copy_cuda(h_array_store, d_array_store, bytes_store);
    
    for (int i = 0; i < num_elems_store; i++)
    {
        if (h_array_store[i] != 6)
        {
            fprintf(stderr, "Sum is not 6.\n");
            exit(-1);
        }
    }

    delete[] h_array_load;
    delete[] h_array_store;
    free_device(d_array_load);
    free_device(d_array_store);

    return 0;
}

void assert_cuda(cudaError_t error, int line)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s, %d\n", cudaGetErrorString(error), line);

        exit(error);
    }
}

cudaError_t copy_cuda
(
    void*  dst,
    void*  src,
    size_t bytes
)
{
    cudaError_t error = cudaMemcpy
    (
        dst,
        src,
        bytes,
        cudaMemcpyDefault
    );

    return error;
}

__host__
void* malloc_device
(
    size_t bytes
)
{
    void* ptr;
    
    cudaMalloc
    (
        &ptr,
        bytes
    );

    return ptr;
}

__host__
cudaError_t free_device
(
    void* ptr
)
{
    return (nullptr != ptr) ? cudaFree(ptr) : cudaSuccess;
}