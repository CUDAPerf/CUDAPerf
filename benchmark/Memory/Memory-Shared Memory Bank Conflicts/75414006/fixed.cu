#include <cuda_runtime.h>

const int rows = 1024*64;
const int columns = 1024*8;
const int tile_dim = 32;

__global__ void transpose(float* in, float* out)
{
  __shared__ float tile[tile_dim][tile_dim + 1];

  int x = blockIdx.x * tile_dim + threadIdx.x;
  int y = blockIdx.y * tile_dim + threadIdx.y;

  tile[threadIdx.y][threadIdx.x] = in[y * columns + x];

  __syncthreads();

  x = blockIdx.y * tile_dim + threadIdx.x;
  y = blockIdx.x * tile_dim + threadIdx.y;

  out[y * columns + x] = tile[threadIdx.x][threadIdx.y];
}

int main()
{
  //cudaSetDevice(1);
  float *in, *out;

  size_t size = rows * columns * sizeof(float);
  cudaMalloc(&in, size);
  cudaMalloc(&out, size);

  dim3 grid(rows / tile_dim, columns / tile_dim);
  dim3 block(tile_dim, tile_dim);
  for(int i =0;i<10;i++){
    transpose<<<grid, block>>>(in, out);
  }

  cudaDeviceSynchronize();

  return 0;
}