#include <cub/cub.cuh>
#include <thrust/copy.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>

#include <random>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <map>
#include <cassert>
#include <string>


//#define USE_DP

struct Op{
    __host__ __device__
    bool operator()(int i) const noexcept{
        return i % 4 == 2;
    }
};

//from thrust example
struct not_my_pointer
{
  not_my_pointer(void* p)
    : message()
  {
    std::stringstream s;
    s << "Pointer `" << p << "` was not allocated by this allocator.";
    message = s.str();
  }

  virtual ~not_my_pointer() {}

  virtual const char* what() const
  {
    return message.c_str();
  }

private:
  std::string message;
};



// A simple allocator for caching cudaMalloc allocations.
struct cached_allocator
{
  typedef char value_type;

  cached_allocator() {}

  ~cached_allocator()
  {
    free_all();
  }

  char *allocate(std::ptrdiff_t num_bytes)
  {
    char *result = 0;

    // Search the cache for a free block.
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if (free_block != free_blocks.end())
    {
      result = free_block->second;

      // Erase from the `free_blocks` map.
      free_blocks.erase(free_block);
    }
    else
    {
      // No allocation of the right size exists, so create a new one with
      // `thrust::cuda::malloc`.
      try
      {
        // Allocate memory and convert the resulting `thrust::cuda::pointer` to
        // a raw pointer.
        result = thrust::cuda::malloc<char>(num_bytes).get();
      }
      catch (std::runtime_error&)
      {
        throw;
      }
    }

    // Insert the allocated pointer into the `allocated_blocks` map.
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return result;
  }

  void deallocate(char *ptr, size_t)
  {
    // Erase the allocated block from the allocated blocks map.
    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);

    if (iter == allocated_blocks.end())
      throw not_my_pointer(reinterpret_cast<void*>(ptr));

    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // Insert the block into the free blocks map.
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

private:
  typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
  typedef std::map<char*, std::ptrdiff_t>      allocated_blocks_type;

  free_blocks_type      free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all()
  {

    // Deallocate all outstanding blocks in both lists.
    for ( free_blocks_type::iterator i = free_blocks.begin()
        ; i != free_blocks.end()
        ; ++i)
    {
      // Transform the pointer to cuda::pointer before calling cuda::free.
      thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
    }

    for( allocated_blocks_type::iterator i = allocated_blocks.begin()
       ; i != allocated_blocks.end()
       ; ++i)
    {
      // Transform the pointer to cuda::pointer before calling cuda::free.
      thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
    }
  }
};

int main()
{
    constexpr int N = 1024*1024/512;
    constexpr int numarrays = 100;
    constexpr int blocksize = 128;
    constexpr int gridsize = (numarrays + blocksize - 1) / blocksize;

    std::mt19937 gen(42);
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,50000);

    cudaError_t status = cudaSuccess;
    
    std::vector<int> h_data(N);
    for(int i = 0; i < N; i++){
        h_data[i] = dist(gen);
    }

    std::vector<int*> d_data1vec(numarrays); 
    std::vector<int*> d_data2vec(numarrays);
    for(int i = 0; i < numarrays; i++){
        cudaMalloc(&d_data1vec[i], sizeof(int) * N);
        cudaMalloc(&d_data2vec[i], sizeof(int) * N);
        cudaMemcpy(d_data1vec[i], h_data.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
    }

    std::vector<int> h_numselected(numarrays);
    int* d_numselected; cudaMalloc(&d_numselected, sizeof(int) * numarrays);

    

    cudaEvent_t event1; cudaEventCreate(&event1);
    cudaEvent_t event2; cudaEventCreate(&event2);

    cudaEventRecord(event1);
    for(int k = 0; k < numarrays; k++){
        thrust::copy_if(thrust::device,d_data1vec[k], d_data1vec[k] + N, d_data2vec[k], Op{});
    }
    cudaEventRecord(event2);
    cudaEventSynchronize(event2);
    float time = 0;
    cudaEventElapsedTime(&time, event1, event2);
    std::cerr << "normal thrust took " << time << " ms\n";

    return 0;
}