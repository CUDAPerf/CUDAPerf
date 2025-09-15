#include <iostream>
#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start=0){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

size_t sz = 1024*1024*512;

template <typename T>
__global__ void write_kernel(T *d, size_t s, T val){
  for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < s; i+=gridDim.x*blockDim.x)
    d[i] = val;
}

template <typename T>
__global__ void read_kernel(const T *d, size_t s, T tval, T *r){
  T val = 0;
  for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < s; i+=gridDim.x*blockDim.x)
    val += d[i];
  if (val == tval) *r = val;
}


using mt = float;

int test1(){
  mt *d = new mt[sz];
  cudaHostRegister(d, sizeof(*d)*sz, /* cudaHostRegisterPortable | */  cudaHostAllocMapped);
  mt *r;
  cudaMalloc(&r, sizeof(mt));
  memset(d, 0, sizeof(*d)*sz);
  cudaMemset(r, 0, sizeof(*r));
  // warm-up
  write_kernel<<<3*58, 512>>>(d, sz, 1.0f);
  cudaDeviceSynchronize();
  read_kernel<<<3*58, 512>>>(d, sz, 1.0f, r);
  cudaDeviceSynchronize();

  unsigned long long dt = dtime_usec(0);
  write_kernel<<<3*58, 512>>>(d, sz, 1.0f);
  cudaDeviceSynchronize();
  dt = dtime_usec(dt);
  std::cout << "write kernel time: " << dt << "μs" << std::endl;
  unsigned long long dt1 = dt;
  dt = dtime_usec(0);
  read_kernel<<<3*58, 512>>>(d, sz, 1.0f, r);
  cudaDeviceSynchronize();
  dt = dtime_usec(dt);
  std::cout << "read kernel time:  " << dt << "μs" << std::endl;
  std::cout << "all kernel time:  " << dt1+dt << "μs" << std::endl;
  return 0;
}

int main(){
  for(int i=0; i<10; i++){
    test1();
  }
}