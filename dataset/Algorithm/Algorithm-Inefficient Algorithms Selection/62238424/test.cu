#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <thrust/execution_policy.h>
#include <time.h>
#include <sys/time.h>
#include <cstdlib>
#define USECPSEC 1000000ULL
long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

struct RadiosityData {
#ifdef USE_VEC
 float4 emission;
 float4 radiosity;
#else
 float emission[4];
 float radiosity[4];
#endif
        float nPixLight;
 float nPixCam;
        float __padding[2];
};

__global__ void copyKernel(RadiosityData *d, float *f, int *i, int n){
 int idx=threadIdx.x+blockDim.x*blockIdx.x;
 if (idx < n){
  f[idx] = d[idx].nPixCam;
  i[idx] = idx;}
}

__host__ __device__ bool operator<(const RadiosityData &lhs, const RadiosityData  &rhs) { return (lhs.nPixCam > rhs.nPixCam); };
struct my_sort_functor
{
 template <typename T1, typename T2>
 __host__ __device__ bool operator()(T1 lhs, T2 rhs) { return (lhs.nPixCam > rhs.nPixCam); };
};
const int N = 1024*64;
int main(){
 RadiosityData *GPUpointer_ssbo, *o;
 int sz = N*sizeof(RadiosityData);
 thrust::device_vector<RadiosityData> ii(N);
 GPUpointer_ssbo = thrust::raw_pointer_cast(ii.data());
 thrust::device_ptr<RadiosityData> dev_ptr = thrust::device_pointer_cast(GPUpointer_ssbo);
//method 1: ordinary thrust sort
 long long dt = dtime_usec(0);
 for(int i=0;i<10;i++){
  thrust::sort(dev_ptr, dev_ptr+N);
  cudaDeviceSynchronize();
 }
 dt = dtime_usec(dt);
 std::cout << "ordinary sort time: " << dt << "us" << std::endl;

//method 2: reduced sort-and-copy
 cudaMalloc(&o, sz);
 thrust::device_ptr<RadiosityData> dev_optr = thrust::device_pointer_cast(o);
 for (int i = 0; i < N; i++) {RadiosityData q{0}; q.nPixCam = rand(); ii[i] = q;}
 float *d;
 int *k;
 cudaMalloc(&d, N*sizeof(float));
 cudaMalloc(&k, N*sizeof(int));
 thrust::device_ptr<int> dev_kptr = thrust::device_pointer_cast(k);
 cudaDeviceSynchronize();
 dt = dtime_usec(0);
 for(int i=0;i<10;i++){
  copyKernel<<<(N+511)/512, 512>>>(GPUpointer_ssbo, d, k, N);
  thrust::sort_by_key(thrust::device, d, d+N, k);
  thrust::copy(thrust::make_permutation_iterator(dev_ptr, dev_kptr), thrust::make_permutation_iterator(dev_ptr, dev_kptr+N), dev_optr);
  cudaMemcpy(GPUpointer_ssbo, o, sz, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
 }
 dt = dtime_usec(dt);
 std::cout << "sort+copy time: " << dt << "us" << std::endl;
}