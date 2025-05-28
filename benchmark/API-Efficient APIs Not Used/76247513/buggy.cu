#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <iostream>
#include <limits>
#include <cuda_runtime.h>

using namespace thrust::placeholders;
int test() {
  int N = 1024;
  int K = 256;

  std::cout << "N " << N << " K " << K << std::endl;
  
  cudaEvent_t start2, stop2;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);

  typedef int mytype;

  thrust::device_vector<mytype> data(N*K);
  thrust::device_vector<mytype> sums(N);

  thrust::sequence(data.begin(),data.end());


  // method 2 (bad)
  cudaEventRecord(start2, 0);
  for (int i=0; i<N; i++) {
    int res = thrust::reduce(data.begin()+K*i, data.begin()+K*i+K,std::numeric_limits<mytype>::max(),thrust::minimum<mytype>());
  }
  cudaEventRecord(stop2, 0);
    float time2;
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&time2, start2, stop2);
 
    printf("Method 2 Time : %f ms\n", time2);
 
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

  // just print the first 10 results
  thrust::copy_n(sums.begin(),10,std::ostream_iterator<mytype>(std::cout, ","));
  std::cout << std::endl;

  return 0;
}

int main(){
  for(int i=0;i<10;i++){
    test();
  }
}