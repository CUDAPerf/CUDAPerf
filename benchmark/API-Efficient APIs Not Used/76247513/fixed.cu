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

  cudaEvent_t start1, stop1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);

  typedef int mytype;

  thrust::device_vector<mytype> data(N*K);
  thrust::device_vector<mytype> sums(N);

  thrust::sequence(data.begin(),data.end());

  // method 1
  cudaEventRecord(start1, 0);

  thrust::reduce_by_key(thrust::device,
                        
  thrust::make_transform_iterator(thrust::counting_iterator<int>(0),  _1/K),
                        
  thrust::make_transform_iterator(thrust::counting_iterator<int>(N*K),_1/K),
                        data.begin(),
                        thrust::discard_iterator<int>(),
                        sums.begin(),
                        thrust::equal_to<int>(),
                        thrust::minimum<mytype>());
    
    cudaEventRecord(stop1, 0);
    float time1;
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&time1, start1, stop1);
 
    printf("Method 1 Time : %f ms\n", time1);
 
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

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