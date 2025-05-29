#include <thrust/device_vector.h>

#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <iomanip>

#include <thrust/transform.h>
#include <thrust/functional.h>


/////////  https://github.com/NVIDIA/thrust/blob/master/examples/expand.cu //////////
template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator>
OutputIterator expand(InputIterator1 first1,
                      InputIterator1 last1,
                      InputIterator2 first2,
                      OutputIterator output)
{
  typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;
  
  difference_type input_size  = thrust::distance(first1, last1);
  difference_type output_size = thrust::reduce(first1, last1);

  // scan the counts to obtain output offsets for each input element
  thrust::device_vector<difference_type> output_offsets(input_size, 0);
  thrust::exclusive_scan(first1, last1, output_offsets.begin()); 

  // scatter the nonzero counts into their corresponding output positions
  thrust::device_vector<difference_type> output_indices(output_size, 0);
  thrust::scatter_if
    (thrust::counting_iterator<difference_type>(0),
     thrust::counting_iterator<difference_type>(input_size),
     output_offsets.begin(),
     first1,
     output_indices.begin());

  // compute max-scan over the output indices, filling in the holes
  thrust::inclusive_scan
    (output_indices.begin(),
     output_indices.end(),
     output_indices.begin(),
     thrust::maximum<difference_type>());

  // gather input values according to index array (output = first2[output_indices])
  thrust::gather(output_indices.begin(),
                 output_indices.end(),
                 first2,
                 output);

  // return output + output_size
  thrust::advance(output, output_size);
  return output;
}

/////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void print_vector(T& vec) {
  for (const auto& elem : vec) {
    std::cout << std::setw(5) << elem; 
  }
  std::cout << std::endl;
}

struct myOp : public thrust::unary_function<thrust::tuple<double,double,int,int,int,double>, double> {
                                  // vals   data   1/K 1%K nums crit
  __host__ __device__             // 0      1      2   3   4    5
    double operator() (const thrust::tuple<double,double,int,int,int, double> &t) const {
      double res;

      if (thrust::get<2>(t) >= thrust::get<4>(t)) {
        res = thrust::get<0>(t);  // do nothing
      }else {
        if (thrust::get<3>(t) >= thrust::get<4>(t)) {
          res = thrust::get<0>(t); // do nothing
        }else {
          double tmp = thrust::get<0>(t);
          if (thrust::get<1>(t) >= thrust::get<5>(t)) { tmp = 0.0; }
          res = tmp;
        }
      }

      return res;
    }
};

__global__ void myOpKernel(double *vals, double *data, int *nums, double *crit, int N, int K) {
  int index = blockIdx.x*blockDim.x + threadIdx.x;

  if (index >= N) return;

  double _crit = crit[index];
  for (int i=0; i<nums[index]; i++) {
    double _res = vals[index*K + i];
    if (data[index*K + i] >= _crit) { _res = 0.0; }  

    vals[index*K + i] = _res;
  }
}


int main(int argc, char **argv) {
  cudaSetDevice(1);
  //using namespace thrust::placeholders;

  int N = atoi(argv[1]); 
  int K = atoi(argv[2]); 

  std::cout << "N " << N << " K " << K << std::endl;

  thrust::host_vector<double> h_vals(N*K);
  thrust::host_vector<double> h_data(N*K);
  thrust::host_vector<double> h_crit(N);
  thrust::host_vector<int>    h_nums(N);

  for (int i=0; i<N; i++) {
    h_crit[i] = 101.0; // arbitrary
    h_nums[i] = 4;   // arbitrary number less than 256
    for (int j=0; j<K; j++) {
        h_vals[i*K + j] = (double)(i*K + j); // arbitrary
        h_data[i*K + j] = (double)(i*K + j); // arbitrary
    }
  }



  cudaEvent_t eventA; cudaEventCreate(&eventA);
  cudaEvent_t eventB; cudaEventCreate(&eventB);
 

//--- 2) kernel
thrust::device_vector<double> vals2 = h_vals;
thrust::device_vector<double> data2 = h_data;
thrust::device_vector<double> crit2 = h_crit;
thrust::device_vector<int>    nums2 = h_nums;

  cudaEventRecord(eventA);

  int Nthreads = 256;
  int Nblocks = (N*K - 1) / Nthreads + 1;
  for(int i=0;i<10;i++){
    myOpKernel<<<Nblocks,Nthreads>>>(vals2.data().get(), data2.data().get(), nums2.data().get(), crit2.data().get(), N, K);
  }

  cudaEventRecord(eventB);
cudaEventSynchronize(eventB);
float myOpKerneltime; cudaEventElapsedTime(&myOpKerneltime, eventA, eventB);
std::cout << "myOpKerneltime " << myOpKerneltime << " ms\n";

  cudaDeviceSynchronize();


  cudaEventDestroy(eventA);
  cudaEventDestroy(eventB);

  return 0;
}