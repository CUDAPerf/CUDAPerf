#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}
const int MY_MAX = 1000;
//const int MY_MIN = 0;
#define LOGLEN_MAX 150
__constant__ int constant_nLogLen;
__constant__ int constant_inArray[LOGLEN_MAX];

__global__ void calculateMin(
    int* dev_outMatrix)
{
    int length = threadIdx.x;
    int startIndex = blockIdx.x;
    int logLen = constant_nLogLen;

    if (startIndex + length >= logLen || startIndex >= logLen || length >= logLen)
    {
        return;
    }

    int min = MY_MAX;
    for (int j = startIndex; j <= startIndex + length;j++)
    {
        if (constant_inArray[j] < min)
        {
            min = constant_inArray[j];
        }
    }

    dev_outMatrix[startIndex*logLen + length+startIndex] = min;
}

template <typename T>
void cpu_test(T *in_array, T *out_matrix, T *out_max, int length){

//Loop 1
for (int start= 0;  start < length;  ++start )
{
        //Loop 2
        for (int end= 0;  end< length;  ++end)
        {
                T nMin = MY_MAX;
//              T nMax = MY_MIN;
                //Loop 3
                for (int n = start;  n <= end;  ++n)
                {
                        if (in_array[n] < nMin)
                        {
                                nMin = in_array[n];
                        }
                                out_matrix[start*length+end] = nMin;
                }
}
}
}
// assumes block consists of an even number of threads
template <typename T>
__global__ void gpu_test(const T * __restrict__ in, T * __restrict__ out_min, T * __restrict__ out_max){

  __shared__ T s[LOGLEN_MAX*LOGLEN_MAX/2];
  const int t = threadIdx.x;
  T my_val = in[t];
  const int my_l = blockDim.x;
  const int my_half = my_l>>1;
  const int my_l_m1 = my_l-1;
  int my_oidx = (blockIdx.x*my_l+my_l_m1)*my_l + t;
  int my_sidx = ((t - (t >= my_half)*my_half)*my_l)+t;
  s[my_sidx] = my_val;
  out_min[my_oidx] = (t == (my_l_m1))*my_val;
  for (int i = 1; i < my_l; i++){
    my_oidx -= my_l;
    my_sidx -= my_l;
    if (my_sidx < 0) my_sidx += my_half*my_l;
    int my_osidx = t+(my_l_m1-i-((i < my_half)*my_half))*my_l;
    __syncthreads();
    if (t >= i)
      s[my_sidx] = my_val = min(my_val,s[my_sidx-1]);
    out_min[my_oidx] = (t > (my_l_m1-i-1))*s[my_osidx];
  }
}

typedef int mt;
int test(){
    const int nblk = 16;
    const int length= 150;
    mt out_matrix[length][length];
    mt in_array[length];
    mt *out = &(out_matrix[0][0]);
    for (int i = 0; i < length; i++){
            in_array[i] = (rand()%(MY_MAX-1))+1;
            for (int j = 0; j < length; j++) out_matrix[i][j] = 0;}

    unsigned long long dt = dtime_usec(0);
    for (int i = 0; i < nblk; i++) cpu_test(in_array, out, out, length);
    dt = dtime_usec(dt);
    std::cout << "cpu time: " << dt << "us" << std::endl;
    cudaMemcpyToSymbol(constant_inArray, in_array, length*sizeof(mt));
    cudaMemcpyToSymbol(constant_nLogLen, &length, sizeof(mt));
    mt *d_out, *h_out;
    cudaMalloc(&d_out, nblk*length*length*sizeof(mt));
    h_out = new mt[length*length];
    cudaMemset(d_out, 0, length*length*sizeof(mt));
    for (int i = 0; i < nblk; i++)
      calculateMin<<<length, length>>>(d_out);
    cudaMemcpy(h_out, d_out, length*length*sizeof(mt), cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++){
      for (int j = 0; j < length; j++)
         if (h_out[i*length+j] != out[i*length+j]) {std::cout << "mismatch0 at: " << i << "," << j << " was: " << h_out[i*length+j] << " should be: " << out[i*length+j] <<  std::endl; return 0;}
    }
    mt *d_in;
    cudaMalloc(&d_in, length*sizeof(mt));
    cudaMemcpy(d_in, in_array, length*sizeof(mt), cudaMemcpyHostToDevice);
    gpu_test<<<nblk, length>>>(d_in, d_out, d_out);
    cudaMemcpy(h_out, d_out, length*length*sizeof(mt), cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++)
      for (int j = 0; j < length; j++)
         if (h_out[i*length+j] != out[i*length+j]) {std::cout << "mismatch1 at: " << i << "," << j << " was: " << h_out[i*length+j] << " should be: " << out[i*length+j] <<  std::endl; return 0;}
  return 0;
}

int main(){
  for(int i=0;i<10;i++){
    test();
  }
}