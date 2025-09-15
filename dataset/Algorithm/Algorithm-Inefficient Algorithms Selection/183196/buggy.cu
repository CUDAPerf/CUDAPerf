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
  return 0;
}

int main(){
  for(int i=0;i<10;i++){
    test();
  }
}