#include<chrono>
#include<iostream>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/reduce.h>


int N = 1024;
int M = 1024;

int before()
{
    thrust::device_vector<float> D(N*M,5.0);
    int sum;
    
    auto start = std::chrono::high_resolution_clock::now();
    sum = thrust::reduce(D.begin(),D.end(),(float)0,thrust::plus<float>());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);

    std::cout<<duration.count()<<"μs  ";
    std::cout<<sum<<std::endl;
    return 0;
}


int main(){
    before();
}
