#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

template <typename T>
void cpufunction(T *result, T *oldArray, size_t size, T k) {
    for (int i = 2; i < size; i++) {
        result[i] = oldArray[i] + k * result[i - 2];
    }
}

struct scan_op // as per blelloch (1.7)
{
    template <typename T1, typename T2>
    __host__ __device__
    T1 operator()(const T1 &t1, const T2 &t2) {
        T1 ret;
        thrust::get<0>(ret) = thrust::get<0>(t1) * thrust::get<2>(t2) + thrust::get<1>(t1) * thrust::get<4>(t2) + thrust::get<0>(t2);
        thrust::get<1>(ret) = thrust::get<0>(t1) * thrust::get<3>(t2) + thrust::get<1>(t1) * thrust::get<5>(t2) + thrust::get<1>(t2);
        thrust::get<2>(ret) = thrust::get<2>(t1) * thrust::get<2>(t2) + thrust::get<3>(t1) * thrust::get<4>(t2);
        thrust::get<3>(ret) = thrust::get<2>(t1) * thrust::get<3>(t2) + thrust::get<3>(t1) * thrust::get<5>(t2);
        thrust::get<4>(ret) = thrust::get<4>(t1) * thrust::get<2>(t2) + thrust::get<5>(t1) * thrust::get<4>(t2);
        thrust::get<5>(ret) = thrust::get<4>(t1) * thrust::get<3>(t2) + thrust::get<5>(t1) * thrust::get<5>(t2);
        return ret;
    }
};

typedef float mt;
const size_t ds = 1024*1024/8;
const mt k = 1.01;
const int snip = 10;

int main() {
    mt *b1  = new mt[ds]; // b as in blelloch (1.5)
    mt *cr = new mt[ds]; // cpu result
    for (int i = 0; i < ds; i++) { b1[i] = rand() / (float)RAND_MAX; }
    cr[0] = b1[0];
    cr[1] = b1[1];

    // Measure CPU time
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu, 0);

    cpufunction(cr, b1, ds, k);

    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);
    float cpu_time = 0;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);

    for (int i = 0; i < snip; i++) std::cout << cr[i] << ",";
    for (int i = ds - snip; i < ds; i++) std::cout << cr[i] << ",";
    std::cout << std::endl;

    thrust::device_vector<mt> db(b1, b1 + ds);
    auto b0 = thrust::constant_iterator<mt>(0);
    auto a0 = thrust::constant_iterator<mt>(0);
    auto a1 = thrust::constant_iterator<mt>(1);
    auto a2 = thrust::constant_iterator<mt>(k);
    auto a3 = thrust::constant_iterator<mt>(0);
    thrust::device_vector<mt> dx1(ds);
    thrust::device_vector<mt> dx0(ds);
    thrust::device_vector<mt> dy0(ds);
    thrust::device_vector<mt> dy1(ds);
    thrust::device_vector<mt> dy2(ds);
    thrust::device_vector<mt> dy3(ds);
    auto my_i_zip = thrust::make_zip_iterator(thrust::make_tuple(db.begin(), b0, a0, a1, a2, a3));
    auto my_o_zip = thrust::make_zip_iterator(thrust::make_tuple(dx1.begin(), dx0.begin(), dy0.begin(), dy1.begin(), dy2.begin(), dy3.begin()));

    // Measure GPU time
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu, 0);

    thrust::inclusive_scan(my_i_zip, my_i_zip + ds, my_o_zip, scan_op());

    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    thrust::host_vector<mt> hx1 = dx1;
    thrust::copy_n(hx1.begin(), snip, std::ostream_iterator<mt>(std::cout, ","));
    thrust::copy_n(hx1.begin() + ds - snip, snip, std::ostream_iterator<mt>(std::cout, ","));
    std::cout << std::endl;

    // Print the times
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    // Cleanup
    delete[] b1;
    delete[] cr;
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}