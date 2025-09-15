#include <cstdlib>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>

const int N = 1024;

struct scan_op {
    template <typename T1, typename T2>
    __host__ __device__
    T1 operator()(const T1 &t1, const T2 &t2) {
        T1 ret;
        thrust::get<0>(ret) = thrust::get<0>(t1) * thrust::get<0>(t2);
        thrust::get<1>(ret) = thrust::get<1>(t1) * thrust::get<0>(t2) + thrust::get<1>(t2);
        return ret;
    }
};

using mt = double;
using namespace thrust::placeholders;

int main() {
    //cudaSetDevice(1);
    mt *h_w, *d_w, *h_b, *d_b, *h_r;
    h_w = new mt[N];
    h_r = new mt[N];
    h_b = new mt[N];

    cudaMalloc(&d_b, N * sizeof(d_b[0]));
    cudaMalloc(&d_w, N * sizeof(d_w[0]));
    for (int i = 0; i < N; i++) {
        h_w[i] = rand() / (double)RAND_MAX;
        h_b[i] = rand() / (double)RAND_MAX;
    }
    cudaMemcpy(d_b, h_b, N * sizeof(d_b[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, N * sizeof(d_w[0]), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Thrust
    thrust::device_vector<mt> db(h_b, h_b + N);
    thrust::device_vector<mt> da(h_w, h_w + N);
    thrust::transform(da.begin(), da.end(), da.begin(), _1 * (-1));
    thrust::device_vector<mt> dy(N);
    thrust::device_vector<mt> dx(N);

    cudaEventRecord(start);
    thrust::inclusive_scan(
        thrust::make_zip_iterator(thrust::make_tuple(da.begin(), db.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(da.end(), db.end())),
        thrust::make_zip_iterator(thrust::make_tuple(dy.begin(), dx.begin())),
        scan_op()
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Thrust execution time: " << milliseconds << " ms" << std::endl;

    thrust::host_vector<mt> hx = dx;
    thrust::copy_n(hx.begin(), 8, std::ostream_iterator<mt>(std::cout, ","));
    std::cout << std::endl;

    // free
    cudaFree(d_b);
    cudaFree(d_w);
    delete[] h_w;
    delete[] h_b;
    delete[] h_r;

    return 0;
}