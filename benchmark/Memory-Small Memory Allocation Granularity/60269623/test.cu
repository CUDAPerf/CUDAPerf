#include <iostream>
#include <cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define N 102400
#define ARR_SZ 100

struct Struct
{
    float* arr;
};

int before()
{
    Struct* struct_arr;

    size_t free_mem, total_mem;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Used memory: " << (double)(total_mem - free_mem) / (1024 * 1024) << " MB" << std::endl;
    double mem1 = (double)(total_mem - free_mem);

    gpuErrchk( cudaMallocManaged((void**)&struct_arr, sizeof(Struct)*N) );
    for(int i = 0; i < N; ++i)
        gpuErrchk( cudaMallocManaged((void**)&(struct_arr[i].arr), sizeof(float)*ARR_SZ) ); //out of memory...

    status = cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Used memory: " << (double)(total_mem - free_mem) / (1024 * 1024) << " MB" << std::endl;
    double mem2 = (double)(total_mem - free_mem);
    std::cout<<"Memory used before: "<<(mem2-mem1)/1024<<"KB"<<std::endl;

    for(int i = 0; i < N; ++i)
        cudaFree(struct_arr[i].arr);
    cudaFree(struct_arr);

    /*float* f;
    gpuErrchk( cudaMallocManaged((void**)&f, sizeof(float)*N*ARR_SZ) ); //this works ok
    cudaFree(f);*/

    return 0;
}

int after(){
    Struct* struct_arr;
    float* f;

    size_t free_mem, total_mem;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Used memory: " << (double)(total_mem - free_mem) / (1024 * 1024) << " MB" << std::endl;
    double mem1 = (double)(total_mem - free_mem);
    gpuErrchk( cudaMallocManaged((void**)&struct_arr, sizeof(Struct)*N) );
    gpuErrchk( cudaMallocManaged((void**)&f, sizeof(float)*N*ARR_SZ) );
    for(int i = 0; i < N; ++i)
        struct_arr[i].arr = f+i*ARR_SZ;

    status = cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Used memory: " << (double)(total_mem - free_mem) / (1024 * 1024) << " MB" << std::endl;
    double mem2 = (double)(total_mem - free_mem);
    std::cout<<"Memory used after: "<<(mem2-mem1)/1024<<"KB"<<std::endl;

    cudaFree(struct_arr);
    cudaFree(f);

    return 0;
}

int main(){
    cudaSetDevice(0);
    before();
    after();
}