url
https://stackoverflow.com/questions/75831866/cuda-kernel-10x-slower-when-operating-on-cudamallocmanaged-memory-even-when-pref
notes
nvcc -o test test.cu / nvcc -o test_managed test.cu -DUSE_MANAGED
./test
test() is an incorrect mix of managed memory and memory. Compiling with nvcc - o test test.cu means using CUDAMalloc for all, while compiling with nvcc - o test_managed test.cu - DUSE-MANAGED means using CUDAMallocManaged for all