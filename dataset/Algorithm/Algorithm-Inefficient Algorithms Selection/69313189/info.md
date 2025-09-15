url
https://stackoverflow.com/questions/69313189/cuda-array-reduction-optimisation
notes
nvcc -o test test.cu
nsys profile --stats=true ./test
CUDA Simple atomicAdd compare to CUDA Custom reduce: Algorithm-Inefficient Algorithms Selection
CUDA Thrust reduce compare to CUDA Custom reduce: API-Efficient APIs Not Used