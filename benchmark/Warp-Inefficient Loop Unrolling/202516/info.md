url
https://forums.developer.nvidia.com/t/significant-speedup-of-opencl-vs-cuda/202516
notes
nvcc -o test test.cu
nsys profile --stats=true ./test