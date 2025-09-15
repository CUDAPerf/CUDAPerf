url
https://forums.developer.nvidia.com/t/2d-n-body-simulation-optimization/111215
notes
nvcc -o test test.cu
nsys profile --stats=true ./test
compute1_1 compare to compute1: Data-Incompatible Data Types
compute2 compare to compute1_1: Warp-Inefficient Loop Unrolling