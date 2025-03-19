url
https://forums.developer.nvidia.com/t/neural-network-code-optimization/175678
notes
nvcc -o test test.cu
nsys profile --stats=true ./test