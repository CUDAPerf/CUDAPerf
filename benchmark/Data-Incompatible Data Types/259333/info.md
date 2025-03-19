url
https://forums.developer.nvidia.com/t/intermediate-multiplications-downgrades-warp-shuffling-performance/259333/2
notes
nvcc -o test test.cu
nsys profile --stats=true ./test