url
https://forums.developer.nvidia.com/t/cuda-vs-vulkan-performance-difference/238633
notes
nvcc -o test test.cu
nsys profile --stats=true ./test