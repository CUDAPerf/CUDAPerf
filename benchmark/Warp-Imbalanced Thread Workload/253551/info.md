url
https://forums.developer.nvidia.com/t/how-to-use-more-efficiently-the-shared-memory-and-2d-tiles/253551
notes
nvcc -o test test.cu
nsys profile --stats=true ./test