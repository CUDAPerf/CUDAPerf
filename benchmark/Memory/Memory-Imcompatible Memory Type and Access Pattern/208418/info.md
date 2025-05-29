url
https://forums.developer.nvidia.com/t/implement-2d-matrix-transpose-using-warp-shuffle-without-local-memory/208418
notes
nvcc -o test test.cu
nsys profile --stats=true ./test 1024