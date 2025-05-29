url
https://forums.developer.nvidia.com/t/kernel-is-slower-after-using-warp-shuffles/286090
notes
nvcc -o test test.cu
nsys profile --stats=true ./test