url
https://forums.developer.nvidia.com/t/loop-unroll-remainder-perf/209443/5
notes
nvcc -o test test.cu
nsys profile --stats=true ./test