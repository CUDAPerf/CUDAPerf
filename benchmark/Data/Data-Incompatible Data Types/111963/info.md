url
https://forums.developer.nvidia.com/t/seemingly-insignificant-changes-result-in-a-100x-kernel-slowdown/111963
notes
nvcc -o test test.cu
nsys profile --stats=true ./test