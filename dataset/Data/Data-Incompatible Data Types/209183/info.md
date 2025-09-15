url
https://forums.developer.nvidia.com/t/is-it-possible-to-replace-integer-division-by-floating-point-division-for-speed/209183/4
notes
nvcc -o test test.cu
nsys profile --stats=true ./test