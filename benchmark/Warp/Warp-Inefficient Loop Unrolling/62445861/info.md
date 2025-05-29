url
https://stackoverflow.com/questions/62445861/cuda-kernel-performance-drops-by-10x-when-increased-loop-count-by-10
notes
nvcc -o test test.cu
nsys profile --stats=true ./test