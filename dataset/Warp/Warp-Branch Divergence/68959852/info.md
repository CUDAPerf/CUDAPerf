url
https://stackoverflow.com/questions/68959852/why-these-two-gpu-kernel-have-massive-performance-difference
notes
nvcc -o test test.cu
nsys profile --stats=true ./test