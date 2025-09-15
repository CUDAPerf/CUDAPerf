url
https://forums.developer.nvidia.com/t/optimization-opportunity-for-large-vector-access/217147
notes
nvcc -o test test.cu
nsys profile --stats=true ./test