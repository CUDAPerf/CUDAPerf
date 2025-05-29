url
https://stackoverflow.com/questions/73155788/efficient-reallocation-of-cuda-memory
notes
nvcc -o test test.cu
nsys profile --stats=true ./test