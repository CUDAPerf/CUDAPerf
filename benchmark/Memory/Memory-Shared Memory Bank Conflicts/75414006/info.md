url
https://stackoverflow.com/questions/75414006/shared-memory-read-is-slower-with-1d-vs-2d-indexing-in-cuda
notes
nvcc -o test test.cu
nsys profile --stats=true ./test