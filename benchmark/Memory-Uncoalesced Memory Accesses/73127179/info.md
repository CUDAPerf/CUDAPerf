url
https://stackoverflow.com/questions/73127179/how-to-parallelize-evaluation-of-a-function-to-each-element-of-a-matrix-in-cuda
notes
nvcc -o test test.cu
nsys profile --stats=true ./test