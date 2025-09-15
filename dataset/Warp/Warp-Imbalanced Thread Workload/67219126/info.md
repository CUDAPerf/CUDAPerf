url
https://stackoverflow.com/questions/67219126/how-to-use-register-memory-for-each-thread-in-cuda
notes
nvcc -o test test.cu
nsys profile --stats=true ./test