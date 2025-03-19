url
https://stackoverflow.com/questions/76269455/further-chance-of-optimization-of-thrust-operation-of-cuda-kernel
notes
nvcc --extended-lambda -O3 test.cu -o test
nsys profile --stats=true ./test  256