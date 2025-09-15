url
https://forums.developer.nvidia.com/t/cublas-matrix-multiplication-is-slower-than-the-naive-one/263823
notes
nvcc -o test test.cu  -lcublas
nsys profile --stats=true ./test
testTiledMatrixMulKernel compare to testMatrixMulKernel: Algorithm-Hardware-Unaware Algorithm Implementation
testCuBLASMatrixMulKernel compare to testMatrixMulKernel: API-Efficient APIs Not Used
