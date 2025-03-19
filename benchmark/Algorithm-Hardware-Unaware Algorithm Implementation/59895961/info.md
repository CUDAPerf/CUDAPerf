url
https://stackoverflow.com/questions/59895961/cuda-xor-single-bitset-with-array-of-bitsets
notes
nvcc -o test test.cu
nsys profile --stats=true ./test
kernelXOR2 compare to kernelXOR: Warp-Improper Kernel Launch Paramters
kernelXOR_imp compare to kernelXOR: Algorithm-Hardware-Unaware Algorithm Implementation