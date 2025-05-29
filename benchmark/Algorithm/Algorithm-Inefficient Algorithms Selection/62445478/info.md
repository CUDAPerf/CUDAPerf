url
https://stackoverflow.com/questions/62445478/how-to-do-the-modulus-of-complex-number-more-efficiently-in-cuda
notes
nvcc -o test test.cu
./test
modulus_kernel1 compare to modulus_kernel: Warp-Redundant Warp Synchronization
modulus_kernel2 compare to modulus_kernel1: Algorithm-Inefficient Algorithms Selection
modulus_kernel3 compare to modulus_kernel2: API-Inefficient API Usage