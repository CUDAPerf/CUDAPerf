url
https://stackoverflow.com/questions/76745515/which-of-the-following-approaches-is-more-suitable-for-cuda-parallelism
notes
nvcc -o test test.cu
nsys profile --stats=true ./test
dotProductKernel compare to dotProductKernel1: Warp-Improper Kernel Launch Paramters
dotProductKernel compare to dotProductKernel2+sumResultKernel: Warp-Imbalanced Thread Workload
