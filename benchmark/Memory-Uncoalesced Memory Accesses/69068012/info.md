url
https://stackoverflow.com/questions/69068012/why-my-vectorized-access-kernel-is-so-slow
notes
nvcc -o test test.cu
nsys profile --stats=true ./test
veccopy2 compare to naivecopy: Memory-Uncoalesced Memory Accesses
veccopy2 compare to veccopy: Warp-Improper Kernel Launch Paramters