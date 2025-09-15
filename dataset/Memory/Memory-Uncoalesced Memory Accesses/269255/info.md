url
https://forums.developer.nvidia.com/t/converting-a-kernel-from-floats-and-ints-to-halfs-is-6x-slower/269255
notes
nvcc -o test test.cu -arch=sm_86 (You need to change - arch to the architecture of your GPU device)
nsys profile --stats=true ./test
kernel2 compare to kernel1: Data-Incompatible Data Types
kernel3 compare to kernel1: Memory-Uncoalesced Memory Accesses