url
https://forums.developer.nvidia.com/t/why-does-the-persistent-thread-approach-reduce-kernel-performance/261299
notes
nvcc -o test test.cu -lcuda
nsys profile --stats=true ./test
Method 6 compare to Method 1: API-Efficient APIs Not Used
Method 2 compare to Method 1: Memory-Uncoalesced Memory Accesses