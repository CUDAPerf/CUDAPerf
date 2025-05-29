url
https://forums.developer.nvidia.com/t/porting-a-complex-polynomial-root-solver-to-cuda-optimizing-kernel-performance/201840
notes
nvcc -o test test.cu
nsys profile --stats=true ./test
Approach1 compare to Approach0: Data-Incompatible Data Types
Approach2 compare to Approach1: Memory-Uncoalesced Memory Accesses
Approach3 compare to Approach2: Warp-Imbalanced Thread Workload