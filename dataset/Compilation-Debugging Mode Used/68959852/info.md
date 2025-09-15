url
https://stackoverflow.com/questions/68959852/why-these-two-gpu-kernel-have-massive-performance-difference
notes
The buggy and fixed version code of the CUDA program are the same, only the difference lies in whether the - G tag is used during compilation
nvcc -o buggy -G buggy.cu
nvcc -o fixed fixed.cu
nsys profile --stats=true ./buggy  
nsys profile --stats=true ./fixed
