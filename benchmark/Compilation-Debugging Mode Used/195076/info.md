url
https://forums.developer.nvidia.com/t/terrible-performance-from-very-simple-kernel/195076
notes
nvcc -o withG -G test.cu
nvcc -o noG test.cu
nsys profile --stats=true ./withG  
nsys profile --stats=true ./noG
