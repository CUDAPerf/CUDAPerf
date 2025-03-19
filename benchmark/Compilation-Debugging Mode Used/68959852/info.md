url
https://stackoverflow.com/questions/68959852/why-these-two-gpu-kernel-have-massive-performance-difference
notes
nvcc -o withG -G test.cu
nvcc -o noG test.cu
nsys profile --stats=true ./withG  
nsys profile --stats=true ./noG
