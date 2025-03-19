url
https://forums.developer.nvidia.com/t/optimizing-memory-coalescence-doesnt-make-my-program-faster/185659
notes
nvcc -o test test.cu
nsys profile --stats=true ./test