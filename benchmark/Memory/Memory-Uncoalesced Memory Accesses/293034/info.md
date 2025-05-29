url
https://forums.developer.nvidia.com/t/strided-vs-non-strided-access/293034
notes
nvcc -o test test.cu
nsys profile --stats=true ./test