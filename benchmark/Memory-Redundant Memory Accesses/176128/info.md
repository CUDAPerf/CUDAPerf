url
https://forums.developer.nvidia.com/t/curand-my-implementation-works-but-i-am-not-sure-its-the-right-way-to-do-it/176128
notes
nvcc -o test test.cu  -lcurand
nsys profile --stats=true ./test