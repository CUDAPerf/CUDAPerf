url
https://stackoverflow.com/questions/76021386/memory-accesses-make-a-cuda-kernel-extremely-slow
notes
nvcc -o test test.cu -arch=sm_86
nsys profile --stats=true ./test